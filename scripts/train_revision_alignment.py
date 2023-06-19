import collections
import itertools
import json
import logging
import os
import sys

import numpy as np
import sacrebleu
import torch
import tqdm
import transformers

import aries.util.data
import aries.util.s2orc
from aries.alignment.biencoder import BiencoderTransformerAligner
from aries.alignment.bm25 import BM25Aligner
from aries.alignment.cross_encoder import PairwiseTransformerAligner
from aries.alignment.doc_edits import DocEdits
from aries.alignment.eval import do_model_eval
from aries.alignment.gpt import GptChatAligner, GptChatFullPaperAligner
from aries.alignment.other import MultiStageAligner
from aries.alignment.precomputed import PrecomputedEditsAligner
from aries.util.data import index_by, iter_jsonl_files
from aries.util.logging import init_logging, pprint_metrics

logger = logging.getLogger(__name__)


try:
    # Needed for SLED models
    import sled
except ImportError:
    sled = None


def _load_transformer(config, cls):
    transformer_model = cls.from_pretrained(config["model_name_or_path"])
    if torch.cuda.device_count() > 0:
        transformer_model = transformer_model.to(torch.device("cuda"))
    if config.get("model_adapter", None) is not None:
        logger.info("initializing adapter: {}".format(config["model_adapter"]))
        transformer_model.load_adapter(config["model_adapter"], source="hf", load_as="adapter", set_active=True)
        logger.info(transformer_model.adapter_summary())
    tokenizer = transformers.AutoTokenizer.from_pretrained(config["model_name_or_path"])

    return transformer_model, tokenizer


def init_model_from_config(config):
    model = None
    if config["model_type"] == "cross_encoder":
        transformer_model, tokenizer = _load_transformer(config, transformers.AutoModelForSequenceClassification)
        model = PairwiseTransformerAligner(config, transformer_model, tokenizer)
    elif config["model_type"] == "biencoder":
        transformer_model, tokenizer = _load_transformer(config, transformers.AutoModel)
        model = BiencoderTransformerAligner(config, transformer_model, tokenizer)
    elif config["model_type"] == "gpt":
        model = GptChatAligner(config)
    elif config["model_type"] == "gpt_full_paper":
        model = GptChatFullPaperAligner(config)
    elif config["model_type"] == "bm25":
        model = BM25Aligner(config)
    elif config["model_type"] == "precomputed":
        model = PrecomputedEditsAligner(config)
    else:
        raise ValueError("Unknown model type: {}".format(config["model_type"]))

    return model


def train_eval_main(config, split_data):
    models = []
    # Initial models are treated as pre-processing filters and do not train
    for model_conf in config["model_pipeline"][:-1]:
        model_conf["output_dir"] = config["output_dir"]
        model_conf["seed"] = config["seed"]
        model = init_model_from_config(model_conf)
        logger.info("model_cls: {}".format(str(model.__class__.__name__)))
        models.append(model)

    model_conf = config["model_pipeline"][-1]
    model_conf["output_dir"] = config["output_dir"]
    model_conf["seed"] = config["seed"]
    model = init_model_from_config(model_conf)
    logger.info("main model_cls: {}".format(str(model.__class__.__name__)))
    model.train(split_data["train"], split_data["dev"])
    models.append(model)

    model = MultiStageAligner(config, models)

    eval_splits = ["dev", "test"]
    if config.get("do_final_evaluation_on_train", False):
        eval_splits.append("train")
    dev_threshold = None
    for name in eval_splits:
        recs = split_data[name]

        metrics, all_results, by_review = do_model_eval(model, recs, custom_decision_threshold=dev_threshold, custom_threshold_name="devthresh")
        if name == "dev":
            dev_threshold = metrics.get("optimal_decision_threshold", None)

        logger.info("Done.  Writing output...")
        with open(os.path.join(config["output_dir"], "{}_inferences.jsonl".format(name)), "w") as f:
            for res in all_results:
                f.write(json.dumps(res) + "\n")

        logger.info("Final {} metrics:".format(name))
        pprint_metrics(metrics, logger, name=name)

        with open(os.path.join(config["output_dir"], "{}_metrics.json".format(name)), "w") as f:
            if "bleu" in metrics and isinstance(metrics["bleu"], sacrebleu.metrics.bleu.BLEUScore):
                metrics["bleu"] = metrics["bleu"].score
            json.dump(metrics, f)

        with open(os.path.join(config["output_dir"], "{}_inferences_by_review.jsonl".format(name)), "w") as f:
            for rec in by_review.values():
                f.write(json.dumps(rec) + "\n")


def make_revision_alignment_prediction_data(
    config,
    review_comments_by_doc,
    paper_edits_by_doc,
    edit_labels_file,
    max_negatives,
    negative_sample_method="same_doc",
    hard_negative_ratio=0.0,
    seed=None,
):
    if seed is None:
        seed = config["seed"]

    edit_labels = list(iter_jsonl_files([edit_labels_file]))
    edit_labels_by_doc = index_by(edit_labels, "doc_id")

    all_split_edits = list(
        itertools.chain(*[[(doc_id, x) for x in paper_edits_by_doc[doc_id].paragraph_edits] for doc_id in edit_labels_by_doc.keys()])
    )

    examples = []

    rng = np.random.default_rng(seed)
    for doc_id in edit_labels_by_doc.keys():
        distractor_idxs = rng.choice(
            len(all_split_edits),
            size=min(len(all_split_edits), config["distractor_reservoir_size"]),
            replace=False,
        )
        distractor_pool = [all_split_edits[i][1] for i in distractor_idxs if all_split_edits[i][0] != doc_id]
        doc_examples = get_alignments_for_revision(
            config,
            rng,
            doc_id,
            review_comments_by_doc[doc_id],
            paper_edits_by_doc[doc_id],
            edit_labels=edit_labels_by_doc[doc_id],
            max_negatives=max_negatives,
            distractor_pool=distractor_pool,
            negative_sample_method=negative_sample_method,
            hard_negative_ratio=hard_negative_ratio,
        )
        examples.extend(doc_examples)

    return examples


def _filter_edits_by_type(edit_list, keep_type, min_length=0):
    newlist = None
    if keep_type == "full_additions":
        newlist = [edit for edit in edit_list if edit.is_full_addition()]
    elif keep_type == "diffs":
        newlist = [edit for edit in edit_list if not edit.is_identical()]
    elif keep_type == "source_diffs":
        newlist = [edit for edit in edit_list if (len(edit.source_idxs) != 0 and not edit.is_identical())]
    else:
        raise ValueError("Invalid candidate edit type {}".format(keep_type))

    if min_length > 0:
        newlist = [edit for edit in newlist if len(edit.get_source_text() + edit.get_target_text()) >= min_length]

    return newlist


def get_alignments_for_revision(
    config,
    rng,
    doc_id,
    review_comments,
    edits,
    edit_labels,
    max_negatives=999999,
    distractor_pool=None,
    negative_sample_method="same_doc",
    hard_negative_ratio=0.0,
):
    review_comments_by_id = index_by(review_comments, "comment_id", one_to_one=True)

    examples = []

    for record in edit_labels:
        positives = [edits.by_id(x) for x in record["positive_edits"]]
        positives = _filter_edits_by_type(positives, config["candidate_edit_type"], min_length=config["candidate_min_chars"])
        pos_ids = set([x.edit_id for x in positives])

        if negative_sample_method == "same_doc":
            # Assume all non-positive the paras from the same doc are negatives (appropriate when positives are high-recall)
            negatives = [x for idx, x in enumerate(edits.paragraph_edits) if x.edit_id not in pos_ids]
        elif negative_sample_method == "other_docs":
            # Only sample negatives from edits to other docs (appropriate when positives are low-recall)
            if distractor_pool is None:
                raise ValueError("Need distractor edits from other docs to use other_doc_edits negative sampling")
            negatives = distractor_pool.copy()

        negatives = _filter_edits_by_type(negatives, config["candidate_edit_type"], min_length=config["candidate_min_chars"])

        rng.shuffle(negatives)

        if len(negatives) <= max_negatives:
            final_negatives = negatives
        elif config["hard_negative_strategy"] == "none" or hard_negative_ratio == 0:
            final_negatives = negatives
            if hard_negative_ratio != 0:
                logger.warning(
                    "hard_negative_ratio was {} but hard_negative_strategy is {}; no hard negatives will be used".format(
                        hard_negative_ratio, config["hard_negative_strategy"]
                    )
                )
        else:
            hard_negatives = _get_hard_negatives(negatives, positives, strategy=config["hard_negative_strategy"])[:max_negatives]
            n_hard = min(len(hard_negatives), int(max_negatives * hard_negative_ratio))
            n_easy = max_negatives - n_hard

            # note: Could potentially duplicate an example between easy and
            # hard negatives since hard are just sorted; maybe try to dedup
            final_negatives = negatives[:n_easy] + hard_negatives[:n_hard]

        final_negatives = final_negatives[:max_negatives]

        comment = review_comments_by_id[record["comment_id"]]

        example = {
            "source_pdf_id": edits.s2orc1["paper_id"],
            "target_pdf_id": edits.s2orc2["paper_id"],
            "doc_id": doc_id,
            "comment_id": comment["comment_id"],
            "review_comment": comment["comment"],
            "context": comment["comment_context"],
            "context_side": comment.get("context_side", None),
            "positives": positives,
            "negatives": final_negatives,
        }
        if example["context_side"] is None:
            if example["context"] != "" and example["context"].strip().startswith("["):
                example["context_side"] = "right"
            else:
                example["context_side"] = "left"

        examples.append(example)

    return examples


def _get_hard_negatives(negatives, positives, strategy="none"):
    """Returns the negatives sorted by hardness, and possibly also filtered by hardness"""
    if len(positives) == 0 or strategy == "none":
        return []
    elif strategy == "length":
        pos_lengths = [len(x.get_target_text()) for x in positives]
        return sorted(negatives, key=lambda x: min(abs(len(x.get_target_text()) - pl) for pl in pos_lengths))
    elif strategy == "aggregate_unigram_overlap":
        all_pos_tokens = collections.Counter(itertools.chain(*[x.get_target_text().lower().split() for x in positives]))
        return sorted(
            negatives, key=lambda x: -aries.util.data.counter_jaccard(all_pos_tokens, collections.Counter(x.get_target_text().lower().split()))
        )

    raise ValueError("Unknown strategy {}".format(strategy))


def init_data(config):
    review_comments = list(iter_jsonl_files([config["review_comments_file"]]))
    review_comments_by_doc = index_by(review_comments, "doc_id")

    paper_edits = iter_jsonl_files([config["paper_edits_file"]])

    paper_edits_by_doc = index_by(paper_edits, "doc_id", one_to_one=True)

    for doc_id, s2orc1, s2orc2 in aries.util.s2orc.iter_s2orc_pairs(config["s2orc_base_path"], [x[1] for x in sorted(paper_edits_by_doc.items())]):
        paper_edits_by_doc[doc_id] = DocEdits.from_list(s2orc1, s2orc2, paper_edits_by_doc[doc_id]["edits"])

    all_data = dict()
    all_data["dev"] = make_revision_alignment_prediction_data(
        config,
        review_comments_by_doc,
        paper_edits_by_doc,
        config["dev_edit_labels_file"],
        max_negatives=config.get("dev_max_negatives", config["max_negatives"]),
        seed=config.get("dev_seed", config["seed"]),
        negative_sample_method=config.get("dev_negative_sample_method", config["default_negative_sample_method"]),
    )
    logger.info("dev data size: {}".format(len(all_data["dev"])))

    all_data["test"] = make_revision_alignment_prediction_data(
        config,
        review_comments_by_doc,
        paper_edits_by_doc,
        config["test_edit_labels_file"],
        max_negatives=9999,
        seed=config["seed"],
        negative_sample_method=config.get("test_negative_sample_method", config["default_negative_sample_method"]),
    )
    logger.info("test data size: {}".format(len(all_data["test"])))

    all_data["train"] = make_revision_alignment_prediction_data(
        config,
        review_comments_by_doc,
        paper_edits_by_doc,
        config["train_edit_labels_file"],
        max_negatives=config["max_negatives"],
        seed=config["seed"],
        negative_sample_method=config.get("train_negative_sample_method", config["default_negative_sample_method"]),
    )
    logger.info("train data size: {}".format(len(all_data["train"])))

    return all_data


def augment_config(config):
    config_defaults = {
        "max_negatives": 999999,
        "candidate_edit_type": "diffs",
        "candidate_min_chars": 100,
        "prune_candidates": False,
        "write_examples_on_eval": True,
        "distractor_reservoir_size": 1000,
        "default_negative_sample_method": "same_doc",
        "train_hard_negative_ratio": 0.0,
        "hard_negative_strategy": ("length" if config.get("train_hard_negative_ratio", 0.0) != 0 else "none"),
    }
    for k, v in config_defaults.items():
        config[k] = config.get(k, v)

    NEEDED_KEYS = [
        "dev_edit_labels_file",
        "test_edit_labels_file",
        "train_edit_labels_file",
        "model_pipeline",
        "output_dir",
        "paper_edits_file",
        "review_comments_file",
        "s2orc_base_path",
        "seed",
    ]
    missing_keys = [x for x in NEEDED_KEYS if x not in config]
    if len(missing_keys) > 0:
        raise ValueError("Missing config keys: %s" % missing_keys)


def main():
    with open(sys.argv[1]) as f:
        config = json.load(f)

    augment_config(config)

    os.makedirs(config["output_dir"], exist_ok=True)

    init_logging(
        logfile=os.path.join(config["output_dir"], "logging_output.log"),
        level=logging.INFO,
    )

    transformers.set_seed(config["seed"])

    all_data = init_data(config)
    train_eval_main(config, all_data)


if __name__ == "__main__":
    main()
