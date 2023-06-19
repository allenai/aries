import difflib
import itertools
import json
import logging
import os
import sys

import gensim
import numpy as np
import tqdm

import aries.util.data
import aries.util.edit
import aries.util.gensim
from aries.alignment.eval import full_tune_optimal_thresholds

logger = logging.getLogger(__name__)


class BM25Aligner:
    def __init__(self, config):
        self.config = config

        self.fixed_pred_threshold = self.config.get("fixed_pred_threshold", None)
        self.pred_threshold = self.config.get("fixed_pred_threshold", 0.5)

        self.fixed_rel_pred_threshold = self.config.get("fixed_rel_pred_threshold", None)
        self.rel_pred_threshold = self.config.get("fixed_rel_pred_threshold", 0.2)

        self.tune_on_dev = self.config.get("tune_on_dev", False)
        self.tuning_minimum_recall = self.config.get("tuning_minimum_recall", 0.0)

        self.query_input_format = self.config["query_input_format"]
        self.edit_input_format = self.config["edit_input_format"]

        self.output_dir = self.config.get("output_dir", None)

        # Check for conflicts between tune_on_dev and fixed_*_thresholds
        if self.tune_on_dev and (self.fixed_pred_threshold is not None or self.fixed_rel_pred_threshold is not None):
            logger.warning("tune_on_dev is set to True, but fixed_pred_threshold and/or fixed_rel_pred_threshold are set. Ignoring fixed thresholds.")

        self.bm25_model = None
        self.bm25_index = None
        self.tfidf_model = None

        self.dictionary = None
        if self.config.get("bm25_dictionary", None) is not None:
            logger.info("Loading dictionary from {}".format(self.config["bm25_dictionary"]))
            self.dictionary = gensim.corpora.Dictionary.load(self.config["bm25_dictionary"])

    def _candidate_record_to_input_text(self, rec):
        if self.query_input_format == "comment_only":
            return rec["review_comment"]
        elif self.query_input_format == "comment_with_canonical":
            return rec["review_comment"] + "\ncanonicalized: " + rec["canonical"]["canonicalized"]
        elif self.query_input_format == "reply_comment_or_extracted_comment":
            return rec.get("reply_comment_line", rec["review_comment"])
        elif self.query_input_format == "reply_comment_or_extracted_comment_with_canonical":
            return rec.get("reply_comment_line", rec["review_comment"]) + "\ncanonicalized: " + rec["canonical"]["canonicalized"]
        elif self.query_input_format == "comment_with_context":
            comment_str = rec["review_comment"].strip()
            if rec.get("context_side", "none") == "left":
                comment_str = rec["context"].strip() + " " + comment_str
            else:
                comment_str = comment_str + " " + rec["context"].strip()

            return "review comment: " + comment_str
        raise ValueError("Unknown query_input_format {}".format(self.query_input_format))

    def _edit_to_input_text(self, edit):
        if self.edit_input_format == "added_tokens":
            return " ".join(edit.get_added_tokens())
        if self.edit_input_format == "source_text":
            return edit.get_source_text()
        if self.edit_input_format == "target_text":
            return edit.get_target_text()
        if self.edit_input_format == "target_text_with_context":
            context = "context: none"
            if len(edit.target_idxs) != 0 and min(edit.target_idxs) != 0:
                context = "context: " + edit.texts2[min(edit.target_idxs) - 1]
            return edit.get_target_text() + "\n\n" + context
        elif self.edit_input_format == "diff":
            return aries.util.edit.make_word_diff(
                edit.get_source_text(),
                edit.get_target_text(),
                color_format="none",
            )
        elif self.edit_input_format == "tokens_union":
            text1 = edit.get_source_text()
            text2 = edit.get_target_text()
            textw = text1.split(" ") if len(text1) != 0 else []
            outtextw = text2.split(" ") if len(text2) != 0 else []
            tokens = []
            for idx, x in enumerate(difflib.ndiff(textw, outtextw)):
                tokens.append(x[2:])
            return " ".join(tokens)
        raise ValueError("Unknown edit_input_format {}".format(self.edit_input_format))

    def train(self, train_recs, dev_recs):
        logger.info("Getting corpus statistics from training documents...")
        # Pull the full doc text from the training set
        all_doc_edits = dict()
        for rec in train_recs:
            # We only need one edit to get the DocEdits for the whole doc
            if rec["doc_id"] in all_doc_edits:
                continue
            edits = rec["positives"] + rec["negatives"]
            if len(edits) == 0:
                continue
            all_doc_edits[rec["doc_id"]] = edits[0].doc_edits

        docs = []
        for doc_id, doc_edits in all_doc_edits.items():
            docs.append("\n\n".join([x["text"] for x in doc_edits.s2orc2["pdf_parse"]["body_text"]]))

        corpus = aries.util.gensim.InMemoryTextCorpus(docs, dictionary=self.dictionary)
        self.dictionary = corpus.dictionary

        # Save dictionary
        self.dictionary.save(os.path.join(self.output_dir, "dictionary.pk"))

        # Tune the thresholds, if needed
        if self.tune_on_dev:
            logger.info("Tuning thresholds on dev set...")
            self.pred_threshold, self.rel_pred_threshold = self._tune_thresholds(dev_recs)
            logger.info("Tuned thresholds: pred_threshold={}, rel_pred_threshold={}".format(self.pred_threshold, self.rel_pred_threshold))

        with open(os.path.join(self.output_dir, "thresholds.json"), "w") as f:
            json.dump(
                {
                    "pred_threshold": self.pred_threshold,
                    "rel_pred_threshold": self.rel_pred_threshold,
                },
                f,
            )

    def _tune_thresholds(self, dev_recs):
        eval_records = []
        for rec in dev_recs:
            eval_records.append(
                {
                    "doc_id": rec["doc_id"],
                    "review_comment": rec["review_comment"],
                    "context": rec["context"],
                    "context_side": rec.get("context_side", "none"),
                    "candidates": rec["positives"] + rec["negatives"] + rec.get("unknowns", []),
                    "candidate_labels": [1] * len(rec["positives"]) + [0] * len(rec["negatives"]) + [None] * len(rec.get("unknowns", [])),
                }
            )
        all_results = self.predict_many(eval_records)

        all_candidates = []
        for rec in all_results:
            for idx, ex in enumerate(rec["predictions"]):
                ex["label"] = rec["input_record"]["candidate_labels"][idx]
                all_candidates.append(ex)

        pred_threshold, rel_pred_threshold, _ = full_tune_optimal_thresholds(
            all_candidates,
            min_recall=self.tuning_minimum_recall,
            num_abs_thresholds=20,
            num_rel_thresholds=20,
            abs_thresh=self.fixed_pred_threshold,
            rel_thresh=self.fixed_rel_pred_threshold,
        )
        return pred_threshold, rel_pred_threshold

    def _init_vector_models(self):
        self.bm25_model = gensim.models.OkapiBM25Model(dictionary=self.dictionary)
        self.tfidf_model = gensim.models.TfidfModel(dictionary=self.dictionary, normalize=True, smartirs="bnn")

    def predict_many(self, *args, **kwargs):
        if self.bm25_model is None:
            self._init_vector_models()

        results = self._predict_many(*args, **kwargs)
        return results

    def _predict_many(self, test_recs, quiet=False):
        out_recs = []

        logger.info("Doing inference with pred_threshold={}, rel_pred_threshold={}".format(self.pred_threshold, self.rel_pred_threshold))

        for rec in tqdm.tqdm(test_recs, "predicting", disable=quiet):
            outrec = {
                "input_record": rec,
                "predictions": [{"edit": cand, "pred": None, "score": None} for cand in rec["candidates"]],
            }
            out_recs.append(outrec)

            if len(outrec["predictions"]) == 0:
                continue

            candidate_texts = [self._edit_to_input_text(x["edit"]) for x in outrec["predictions"]]
            corpus = aries.util.gensim.InMemoryTextCorpus(candidate_texts, dictionary=self.dictionary)

            query_vec = self.tfidf_model[self.dictionary.doc2bow(corpus.preprocess_text(self._candidate_record_to_input_text(rec)))]

            candidate_vectors = self.bm25_model[list(corpus)]
            bm25_index = gensim.similarities.SparseMatrixSimilarity(None)
            bm25_index.normalize = True
            bm25_index.index = gensim.matutils.corpus2csc(candidate_vectors, num_docs=len(corpus), num_terms=len(self.dictionary), dtype=float).T

            cosine_similarities = bm25_index[query_vec].tolist()

            best_candidxs = np.argsort(cosine_similarities).tolist()
            best_candidx_score = cosine_similarities[best_candidxs[-1]]

            for candidx, predr in enumerate(outrec["predictions"]):
                predr["best_group_score"] = best_candidx_score
                predr["cosine_score"] = cosine_similarities[candidx]

                predr["pred"] = (
                    1
                    if cosine_similarities[candidx] > self.pred_threshold
                    and cosine_similarities[candidx] >= (best_candidx_score - self.rel_pred_threshold)
                    else 0
                )
                predr["score"] = cosine_similarities[candidx]

        return out_recs
