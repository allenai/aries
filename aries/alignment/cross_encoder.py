import functools
import logging
import os
import sys

import datasets
import numpy as np
import torch
import tqdm
import transformers

from aries.alignment.eval import AlignerEvalCallback
from aries.util.edit import make_word_diff
from aries.util.training import TrainLoggerCallback

logger = logging.getLogger(__name__)


class PairwiseTransformerAligner:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = self.config["max_seq_length"]

    @staticmethod
    def preprocess_fn(examples, tokenizer, max_length):
        model_inputs = tokenizer(
            examples["first_text"],
            max_length=max_length,
            padding=False,
            truncation=True,
        )
        model_inputs["labels"] = examples["label"]
        return model_inputs

    def _candidate_record_to_input_text(self, rec):
        if self.config["query_input_format"] == "comment_only":
            return rec["review_comment"]
        elif self.config["query_input_format"] == "comment_with_canonical":
            return rec["review_comment"] + "\ncanonicalized: " + rec["canonical"]["canonicalized"]
        elif self.config["query_input_format"] == "reply_comment_or_extracted_comment":
            return rec.get("reply_comment_line", rec["review_comment"])
        elif self.config["query_input_format"] == "reply_comment_or_extracted_comment_with_canonical":
            return rec.get("reply_comment_line", rec["review_comment"]) + "\ncanonicalized: " + rec["canonical"]["canonicalized"]
        elif self.config["query_input_format"] == "comment_with_context":
            comment_str = rec["review_comment"].strip()
            if rec.get("context_side", "none") == "left":
                comment_str = rec["context"].strip() + " " + comment_str
            else:
                comment_str = comment_str + " " + rec["context"].strip()

            return "review comment: " + comment_str
        raise ValueError("Unknown query_input_format {}".format(self.config["query_input_format"]))

    def _edit_to_input_text(self, edit):
        if self.config["edit_input_format"] == "added_tokens":
            return " ".join(edit.get_added_tokens())
        if self.config["edit_input_format"] == "source_text":
            return edit.get_source_text()
        if self.config["edit_input_format"] == "target_text":
            return edit.get_target_text()
        if self.config["edit_input_format"] == "target_text_with_context":
            context = "context: none"
            if len(edit.target_idxs) != 0 and min(edit.target_idxs) != 0:
                context = "context: " + edit.texts2[min(edit.target_idxs) - 1]
            return edit.get_target_text() + "\n\n" + context
        elif self.config["edit_input_format"] == "diff":
            return make_word_diff(
                edit.get_source_text(),
                edit.get_target_text(),
                color_format="none",
            )
        raise ValueError("Unknown edit_input_format {}".format(self.config["edit_input_format"]))

    def _make_example_for_rec_edit(self, rec, edit, label=None):
        query_text = self._candidate_record_to_input_text(rec)
        edit_text = self._edit_to_input_text(edit)
        return {
            "doc_id": rec["doc_id"],
            "source_pdf_id": rec["source_pdf_id"],
            "target_pdf_id": rec["target_pdf_id"],
            "review_comment": rec["review_comment"],
            "first_text": "review comment: {}\n\nparagraph: {}".format(query_text, edit_text),
            "label": label,
        }

    def _make_dataset(self, recs, name="dataset", shuffle=False):
        if isinstance(recs, dict):
            recs = list(recs.values())
        exs = []

        for rec in recs:
            edit_with_labels = []
            edit_with_labels.extend([(x, 1) for x in rec["positives"]])
            edit_with_labels.extend([(x, 0) for x in rec["negatives"]])
            for edit, label in edit_with_labels:
                exs.append(self._make_example_for_rec_edit(rec, edit, label=label))

        tmp = {k: [] for k in exs[0].keys()}
        for ex in exs:
            for k, v in ex.items():
                tmp[k].append(v)
        dset = datasets.Dataset.from_dict(tmp)

        if shuffle:
            dset = dset.shuffle()

        dset = dset.map(
            functools.partial(PairwiseTransformerAligner.preprocess_fn, tokenizer=self.tokenizer, max_length=self.max_length),
            batched=True,
            num_proc=4,
            load_from_cache_file=False,
            desc="Processing {}".format(name),
        )
        return dset

    def train(self, train_recs, dev_recs):
        if len(train_recs) == 0:
            raise ValueError("Got empty train_recs")
        if len(dev_recs) == 0:
            raise ValueError("Got empty dev_recs")

        training_args_dict = transformers.TrainingArguments(output_dir=self.config["output_dir"], log_level="passive").to_dict()
        training_args_dict.update(self.config.get("training_args", dict()))
        training_args = transformers.HfArgumentParser(transformers.TrainingArguments).parse_dict(training_args_dict)[0]

        self.rng = np.random.default_rng(self.config["seed"])
        for rec in train_recs:
            rec["negatives"] = [x for x in rec["negatives"] if x.is_full_addition()]
        train_dset = self._make_dataset(train_recs, shuffle=True)

        self.rng = np.random.default_rng(self.config["seed"])
        dev_dset = self._make_dataset(dev_recs)

        logger.info("{} | {}".format(self.tokenizer.decode(train_dset["input_ids"][0]), self.tokenizer.decode(train_dset["labels"][0])))

        data_collator = transformers.DataCollatorWithPadding(
            self.tokenizer,
            pad_to_multiple_of=None,
        )

        model_selector_callback = AlignerEvalCallback(
            self.config,
            self,
            dev_recs,
            model_selection_metric_fn=lambda x: x["optimal_f1"],
        )

        # TODO: Make training args configurable from model_config
        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dset,
            eval_dataset=dev_dset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[model_selector_callback, TrainLoggerCallback(logger)],
            compute_metrics=None,
        )

        _ = trainer.train()

        self.model.load_state_dict(model_selector_callback._best_model_state)
        self.model.save_pretrained(os.path.join(self.config["output_dir"], "ptmodel"))
        self.tokenizer.save_pretrained(os.path.join(self.config["output_dir"], "ptmodel"))

    def predict_many(self, test_recs):
        was_training = self.model.training
        self.model.eval()

        out_recs = []
        with tqdm.trange(sum(len(x["candidates"]) for x in test_recs), miniters=1, desc="{}.predict_many".format(self.__class__.__name__)) as pbar:
            with torch.no_grad():
                for rec in test_recs:
                    outrec = {
                        "input_record": rec,
                        "predictions": [{"edit": cand, "pred": None, "score": None} for cand in rec["candidates"]],
                    }

                    out_recs.append(outrec)

                    for pred_rec in outrec["predictions"]:
                        tensors = self.tokenizer(
                            self._make_example_for_rec_edit(rec, pred_rec["edit"])["first_text"],
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                        )
                        out = self.model(
                            input_ids=torch.tensor(tensors["input_ids"], device=self.model.device, dtype=torch.long).unsqueeze(0),
                            attention_mask=torch.tensor(tensors["attention_mask"], device=self.model.device, dtype=torch.long).unsqueeze(0),
                        )
                        pred_rec["pred"] = torch.argmax(out.logits, dim=-1)[0].item()
                        pred_rec["score"] = torch.nn.functional.softmax(out.logits, dim=-1)[0].tolist()[1]
                        pred_rec["logits"] = [out.logits[0][0].item(), out.logits[0][1].item()]

                        pbar.update(1)

        self.model.train(was_training)

        return out_recs
