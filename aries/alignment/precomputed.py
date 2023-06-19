import json
import logging
import os
import sys

from aries.util.data import index_by, openc

logger = logging.getLogger(__name__)


class PrecomputedEditsAligner:
    def __init__(self, config):
        self.config = config

    def train(self, train_recs, dev_recs):
        logger.warning("{} doesn't train; ignoring call to train()".format(self.__class__.__name__))

    def predict_many(self, test_recs):
        out_recs = []

        predictions_by_docid = dict()
        with openc(self.config["precomputed_predictions_jsonl_path"], "rt") as f:
            predictions_by_docid = index_by(map(json.loads, f), "doc_id")

        warned_docs = set()
        for rec in test_recs:
            outrec = {
                "input_record": rec,
                "predictions": [{"edit": cand, "pred": None, "score": None} for cand in rec["candidates"]],
            }
            out_recs.append(outrec)

            if rec["doc_id"] not in predictions_by_docid:
                if rec["doc_id"] not in warned_docs:
                    logger.warning("missing prediction for doc: {}".format(rec["doc_id"]))
                    warned_docs.add(rec["doc_id"])
                for cand_rec in outrec["predictions"]:
                    cand_rec["pred"] = 0
                    cand_rec["score"] = 0
                continue

            pred_recs = predictions_by_docid[rec["doc_id"]]
            pred_rec = None
            for rec2 in pred_recs:
                # if rec["review_comment"] == dset_rec["review_comment"]:
                # if rec["review_comment"].strip(".\n ") == rec2["review_comment"].strip(".\n "):
                if rec["review_comment"].strip() == rec2["comment"].strip():
                    pred_rec = rec2
                    break
            if pred_rec is None:
                logger.warning("Missing prediction match for comment: {}".format(rec["review_comment"]))

            for cand_rec in outrec["predictions"]:
                pred_label = 0
                for edit_id in pred_rec["positive_edits"]:
                    if edit_id == cand_rec["edit"].edit_id:
                        pred_label = 1
                        break
                if cand_rec["edit"].is_identical():
                    pred_label = 0
                cand_rec["pred"] = pred_label
                cand_rec["score"] = pred_label

        return out_recs
