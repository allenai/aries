import collections
import json
import logging
import os
import sys

import numpy as np
import sklearn.exceptions
import sklearn.metrics
import transformers

from aries.util.data import index_by
from aries.util.logging import pprint_metrics

logger = logging.getLogger(__name__)


class AlignerEvalCallback(transformers.TrainerCallback):
    def __init__(self, config, model, eval_records, model_selection_metric_fn=None, model_to_save=None):
        self.config = config
        self.model = model
        self.model_to_save = model_to_save or model.model
        self.eval_records = eval_records
        # self.eval_precached_dataset = self.model._make_dataset(self.eval_records)
        self.eval_precached_dataset = None

        self.model_selection_metric_fn = model_selection_metric_fn
        if isinstance(model_selection_metric_fn, str):
            self.model_selection_metric_fn = lambda x: x[model_selection_metric_fn]

        self._best_metric_val = float("-inf")
        self._best_model_state = None

    @staticmethod
    def _clone_cpu_model_state_dict(model):
        return collections.OrderedDict((k, v.clone().cpu().detach()) for k, v in model.state_dict().items())

    def on_evaluate(self, args, state, control, **kwargs):
        metrics, all_results, _ = do_model_eval(self.model, self.eval_records, eval_precached_dataset=self.eval_precached_dataset)

        if self.config.get("write_examples_on_eval", False):
            with open(os.path.join(self.config["output_dir"], "{}_inferences.jsonl".format("tmp_mid_eval")), "w") as f:
                for res in all_results:
                    f.write(json.dumps(res) + "\n")

        pprint_metrics(metrics, logger, name="dev (mid-train)")
        metrics["global_step"] = state.global_step
        metrics["epoch"] = state.epoch
        metrics["total_flos"] = state.total_flos
        with open(os.path.join(self.config["output_dir"], "{}_metrics.jsonl".format("mid_eval")), "a") as f:
            f.write(json.dumps(metrics) + "\n")

        if self.model_selection_metric_fn is not None:
            metric_val = self.model_selection_metric_fn(metrics)
            if metric_val > self._best_metric_val:
                logger.info(
                    "Got new best model at global step {} (epoch {}, {:0.2f} TFLOs)".format(state.global_step, state.epoch, state.total_flos / 1e12)
                )
                state.best_metric = metric_val
                self._best_metric_val = metric_val
                self._best_model_state = AlignerEvalCallback._clone_cpu_model_state_dict(self.model_to_save)


def _get_possible_optimal_thresholds(all_candidates):
    return _get_possible_optimal_thresholds_smart(all_candidates)


def _get_possible_optimal_thresholds_smart(all_candidates):
    """Gets the thresholds that have a chance of maximizing f1; that is,
    thresholds at positive-negative boundaries (in descending order of score)
    and thresholds at the extremes."""
    # Sort by descending score
    all_scored_candidates = sorted([x for x in all_candidates if x["score"] is not None], key=lambda x: x["score"], reverse=True)

    if len(all_scored_candidates) == 0:
        return []

    # return list(range(min(x['score'] for x in all_scored_candidates), max(x['score'] for x in all_scored_candidates), 0.05))

    # The possible thresholds should be the midpoints between each pos-label score and the next-lowest-scoring point, plus the endpoints
    possible_thresholds = []
    possible_thresholds.append(all_scored_candidates[0]["score"] + 0.0001)
    possible_thresholds.append(all_scored_candidates[-1]["score"] - 0.0001)
    # We only need to consider pos-neg boundaries; if there is a run of
    # consecutive positive examples, it is never worse to include all of them.
    for candidx in range(len(all_scored_candidates)):
        cand0 = all_scored_candidates[candidx - 1]
        cand1 = all_scored_candidates[candidx]
        if cand0["label"] == 1 and cand1["label"] == 0:
            thresh = (cand0["score"] + cand1["score"]) / 2
            if thresh not in possible_thresholds:
                possible_thresholds.append(thresh)

    return possible_thresholds


def get_pred_labels_for_threshold(thresh, all_candidates, rel_thresh=0.2):
    pred_labels = []
    for x in all_candidates:
        if "score" not in x or x.get("base_pred", None) == 0:
            pred_labels.append(x["pred"])
        elif "best_group_score" in x:
            pred_labels.append(1 if x["score"] > thresh and x["score"] >= (x["best_group_score"] - rel_thresh) else 0)
        else:
            pred_labels.append(1 if x["score"] > thresh else 0)
    return pred_labels


def tune_optimal_f1_threshold(all_candidates):
    """Find the absolute decision threshold that maximizes F1."""
    if len(all_candidates) == 0:
        return None, []

    possible_thresholds = _get_possible_optimal_thresholds(all_candidates)

    if len(possible_thresholds) == 0:
        logger.info("Couldn't get optimal threshold because there were no scores on positive examples")
        return None, [x["pred"] for x in all_candidates]
    possible_thresholds = sorted(possible_thresholds)

    true_labels = [x["label"] for x in all_candidates]
    best = (-float("inf"), None, None)
    for thresh in possible_thresholds:
        pred_labels = get_pred_labels_for_threshold(thresh, all_candidates)

        f1 = sklearn.metrics.f1_score(true_labels, pred_labels)
        if f1 > best[0]:
            best = (f1, thresh, pred_labels)

    return best[1], best[2]


def full_tune_optimal_thresholds(all_candidates, min_recall=None, num_abs_thresholds=100, num_rel_thresholds=100, abs_thresh=None, rel_thresh=None):
    """Find the combination of absolute and relative decision thresholds that
    maximize F1.  If abs_thresh or rel_thresh are set, only the other one will
    be tuned. However, note that this is less efficient and precise than
    tune_optimal_f1_threshold if only the absolute threshold needs to be tuned.
    To tune the relative threshold, records in all_candidates must have
    a "best_group_score" field set."""

    if len(all_candidates) == 0:
        return None, None, []

    if abs_thresh is not None and rel_thresh is not None:
        raise ValueError("Cannot specify both abs_thresh and rel_thresh")

    possible_abs_threshs = [abs_thresh]
    possible_rel_threshs = [rel_thresh]

    if abs_thresh is None:
        # First, find the maximum pred_threshold that achieves the minimum recall
        max_threshold = max(x["score"] for x in all_candidates)
        if min_recall > 0:
            # We can be efficient by just going down the list in score order
            # until we have enough positives (min_recall
            # * num positives in all_candidates)
            all_candidates.sort(key=lambda x: x["score"], reverse=True)
            num_positives = sum(x["label"] == 1 for x in all_candidates)
            num_positives_needed = min_recall * num_positives
            num_positives_found = 0
            for idx, x in enumerate(all_candidates):
                if x["label"] == 1:
                    num_positives_found += 1
                if num_positives_found >= num_positives_needed:
                    max_threshold = x["score"]
                    break
            if num_positives_found < num_positives_needed:
                logger.warning("Unable to find enough positives to achieve tuning_minimum_recall of {}".format(min_recall))
                # We're done; thresholds must be low enough to predict positive for everything
                min_score = min(x["score"] for x in all_candidates)
                max_score = max(x["score"] for x in all_candidates)
                return min_score, (max_score - min_score), [1] * len(all_candidates)
        possible_abs_threshs = np.linspace(0, max_threshold, num_abs_thresholds)

    if rel_thresh is None:
        max_rel_pred_threshold = max(x["score"] for x in all_candidates) - max_threshold
        # Iterate rel thresholds from high to low; if we miss the recall target
        # we can exit early
        possible_rel_threshs = np.linspace(max_rel_pred_threshold, 0, num_rel_thresholds)

    # Now find the combination of pred_threshold and rel_pred_threshold
    # that maximizes f1 while achieving the minimum recall
    best_f1 = 0
    best_thresholds = (0, 0)
    best_pred_labels = []
    for pred_threshold in possible_abs_threshs:
        for rel_pred_threshold in possible_rel_threshs:
            labels = [x["label"] for x in all_candidates]
            pred_labels = get_pred_labels_for_threshold(pred_threshold, all_candidates, rel_pred_threshold)

            recall = sklearn.metrics.recall_score(labels, pred_labels)
            if recall < min_recall:
                break

            f1 = sklearn.metrics.f1_score(labels, pred_labels)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds = (pred_threshold, rel_pred_threshold)
                best_pred_labels = pred_labels

    return best_thresholds[0], best_thresholds[1], best_pred_labels


def group_macro_prf1(labels, preds, group_ids, include_empty=False):
    grouped_comments = {gid: [] for gid in set(group_ids)}
    if not (len(labels) == len(preds)) and (len(labels) == len(group_ids)):
        raise ValueError("need len(labels) ({}) == len(preds) ({}) == len(group_ids) ({})".format(len(labels), len(preds), len(group_ids)))

    if len(labels) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    for idx in range(len(labels)):
        grouped_comments[group_ids[idx]].append((labels[idx], preds[idx]))
    group_prf1s = []
    group_ps = []
    group_rs = []
    group_f1s = []
    group_ems = []
    for gid, group in sorted(grouped_comments.items()):
        labels, preds = list(zip(*group))
        if any(x == 1 for x in preds):
            p = sklearn.metrics.precision_score(labels, preds)
            group_ps.append(p)
        else:
            p = 1
            if include_empty:
                group_ps.append(p)

        if any(x == 1 for x in labels):
            r = sklearn.metrics.recall_score(labels, preds)
            group_rs.append(r)
        else:
            r = 1
            if include_empty:
                group_rs.append(r)

        if any(x == 1 for x in preds) or any(x == 1 for x in labels):
            f1 = sklearn.metrics.f1_score(labels, preds, zero_division="warn")
            group_f1s.append(f1)
        else:
            f1 = 1
            if include_empty:
                group_f1s.append(f1)

        group_ems.append(1 if all(x == y for x, y in zip(labels, preds)) else 0)

        group_prf1s.append(
            (
                p,
                r,
                sklearn.metrics.f1_score(labels, preds, zero_division=1),
            )
        )

    if include_empty:
        pmean, rmean, f1mean = np.mean(np.array(group_prf1s), axis=0).tolist()
    else:
        pmean = np.mean(group_ps).tolist()
        rmean = np.mean(group_rs).tolist()
        f1mean = np.mean(group_f1s).tolist()

    return pmean, rmean, f1mean, np.mean(group_ems).tolist()


def do_model_eval(model, eval_records, eval_precached_dataset=None, custom_decision_threshold=None, custom_threshold_name="custom_threshold"):
    for rec in eval_records:
        rec["candidates"] = rec["positives"] + rec["negatives"] + rec.get("unknowns", [])
        rec["candidate_labels"] = [1] * len(rec["positives"]) + [0] * len(rec["negatives"]) + [None] * len(rec.get("unknowns", []))
    all_results = model.predict_many(eval_records)

    if len(all_results) != len(eval_records):
        raise ValueError("Number of results ({}) does not match number of records ({})".format(len(all_results), len(eval_records)))

    comment2id = dict()
    all_candidates = []
    candidate_comment_ids = []
    for rec in all_results:
        if rec["input_record"]["review_comment"] not in comment2id:
            comment2id[rec["input_record"]["review_comment"]] = len(comment2id)

        for idx, ex in enumerate(rec["predictions"]):
            ex["label"] = rec["input_record"]["candidate_labels"][idx]
            all_candidates.append(ex)
            candidate_comment_ids.append(comment2id[rec["input_record"]["review_comment"]])

    true_labels = [x["label"] for x in all_candidates]

    def metrics_for_predictions(pred_labels, prefix=""):
        nonlocal true_labels
        _, _, _, exactmatch = group_macro_prf1(true_labels, pred_labels, candidate_comment_ids, include_empty=False)
        ie_macro_p, ie_macro_r, ie_macro_f1, _ = group_macro_prf1(true_labels, pred_labels, candidate_comment_ids, include_empty=True)
        metrics = {
            "accuracy": sklearn.metrics.accuracy_score(true_labels, pred_labels),
            "precision": sklearn.metrics.precision_score(true_labels, pred_labels),
            "recall": sklearn.metrics.recall_score(true_labels, pred_labels),
            "f1": sklearn.metrics.f1_score(true_labels, pred_labels),
            "macro_precision": ie_macro_p,
            "macro_recall": ie_macro_r,
            "macro_f1": ie_macro_f1,
            "exact_match": exactmatch,
            "n_pred_positive": sum(1 for x in pred_labels if x == 1),
        }

        return {(prefix + k): v for k, v in metrics.items()}

    metrics = dict()
    pred_labels = [x["pred"] for x in all_candidates]
    metrics.update(metrics_for_predictions(pred_labels, prefix=""))

    optimal_threshold, optimal_pred_labels = tune_optimal_f1_threshold(all_candidates)
    if optimal_threshold is not None:
        logger.info("Got optimal threshold: {:0.3f}".format(optimal_threshold))
        metrics.update(metrics_for_predictions(optimal_pred_labels, prefix="optimal_"))
        metrics["optimal_decision_threshold"] = optimal_threshold

    if custom_decision_threshold is not None:
        custom_pred_labels = get_pred_labels_for_threshold(custom_decision_threshold, all_candidates)
        metrics.update(metrics_for_predictions(custom_pred_labels, prefix=(custom_threshold_name + "_")))
        metrics[(custom_threshold_name + "_decision_threshold")] = custom_decision_threshold

    metrics.update(
        {
            "n_true_positive": sum(1 for x in all_candidates if x["label"] == 1),
            "n_candidates": len(all_candidates),
            "n_comments": len(eval_records),
        }
    )

    serializable_results = []
    for res in all_results:
        sres = dict()
        for k, v in res["input_record"].items():
            try:
                json.dumps(v)
                sres[k] = v
            except TypeError:
                pass

        cands = []
        for pred_rec in res["predictions"]:
            edit = pred_rec["edit"]
            scand = {k: v for k, v in pred_rec.items() if k not in ["edit"]}
            scand["edit_source_idxs"] = edit.source_idxs
            scand["edit_target_idxs"] = edit.target_idxs
            if any(x >= len(edit.doc_edits.s2orc2["pdf_parse"]["body_text"]) and x != 9999 for x in edit.target_idxs):
                raise KeyError(
                    "Out of bounds!  {}  {}  {}  {}".format(
                        edit.doc_edits.s2orc2["paper_id"],
                        len(edit.doc_edits.s2orc2["pdf_parse"]["body_text"]),
                        str(edit.target_idxs),
                        edit.get_target_text(),
                    )
                )
            scand["edit_source_pdf_id"] = edit.doc_edits.s2orc1["paper_id"]
            scand["edit_target_pdf_id"] = edit.doc_edits.s2orc2["paper_id"]
            cands.append(scand)
        sres["candidates"] = cands

        serializable_results.append(sres)

    return (
        metrics,
        serializable_results,
        # pair_results,
        index_by(
            serializable_results,
            lambda x: (x["doc_id"], x["source_pdf_id"], x["target_pdf_id"], x["review_comment"]),
        ),
    )
