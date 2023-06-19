import logging

logger = logging.getLogger(__name__)


class MultiStageAligner:
    def __init__(self, config, aligners):
        self.config = config
        self.aligners = aligners

        self.prune_candidates = config.get("prune_candidates", False)

    def train(self, train_recs, dev_recs):
        logger.info("Multi-stage aligner doesn't train; skipping...")

    def _update_candidate_scores(self, candidate):
        # Fill base_pred, pred, and score based on the stack of aligner predictions
        candidate["base_pred"] = None
        candidate["pred"] = None
        candidate["score"] = None
        if len(candidate["predictions"]) == 0:
            return

        # If any aligner predicts 0, then the candidate's pred is 0.  The
        # base_pred is 0 if any aligner other than the last one predicts 0 (1 otherwise).
        # The score is the final aligner's score.
        for pred_idx, pred_rec in enumerate(candidate["predictions"]):
            if pred_rec is None:
                continue
            if pred_rec["pred"] == 0:
                if pred_idx < len(candidate["predictions"]) - 1:
                    candidate["base_pred"] = 0
                candidate["pred"] = 0
            elif pred_rec["pred"] == 1 and candidate["base_pred"] is None:
                if pred_idx < len(candidate["predictions"]) - 1:
                    candidate["base_pred"] = 1
                candidate["pred"] = 1

        if candidate["predictions"][-1] is not None:
            candidate["score"] = candidate["predictions"][-1]["score"]

    def predict_many(self, *args, **kwargs):
        results = self._predict_many(*args, **kwargs)
        return results

    def _predict_many(self, test_recs):
        out_recs = []

        for rec in test_recs:
            out_recs.append(
                {
                    "input_record": rec,
                    "predictions": [{"edit": x, "predictions": [], "base_pred": None, "pred": None, "score": None} for x in rec["candidates"]],
                }
            )

        backmaps = [list(range(len(x["candidates"]))) for x in test_recs]

        # Don't modify the input test_recs if we need to prune
        cur_recs = test_recs
        if self.prune_candidates:
            cur_recs = [x.copy() for x in test_recs]
            for rec in cur_recs:
                rec["candidates"] = rec["candidates"].copy()

        pruned_idxs = [set() for x in test_recs]
        for aligner_idx, aligner in enumerate(self.aligners):
            logger.info(f"Running aligner {aligner_idx + 1} of {len(self.aligners)} ({aligner.__class__.__name__})")
            predictions = aligner.predict_many(cur_recs)

            # Update the corresponding prediction lists, keeping track of the
            # back-mappings from pruned candidates
            for recidx, rec in enumerate(predictions):
                for candidx, cand in enumerate(rec["predictions"]):
                    out_cand = out_recs[recidx]["predictions"][backmaps[recidx][candidx]]

                    # Hack: need to remove 'edit' to make the cands
                    # JSON-serializable
                    assert out_cand["edit"] == cand["edit"]
                    del cand["edit"]

                    out_cand["predictions"].append(cand)
                    self._update_candidate_scores(out_cand)
                    if out_cand["pred"] is None:
                        breakpoint()
                        print(out_cand["pred"])

            if self.prune_candidates:
                # Append None to predictions for any candidates that were pruned
                # by previous aligners
                for recidx, rec in enumerate(out_recs):
                    for candidx in pruned_idxs[recidx]:
                        rec["predictions"][candidx]["predictions"].append(None)
                        self._update_candidate_scores(rec["predictions"][candidx])

                if aligner_idx < len(self.aligners) - 1:
                    # Prune anything that was predicted to be 0
                    candidates_to_prune = []
                    for recidx, rec in enumerate(predictions):
                        for candidx, cand in enumerate(rec["predictions"]):
                            if cand["pred"] == 0:
                                candidates_to_prune.append((recidx, candidx))

                    # Reverse sort is important to ensure indices don't shift as we prune them
                    for recidx, candidx in sorted(candidates_to_prune, key=lambda x: x[1], reverse=True):
                        backmaps[recidx].pop(candidx)
                        cur_recs[recidx]["candidates"].pop(candidx)
                        pruned_idxs[recidx].add(candidx)

        return out_recs
