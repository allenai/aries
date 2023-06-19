import collections
import json
import logging
import os

import torch
import transformers

from .logging import pprint_metrics

logger = logging.getLogger(__name__)


class TrainLoggerCallback(transformers.TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        self.logger.info(
            "Logs at global step {} (epoch {}, {:0.2f} TFLOs): {}".format(state.global_step, state.epoch, state.total_flos / 1e12, json.dumps(logs))
        )


class Seq2SeqEvalCallback(transformers.TrainerCallback):
    def __init__(self, config, model, eval_records, model_eval_fn, model_selection_metric_fn=None):
        self.config = config
        self.model = model
        self.eval_records = eval_records
        self.model_eval_fn = model_eval_fn
        self.eval_precached_dataset = self.model._make_dataset(self.eval_records)

        self.model_selection_metric_fn = model_selection_metric_fn
        if isinstance(model_selection_metric_fn, str):
            self.model_selection_metric_fn = lambda x: x[model_selection_metric_fn]

        self._best_metric_val = float("-inf")
        self._best_model_state = None

    @staticmethod
    def _clone_cpu_model_state_dict(model):
        return collections.OrderedDict((k, v.clone().cpu().detach()) for k, v in model.state_dict().items())

    def on_evaluate(self, args, state, control, **kwargs):
        metrics, all_results, _ = self.model_eval_fn(self.model, self.eval_records, eval_precached_dataset=self.eval_precached_dataset)

        if self.config.get("write_examples_on_eval", False):
            with open(os.path.join(self.config["output_dir"], "{}_inferences.jsonl".format("tmp_mid_eval")), "w") as f:
                for res in all_results:
                    f.write(json.dumps(res) + "\n")

        pprint_metrics(metrics, logger, name="dev (mid-train)")
        if self.model_selection_metric_fn is not None:
            metric_val = self.model_selection_metric_fn(metrics)
            if metric_val > self._best_metric_val:
                logger.info(
                    "Got new best model at global step {} (epoch {}, {:0.2f} TFLOs)".format(state.global_step, state.epoch, state.total_flos / 1e12)
                )
                state.best_metric = metric_val
                self._best_metric_val = metric_val
                self._best_model_state = Seq2SeqEvalCallback._clone_cpu_model_state_dict(self.model.model)


class SequentialTrainer(transformers.Trainer):
    def _get_train_sampler(self):
        if self.train_dataset is None:
            return None
        return torch.utils.data.SequentialSampler(self.train_dataset)
