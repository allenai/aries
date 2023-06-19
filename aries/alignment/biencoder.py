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
from aries.util.data import batch_iter
from aries.util.edit import make_word_diff
from aries.util.training import TrainLoggerCallback

logger = logging.getLogger(__name__)


class BiencoderTransformerAlignerModel(torch.nn.Module):
    def __init__(self, embedding_model, projection_size=None):
        super().__init__()
        self.embedding_model = embedding_model
        self.projector = None
        if projection_size is not None:
            self.projector = torch.nn.Linear(embedding_model.config.hidden_size, projection_size)
        self.pred_threshold = 0

    def forward(
        self,
        query_input_ids,
        positive_input_ids,
        negative_input_ids,
        query_attention_mask=None,
        positive_attention_mask=None,
        negative_attention_mask=None,
    ):
        if not len(query_input_ids.shape) == 2:
            raise ValueError("Expected query_input_ids to be 2D, got {}".format(query_input_ids.shape))
        if query_input_ids.shape[0] != positive_input_ids.shape[0] or query_input_ids.shape[0] != negative_input_ids.shape[0]:
            raise ValueError(
                "Expected query_input_ids, positive_input_ids, and negative_input_ids to have the same batch size, got {} vs {} vs {}".format(
                    query_input_ids.shape[0], positive_input_ids.shape[0], negative_input_ids.shape[0]
                )
            )
        query_embeddings = self.embedding_model(query_input_ids, attention_mask=query_attention_mask).last_hidden_state[:, 0, :]
        positive_embeddings = self.embedding_model(positive_input_ids, attention_mask=positive_attention_mask).last_hidden_state[:, 0, :]
        negative_embeddings = self.embedding_model(negative_input_ids, attention_mask=negative_attention_mask).last_hidden_state[:, 0, :]

        if self.projector:
            query_embeddings = self.projector(query_embeddings)
            positive_embeddings = self.projector(positive_embeddings)
            negative_embeddings = self.projector(negative_embeddings)

        loss = self._compute_loss(query_embeddings, positive_embeddings, negative_embeddings)

        loss = torch.mean(loss)

        return {"loss": loss}

    def _compute_loss(self, query_embeddings, positive_embeddings, negative_embeddings):
        """Margin loss on the triplets"""
        query_embeddings = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)
        positive_embeddings = positive_embeddings / positive_embeddings.norm(dim=-1, keepdim=True)
        negative_embeddings = negative_embeddings / negative_embeddings.norm(dim=-1, keepdim=True)

        positive_scores = (query_embeddings * positive_embeddings).sum(dim=-1)
        negative_scores = (query_embeddings * negative_embeddings).sum(dim=-1)

        # TODO: Hacky; just gives us a rough estimate of the pos/neg class divider on a single train batch
        # new_pred_threshold = (positive_scores.mean().item() + negative_scores.mean().item()) / 2
        new_pred_threshold = min(positive_scores.tolist())
        delta = new_pred_threshold - self.pred_threshold
        delta = delta * 0.01 + np.sign(delta) * 0.01
        self.pred_threshold += delta

        # scores_as_logits = torch.stack([positive_scores, negative_scores], axis=1)
        # loss = torch.nn.functional.cross_entropy(scores_as_logits, torch.zeros(positive_scores.shape[0], dtype=torch.long, device=scores_as_logits.device), reduction='none')

        # loss = torch.nn.functional.relu(1.0 - positive_scores + negative_scores).mean()
        loss = torch.nn.functional.relu(0.5 - positive_scores + negative_scores).mean()
        return loss


class TripletDataCollator:
    def __init__(self, tokenizer, max_length, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        collated_batch = dict()

        tensors = self.tokenizer.pad(
            {"input_ids": [x["query_input_ids"] for x in batch]},
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            padding="longest",
        )
        collated_batch["query_input_ids"] = tensors["input_ids"]
        collated_batch["query_attention_mask"] = tensors["attention_mask"]

        tensors = self.tokenizer.pad(
            {"input_ids": [x["positive_input_ids"] for x in batch]},
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            padding="longest",
        )
        collated_batch["positive_input_ids"] = tensors["input_ids"]
        collated_batch["positive_attention_mask"] = tensors["attention_mask"]

        tensors = self.tokenizer.pad(
            {"input_ids": [x["negative_input_ids"] for x in batch]},
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            padding="longest",
        )
        collated_batch["negative_input_ids"] = tensors["input_ids"]
        collated_batch["negative_attention_mask"] = tensors["attention_mask"]

        return collated_batch


class BiencoderTransformerAligner:
    def __init__(self, config, embedding_model, tokenizer):
        self.config = config
        self.model = BiencoderTransformerAlignerModel(embedding_model, projection_size=config.get("embedding_projection_size", None))
        self.tokenizer = tokenizer
        if self.config["add_diff_tokens"]:
            self.tokenizer.add_tokens(["[+", "+]", "[-", "-]"])
            embedding_model.resize_token_embeddings(len(self.tokenizer))
        self.max_length = self.config["max_seq_length"]

    @staticmethod
    def preprocess_fn(examples, tokenizer, max_length):
        model_inputs = dict()
        query_inputs = tokenizer(
            examples["query_text"],
            max_length=max_length,
            padding=False,
            truncation=True,
        )
        model_inputs["query_input_ids"] = query_inputs["input_ids"]

        positive_inputs = tokenizer(
            examples["positive_text"],
            max_length=max_length,
            padding=False,
            truncation=True,
        )
        model_inputs["positive_input_ids"] = positive_inputs["input_ids"]

        negative_inputs = tokenizer(
            examples["negative_text"],
            max_length=max_length,
            padding=False,
            truncation=True,
        )
        model_inputs["negative_input_ids"] = negative_inputs["input_ids"]

        return model_inputs

    def _base_rec_example_fields(self, rec):
        return {
            "doc_id": rec["doc_id"],
            "source_pdf_id": rec["source_pdf_id"],
            "target_pdf_id": rec["target_pdf_id"],
            "review_comment": rec["review_comment"],
        }

    def _make_example_for_rec_triplet(self, rec, triplet):
        ex = self._base_rec_example_fields(rec)
        ex.update(
            {
                "query_text": triplet["query"],
                "positive_text": self._edit_to_input_text(triplet["positive"]),
                "negative_text": self._edit_to_input_text(triplet["negative"]),
            }
        )
        return ex

    def _candidate_record_to_input_text(self, rec):
        if self.config["query_input_format"] == "comment_only":
            return "review comment: " + rec["review_comment"]
        elif self.config["query_input_format"] == "comment_with_canonical":
            return "review comment: " + rec["review_comment"] + "\ncanonicalized: " + rec["canonical"]["canonicalized"]
        elif self.config["query_input_format"] == "reply_comment_or_extracted_comment":
            return "review comment: " + rec.get("reply_comment_line", rec["review_comment"])
            # return rec.get("reply_comment_line", rec["review_comment"])
        elif self.config["query_input_format"] == "reply_comment_or_extracted_comment_with_canonical":
            return "review comment: " + rec.get("reply_comment_line", rec["review_comment"]) + "\ncanonicalized: " + rec["canonical"]["canonicalized"]
        elif self.config["query_input_format"] == "comment_with_context":
            comment_str = rec["review_comment"].strip()
            if rec.get("context_side", "none") == "left":
                comment_str = rec["context"].strip() + " " + comment_str
            else:
                comment_str = comment_str + " " + rec["context"].strip()

            return "review comment: " + comment_str
        raise ValueError("Unknown query_input_format {}".format(self.config["query_input_format"]))

    def _edit_to_input_text(self, edit):
        if self.config["edit_input_format"] == "target_text":
            return edit.get_target_text()
        elif self.config["edit_input_format"] == "target_text_with_context":
            context = "context: none"
            if len(edit.target_idxs) != 0 and min(edit.target_idxs) != 0:
                context = "context: " + edit.texts2[min(edit.target_idxs) - 1]
            return edit.get_target_text() + "\n\n" + context
        elif self.config["edit_input_format"] == "source_text":
            return edit.get_source_text()
        elif self.config["edit_input_format"] == "diff":
            return make_word_diff(
                edit.get_source_text(),
                edit.get_target_text(),
                color_format="none",
            )
        raise ValueError("Unknown edit_input_format {}".format(self.config["edit_input_format"]))

    def _make_triplets_for_rec(self, rec, max_negatives=10000):
        query_text = self._candidate_record_to_input_text(rec)
        triplets = []
        for positive_edit in rec["positives"]:
            for negative_edit in sorted(rec["negatives"], key=lambda x: -len(x.get_target_text()))[:max_negatives]:
                triplets.append({"query": query_text, "positive": positive_edit, "negative": negative_edit})
        return triplets

    def _make_dataset(self, recs, name="dataset", shuffle=False):
        if isinstance(recs, dict):
            recs = list(recs.values())
        exs = []

        for rec in tqdm.tqdm(recs, desc="building triplets for records"):
            edit_with_labels = []
            edit_with_labels.extend([(x, 1) for x in rec["positives"]])
            edit_with_labels.extend([(x, 0) for x in rec["negatives"]])
            for triplet in self._make_triplets_for_rec(rec):
                exs.append(self._make_example_for_rec_triplet(rec, triplet))

        tmp = {k: [] for k in exs[0].keys()}
        for ex in exs:
            for k, v in ex.items():
                tmp[k].append(v)
        dset = datasets.Dataset.from_dict(tmp)

        if shuffle:
            dset = dset.shuffle()

        dset = dset.map(
            functools.partial(BiencoderTransformerAligner.preprocess_fn, tokenizer=self.tokenizer, max_length=self.max_length),
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

        if training_args.max_steps == 0 and training_args.num_train_epochs <= 0:
            logger.info("Got 0 train steps; skipping training")
            return

        self.rng = np.random.default_rng(self.config["seed"])

        train_dset = self._make_dataset(train_recs, shuffle=False)

        self.rng = np.random.default_rng(self.config["seed"])
        dev_dset = self._make_dataset(dev_recs, shuffle=False)

        logger.info("Train data size: {}  Dev data size: {}".format(len(train_dset), len(dev_dset)))

        logger.info(
            "{} | {} | {}\n".format(
                self.tokenizer.decode(train_dset["query_input_ids"][0]),
                self.tokenizer.decode(train_dset["positive_input_ids"][0]),
                self.tokenizer.decode(train_dset["negative_input_ids"][0]),
            )
        )

        data_collator = TripletDataCollator(
            self.tokenizer,
            self.max_length,
            pad_to_multiple_of=None,
        )

        model_selector_callback = AlignerEvalCallback(
            self.config,
            self,
            dev_recs,
            model_selection_metric_fn=lambda x: x["optimal_f1"],
            model_to_save=self.model.embedding_model,
        )

        logger.info("Subsampling dev_dset for loss computation")
        sub_dev_dset = dev_dset.select(np.random.default_rng(42).choice(len(dev_dset), size=min(200, len(dev_dset)), replace=False))

        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dset,
            eval_dataset=sub_dev_dset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[model_selector_callback, TrainLoggerCallback(logger)],
            compute_metrics=None,
        )

        _ = trainer.train()

        self.model.embedding_model.load_state_dict(model_selector_callback._best_model_state)
        self.model.embedding_model.save_pretrained(os.path.join(self.config["output_dir"], "ptmodel"))
        self.tokenizer.save_pretrained(os.path.join(self.config["output_dir"], "ptmodel"))

    def _eval_batch_size(self):
        td = self.config.get("training_args", dict())
        return td.get("per_device_eval_batch_size", td.get("per_device_train_batch_size", 16))

    def _embed_texts(self, texts, batch_size=None, cache_dict=None, use_tqdm=False):
        """cache_dict contains cached embeddings for some texts; missing ones
        will be computed and the cache_dict will be updated"""
        if len(texts) == 0:
            return np.array([])

        if batch_size is None:
            batch_size = self._eval_batch_size()

        if cache_dict is None:
            cache_dict = dict()
        final_embeddings = [cache_dict.get(x, None) for x in texts]
        text_idxs = [idx for idx, x in enumerate(final_embeddings) if x is None]
        orig_texts = texts
        texts = [orig_texts[idx] for idx in text_idxs]

        if len(text_idxs) != 0:
            embeddings = []
            pbar = tqdm.tqdm(
                batch_iter(texts, batch_size=batch_size),
                total=np.ceil(len(texts) / batch_size),
                disable=(not use_tqdm),
                desc="embedding text batches",
            )
            for text_batch in pbar:
                inputs = self.tokenizer(
                    text_batch,
                    return_tensors="pt",
                    max_length=self.max_length,
                    padding="longest",
                    truncation=True,
                )
                inputs = {k: v.to(self.model.embedding_model.device) for k, v in inputs.items()}
                batch_embeddings = self.model.embedding_model(**inputs).last_hidden_state[:, 0, :]
                if self.model.projector:
                    batch_embeddings = self.model.projector(batch_embeddings)
                batch_embeddings = batch_embeddings.detach().cpu().numpy()
                embeddings.append(batch_embeddings)
            embeddings = np.concatenate(embeddings, axis=0)

            for idx, embed in zip(text_idxs, embeddings):
                final_embeddings[idx] = embed
                cache_dict[orig_texts[idx]] = embed
        final_embeddings = np.stack(final_embeddings, axis=0)
        return final_embeddings

    def predict_many(self, *args, **kwargs):
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            results = self._predict_many(*args, **kwargs)

        self.model.train(was_training)
        return results

    def _predict_many(self, test_recs, quiet=False):
        out_recs = []

        pred_threshold = self.config.get("fixed_pred_threshold", None)
        if pred_threshold is None:
            pred_threshold = self.model.pred_threshold

        if not quiet:
            logger.info("Predicting with pred_threshold={}".format(pred_threshold))

        embed_cache_dict = dict()

        for rec in tqdm.tqdm(test_recs, "predicting", disable=quiet):
            outrec = {
                "input_record": rec,
                "predictions": [{"edit": cand, "pred": None, "score": None} for cand in rec["candidates"]],
            }

            query_embedding = self._embed_texts([self._candidate_record_to_input_text(rec)])[0]
            query_embedding /= np.linalg.norm(query_embedding)

            if len(outrec["predictions"]) != 0:
                candidate_embeddings = self._embed_texts(
                    [self._edit_to_input_text(cand["edit"]) for cand in outrec["predictions"]], cache_dict=embed_cache_dict
                )
                candidate_embeddings /= np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)

                cosine_similarities = query_embedding.dot(candidate_embeddings.T).tolist()

                best_candidxs = np.argsort(cosine_similarities).tolist()
                best_candidx_score = cosine_similarities[best_candidxs[-1]]

            for candidx, cand in enumerate(outrec["predictions"]):
                cand["best_group_score"] = best_candidx_score
                cand["cosine_score"] = cosine_similarities[candidx]
                cand["pred"] = (
                    1 if cosine_similarities[candidx] > pred_threshold and cosine_similarities[candidx] >= (best_candidx_score - 0.2) else 0
                )
                cand["score"] = cosine_similarities[candidx]
            out_recs.append(outrec)

        return out_recs
