import itertools
import json
import logging
import os
import sys

import tqdm

from aries.util.data import index_by
from aries.util.edit import make_word_diff
from aries.util.gpt3 import Gpt3CacheClient

logger = logging.getLogger(__name__)


class GptChatAligner:
    def __init__(self, config):
        self.config = config
        self.system_prompt = self.config["gpt_system_prompt"]
        self.prompt_template = self.config["gpt_prompt_template"]
        self.model_name = self.config["gpt_model"]
        self.max_length = self.config["gpt_max_length"]
        self.cache_db_path = self.config["cache_db_path"]

    def train(self, train_recs, dev_recs):
        logger.warning("GPT doesn't train; ignoring call to train()")

    def _predict_one(self, comment, edit, gptcli):
        tags = {
            "{{review_comment}}": comment,
            "{{target_paragraph}}": edit.get_target_text(),
            "{{source_paragraph}}": edit.get_source_text(),
            "{{diff_paragraph}}": make_word_diff(
                edit.get_source_text(),
                edit.get_target_text(),
                color_format="none",
            ),
        }
        msg = self.prompt_template
        for k, v in sorted(tags.items()):
            msg = msg.replace(k, v)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg},
        ]
        resp = gptcli.chat_completion(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=self.max_length,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        result_text = resp["choices"][0]["message"]["content"]
        result_words = set(result_text.lower().replace(".", " ").replace(",", " ").replace("\n", " ").replace('"', " ").replace("'", " ").split(" "))
        # Extract yes/no answer from response text
        has_yes = "yes" in result_words or "answer=yes" in result_words
        has_no = "no" in result_words or "answer=no" in result_words
        pred = None
        if has_yes and has_no:
            pred = None
            raise ValueError("Got both yes and no in response")
        elif has_yes:
            pred = 1
        elif has_no:
            pred = 0
        else:
            logger.error("Bad response: {}".format(result_text))
            raise ValueError("Got neither yes nor no in response")
        return pred, resp

    def predict_many(self, test_recs):
        out_recs = []

        total_tokens, uncached_total_tokens = 0, 0
        loopname = "{}.predict_many".format(self.__class__.__name__)
        with tqdm.trange(sum(len(x["candidates"]) for x in test_recs), miniters=1, desc=loopname) as pbar:
            with Gpt3CacheClient(self.cache_db_path) as gptcli:
                for rec in test_recs:
                    outrec = {
                        "input_record": rec,
                        "predictions": [{"edit": cand, "pred": None, "score": None} for cand in rec["candidates"]],
                    }
                    out_recs.append(outrec)

                    for pred_rec in outrec["predictions"]:
                        pred_label, resp = self._predict_one(rec["review_comment"], pred_rec["edit"], gptcli)
                        total_tokens += resp["usage"]["total_tokens"]
                        uncached_total_tokens += resp["usage"]["uncached_total_tokens"]

                        pred_rec["pred"] = pred_label
                        pred_rec["score"] = pred_label

                        pbar.set_description(f"{loopname} tt={total_tokens} utt={uncached_total_tokens}", refresh=False)
                        pbar.update(1)

        return out_recs


class GptChatFullPaperAligner:
    def __init__(self, config):
        self.config = config
        self.system_prompt = self.config["gpt_system_prompt"]
        self.prompt_template = self.config["gpt_prompt_template"]
        self.model_name = self.config["gpt_model"]
        self.max_length = self.config["gpt_max_length"]
        self.cache_db_path = self.config["cache_db_path"]
        self.output_dir = self.config.get("output_dir", None)
        self.max_response_length = 500

        self.raw_responses = []

    def train(self, train_recs, dev_recs):
        logger.warning("GPT doesn't train; ignoring call to train()")

    def _make_chunked_paper_diff(self, doc_edits, chunk_size, gptcli):
        full_diff_string, edits_by_id = doc_edits.make_paper_diff_string(
            color_format="none",
            print_ids_only=True,
            return_edit_ids=True,
        )

        para_chunks = full_diff_string.split("\n\n")

        diff_chunks = []
        cur_chunk = []
        cur_chunk_len = 0
        # Note: we don't account for individual paras being bigger than
        # chunk_size; that probably never happens anyway
        for para_chunk in para_chunks:
            # Add 2 for the stripped \n\n
            new_chunk_len = gptcli.estimate_num_tokens(para_chunk, self.model_name) + 2
            if cur_chunk_len + new_chunk_len > chunk_size:
                diff_chunks.append("\n\n".join(cur_chunk))
                cur_chunk = []
                cur_chunk_len = 0
            cur_chunk.append(para_chunk)
            cur_chunk_len += new_chunk_len

        if len(cur_chunk) != 0:
            diff_chunks.append("\n\n".join(cur_chunk))

        return diff_chunks, edits_by_id

    def _make_comments_text_blob(self, recs):
        comments_text_blob = ""
        for idx, comment in enumerate(recs):
            comments_text_blob += comment.replace("\n", " ") + "\ncomment id: {}\n\n".format(idx)
        return comments_text_blob

    def _predict_one_doc(self, doc_edits, comments, gptcli):
        comments_text = self._make_comments_text_blob(comments)

        base_length = gptcli.estimate_num_tokens(self.prompt_template, self.model_name) + gptcli.estimate_num_tokens(
            self.system_prompt, self.model_name
        )
        if "{{review_comments}}" in self.prompt_template:
            base_length += gptcli.estimate_num_tokens(comments_text, self.model_name)
        chunk_size = self.max_length - base_length - self.max_response_length

        diff_chunks, edits_by_id = self._make_chunked_paper_diff(doc_edits, chunk_size=chunk_size, gptcli=gptcli)

        all_response_lines_by_comment = {idx: [] for idx in range(len(comments))}
        total_tokens, uncached_total_tokens = 0, 0
        for chunk in diff_chunks:
            tags = {
                "{{review_comments}}": comments_text,
                "{{paper_diff_chunk}}": chunk,
            }
            msg = self.prompt_template
            for k, v in sorted(tags.items()):
                msg = msg.replace(k, v)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": msg},
            ]
            if base_length + gptcli.estimate_num_tokens(chunk, self.model_name) + self.max_response_length > 8150:
                print(base_length, gptcli.estimate_num_tokens(chunk, self.model_name), self.max_response_length)
                print()
            try:
                resp = gptcli.chat_completion(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    # max_tokens=self.max_length,
                    max_tokens=self.max_response_length,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
            except Exception as e:
                breakpoint()
                print(e)
            total_tokens += resp["usage"]["total_tokens"]
            uncached_total_tokens += resp["usage"]["uncached_total_tokens"]
            result_text = resp["choices"][0]["message"]["content"]

            self.raw_responses.append(
                {
                    # "doc_id": doc_id,
                    "source_pdf_id": doc_edits.s2orc1["paper_id"],
                    "target_pdf_id": doc_edits.s2orc2["paper_id"],
                    "comments": comments,
                    "comments_text": comments_text,
                    "response_text": result_text,
                }
            )

            for line in result_text.split("\n"):
                # Imperfect but good-enough detection of JSON lines
                if not line.startswith("{"):
                    continue

                # Hacky; fix some specific known failures
                line = line.replace(" \\phi", " \\\\phi")
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse JSON line: {}".format(line))
                    # raise e
                    continue
                all_response_lines_by_comment[obj["comment_id"]].append(obj)

        results = []
        for comment_id, resps in all_response_lines_by_comment.items():
            # Ignore the abstract (9999) since it isn't diffed in DocEdits
            all_edit_ids = sorted(set(itertools.chain(*[x["edit_ids"] for x in resps])) - {9999})
            results.append(
                {
                    "review_comment": comments[comment_id],
                    "predicted_positive_edits": [
                        {
                            "source_idxs": edits_by_id[x].source_idxs,
                            "target_idxs": edits_by_id[x].target_idxs,
                        }
                        for x in all_edit_ids
                    ],
                }
            )

        usage_info = {
            "total_tokens": total_tokens,
            "uncached_total_tokens": uncached_total_tokens,
        }
        return results, usage_info

    def predict_many(self, test_recs):
        out_recs = []

        # We need to run the model for the pdf pair of each *candidate*, since
        # it is possible to have candidates sampled from other documents than
        # the one the comment was for.
        comment_pdf_pairs = []
        for rec in test_recs:
            for edit in rec["candidates"]:
                comment_pdf_pairs.append(
                    {
                        "comment": rec["review_comment"],
                        "pdf_pair": (edit.doc_edits.s2orc1["paper_id"], edit.doc_edits.s2orc2["paper_id"]),
                        "doc_edits": edit.doc_edits,
                    }
                )
            # For consistency, include comments in the many-to-many alignment
            # even when no candidates are given
            if "source_pdf_id" in rec and "target_pdf_id" in rec:
                comment_pdf_pairs.append(
                    {
                        "comment": rec["review_comment"],
                        "pdf_pair": (rec["source_pdf_id"], rec["target_pdf_id"]),
                        "doc_edits": None,
                    }
                )

        comment_pdf_pairs_by_pdf = index_by(comment_pdf_pairs, "pdf_pair")

        total_tokens, uncached_total_tokens = 0, 0
        with Gpt3CacheClient(self.cache_db_path) as gptcli:
            loopname = "{}.predict_many".format(self.__class__.__name__)
            predictions_by_pdf = dict()
            pbar = tqdm.tqdm(comment_pdf_pairs_by_pdf.items(), miniters=1, desc=loopname)
            for pdf_pair, comment_recs in pbar:
                if all(x["doc_edits"] is None for x in comment_recs):
                    # No candidates for this pdf pair, so skip it
                    continue
                predictions_by_pdf[pdf_pair], token_usage = self._predict_one_doc(
                    [x for x in comment_recs if x["doc_edits"] is not None][0]["doc_edits"],
                    sorted(set([x["comment"] for x in comment_recs])),
                    gptcli,
                )
                predictions_by_pdf[pdf_pair] = index_by(predictions_by_pdf[pdf_pair], "review_comment", one_to_one=True)

                total_tokens += token_usage["total_tokens"]
                uncached_total_tokens += token_usage["uncached_total_tokens"]
                pbar.set_description(f"{loopname} tt={total_tokens} utt={uncached_total_tokens}", refresh=False)

        for rec in test_recs:
            outrec = {
                "input_record": rec,
                "predictions": [{"edit": cand, "pred": None, "score": None} for cand in rec["candidates"]],
            }

            out_recs.append(outrec)

            for pred in outrec["predictions"]:
                pred_rec = predictions_by_pdf[(pred["edit"].doc_edits.s2orc1["paper_id"], pred["edit"].doc_edits.s2orc2["paper_id"])][
                    rec["review_comment"]
                ]
                pos_edits = [] if pred_rec is None else pred_rec["predicted_positive_edits"]
                pred_label = 0
                for edit in pos_edits:
                    if (sorted(edit["source_idxs"]) == sorted(pred["edit"].source_idxs)) and (
                        sorted(edit["target_idxs"]) == sorted(pred["edit"].target_idxs)
                    ):
                        pred_label = 1
                        break
                pred["pred"] = pred_label
                pred["score"] = pred_label

        if self.output_dir is not None:
            with open(os.path.join(self.output_dir, "raw_gpt_outputs.json"), "w") as f:
                json.dump(self.raw_responses, f)

        return out_recs
