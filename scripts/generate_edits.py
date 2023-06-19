import json
import logging
import os
import sys

import tqdm

import aries.util.data
import aries.util.s2orc
from aries.util.data import index_by, iter_jsonl_files
from aries.util.gpt3 import Gpt3CacheClient
from aries.util.logging import init_logging

logger = logging.getLogger(__name__)


def generate_edits_for_doc_comment(
    doc_s2orc,
    comment_record,
    prompt_template,
    gptcli,
):
    prompt = make_gpt_prompt(doc_s2orc, comment_record, prompt_template, gptcli)
    messages = [
        {
            "role": "system",
            "content": "You are SciGPT, a research assistant that specializes in helping authors to improve their scientific papers.  Follow the user's instructions carefully.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    result = {
        "source_pdf_id": doc_s2orc["paper_id"],
        "comment_record": comment_record,
        "openreview_base_pdf": "https://openreview.net/references/pdf?id={}".format(doc_s2orc["paper_id"]),
        "gpt_edit": None,
    }
    try:
        response = gptcli.chat_completion(
            model="gpt-4-0314",
            messages=messages,
            temperature=0,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        result_text = response["choices"][0]["message"]["content"]
    except Exception:
        logging.exception("Error generating edit for doc_id={}".format(doc_s2orc["paper_id"]))
        return result, None
    parsed_result = parse_result_text(result_text)

    result["gpt_edit"] = result_text[result_text.find("Location:") :]

    return result, response


def make_gpt_prompt(doc_s2orc, comment_record, template, gptcli):
    abstract = doc_s2orc["pdf_parse"]["abstract"][0]["text"]
    body_text_blob = ""
    prev_section = "unknown"
    for idx, x in enumerate(doc_s2orc["pdf_parse"]["body_text"]):
        secheader = ""
        secname = x["section"] if x["section"] else "unknown"
        if secname != prev_section:
            secheader = "section: {}\n".format(secname)
        prev_section = secname
        newtext = "{}{}\nparagraph id: {}\n\n".format(secheader, x["text"], idx)
        if gptcli.estimate_num_tokens(body_text_blob + newtext, model="gpt-4") < 6 * (2**10):
            body_text_blob += newtext

    comment_with_context = comment_record["comment"].strip()
    if comment_record["comment_context"] != "":
        comment_with_context += "\ncontext: {}".format(comment_record["comment_context"])

    variables = {
        "__abstract": abstract,
        #'__comment': comment_record['comment'],
        "__comment_with_context": comment_with_context,
        "__body_chunk": body_text_blob,
        #'__full_review': None,
    }
    s = template.format(**variables)
    return s


def parse_result_text(result_text):
    result = {"response": "", "edits": []}
    lines = result_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Response:"):
            if result["response"] != "":
                raise ValueError("Multiple 'Response' tags")
            result["response"] = line[9:].strip()
            i += 1
        elif line.startswith("Location:"):
            location = line[9:].strip()
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("Edit:"):
                location += "\n" + lines[i].strip()
                i += 1
            if i < len(lines) and lines[i].strip().startswith("Edit:"):
                edit = lines[i][5:].strip()
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("Location:"):
                    edit += "\n" + lines[i].strip()
                    i += 1
                result["edits"].append({"location": location.strip(), "edit": edit.strip()})
        else:
            i += 1
    return result


def augment_config(config):
    config_defaults = {
        "seed": 42,
    }
    for k, v in config_defaults.items():
        config[k] = config.get(k, v)

    NEEDED_KEYS = ["s2orc_base_path", "cache_db_path", "output_dir", "split_ids_file", "split_name", "review_comments_file"]
    missing_keys = [x for x in NEEDED_KEYS if x not in config]
    if len(missing_keys) > 0:
        raise ValueError("Missing config keys: %s" % missing_keys)


def main():
    with open(sys.argv[1]) as f:
        config = json.load(f)

    augment_config(config)

    init_logging(
        logfile=os.path.join(config["output_dir"], "logging_output.log"),
        level=logging.INFO,
    )

    with open(config["split_ids_file"]) as f:
        pdf_pair_ids = json.load(f)[config["split_name"]]

    pair_ids_by_doc = index_by(pdf_pair_ids, "doc_id", one_to_one=True)

    review_comments = [x for x in iter_jsonl_files([config["review_comments_file"]]) if x["doc_id"] in pair_ids_by_doc]
    review_comments_by_docid = index_by(review_comments, "doc_id")

    all_outputs = []
    tt = 0
    utt = 0

    with aries.util.s2orc.S2orcFetcherSqlite(
        config.get("s2orc_db_path", ":memory:"),
        fallback_fetcher=aries.util.s2orc.S2orcFetcherFilesystem(config["s2orc_base_path"]),
        update_db=False,
    ) as fetcher:
        with Gpt3CacheClient(config["cache_db_path"]) as gptcli:
            with tqdm.trange(sum(map(len, review_comments_by_docid.values()))) as pbar:
                for doc_id, comment_records in review_comments_by_docid.items():
                    doc_s2orc = aries.util.s2orc.load_s2orc(pair_ids_by_doc[doc_id]["source_pdf_id"], fetcher)
                    for idx, comment_record in enumerate(comment_records):
                        record, response = generate_edits_for_doc_comment(doc_s2orc, comment_record, config["prompt_template"], gptcli)
                        all_outputs.append(record)
                        if response is None:
                            raise ValueError("GPT returned no response")
                        tt += response["usage"]["total_tokens"]
                        utt += response["usage"]["uncached_total_tokens"]
                        pbar.set_description("tt={} utt={}, doc={}".format(tt, utt, doc_s2orc["paper_id"]))
                        pbar.update(1)

    with open(os.path.join(config["output_dir"], "edits.jsonl"), "w") as f:
        for record in all_outputs:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
