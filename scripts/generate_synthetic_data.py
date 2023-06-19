import collections
import itertools
import json
import logging
import os
import re
import sys
import time

import nltk
import nltk.util
import numpy as np
import tqdm
from nltk.util import ngrams

import aries.util.data
import aries.util.s2orc
from aries.alignment.doc_edits import make_full_aligns
from aries.util.data import deduplicate_by, index_by, iter_jsonl_files
from aries.util.edit import find_overlapping_substrings
from aries.util.logging import init_logging

logger = logging.getLogger(__name__)

STOPWORDS = set(nltk.corpus.stopwords.words("english")) | set(",./<>?!@#$%^&*()_+-={}|[]\\,")


def get_author_replies(review_note, forum_notes):
    replies = [x for x in forum_notes if x["replyto"] == review_note["id"] and any("author" in xx.lower() for xx in x["writers"])]
    # Sometimes authors break up their response by replying to their own message
    nested_replies = []
    for reply in replies:
        nested_replies.extend(get_author_replies(reply, forum_notes))
    return replies + nested_replies


def _get_combined_text2_overlap_spans_overlapping_span(all_overlaps, span, is_sorted=False):
    """Given a set of overlap span pairs, find all of the ones for which the
    text2 span overlaps the given span, and return a list of just the text2
    spans of those overlaps, merged such that any partially-overlapping spans
    are collapsed into a single span in the list."""
    if not is_sorted:
        # Sort by text2 idxs; this allows fast lookups for overlaps contained within a span
        all_overlaps = sorted(all_overlaps, key=lambda x: x[1])

    overlaps = []
    for ov in all_overlaps:
        if ov[1][0] >= span[0] and ov[1][0] < span[1]:
            overlaps.append(ov)
        elif ov[1][1] > span[0] and ov[1][1] <= span[1]:
            overlaps.append(ov)

    if len(overlaps) <= 1:
        return [x[1] for x in overlaps]

    combined = []
    last_span = overlaps[0][1]
    for ov in overlaps[1:]:
        if ov[1][0] < last_span[1]:
            last_span = (last_span[0], ov[1][1])
        else:
            combined.append(last_span)
            last_span = ov[1]
    combined.append(last_span)
    return combined


TextReplyMatch = collections.namedtuple("TextReplyMatch", "doc_id review_id reply_id match_text spans line_span next_line_span reply_text")


def get_tight_span(toks, line_overlaps, prevnewline, nextnewline):
    # There are two main ways authors mess with the text at this
    # point: (1) they correct typos, causing non-exact matches, and
    # (2) they add prefixes or quotation marks to the text (e.g.,
    # "Comment 1:").  To deal with the first case, we want to
    # include the whole line even if there are some non-matched
    # spans in the middle.  But, to deal with the second case we
    # want to omit the start or end of the line if those aren't
    # matched and they don't occur in the middle of a word.
    tight_span = [max(min(x[0] for x in line_overlaps), prevnewline), min(max(x[1] for x in line_overlaps), nextnewline)]
    while toks[tight_span[0]].isspace() or toks[tight_span[0]] in ".:)*":
        tight_span[0] += 1

    # Expand back if the span started mid-word (usually a capitalization difference)
    while toks[tight_span[0] - 1].isalpha():
        tight_span[0] -= 1

    # Citations are weird; never prune when we have one
    if re.search(r" et |[0-9]{4}", toks[prevnewline : tight_span[0]]):
        tight_span[0] = prevnewline

    while toks[tight_span[1] - 1].isspace():
        tight_span[1] -= 1
    if tight_span[0] > prevnewline + 20:
        tight_span[0] = prevnewline
    if tight_span[1] < nextnewline - 10:
        tight_span[1] = nextnewline

    return tight_span


def _get_match_for_overlap(toks1, toks2, overlap, all_overlaps, min_line_overlap_ratio, doc_id, review_id, reply_id):
    # Check if it takes up most of a line (quotes usually do)
    prevnewline = max(0, toks2.rfind("\n", 0, overlap[1][0]))

    nextnewline = toks2.find("\n", overlap[1][1] - 1)
    nextnewline = nextnewline if nextnewline >= 0 else len(toks2)
    while nextnewline > prevnewline and toks2[nextnewline - 1] == "\n":
        nextnewline -= 1

    if nextnewline == prevnewline:
        print(nextnewline, prevnewline, overlap, len(toks2))
    line_overlaps = _get_combined_text2_overlap_spans_overlapping_span(all_overlaps, (prevnewline, nextnewline))
    total_line_overlap = sum(max(x[1], prevnewline) - min(x[0], nextnewline) for x in line_overlaps)
    lineratio = total_line_overlap / (nextnewline - prevnewline)

    if lineratio < min_line_overlap_ratio:
        return None

    tight_span = get_tight_span(toks2, line_overlaps, prevnewline, nextnewline)
    # if abs(tight_span[0] - prevnewline) > 2 or abs(tight_span[0] - nextnewline) > 2:
    #    print('difference! oldline={},\nnewline={}'.format(toks2[prevnewline:nextnewline], toks2[tight_span[0]:tight_span[1]],))

    nextnextnewline = nextnewline
    while nextnextnewline < len(toks2) and toks2[nextnextnewline] == "\n":
        nextnextnewline += 1
    nnlend = nextnextnewline
    while nextnextnewline < len(toks2) and toks2[nextnextnewline] != "\n":
        nextnextnewline += 1
    # print(toks1[overlap[0][0]:overlap[0][1]])
    # all_matches.append((toks1[overlap[0][0]:overlap[0][1]], docrec['docid'], revrec['id'], reply['id'], overlap))
    # all_matches.append((toks2[prevnewline:nextnewline], docrec["docid"], revrec["id"], reply["id"], overlap, (prevnewline, nextnewline), toks2))
    return TextReplyMatch(
        doc_id,
        review_id,
        reply_id,
        # None,
        # None,
        # None,
        # docrec["docid"],
        # revrec["id"],
        # reply["id"],
        # toks2[prevnewline:nextnewline],
        toks2[tight_span[0] : tight_span[1]],
        overlap,
        # (prevnewline, nextnewline),
        tuple(tight_span),
        (nnlend, nextnextnewline),
        toks2,
    )


def get_author_comment_replies_for_doc(forum_id, review_replies, min_length=80, min_line_overlap_ratio=0.9):
    all_matches = []
    for review_rec in review_replies:
        replies = review_rec["author_replies"]
        used_spans = set()
        for reply in replies:
            toks1 = "\n".join([str(x) for x in review_rec["content"].values()])
            toks2 = reply["content"]["comment"]

            overlaps = find_overlapping_substrings(toks1, toks2, min_length=min_length)
            overlaps.sort(key=lambda x: x[1])

            for overlap in overlaps:
                m = _get_match_for_overlap(
                    toks1, toks2, overlap, overlaps, min_line_overlap_ratio, review_rec["forum"], review_rec["id"], reply["id"]
                )
                if m is not None:
                    sp = (m.doc_id, m.review_id, m.reply_id, m.line_span)
                    if sp not in used_spans:
                        all_matches.append(m)
                        used_spans.add(sp)
                    else:
                        logger.debug("Skipping duplicate match: %s", sp)
    return all_matches


def make_bow(txt):
    return collections.Counter(ngrams([x for x in txt.split() if x.lower() not in STOPWORDS], 1))


def _similarity_many_many_minl(txts1, txts2, match_denom=False):
    ngs1 = [make_bow(txt) for txt in txts1]
    ngs2 = [make_bow(txt) for txt in txts2]
    sim_mat = np.zeros((len(txts1), len(txts2)))

    if match_denom:
        denom = max(sum(x.values()) for x in itertools.chain(ngs1, ngs2))

        def sim_fn(counter1, counter2):
            return sum((counter1 & counter2).values()) / denom

    else:

        def sim_fn(counter1, counter2):
            return sum((counter1 & counter2).values()) / max(40, min(sum(counter1.values()), sum(counter2.values())))

    for idx1, idx2 in itertools.product(range(len(txts1)), range(len(txts2))):
        ng1 = ngs1[idx1]
        ng2 = ngs2[idx2]
        if len(ng1) == 0 and len(ng2) == 0:
            sim_mat[idx1, idx2] = sim_fn(collections.Counter(txts1[idx1]), collections.Counter(txts2[idx2]))
        sim_mat[idx1, idx2] = sim_fn(ng1, ng2)
    return sim_mat


def _get_high_similarity_comment_edit_texts(comments, edits, sim_threshold):
    output_matches = []

    t2s = []
    for edit_idx, edit in enumerate(edits):
        try:
            output_text = " ".join(edit.get_added_tokens())
        except RecursionError:
            logger.error("Recursion error for edit %s", edit_idx)
            output_text = ""
        t2s.append(output_text)
    sim_mat = _similarity_many_many_minl(comments, t2s, match_denom=False)
    for cidx, rec in enumerate(comments):
        # If there are multiple matches, take only the best; others are sometimes spurious
        best = None
        for eidx in range(len(t2s)):
            if sim_mat[cidx, eidx] <= sim_threshold:
                continue

            edit = edits[eidx]
            # We allow a little wiggle room for off-by-1's (could come from bad splits/parses),
            # but it's unlikely to be correct if there were many non-consecutive matches
            if (sorted(edit.target_idxs)[-1] - sorted(edit.target_idxs)[0]) >= (len(edit.target_idxs) * 2 - 1):
                continue

            if edits[eidx].is_full_deletion() or edits[eidx].is_identical() or len(edits[eidx].get_added_tokens()) < 5:
                continue

            if best is None or best[2] < sim_mat[cidx, eidx]:
                best = (cidx, eidx, sim_mat[cidx, eidx])

        if best is not None:
            output_matches.append(best)
    return output_matches


def get_high_precision_reply_based_alignments_for_doc(
    pdf_pair_record,
    review_replies,
    sim_threshold,
    min_reply_overlap_chars,
    min_line_overlap_ratio,
    s2orc_fetcher,
):
    doc_id = pdf_pair_record["doc_id"]
    s2orc1 = aries.util.s2orc.load_s2orc(pdf_pair_record["source_pdf_id"], s2orc_fetcher)
    s2orc2 = aries.util.s2orc.load_s2orc(pdf_pair_record["target_pdf_id"], s2orc_fetcher)

    if s2orc1 is None or s2orc2 is None or s2orc1["paper_id"] == s2orc2["paper_id"]:
        return []

    forum_replies = get_author_comment_replies_for_doc(
        doc_id,
        review_replies,
        min_length=min_reply_overlap_chars,
        min_line_overlap_ratio=min_line_overlap_ratio,
    )

    review_recs = [
        {
            "review_id": x.review_id,
            "review_comment": x.match_text,
            "reply": x.reply_text[x.next_line_span[0] : x.next_line_span[1]],
            "full_match": x,
        }
        for x in forum_replies
    ]

    # If we don't even have enough tokens to form a sentence, it's probably invalid
    review_recs = [x for x in review_recs if len(x["review_comment"].split()) >= 4]

    review_recs = deduplicate_by(review_recs, "review_comment")

    aligns = make_full_aligns(s2orc1, s2orc2)

    aug_review_comments = [x["reply"] for x in review_recs]

    output_matches = []

    for cidx, eidx, sim in _get_high_similarity_comment_edit_texts(aug_review_comments, aligns.paragraph_edits, sim_threshold):
        edit = aligns.paragraph_edits[eidx]

        output_matches.append(
            {
                "doc_id": doc_id,
                "source_pdf_id": pdf_pair_record["source_pdf_id"],
                "target_pdf_id": pdf_pair_record["target_pdf_id"],
                "review_id": review_recs[cidx]["review_id"],
                "edit": edit,
                "review_comment": review_recs[cidx]["review_comment"],
                "reply": review_recs[cidx]["reply"],
                "forum_reply": review_recs[cidx]["full_match"],
                "similarity": sim,
            }
        )

    paper_edit_record = aligns.to_json()
    paper_edit_record["doc_id"] = doc_id

    review_comment_records = [
        {
            "comment_id": cidx,
            "doc_id": doc_id,
            "annotation": "synthetic",
            "comment": x["review_comment"],
            "comment_context": "",
            "review_id": x["review_id"],
        }
        for cidx, x in enumerate(output_matches)
    ]

    edit_label_records = [
        {"doc_id": doc_id, "comment_id": cidx, "positive_edits": [x["edit"].edit_id], "negative_edits": [], "annotation": "synthetic"}
        for cidx, x in enumerate(output_matches)
    ]
    return paper_edit_record, review_comment_records, edit_label_records, output_matches


def augment_config(config):
    config_defaults = {
        "similarity_threshold": 0.26,
        "min_reply_overlap_chars": 40,
        "min_line_overlap_ratio": 0.9,
        "seed": 42,
    }
    for k, v in config_defaults.items():
        config[k] = config.get(k, v)

    NEEDED_KEYS = ["s2orc_base_path", "output_dir", "split_ids_file", "split_name", "review_replies_file"]
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

    review_replies = list(iter_jsonl_files([config["review_replies_file"]]))
    review_replies_by_docid = index_by(review_replies, "forum")

    paper_edit_records = []
    review_comment_records = []
    edit_label_records = []
    full_match_records = []

    pbar = tqdm.tqdm(pdf_pair_ids)
    with aries.util.s2orc.S2orcFetcherSqlite(
        config.get("s2orc_db_path", ":memory:"),
        fallback_fetcher=aries.util.s2orc.S2orcFetcherFilesystem(config["s2orc_base_path"]),
        update_db=False,
    ) as fetcher:
        for pdf_pair in pbar:
            if pdf_pair["doc_id"] not in review_replies_by_docid:
                continue

            per, rcr, elr, fmr = get_high_precision_reply_based_alignments_for_doc(
                pdf_pair,
                review_replies_by_docid[pdf_pair["doc_id"]],
                sim_threshold=config["similarity_threshold"],
                min_reply_overlap_chars=config["min_reply_overlap_chars"],
                min_line_overlap_ratio=config["min_line_overlap_ratio"],
                s2orc_fetcher=fetcher,
            )

            paper_edit_records.append(per)
            review_comment_records.extend(rcr)

            for elr_i in elr:
                elr_i["split"] = config["split_name"]
            edit_label_records.extend(elr)

            full_match_records.extend(fmr)

            pbar.set_description("n_results={}".format(len(edit_label_records)), refresh=False)

    with open(os.path.join(config["output_dir"], "paper_edits.jsonl"), "w") as f:
        for rec in paper_edit_records:
            f.write(json.dumps(rec) + "\n")

    with open(os.path.join(config["output_dir"], "review_comments.jsonl"), "w") as f:
        for rec in review_comment_records:
            f.write(json.dumps(rec) + "\n")

    with open(os.path.join(config["output_dir"], "edit_labels.jsonl"), "w") as f:
        for rec in edit_label_records:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
