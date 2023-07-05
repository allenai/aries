# ARIES

Data and code for [ARIES: A Corpus of Scientific Paper Edits Made in Response to Peer Reviews](https://arxiv.org/pdf/2306.12587.pdf)

## Dataset

To download the dataset, run `aws s3 sync --no-sign-request s3://ai2-s2-research-public/aries/ data/aries/`. Below, we provide an overview of the files contained in ARIES.

`s2orc.tar.gz` contains the S2ORC (full-text) parses of papers with comment-aligned edits, can be extracted with `tar -C data/aries -xf data/aries/s2orc.tar.gz`.  

`paper_edits.jsonl` contains edits for each document.  Each document has a source pdf id and a target pdf id corresponding to the original and revised versions, respectively; these are the same as OpenReview pdf ids, so the corresponding PDFs are available at https://openreview.net/references/pdf?id=PDF_ID.  Each edit consists of an edit id, a list of source paragraph ids (which are indexes into the "body_text" list of the corresponding paper S2ORC file), and a list of target paragraph ids.

`review_comments.jsonl` contains the review comments, each uniquely identified by its `(doc_id, comment_id)` tuple.

`edit_labels_*.jsonl` files contain the comment-edit alignment labels for each data split (train, dev, test).  Each has a doc_id and comment id (corresponding to a comment in `review_comments.jsonl`), a list of aligned (positive) edit ids, and a list of negative edit ids.  For the synthetic data (train and dev) the negative ids are empty, indicating that there are no edits from the same document that can safely be treated as negative (due to the low recall of the data).  Code that loads these files and merges them with comments and paper_edits is in `scripts/train_revision_alignment.py`.

`alignment_human_eval.jsonl` contains the human evaluation labels (created without PDFs or author responses, only using the same information available to models), formatted in the same way as the `edit_labels_*` files.

`generated_edits.jsonl` contains edits generated by GPT for each comment in the test split.  It is the output of `scripts/generate_edits.py`, although changes in the OpenAI API may cause fluctuations.  Some records in the file additionally have an "annotations" field, which contains labels for the edit generation analysis in the ARIES paper.

`raw_split_ids.json` and `review_replies.jsonl` contains the raw split document ids and author responses/reviews used to construct the synthetic dataset.  They are consumed by `scripts/generate_synthetic_data.py`, although the outputs of that script are already available in the train and dev label files.

`gpt3_cache.sqlite` is a cache of the GPT inputs and responses that should be necessary to reproduce the main paper results.  To use it, make sure it is in the path pointed by `"cache_db_path"` in the GPT experiment configs (by default, it isn't, so GPT responses would be re-generated).


## Running experiments

This codebase is intended to be installed as a module that can be imported by the scripts in `scripts`.  After cloning the repo, first install dependencies with `pip install -r requirements.txt`, and then install this repository with `pip install -e .`.  Alternatively, the provided Dockerfile can be used to build a suitable environment.  Please also note that these instructions assume the dataset has been downloaded and extracted as described above into the `data/aries` directory.

To train models on the alignment task, run `python scripts/train_revision_alignment.py <config.json>` with an appropriate config file.  Config files corresponding to the experiments in the paper are in the `data/configs` directory.  Results for the experiment are stored in the path specified by the `output_dir` of the config, and the metrics are stored in the `test_metrics.json` file in that directory.  The main metrics to consider are the `devthresh_*` metrics, which are based on the decision threshold that maximizes f1 on the dev set.


## Inference

To predict which edits of a document align to a given comment, use the predict_many method of an aligner.  Example using a SPECTER bi-encoder:

```python
import transformers

import aries.util.s2orc
import aries.util.edit
from aries.alignment.doc_edits import DocEdits
from aries.util.data import iter_jsonl_files, index_by
from aries.alignment.biencoder import BiencoderTransformerAligner

doc_id = "EYCm0AFjaSS"
paper_edits = index_by(iter_jsonl_files(["data/aries/paper_edits.jsonl"]), "doc_id", one_to_one=True)[doc_id]
# Use aries.util.s2orc loader to handle back_matter merging
with aries.util.s2orc.S2orcFetcherFilesystem("data/aries/s2orc/") as fetcher:
    s2orc1 = aries.util.s2orc.load_s2orc(paper_edits["source_pdf_id"], fetcher)
    s2orc2 = aries.util.s2orc.load_s2orc(paper_edits["target_pdf_id"], fetcher)
doc_edits = DocEdits.from_list(s2orc1, s2orc2, paper_edits["edits"])
candidate_edits = [edit for edit in doc_edits.paragraph_edits if not edit.is_identical()]

comment = index_by(iter_jsonl_files(["data/aries/review_comments.jsonl"]), "doc_id")[doc_id][0]

aligner = BiencoderTransformerAligner(
    {
        "edit_input_format": "diff",
        "query_input_format": "comment_with_context",
        "add_diff_tokens": False,
        "max_seq_length": 512,
    },
    transformers.AutoModel.from_pretrained("allenai/specter"),
    transformers.AutoTokenizer.from_pretrained("allenai/specter"),
)

predictions = aligner.predict_many(
    [
        {
            "review_comment": comment["comment"],
            "context": comment["comment_context"],
            "candidates": candidate_edits,
        }
    ]
)[0]["predictions"]

predicted_edits = [(record["score"], record["edit"]) for record in predictions if record["pred"] == 1]
print("Comment:", comment["comment"])
# Expected result: edits 75, 78, 77
for score, edit in sorted(predicted_edits, key=lambda x: x[0], reverse=True)[:3]:
    print("\nEdit {} ({:0.2f}):".format(edit.edit_id, score))
    aries.util.edit.print_word_diff(edit.get_source_text(), edit.get_target_text(), color_format="ansi")
```

## Edit Generation

Edits can be generated with GPT models using `scripts/generate_edits.py`.  We provide an example config with the prompt used in the paper, which can be run with `python scripts/generate_edits.py configs/edit_generation_paper.json`.  However, to get the actual edits used for the paper analysis we recommend using the `generated_edits.jsonl` file in the dataset.

## Citation

```
@misc{darcy2023aries,
      title={ARIES: A Corpus of Scientific Paper Edits Made in Response to Peer Reviews}, 
      author={Mike D'Arcy and Alexis Ross and Erin Bransom and Bailey Kuehl and Jonathan Bragg and Tom Hope and Doug Downey},
      year={2023},
      eprint={2306.12587},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

The ARIES dataset is licensed under [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/).  The code in this repo is licensed under [Apache 2.0](https://apache.org/licenses/LICENSE-2.0).
