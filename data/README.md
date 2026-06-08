# Data directory

The repository keeps folder structure but ignores raw/private datasets by default.

- `raw/instagram/metadata`: raw JSON/JSONL crawl metadata.
- `raw/instagram/screenshots`: raw Instagram screenshots.
- `processed/normalized`: normalized corpus.
- `processed/annotations`: human NER/RE annotation exports.
- `processed/splits`: binary gold evaluation split and silver pool.
- `processed/ner_re`: NER/RE CV metrics and enriched prediction CSVs.
- `processed/weak_labeling`: weak-labeled silver train and debug artifacts.
- `processed/modeling_inputs`: optional manually curated final model inputs.


## Prepared inputs

`data/prepared/` contains curated input files that allow the project to be run without crawling Instagram again:

- `normalized_posts.csv`
- `silver_pool.csv`
- `binary_gold_eval.csv`
- `ner_re_gold_annotated_subset.json`

These files are useful for reproducibility, but they may contain public Instagram captions, URLs, account names, and names appearing in testimonial captions. Keep the full prepared data only in a private repository unless your data-release plan explicitly permits public release.
