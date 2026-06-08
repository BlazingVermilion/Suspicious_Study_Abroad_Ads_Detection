# Data schema

## `data/processed/normalized/normalized_posts.csv`

Canonical columns produced by `src/preprocessing/build_normalized_dataset.py`:

```text
post_id, post_url, platform, account_name, source_file,
caption_text, clean_text, model_text,
hashtags, mentions, url_count, emoji_count, external_link,
screenshot_url, posting_time, language, seed_label
```

Policy:

- `clean_text`: core caption only; used for NER/RE and rule scoring.
- `model_text`: caption with abstracted URL/mention/emoji markers and normalized hashtags; useful for BERT/MLP experiments.
- `account_name`: metadata only; not injected into model text.
- `seed_label`: before weak labeling, only `none` or `legitimate`.

## Binary labels

`data/processed/splits/binary_gold_eval.csv` and `data/processed/weak_labeling/silver_train.csv` use `seed_label` as the binary label column with values:

```text
normal, suspicious
```
