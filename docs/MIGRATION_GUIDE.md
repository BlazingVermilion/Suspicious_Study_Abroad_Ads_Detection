# Migration guide from the old project layout

| Old path/name | New path/name |
|---|---|
| `src/ner_re_model/ner_re_extraction_pipeline.py` | `src/extraction/train_ner_re_extractor.py` |
| `src/data_handling/cleaning_dataset.py` | `src/preprocessing/build_normalized_dataset.py` |
| `src/data_handling/goldsubset_valdidation.py` | `src/annotation/build_binary_gold_split.py` |
| `src/data_handling/weak_labeling_pipeline.py` | `src/weak_labeling/build_weak_labels.py` |
| `src/crawler/google_search_for_suspicious_account.py` | `src/collection/discover_suspicious_accounts.py` |
| `src/crawler/crawl_protential_suspicious_posts.py` | `src/collection/crawl_suspicious_posts.py` |
| `src/crawler/crawl_legitimate_posts.py` | `src/collection/crawl_legitimate_posts.py` |
| `src/classifier_model/multi_training_model.py` | `src/modeling/train_classifier.py` |
| `instagram_session.json` | `secrets/instagram_session.json` |
| `data/raw/metadata/pre_labeled_data.json` | `data/raw/instagram/metadata/suspicious_candidate_posts.json` |
| `data/raw/metadata/legitimate_seed_data.json` | `data/raw/instagram/metadata/legitimate_seed_posts.json` |
| `data/processed/normalize_dataset.csv` | `data/processed/normalized/normalized_posts.csv` |
| `data/processed/gold_NER_RE_annotated_subset.json` | `data/processed/annotations/ner_re_gold_annotated_subset.json` |
| `data/processed/binary_gold_subset_173.csv` | `data/processed/splits/binary_gold_eval.csv` |
| `data/processed/pre_labeled_silver_subset.csv` | `data/processed/splits/silver_pool.csv` |
| `data/processed/ner_re_extraction_outputs/` | `data/processed/ner_re/` |
| `data/processed/silver_training_dataset.csv` | `data/processed/weak_labeling/silver_train.csv` |
| `data/processed/uncertain.csv` | `data/processed/weak_labeling/uncertain_posts.csv` |
| `data/processed/all_scored_dataset_debug.csv` | `data/processed/weak_labeling/weak_labeling_scored_posts_debug.csv` |
| `data/processed/phase9_10_11_outputs_multiseed/` | `outputs/classifier_evaluation/` |

Use `scripts/migrate_existing_project.py` to copy common files from an old local folder into the new layout.
