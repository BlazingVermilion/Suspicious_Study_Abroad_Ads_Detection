# Pipeline input/output audit

This file documents the expected input and output contract for each script.

| Script | Main input | Main output | Notes |
|---|---|---|---|
| `src/collection/save_instagram_session.py` | Manual Instagram login | `secrets/instagram_session.json` | Gitignored. |
| `src/collection/discover_suspicious_accounts.py` | Google/Instagram search results | `data/raw/instagram/metadata/account_registry.json` | Also writes `suspicious_account_discovery_log.jsonl`. |
| `src/collection/crawl_suspicious_posts.py` | `account_registry.json`, Instagram session | `suspicious_candidate_posts.json` | Raw unlabeled candidate posts; `seed_label=none`. |
| `src/collection/crawl_legitimate_posts.py` | Trusted account config, Instagram session | `legitimate_seed_posts.json` | Raw legitimate seed posts; `seed_label=legitimate`. |
| `src/preprocessing/build_normalized_dataset.py` | Raw metadata JSON directory | `normalized_posts.csv` | Filters English, deduplicates by `post_id`, outputs stable schema. |
| `src/annotation/build_binary_gold_split.py` | `normalized_posts.csv`, NER/RE annotation keys | `binary_gold_eval.csv`, `silver_pool.csv` | Excludes NER/RE annotation subset from binary gold candidates. JSON annotation exports are supported. |
| `src/extraction/train_ner_re_extractor.py` | NER/RE JSON, silver pool, binary gold | `data/processed/ner_re/cv_results/*`, `predictions/*` | Uses OOF predictions for annotated silver rows and ensemble predictions otherwise. |
| `src/weak_labeling/build_weak_labels.py` | silver+NER/RE predictions, NER/RE CV metrics, annotation JSON | `data/processed/weak_labeling/silver_train.csv` | Removes uncertain rows from classifier training output; debug files stay separate. |
| `src/modeling/train_classifier.py` | `silver_train.csv`, binary gold with NER/RE | `outputs/classifier_evaluation/*` | Runs baselines, ablations, DistilBERT/NER-RE fusion, and gold error analysis. |
