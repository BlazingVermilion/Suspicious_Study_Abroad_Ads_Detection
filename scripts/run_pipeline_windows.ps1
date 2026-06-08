$env:PROJECT_ROOT = (Get-Location).Path

python src/preprocessing/build_normalized_dataset.py `
  --input-dir data/raw/instagram/metadata `
  --output data/processed/normalized/normalized_posts.csv

python src/annotation/build_binary_gold_split.py `
  --normalize data/processed/normalized/normalized_posts.csv `
  --annotation-subset data/processed/annotations/ner_re_gold_annotated_subset.json `
  --out-dir data/processed/splits

python src/extraction/train_ner_re_extractor.py `
  --nerre-json data/processed/annotations/ner_re_gold_annotated_subset.json `
  --silver-csv data/processed/splits/silver_pool.csv `
  --binary-gold-csv data/processed/splits/binary_gold_eval.csv `
  --out-dir data/processed/ner_re

python src/weak_labeling/build_weak_labels.py

python src/modeling/train_classifier.py `
  --train data/processed/weak_labeling/silver_train.csv `
  --gold data/processed/ner_re/predictions/binary_gold_with_ner_re_ensemble.csv `
  --output_dir outputs/classifier_evaluation
