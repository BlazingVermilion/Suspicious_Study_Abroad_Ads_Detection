# Project structure

The v2 layout intentionally avoids the extra `src/suspicious_ads/` nesting. Scripts are grouped directly under `src/` by pipeline role:

```text
src/collection/
src/preprocessing/
src/annotation/
src/extraction/
src/weak_labeling/
src/modeling/
src/utils/
```

This keeps import paths and CLI paths shorter while still making the pipeline readable.
