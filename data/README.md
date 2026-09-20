# Data Layout

This code repository does not vendor benchmark-scale datasets.

The full LME-Gov release is hosted in the separate Hugging Face dataset
repository:

```text
https://huggingface.co/datasets/siufgdaias/lme-gov
```

For local scoring with `scripts/score_lme_gov_predictions.py`, download that
repository and place its `data/lme_gov/` directory here so one of the following
paths exists:

```text
data/lme_gov/lme_gov.json
data/lme_gov/splits/scenarios/test.jsonl.gz
```

Keeping the dataset outside the code repository avoids large-file limits.
Other third-party evaluation benchmarks used in the manuscript, including
LoCoMo, should be obtained from their original project sources and are not
redistributed here.

For direct Python loading, use:

```python
from datasets import load_dataset

scenarios = load_dataset("siufgdaias/lme-gov", "scenarios")
base_tasks = load_dataset("siufgdaias/lme-gov", "base_tasks")
```
