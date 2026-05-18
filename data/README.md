# Data Layout

This code repository does not vendor benchmark-scale datasets.

The full LME-Gov release is hosted separately. During anonymous review, use
the anonymized dataset identifier supplied with the submission or set:

```text
LME_GOV_DATASET_ID=<anonymized-dataset-id>
```

For local scoring with `scripts/score_lme_gov_predictions.py`, download that
repository and place its `data/lme_gov/` directory here so one of the following
paths exists:

```text
data/lme_gov/lme_gov.json
data/lme_gov/splits/scenarios/test.jsonl.gz
```

Keeping the dataset outside the GitHub code repository avoids large-file
failures and keeps this code artifact compact for review and replication.
Other third-party evaluation benchmarks used in the manuscript, including
LoCoMo, should be obtained from their original project sources and are not
redistributed here.

For direct Python loading, use:

```python
from datasets import load_dataset
import os

dataset_id = os.environ["LME_GOV_DATASET_ID"]
scenarios = load_dataset(dataset_id, "scenarios")
base_tasks = load_dataset(dataset_id, "base_tasks")
```
