# LME-Gov External Dataset Pointer

The full LME-Gov dataset is not stored in this code repository. It is hosted
separately as a Hugging Face dataset:

```text
https://huggingface.co/datasets/siufgdaias/lme-gov
```

After downloading or cloning the dataset repository, restore the files under
this relative path if you want to run the local scoring script and direct
Python loading:

```text
data/lme_gov/lme_gov.json
data/lme_gov/schema.json
data/lme_gov/splits/manifest.json
data/lme_gov/splits/scenarios/{train,dev,test}.jsonl.gz
data/lme_gov/splits/base_tasks/{train,dev,test}.jsonl.gz
```

This directory documents the expected local path layout used by the scoring
script and by direct Python loading. The full dataset is distributed
separately through the Hugging Face release.

For direct Python loading:

```python
from datasets import load_dataset

scenarios = load_dataset("siufgdaias/lme-gov", "scenarios")
base_tasks = load_dataset("siufgdaias/lme-gov", "base_tasks")
```
