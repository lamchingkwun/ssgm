# SSGM: Stability and Safety-Governed Memory

SSGM is a governed-memory middleware layer for LLM agents with writable
long-term memory. It sits between runtime memory extraction and the backing
memory store, and implements write admission, scoped retrieval,
provenance-aware checks, contradiction handling, and ledger-backed repair.

The full LME-Gov dataset is hosted separately. During anonymous review, use
the anonymized dataset identifier supplied with the submission or set:

```text
LME_GOV_DATASET_ID=<anonymized-dataset-id>
```

## Repository Contents

- `ssgm/`: core SSGM runtime implementation.
- `ssgm/prompts/`: judge and NLI prompt templates used by API-backed runs.
- `scripts/score_lme_gov_predictions.py`: scorer for LME-Gov prediction files.
- `data/`: lightweight notes pointing to the external LME-Gov dataset.

The repository intentionally excludes the full dataset, experiment outputs,
model logs, manuscript source, and table-generation artifacts. External
benchmarks used in the paper, including LoCoMo, should be obtained from their
original project sources and are not redistributed here.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default SSGM smoke path does not require an API key. API-backed judges and
NLI components read credentials from environment variables or explicit
constructor arguments.

## Minimal SSGM Example

```python
from ssgm import AccessContext, MemoryRecord, SSGMEngine

engine = SSGMEngine(mode="full_ssgm", stale_after=3)

accepted = engine.write(MemoryRecord(
    key="alice:preference:coffee",
    value="Alice prefers espresso.",
    tenant_id="alice",
    source="user",
    timestamp=1,
    provenance_ok=True,
))
print("accepted:", accepted)
print("write reason:", engine.last_write_result.reason)

ctx = AccessContext(actor_id="alice", tenant_id="alice", now_ts=2)
records = engine.retrieve("coffee preference", ctx, top_k=5)
print([record.key for record in records])
```

Run it directly:

```bash
python - <<'PY'
from ssgm import AccessContext, MemoryRecord, SSGMEngine
engine = SSGMEngine(mode="full_ssgm", stale_after=3)
engine.write(MemoryRecord(key="alice:preference:coffee", value="Alice prefers espresso.", tenant_id="alice", source="user", timestamp=1, provenance_ok=True))
ctx = AccessContext(actor_id="alice", tenant_id="alice", now_ts=2)
print([record.key for record in engine.retrieve("coffee preference", ctx, top_k=5)])
PY
```

## Loading LME-Gov

LME-Gov is published as two Hugging Face configs:

- `scenarios`: derived governance scenarios used for write admission, leakage,
  freshness, contradiction, and scoped-read evaluation.
- `base_tasks`: LongMemEval-derived base histories before governance
  perturbations.

Use the Hugging Face `datasets` package. During anonymous review, set
`LME_GOV_DATASET_ID` to the anonymized dataset mirror supplied with the
submission; after review, replace it with the final public dataset identifier.

```python
import os
from datasets import load_dataset

dataset_id = os.environ["LME_GOV_DATASET_ID"]
scenarios = load_dataset(dataset_id, "scenarios")
print(scenarios)
print(scenarios["test"][0]["scenario_uid"])

base_tasks = load_dataset(dataset_id, "base_tasks")
print(base_tasks)
```

For memory-limited machines, stream rows instead of materializing the full
split:

```python
from datasets import load_dataset

dataset_id = os.environ["LME_GOV_DATASET_ID"]
stream = load_dataset(
    dataset_id,
    "scenarios",
    split="test",
    streaming=True,
)
first = next(iter(stream))
print(first["scenario_uid"], len(first["writes"]))
```

The split JSONL files expose `expected_answer_items`, a stable list of
`{"key", "value"}` objects. The full nested source JSON in the dataset
repository preserves the original builder structure.

## Running SSGM On One LME-Gov Scenario

```python
import os
from datasets import load_dataset
from ssgm import AccessContext, MemoryRecord, SSGMEngine


def to_record(write):
    return MemoryRecord(
        key=write["key"],
        value=write["value"],
        tenant_id=write["tenant_id"],
        source=write["source"],
        timestamp=int(write["timestamp"]),
        confidence=float(write.get("confidence", 1.0)),
        mutable=bool(write.get("mutable", True)),
        provenance_ok=bool(write.get("provenance_ok", True)),
        provenance_attested=write.get("provenance_attested"),
        tags=list(write.get("tags") or []),
        memory_class=write.get("memory_class") or "ordinary",
        conflict_policy=write.get("conflict_policy") or "auto_update",
    )


dataset_id = os.environ["LME_GOV_DATASET_ID"]
row = load_dataset(dataset_id, "scenarios", split="test[:1]")[0]
engine = SSGMEngine(mode="full_ssgm", stale_after=int(row["stale_after"]))

accepted_keys = []
for write in row["writes"]:
    record = to_record(write)
    if engine.write(record, now_ts=int(row["now_ts"])):
        accepted_keys.append(record.key)

ctx = AccessContext(
    actor_id=row["probe_context"]["actor_id"],
    tenant_id=row["probe_context"]["tenant_id"],
    now_ts=int(row["now_ts"]),
)
retrieved = engine.retrieve(row["question"], ctx, top_k=5)

print("scenario:", row["scenario_uid"])
print("accepted writes:", len(accepted_keys))
print("retrieved keys:", [record.key for record in retrieved])
```

This example uses structured candidate records from LME-Gov. Raw-to-record
induction is a separate setting: raw text is first converted into candidate
records, then the same SSGM write gate is applied.

## Scoring LME-Gov Predictions

The scorer expects predictions keyed by `scenario_uid`. Each prediction row can
provide:

- `accepted_write_keys`: list of write keys admitted by the tested memory
  system.
- `retrieved_keys`: list of keys exposed by retrieval/read operations.
- Optional `system` or `mode`: system name used in the score summary.

Example prediction file:

```json
{"scenario_uid":"example_uid","system":"my_system","accepted_write_keys":["alice:key1"],"retrieved_keys":["alice:key1"]}
```

Score against the full nested dataset after downloading it from Hugging Face:

```bash
python scripts/score_lme_gov_predictions.py \
  --dataset data/lme_gov/lme_gov.json \
  --predictions outputs/my_predictions.jsonl \
  --split test \
  --output outputs/my_scores.json
```

You can also score against a scenario split shard, for example:

```bash
python scripts/score_lme_gov_predictions.py \
  --dataset data/lme_gov/splits/scenarios/test.jsonl.gz \
  --predictions outputs/my_predictions.jsonl \
  --output outputs/my_scores.json
```

The output reports benign-write acceptance, risky-write acceptance and block
rate, leakage success, and stale exposure, both overall and by scenario family.

## API-Backed Runs

`create_full_engine` wires the full SSGM stack with optional NLI and judge
components:

```python
from ssgm import create_full_engine

engine = create_full_engine(
    mode="full_ssgm",
    model="qwen3.5:9b",
    base_url="http://localhost:11434/v1",
    strict_api_failures=True,
)
```

For hosted OpenAI-compatible providers, pass `api_key`, `base_url`, and model
name explicitly or through environment variables. Strict API failure mode raises
on provider failures instead of silently falling back.

## Dataset Notes

LME-Gov is a project-maintained, LongMemEval-derived governance evaluation
suite. It should not be described as an independently maintained community
benchmark. When reporting results, include the split, sample size, sampling
seed, system configuration, judge/NLI configuration if used, and whether inputs
were structured records or raw text followed by record induction.

## Privacy And Secrets

The repository uses relative project paths. API credentials are read from
environment variables and are not stored in the repository. Some benchmark
fixtures contain synthetic attack strings such as fake `sk-...` tokens or fake
local paths; these are not deployment credentials. LME-Gov inherits public
LongMemEval-cleaned conversation content, including user-like snippets such as
code outputs, paths, and web references from the upstream public benchmark.

## Licensing And Attribution

The SSGM code is released under the repository license. LME-Gov is derived from
LongMemEval and is distributed through the separate Hugging Face dataset
repository with the upstream LongMemEval attribution and license notice; see
`THIRD_PARTY_NOTICE.md`. If you use LME-Gov, cite both SSGM and LongMemEval.
