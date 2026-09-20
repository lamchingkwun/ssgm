# SSGM: Stability and Safety Governed Memory

SSGM governs writes, reads and repairs in persistent agent memory. It provides
structured-record admission, scoped retrieval, contradiction handling and an
auditable evidence ledger.

**Version 1.0 · Python 3.10+**

See [CHANGELOG.md](CHANGELOG.md) for changes and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for experiment versions and metrics.

Authors: Chingkwun Lam, Jiaxin Li, KoPang, Lingfei Zhang and Zhao Kuo,
Jinan University. Corresponding author: Zhao Kuo.

## Install and test

From the unpacked code directory:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## No-model smoke example

```python
from ssgm import AccessContext, MemoryRecord, SSGMEngine

engine = SSGMEngine(mode="full_ssgm", stale_after=3, use_embeddings=False)
record = MemoryRecord(
    key="alice:preference:coffee", value="Alice prefers espresso.",
    tenant_id="alice", source="user", timestamp=1, provenance_ok=True,
)
print(engine.write(record))
context = AccessContext(actor_id="alice", tenant_id="alice", now_ts=2)
print([r.key for r in engine.retrieve("coffee preference", context, top_k=5)])
```

Expected output: `True` and `['alice:preference:coffee']`.
This example runs without models or network access. With embeddings disabled,
retrieval uses recency ordering and provenance checks use record confidence.

## Ollama judge and NLI

Install and start Ollama separately, then make the required models available:

```bash
ollama pull qwen3.5:9b
ollama pull nomic-embed-text-v2-moe
```

```python
from ssgm import create_full_engine

engine = create_full_engine(
    judge_backend="ollama",
    model="qwen3.5:9b",
    base_url="http://localhost:11434/v1",
    strict_api_failures=True,
)
```

The factory defaults to Ollama, including NLI with abstention
threshold 0.4. Set `enable_nli=False` if only the write judge is wanted.
Embeddings use a separate Ollama endpoint, configured through
`embedding_base_url` or `OLLAMA_BASE_URL`. Failures raise by default.
Setting `allow_embedding_fallback=True` enables recency-based retrieval and
confidence-based provenance checks when embeddings fail.

## Compact NLI alternative

```bash
python -m pip install "torch>=2.2" "transformers>=4.40,<5" "sentencepiece>=0.2"
```

```python
engine = create_full_engine(
    judge_backend="compact_nli",
    model="cross-encoder/nli-deberta-v3-small",
    use_embeddings=False,
)
```

The first model use may download Hugging Face weights. Offline execution
requires those weights to be available locally. This config uses compact NLI
for both judging and contradiction checks; it does not contact Ollama.

For hosted models, select `judge_backend="openai_responses"` or
`"minimax"`, choose the model, and set `OPENAI_API_KEY`
or `MINIMAX_API_KEY` in your environment. The OpenAI backend accepts a
matching Responses endpoint through `base_url`. The MiniMax write judge uses
its built-in MiniMax endpoint; a custom `base_url` only configures its NLI
component. The OpenAI backend requires the Responses API. Store credentials
in environment variables.

## Data and evaluation

The dataset is distributed separately:
https://huggingface.co/datasets/siufgdaias/lme-gov

```python
from datasets import load_dataset

rows = load_dataset(
    "siufgdaias/lme-gov", "scenarios", split="test", streaming=True,
)
first = next(iter(rows))
print(first["scenario_uid"], len(first["writes"]))
```

Dataset content version remains **1.0.0** (500 base histories, 6,500 scenarios).
Tooling version is **1.0**. The data are structured governance scenarios
derived from LongMemEval, not a general QA or unstructured induction benchmark.

### Prediction format

Each scenario needs write decisions and explicit results for every designated
leakage/stale probe. A returned record means exposure; a tested read that
returns no record is `false`.

```json
{"scenario_uid":"example_uid","system":"my_system","write_decisions":{"alice:key1":"allow","alice:risky":"quarantine"},"probe_results":{"bob:private":false}}
```

Use the actual IDs and keys in the dataset. `accepted_write_keys` may replace
`write_decisions`; that list is a complete statement of admitted keys.
For `write_decisions`, allow/accept is accepted; quarantine/block/defer is
not admitted. Unknown strings are errors.

`retrieved_keys` can prove positive exposure. A key missing from that list
does not prove its read probe ran, so explicit negative `probe_results` are
required.

```bash
python scripts/score_lme_gov_predictions.py --dataset /path/to/test.jsonl.gz --predictions predictions.jsonl --output scores.json
```

The scorer accepts nested JSON, JSONL and gzip JSONL. It rejects duplicate or
unknown scenario IDs and incomplete coverage by default. For an intentional
subset add `--allow-partial`; the output reports coverage. Missing-decision
policies are recorded in the output so assumed outcomes remain identifiable.

Outputs include benign/risky acceptance, **risky non-admission**, leakage and
stale exposure. Non-admission combines quarantine and block. Metrics without a
denominator are `null`, not zero or 100%. Errors return a nonzero exit status
and suppress summary scores.

## Scope and limitations

- This package contains the core runtime, two model prompt templates, scorer
  and regression tests. It does not include the full paper experiment drivers,
  saved outputs, table generation, or the LME-Gov construction pipeline.
- Source/tenant/attestation fields must come from trusted infrastructure.
  The middleware does not authenticate them cryptographically.
- Quarantine excludes records from normal retrieval. A human approval UI,
  concurrent-write coordination and verifiable data erasure are not provided.
- LLM judge requests use up to 1,000 characters in single calls and 400 in
  batch calls. Cache keys cover the full input; model decisions cover the
  text sent in the request.
- This is research software. Model-dependent behavior requires evaluation
  under the intended deployment configuration.

## License and attribution

Code is MIT licensed; see [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md). Upstream data retain their
original license and attribution. Cite SSGM and LongMemEval when using LME-Gov;
[CITATION.cff](CITATION.cff) records the SSGM paper title and author list.

AI tools assisted code development and internal manuscript review. The authors
are responsible for the released code and research claims.

## Verification

Checks passed in a clean Windows Python 3.12 environment:

- 12 regression tests covering cache isolation, configuration, access scope,
  quarantine and scoring.
- The no-model smoke example.
- `pip check`.

Install the runtime and data-loading dependencies from `requirements.txt`.
Compact NLI requires the additional packages listed above. Model responses are
mocked in the regression tests.
