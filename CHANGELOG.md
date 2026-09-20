# Changelog

## 1.0 — 2026-09-19

- Cache full candidate content, source, attestation requirements and judging
  policy/model identity consistently in single and batch paths.
- Select Ollama explicitly by default; configure compact NLI separately.
- Make embedding failures explicit unless fallback is deliberately enabled.
- Accept native `allow` decisions in the scorer; reject unknown decisions,
  duplicate/unknown scenarios and incomplete coverage by default.
- Require evidence for negative read probes. Rename risky block rate to
  risky non-admission rate and use null for undefined denominators.
- Use the same scorer for the code and dataset packages, including gzip JSONL.
- Add regression tests and upstream license text.
- Use one requirements.txt for runtime and dataset loading; document compact
  NLI's optional model dependencies in the README.

Runtime and scoring behavior changes are listed above. Experiment versions are
documented in REPRODUCIBILITY.md. Dataset content remains version 1.0.0.
