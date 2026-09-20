# Experiment versions and metrics

The core code in release 1.0 starts from the code-release snapshot
`87f7f6d409c023a7b5277bfba6073e57532132de` and applies the fixes listed in
CHANGELOG.md. The companion dataset tooling starts from dataset repository
snapshot `b83c733df761ebe0609f50ea53fe04294721542f`.

Paper results correspond to the archived experiment implementation and its
saved-result manifests. Version 1.0 provides the maintained runtime and
scorer; the paper's experiment drivers, configurations and saved outputs are
separate artifacts.

For the paper's LongMemEval results, the Utility column is a family-specific
evidence-availability proxy, not generated-answer accuracy. The BM25 reference
uses the original corpus; Support uses answer-bearing turn annotations.
LoCoMo is a separate generated-answer F1 evaluation.

LME-Gov's three 500-scenario paper resamples overlap. A full test split contains
702 scenarios from 54 base histories. Record scenario IDs, sampling seeds,
distinct source-history counts and model/runtime versions for each experiment.
Uncertainty estimates should account for repeated scenarios and shared histories.

The scorer validates submitted prediction records and reports coverage;
execution traces remain the responsibility of the experiment runner.
