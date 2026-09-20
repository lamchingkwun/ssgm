# Third-Party Notices

The original LongMemEval copyright and permission notice is reproduced in
`third_party/LongMemEval-LICENSE`. It applies to the inherited LongMemEval
material; this release does not replace it with the SSGM copyright notice.

This repository contains the SSGM code release. The full LME-Gov dataset is
hosted separately in the Hugging Face dataset repository.

## LongMemEval

LME-Gov is derived from LongMemEval native histories. The separate dataset
repository contains LongMemEval-derived histories plus SSGM governance
perturbations, schema fields, and labels. The original LongMemEval project is
released under the MIT License. SSGM-authored perturbations, label protocols,
schema, and tooling are released under this repository's MIT License.

- Upstream project: https://github.com/xiaowu0162/LongMemEval
- Upstream dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

If you use LME-Gov, cite both SSGM and LongMemEval.

The LongMemEval-derived text may include user-like snippets from the public
LongMemEval-cleaned release, such as code outputs, file-path-like strings, web
references, and answer-bearing facts. These should be treated as benchmark
content inherited from the upstream public dataset.

## Synthetic Security Fixtures

Some examples and generated records contain synthetic attack strings, fake
secret-like tokens, or fake local paths. These are benchmark fixtures, not
credentials or real deployment secrets.
