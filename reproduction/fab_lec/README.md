# Controlled FAB and LEC examples

This release contains the exact inputs and visible outputs for the 12-task FAB/LEC component study reported in the PolySea reviewer response and Supplementary Note 3. Each task was run under four component settings with two repetitions: 96 answer calls in total, preceded by four memory-extraction calls.

Start with [case C1](cases/C1.md), which combines a saved Kelvin reporting convention with the PolyNC polyimide classification task. Every case page includes both repetitions, the actual system/user inputs, visible answers, stop reasons and primary/secondary scores. Expand the input sections to inspect the injected memory and literature. No answer has been regenerated or rewritten for this release.

## Files

| File | Contents |
|---|---|
| [study_results.json](study_results.json) | All 12 tasks, four prior-session instructions, reference values, protocol settings, 96 visible output records and both sets of per-run scores. This is an exact copy of the archived study bundle. |
| [requests.json](requests.json) | All 96 actual system/user requests and context traces, including the memory and literature excerpts supplied to each answer call. |
| [memory_extraction.json](memory_extraction.json) | Inputs, visible outputs, model alias, stop reasons and usage for the four preparatory memory-extraction calls. |
| [evidence_index.json](evidence_index.json) | Source identifiers and publication/segment metadata for all 208 evidence records; supports the frozen source-identifier check. |
| [frozen_scorer.py](frozen_scorer.py) | Byte-identical original prespecified scorer. Use the wrapper below to evaluate this public bundle. |
| [recompute_scores.py](recompute_scores.py) | Reconstructs parsed results and verifies every published primary and secondary score without model/API calls. |
| [sha256_manifest.json](sha256_manifest.json) | Hashes of the release files. |

`run_id` joins a request, visible output and both score records. Every task has eight runs; each condition has 24. All 96 attempts are retained, including 25 empty visible outputs and 28 output-limit stops. A length-limited response may still contain parseable JSON. Missing or incomplete answers remain in the reported denominators.

## Recompute the scores

Python 3.10 or later and the standard library are sufficient:

```bash
cd reproduction/fab_lec
python recompute_scores.py
```

The command verifies all 96 primary scores and all 96 secondary-audit scores, then reports:

| Setting | All checks passed | Parseable JSON |
|---|---:|---:|
| FAB + LEC | 18/24 | 21/24 |
| No FAB | 8/24 | 21/24 |
| No LEC | 8/24 | 16/24 |
| Neither | 1/24 | 12/24 |

Use `python recompute_scores.py --json` for machine-readable summary output. This reproduces scoring of the saved outputs; it does not generate fresh model answers.

The manuscript uses the frozen **primary** scores. The secondary audit uniformly accepts an already-present top-level results object and extracts existing `class` values from richer class objects. It changes four partial scores without changing any complete-task outcome. Both versions remain available; source identification alone is not a check of every free-text claim.

## Protocol and model record

The experiment ran on 14 October 2026 with temperature 0.2, a 4,096-token output budget, two concurrent calls and a fresh session for every answer. Request order used seed 20261014. Enabled settings received the same saved memory and retrieved evidence. Retrieval used IDF-weighted overlap and the top three segments from the fixed local corpus. The exact requests, rather than a newly reconstructed approximation, are released here.

The requested/returned model alias was `/preset-models`. `deepseek-v4.1-falsh` is the deployment label provided for the experiment, not an independently verified checkpoint identifier. Although a request setting asked to disable thinking, the service still reported a thinking-type block; non-thinking execution was not verified. Only the visible final answers are published, together with stop reasons and usage. This is a fixed-corpus component study; the package contains no new PolyOpus checkpoint evaluation or live-web retrieval experiment.

The 96 runs are repeated measurements of 12 tasks, not 96 independent scientific questions. The results describe this controlled set and output budget. Empty answers, unsuccessful tasks, retrieval mismatches and formatting deviations can be inspected alongside successful cases.

## Literature sources

- PolyNC: [10.1039/D3SC05079C](https://doi.org/10.1039/D3SC05079C).
- Polymer informatics with multi-task learning: [10.1016/j.patter.2021.100238](https://doi.org/10.1016/j.patter.2021.100238).
- PolyTAO: [10.1038/s41524-024-01466-5](https://doi.org/10.1038/s41524-024-01466-5).

Requests retain the text actually supplied to the model, including extraction/encoding artifacts. Excerpts and table transcriptions are attributed to their source publications and retain the original publications' rights; the repository's software license does not relicense third-party article text. The release includes the excerpts used in the actual inputs, not complete article files.

## Case index

| Case | Group | Task |
|---|---|---|
| [M1](cases/M1.md) | memory | Temperature reporting convention |
| [M2](cases/M2.md) | memory | Halogen-free screening |
| [M3](cases/M3.md) | memory | Reserved batch and parent grouping |
| [M4](cases/M4.md) | memory | Nominal band-gap objective |
| [E1](cases/E1.md) | literature | PolyNC split and augmentation |
| [E2](cases/E2.md) | literature | Atomization-energy provenance |
| [E3](cases/E3.md) | literature | Polyimide class thresholds |
| [E4](cases/E4.md) | literature | PolyTAO conditioning descriptors |
| [C1](cases/C1.md) | combined | Temperature convention and polyimide classification |
| [C2](cases/C2.md) | combined | Reserved batch and published split method |
| [C3](cases/C3.md) | combined | Screening constraint and conditioning labels |
| [C4](cases/C4.md) | combined | Design objective and descriptor provenance |
