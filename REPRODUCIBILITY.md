# Reproducibility of the PolySea revision

The backend includes the revision's capability policy, persistent-memory fixes and
shared context assembly with FAB/LEC switches. Source files are in
`backend/app/services`; the two files in `tests` contain 17 local checks. Run
`python -m unittest discover -s tests -v` after installing backend dependencies.
Tests use temporary memory and mocked model responses; they make no external calls.

Final research assets are provided separately:

- [PolyOpus training and test inputs, final predictions and scoring](dataset/polyopus/README.md).
- [Complete merged PolyOpus weights](https://huggingface.co/hkqiu/PolyOpus/tree/2b01aae91c3ac81e4f06d9384b18e0aad045d775).
- [Completed GPT-6-Astra evaluation](reproduction/current_models/gpt6_astra_20261015/README.md).
- [New GraphConv checkpoint and CPU generation assessment](reproduction/gnn/revision_run_20261015/README.md).

The local evidence example has seven attributed records. The reported real-model
FAB/LEC study used a different fixed 208-segment corpus and preassembled context;
the manuscript reports its 4 memory-extraction calls and 96 formal answers. The
17 software checks do not reproduce or replace those model-answer outcomes.

Capability labels are rule-based and uncalibrated. Memory matching and reflection
are deterministic; automatic scientific conflict resolution is not claimed. The
standalone GNN reconstruction remains separate from the unfinished online
`deepchem_gnn` option. Historical DFT inputs, original NHscore implementation and
judge API response provenance are not represented as recovered by this release.

Runtime memory, user uploads, credentials, generated models and process logs stay
outside the publication. Public prediction tables contain final results rather
than per-request logs. See each dataset or model's source attribution and terms.


[Structural overlap and duplicate audit](dataset/polyopus/structure_audit/README.md) and [PolyNC source license](dataset/polyopus/PolyNC_LICENSE.txt).
