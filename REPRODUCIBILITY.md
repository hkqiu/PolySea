# PolySea research assets and reproducibility

- [PolyOpus training/test data, final predictions and scoring](dataset/polyopus/README.md).
- [Complete merged PolyOpus weights](https://huggingface.co/hkqiu/PolyOpus/tree/2b01aae91c3ac81e4f06d9384b18e0aad045d775).
- [Structure overlap and fixed-representative metrics](dataset/polyopus/structure_audit/README.md).
- [GPT-6-Astra evaluation](reproduction/current_models/gpt6_astra_20261015/README.md).
- [GraphConv checkpoint and CPU generation assessment](reproduction/gnn/revision_run_20261015/README.md).

The backend implements capability routing, persistent memory and shared FAB/LEC
context assembly in `backend/app/services`. Capability labels are rule-based;
memory matching and reflection are deterministic. The standalone trained GNN
is under `reproduction/gnn`; the online `deepchem_gnn` option remains unimplemented.

After installing backend dependencies, run the 17 local software checks:

```bash
python -m unittest discover -s tests -v
```

Tests cover routing, notices, context/tool switches and memory behavior using
temporary memory and mocked responses. The backend includes a seven-record local
evidence example. The reported FAB/LEC answer study used a fixed 208-segment corpus,
preassembled context, four memory-extraction calls and 96 formal answers. The [complete study release](reproduction/fab_lec/README.md) provides the actual inputs, all visible outputs, per-run scores and a scoring replay script. The [C1 comparison](reproduction/fab_lec/cases/C1.md) shows a concrete four-setting example; [all 12 cases](reproduction/fab_lec/README.md#case-index) are available for inspection.

Dataset and model pages provide their replay commands, methods, source attribution
and licenses.
