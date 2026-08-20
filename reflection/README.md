# Reflection

`Reflection` is a generated self-critique workspace built from `soul` plus local runtime evidence.

Run it with:

```bash
python3 scripts/reflection_dossier.py
python3 scripts/reflection_prepare_candidate.py
```

Each run writes:
- `reflection/runs/<timestamp>_<hash>/manifest.json`
- `reflection/runs/<timestamp>_<hash>/observations.json`
- `reflection/runs/<timestamp>_<hash>/dossiers.json`
- `reflection/runs/<timestamp>_<hash>/summary.md`

Preparing a candidate writes:
- `reflection/candidates/<candidate_id>/manifest.json`
- `reflection/candidates/<candidate_id>/dossier.json`
- `reflection/candidates/<candidate_id>/proposal.json`
- `reflection/candidates/<candidate_id>/instructions.md`
- `reflection/candidates/<candidate_id>/context/`
- `reflection/candidates/<candidate_id>/worktree/`

The dossier pipeline is read-only. Candidate preparation is also read-only with respect to production code; it creates an isolated workspace under `reflection/`.
