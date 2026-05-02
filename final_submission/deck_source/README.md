# Deck Source

This folder contains the artifact-tool slide modules used to generate:

`../confidence_triggered_swarm_final_presentation.pptx`

The modules use the canonical generated figures in `../../runs/professor_ready/`.
They do not introduce new experiment claims.

## Rebuild

From the repository root:

```bash
/Users/tanushsavadi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
  /Users/tanushsavadi/.codex/plugins/cache/openai-primary-runtime/presentations/26.430.10722/skills/presentations/scripts/build_artifact_deck.mjs \
  --workspace /private/tmp/codex-presentations/manual-cs690nn-swarm-deck-10-12 \
  --slides-dir "/Users/tanushsavadi/Documents/CS 690NN/Final Project/final_submission/deck_source" \
  --out "/Users/tanushsavadi/Documents/CS 690NN/Final Project/final_submission/confidence_triggered_swarm_final_presentation.pptx" \
  --preview-dir /private/tmp/codex-presentations/manual-cs690nn-swarm-deck-10-12/preview \
  --layout-dir /private/tmp/codex-presentations/manual-cs690nn-swarm-deck-10-12/layout/final \
  --contact-sheet /private/tmp/codex-presentations/manual-cs690nn-swarm-deck-10-12/preview/contact-sheet.png \
  --slide-count 12
```

Then check layout quality:

```bash
/Users/tanushsavadi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
  /Users/tanushsavadi/.codex/plugins/cache/openai-primary-runtime/presentations/26.430.10722/skills/presentations/scripts/check_layout_quality.mjs \
  --layout /private/tmp/codex-presentations/manual-cs690nn-swarm-deck-10-12/layout/final \
  --warn-only
```

The last verified build had 0 layout errors and 0 warnings.
