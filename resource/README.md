# `resource/` — Semantic text data

This directory holds the entity/relation text files consumed by SCR's semantic
encoding (BERT). The files are **not bundled** in the repository — download them
from the Google Drive link in the top-level [README](../README.md#data-acquisition)
and place them here:

```
resource/
├── FB-entity2text.txt
├── FB-relation2text.txt
├── WK-entity2text.txt
├── WK-relation2text.txt
├── WN-entity2text.txt
└── WN-relation2text.txt
```

The derived embedding caches (`*_entembeds.pth`, `*_graph_data.pth`) are generated
automatically on the first run and are gitignored.
