# Dataset

Pinned source: OCR-Data/ocr_data at 4cf05556e21bc54aca54a8b1f81ce172f3ef0477.
427,825,934,227 bytes; 414 files; 103 originals and 309 augmentation TAR shards.
Exact sample count is pending a complete scan. Repository-described per-shard
maximum is not treated as an exact count.

Inspected original sample_0000000.png/sample_0000000.json and three annotation
pairs from original and augmented shard 001. The JSON markdown field is the
ground-truth target, including Arabic text, diacritics and math. Metadata includes
dimensions, layout blocks, template, fonts, language, script, direction, confidence
scores and augmentation parameters. Original augmentation is null. Augmented
sample IDs include suffix _01 for the inspected aug1 shard.

Preparation must validate naming across every shard before canonicalizing source
document IDs. Keep all versions of a document in one split, then check duplicate
images/targets across splits. Proposed split is deterministic 90/5/5 with seed 42;
not yet generated. Preserve the original raw files without alteration.
