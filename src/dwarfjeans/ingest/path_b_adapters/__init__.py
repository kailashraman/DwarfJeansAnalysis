"""Per-paper adapters for Stage 0b Path B ingest.

Each module exposes a `load(staged_dir, registry_row) -> (arrays, meta_extra)`
function that maps the paper's source schema to our canonical column names.
The driver `dwarfjeans.ingest.stage0b_pathb` looks up the adapter by `<bibkey>`
(derived from the LVDB `ref_vlos` column: lowercased first_author + year).
"""
