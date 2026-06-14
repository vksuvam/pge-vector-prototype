# Data Folder

This folder is not tracked by Git. You need to populate it manually before running ingestion.

## Expected Structure

```
data/
├── greenbook.pdf
└── tariffs/
    ├── (tariff_001.pdf)
    ├── (tariff_002.pdf)
    └── ... (first 50 sorted by filename)
```

## Where to Get the Files

### PG&E Greenbook
The Electric Rule 2 / Greenbook (PG&E Service Planning & Design standards) is publicly available:
- https://www.pge.com/tariffs/ERS.SHTML
- Direct search: "PG&E Greenbook electric rules"
- Download the main standards document and save as `data/greenbook.pdf`

### Tariff Documents
All 823 PG&E tariff PDFs are publicly available at:
- https://www.pge.com/tariffs/ERS.SHTML
- For this prototype, sort by filename and use the first 50
- Place all tariff PDFs inside `data/tariffs/`

## After Populating

Run ingestion once:
```bash
python run_ingestion.py
```

Qdrant storage will be created at `qdrant_storage/` (also gitignored).
You only need to run ingestion once unless the documents change.
