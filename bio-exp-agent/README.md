# Bio Experiment Agent (MVP)

This project ingests biology experiment papers (PDF), extracts text (with optional OCR), keeps protocol sections intact, summarizes the rest, and generates experiment design ideas from user rules.

## Setup (Conda)

```bash
conda env create -f environment.yml
conda activate bio-exp-agent
pip install -e .
```

## Quick start

1) Put PDFs under `data/papers/`
2) Ingest + summarize:

```bash
python -m bio_exp_agent ingest --input data/papers --out outputs
```

3) Generate a design from rules:

```bash
python -m bio_exp_agent design --index outputs/index.json --rules-file data/rules/design_rules.txt --out outputs
```

## Local web UI

```bash
python -m bio_exp_agent.web_app
```

Then open `http://127.0.0.1:5000` in your browser.

## Notes
- Protocol sections are preserved and not summarized.
- OCR via PaddleOCR is used only when a PDF page has little/no text.
- Default models are small and local. You can change them in `src/bio_exp_agent/config.py`.

## Layout
- `src/bio_exp_agent/` application code
- `data/papers/` input PDFs
- `outputs/` summaries, extracted protocol blocks, and index
- `data/rules/` design rules text files
