from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from .config import DEFAULT_MODEL_KEY, MODEL_CATALOG
from .design import generate_design
from .ingest import ingest_pdfs


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "papers"
RULES_DIR = ROOT / "data" / "rules"
OUTPUT_DIR = ROOT / "outputs"
INDEX_PATH = OUTPUT_DIR / "index.json"
ALLOWED_EXT = {".pdf"}

app = Flask(__name__, template_folder=str(ROOT / "src" / "bio_exp_agent" / "templates"),
           static_folder=str(ROOT / "src" / "bio_exp_agent" / "static"))

def _cleanup_dirs() -> None:
    """Delete input and output folders to start fresh."""
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RULES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_uploads(files) -> List[str]:
    saved = []
    for file in files:
        if not file or not file.filename:
            continue
        filename = secure_filename(file.filename)
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            continue
        dest = DATA_DIR / filename
        file.save(dest)
        saved.append(filename)
    return saved


def _clear_inputs() -> None:
    for pdf in DATA_DIR.glob("*.pdf"):
        try:
            pdf.unlink()
        except OSError:
            pass
    for rule_file in RULES_DIR.glob("*.txt"):
        try:
            rule_file.unlink()
        except OSError:
            pass


@app.route("/", methods=["GET", "POST"])
def index():
    _ensure_dirs()
    message = ""
    design = ""
    uploaded = []

    if request.method == "POST":
        rules_text = (request.form.get("rules") or "").strip()
        model_key = (request.form.get("model_key") or DEFAULT_MODEL_KEY).strip()
        uploaded = _save_uploads(request.files.getlist("pdfs"))
        if not rules_text:
            message = "Please add design rules."
        elif not list(DATA_DIR.glob("*.pdf")):
            message = "Please upload at least one PDF."
        else:
            RULES_DIR.joinpath("design_rules.txt").write_text(rules_text, encoding="utf-8")
            try:
                ingest_pdfs(DATA_DIR, OUTPUT_DIR, model_key=model_key)
                design = generate_design(INDEX_PATH, rules_text, model_key=model_key)
                OUTPUT_DIR.joinpath("experiment_design.txt").write_text(design, encoding="utf-8")
                _clear_inputs()
                message = "Design generated."
            except Exception as exc:
                message = f"Error: {exc}"

    return render_template(
        "index.html",
        message=message,
        design=design,
        uploaded=uploaded,
        model_choices=MODEL_CATALOG,
        default_model=DEFAULT_MODEL_KEY,
    )


def main() -> None:
    _cleanup_dirs()
    _ensure_dirs()
    app.run(host="127.0.0.1", port=5000, debug=False)



if __name__ == "__main__":
    main()
