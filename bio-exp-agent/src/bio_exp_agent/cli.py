import argparse
import json
from pathlib import Path

from .ingest import ingest_pdfs
from .design import generate_design


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cmd_ingest(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    out_dir = Path(args.out)
    _ensure_dir(out_dir)
    index_path = ingest_pdfs(input_dir, out_dir)
    print(f"Index written to {index_path}")


def cmd_design(args: argparse.Namespace) -> None:
    index_path = Path(args.index)
    out_dir = Path(args.out)
    _ensure_dir(out_dir)
    rules_text = args.rules
    if args.rules_file:
        rules_text = Path(args.rules_file).read_text(encoding="utf-8")
    design = generate_design(index_path, rules_text)
    out_path = out_dir / "experiment_design.txt"
    out_path.write_text(design, encoding="utf-8")
    print(f"Design written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bio experiment agent")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest PDFs and build index")
    ingest.add_argument("--input", required=True, help="Input PDF folder")
    ingest.add_argument("--out", required=True, help="Output folder")
    ingest.set_defaults(func=cmd_ingest)

    design = sub.add_parser("design", help="Generate experiment design")
    design.add_argument("--index", required=True, help="Path to index.json")
    rules_group = design.add_mutually_exclusive_group(required=True)
    rules_group.add_argument("--rules", help="Design rules as text")
    rules_group.add_argument("--rules-file", help="Path to rules .txt file")
    design.add_argument("--out", required=True, help="Output folder")
    design.set_defaults(func=cmd_design)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
