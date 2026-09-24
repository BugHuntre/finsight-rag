"""Download report PDFs listed in a manifest into data/raw/.

The manifest (default: scripts/sources.csv) has the columns::

    company,fiscal_year,doc_type,url

For example::

    Infosys,FY2025,annual_report,https://<investor-relations-link>/annual-report.pdf

Files are saved as ``<Company>_<FY####>_<doc_type>.pdf`` (spaces in the company become
hyphens), which is the naming convention the ingestion pipeline expects. Copy the
links from each company's investor-relations page, NSE or BSE.
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "Mozilla/5.0 (FinSight RAG data downloader)"


def download(url: str, dest: Path) -> None:
    """Fetch ``url`` to ``dest`` (written atomically via a .part file)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)
    tmp.replace(dest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--manifest", type=Path, default=ROOT / "scripts" / "sources.csv")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = f"{row['company'].strip().replace(' ', '-')}_{row['fiscal_year'].strip()}"
            dest = args.out / f"{name}_{row['doc_type'].strip()}.pdf"
            if dest.exists():
                print(f"[skip] {dest.name}")
                continue
            try:
                download(row["url"].strip(), dest)
                print(f"[ok]   {dest.name}")
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"[fail] {dest.name}: {exc}")


if __name__ == "__main__":
    main()
