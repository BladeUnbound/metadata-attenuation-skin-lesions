"""Download one public ISIC Archive collection (images + flattened metadata) via the open API.

Usage: python fetch_isic_collection.py <collection_id> <name>
Writes research/data/<name>/metadata.csv and research/data/<name>/images/<isic_id>.jpg.
Resume-safe: existing images are skipped. Licence is recorded per image.
"""
import csv
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
API = "https://api.isic-archive.com/api/v2/images/search/?collections={cid}&limit=100"


def get(url, tries=5):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))


def main(cid, name):
    out = RESEARCH / f"data/{name}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    url, rows = API.format(cid=cid), []
    while url:
        page = json.loads(get(url))
        for r in page["results"]:
            row = {"isic_id": r["isic_id"], "licence": r.get("copyright_license"),
                   "url": r["files"]["full"]["url"]}
            for group, fields in r.get("metadata", {}).items():
                if isinstance(fields, dict):
                    row.update({k: v for k, v in fields.items()})
            rows.append(row)
        url = page.get("next")
        print(f"metadata: {len(rows)}/{page.get('count')}", flush=True)
    cols = sorted({k for r in rows for k in r})
    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    def fetch(r):
        p = out / "images" / f"{r['isic_id']}.jpg"
        if p.exists() and p.stat().st_size > 0:
            return 0
        p.write_bytes(get(r["url"]))
        return 1

    done = 0
    with ThreadPoolExecutor(8) as ex:
        for i, n in enumerate(ex.map(fetch, rows), 1):
            done += n
            if i % 500 == 0:
                print(f"images: {i}/{len(rows)}", flush=True)
    print(f"finished {name}: {len(rows)} rows, {done} new images", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
