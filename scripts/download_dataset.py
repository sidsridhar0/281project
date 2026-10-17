#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ECOMMONS_ITEM = "https://ecommons.cornell.edu/items/6d470f13-bac5-4b0e-94f2-4eac93d677cb"
CONTENT = "https://ecommons.cornell.edu/server/api/core/bitstreams/{uuid}/content"

FILES = {
    "readme": {
        "name": "Bax_LiteratureMaps_20251114_README.txt",
        "uuid": "66d3e2fe-ee5a-4b40-9e5c-42c507f640ad",
        "size_mb": 0.01,
        "unzip": False,
        "default": True,
    },
    "gemini": {
        "name": "Gemini.zip",
        "uuid": "a4a1c2de-2b2f-4aa0-b2cb-6ffa0af36734",
        "size_mb": 0.3,
        "unzip": True,
        "default": True,
    },
    "non_map_sample": {
        "name": "non-map_sample.zip",
        "uuid": "9452ea24-3b85-43db-bc30-23d33cfbc263",
        "size_mb": 0.2,
        "unzip": True,
        "default": True,
    },
    "clip": {
        "name": "CLIP.zip",
        "uuid": "7fa90d74-034c-42b2-b744-e1bc6648b19f",
        "size_mb": 6.3,
        "unzip": True,
        "default": True,
    },
    "marc": {
        "name": "MARC.zip",
        "uuid": "9a88a16a-a788-4a5b-afce-187e7564bfee",
        "size_mb": 7.1,
        "unzip": True,
        "default": True,
    },
    "maps": {
        "name": "final_maps.zip",
        "uuid": "49a4af7c-5969-48db-bba0-181d08a2a0ba",
        "size_mb": 1203.0,
        "unzip": True,
        "default": True,
    },
    "efficientnet": {
        "name": "EfficientNet.zip",
        "uuid": "10587132-fd4a-4ca2-a30d-346235a71b72",
        "size_mb": 1727.0,
        "unzip": True,
        "default": False,
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dest_dir() -> Path:
    path = repo_root() / "data" / "ecommons"
    path.mkdir(parents=True, exist_ok=True)
    return path


def progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100.0, downloaded * 100.0 / total_size)
    mb = min(downloaded, total_size) / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    sys.stdout.write(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct:.1f}%)")
    sys.stdout.flush()


def download_file(key: str, force: bool) -> Path:
    spec = FILES[key]
    out_dir = dest_dir()
    path = out_dir / spec["name"]
    if path.exists() and not force:
        print(f"[skip] {spec['name']} already exists ({path.stat().st_size / (1024 * 1024):.1f} MB)")
        return path

    url = CONTENT.format(uuid=spec["uuid"])
    print(f"[download] {spec['name']} (~{spec['size_mb']} MB)")
    print(f"  {url}")
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        urlretrieve(url, tmp, reporthook=progress)
        print()
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def unzip_file(zip_path: Path) -> Path:
    target = dest_dir() / zip_path.stem
    if target.exists() and any(target.iterdir()):
        print(f"[skip] already unzipped: {target}")
        return target
    print(f"[unzip] {zip_path.name} -> {target}")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m for m in zf.infolist()
            if not m.filename.startswith("__MACOSX/") and not Path(m.filename).name.startswith("._")
        ]
        zf.extractall(target, members=[m.filename for m in members])
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-models",
        action="store_true",
        help="Also download EfficientNet.zip (~1.8 GB of pretrained weights).",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip final_maps.zip and EfficientNet.zip.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = [key for key, spec in FILES.items() if spec["default"]]
    if args.metadata_only:
        keys = [key for key in keys if key not in {"maps", "efficientnet"}]
    if args.with_models:
        keys.append("efficientnet")

    print(f"Dataset item: {ECOMMONS_ITEM}")
    print(f"Saving to: {dest_dir()}")
    for key in keys:
        path = download_file(key, force=args.force)
        if FILES[key]["unzip"] and path.suffix == ".zip":
            unzip_file(path)


if __name__ == "__main__":
    main()
