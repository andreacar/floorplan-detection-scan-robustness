import hashlib
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, Optional


RELEASE_RECORD_ID = "18890484"
RELEASE_VERSION = "v1.0.0"
RELEASE_ARCHIVE_NAME = "CubiCasa5k-ScanShift_v1.tar.zst"
RELEASE_BASE_URL = f"https://zenodo.org/records/{RELEASE_RECORD_ID}/files"
RELEASE_ARCHIVE_URL = f"{RELEASE_BASE_URL}/{RELEASE_ARCHIVE_NAME}?download=1"
RELEASE_SUMMARY_NAME = "DATASET_STAGE_SUMMARY.json"
RELEASE_SUMMARY_URL = f"{RELEASE_BASE_URL}/{RELEASE_SUMMARY_NAME}?download=1"
RELEASE_SHA256SUMS_NAME = "SHA256SUMS.txt"
RELEASE_SHA256SUMS_URL = f"{RELEASE_BASE_URL}/{RELEASE_SHA256SUMS_NAME}?download=1"

DATASET_CACHE_DIR_ENV = "FLOORPLAN_DATASET_CACHE_DIR"

_SPLIT_FILES = ("train.txt", "val.txt", "test.txt")


def default_cache_dir() -> Path:
    override = os.environ.get(DATASET_CACHE_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".cache" / "floorplan-detection-scan-robustness" / "datasets").resolve()


def find_dataset_root(search_root: Path) -> Optional[Path]:
    search_root = search_root.expanduser().resolve()
    if _looks_like_dataset_root(search_root):
        return search_root

    if not search_root.exists():
        return None

    for child in sorted(search_root.iterdir()):
        if child.is_dir() and _looks_like_dataset_root(child):
            return child
    return None


def resolve_default_dataset_base_dir() -> str:
    cache_dir = default_cache_dir()
    existing = find_dataset_root(cache_dir)
    if existing is not None:
        return str(existing)
    return ""


def ensure_dataset_available(cache_dir: Optional[Path] = None) -> Path:
    cache_dir = (cache_dir or default_cache_dir()).expanduser().resolve()
    existing = find_dataset_root(cache_dir)
    if existing is not None:
        return existing

    cache_dir.mkdir(parents=True, exist_ok=True)
    checksums_path = cache_dir / RELEASE_SHA256SUMS_NAME
    summary_path = cache_dir / RELEASE_SUMMARY_NAME
    archive_path = cache_dir / RELEASE_ARCHIVE_NAME

    if not checksums_path.exists():
        _download_file(RELEASE_SHA256SUMS_URL, checksums_path)
    expected = _expected_sha256(checksums_path, RELEASE_ARCHIVE_NAME)
    if not expected:
        raise RuntimeError(f"Missing {RELEASE_ARCHIVE_NAME} entry in {checksums_path}")

    if archive_path.exists():
        _verify_sha256(archive_path, expected)
    else:
        _download_file(RELEASE_ARCHIVE_URL, archive_path)
        _verify_sha256(archive_path, expected)

    if not summary_path.exists():
        _download_file(RELEASE_SUMMARY_URL, summary_path)

    _extract_archive(archive_path, cache_dir)
    extracted = find_dataset_root(cache_dir)
    if extracted is None:
        raise RuntimeError(
            f"Extracted {archive_path.name}, but no dataset root with {', '.join(_SPLIT_FILES)} was found in {cache_dir}"
        )
    return extracted


def release_metadata() -> Dict[str, str]:
    return {
        "record_id": RELEASE_RECORD_ID,
        "version": RELEASE_VERSION,
        "archive_name": RELEASE_ARCHIVE_NAME,
        "archive_url": RELEASE_ARCHIVE_URL,
        "summary_name": RELEASE_SUMMARY_NAME,
        "summary_url": RELEASE_SUMMARY_URL,
        "sha256sums_name": RELEASE_SHA256SUMS_NAME,
        "sha256sums_url": RELEASE_SHA256SUMS_URL,
        "cache_dir_env": DATASET_CACHE_DIR_ENV,
    }


def verify_remote_release() -> Dict[str, Dict[str, str]]:
    urls = {
        RELEASE_ARCHIVE_NAME: RELEASE_ARCHIVE_URL,
        RELEASE_SUMMARY_NAME: RELEASE_SUMMARY_URL,
        RELEASE_SHA256SUMS_NAME: RELEASE_SHA256SUMS_URL,
    }
    out: Dict[str, Dict[str, str]] = {}
    for name, url in urls.items():
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as response:
            out[name] = {
                "url": url,
                "status": str(response.status),
                "content_length": response.headers.get("Content-Length", ""),
                "content_type": response.headers.get("Content-Type", ""),
            }
    return out


def _looks_like_dataset_root(path: Path) -> bool:
    return path.is_dir() and all((path / split).is_file() for split in _SPLIT_FILES)


def _download_file(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _expected_sha256(checksums_path: Path, filename: str) -> str:
    for raw in checksums_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) >= 2 and parts[-1] == filename:
            return parts[0]
    return ""


def _verify_sha256(path: Path, expected: str) -> None:
    actual = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            actual.update(chunk)
    digest = actual.hexdigest()
    if digest != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected}, got {digest}")


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    tar = shutil.which("tar")
    if not tar:
        raise RuntimeError("`tar` is required to extract the dataset archive.")

    probe = subprocess.run([tar, "--help"], capture_output=True, text=True, check=True)
    if "--zstd" not in probe.stdout:
        raise RuntimeError("The system `tar` does not support `--zstd`, which is required to extract the dataset archive.")

    subprocess.run(
        [tar, "--zstd", "-xf", str(archive_path), "-C", str(dest_dir)],
        check=True,
    )
