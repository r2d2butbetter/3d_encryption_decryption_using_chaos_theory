"""Dataset discovery and optional download/bootstrap utilities."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


MODELNET40_URLS = [
    "http://modelnet.cs.princeton.edu/ModelNet40.zip",
    "https://modelnet.cs.princeton.edu/ModelNet40.zip",
]


SCANOBJECTNN_URL = "https://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip"


def _project_data_dir() -> Path:
    return Path(__file__).parent / 'data'


def candidate_dataset_roots(dataset: str) -> list[Path]:
    base = _project_data_dir()
    mapping = {
        'modelnet10': [
            base / 'ModelNet10',
            base / 'ModelNet10' / 'ModelNet10',
        ],
        'modelnet40': [
            base / 'ModelNet40',
            base / 'ModelNet40' / 'ModelNet40',
        ],
        'scanobjectnn': [
            base / 'ScanObjectNN',
            base / 'scanobjectnn',
            base / 'ScanObjectNN' / 'OFF',
            base / 'scanobjectnn' / 'OFF',
            base / 'ScanObjectNN' / 'h5_files',
            base / 'scanobjectnn' / 'h5_files',
        ],
    }
    return mapping.get(dataset, [])


def _resolve_existing_root(dataset: str, data_root: str | None) -> Path | None:
    if data_root:
        p = Path(data_root)
        return p if p.exists() else None

    for c in candidate_dataset_roots(dataset):
        if c.exists():
            return c
    return None


def _contains_off_files(root: Path) -> bool:
    try:
        next(root.rglob('*.off'))
        return True
    except StopIteration:
        return False


def _write_off_point_cloud(filepath: Path, points: np.ndarray):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('OFF\n')
        f.write(f"{len(points)} 0 0\n")
        for p in points:
            f.write(f"{float(p[0]):.8f} {float(p[1]):.8f} {float(p[2]):.8f}\n")


def _find_h5_files(root: Path) -> list[Path]:
    return sorted(root.rglob('*.h5'))


def _guess_split_from_filename(name: str) -> str:
    low = name.lower()
    if 'train' in low:
        return 'train'
    if 'test' in low or 'val' in low:
        return 'test'
    return 'all'


def _load_h5_data_labels(h5f):
    # Some ScanObjectNN files store datasets in nested groups and/or use
    # non-standard names. Discover candidates recursively, then choose points
    # and labels based on tensor shape compatibility.
    dataset_items = []

    def _collector(name, obj):
        if hasattr(obj, 'shape'):
            dataset_items.append((name, tuple(obj.shape)))

    h5f.visititems(_collector)
    if not dataset_items:
        raise ValueError("No datasets found in h5 file")

    point_candidates = []
    label_candidates = []
    for name, shape in dataset_items:
        if len(shape) == 3 and shape[-1] >= 3 and shape[0] > 0:
            point_candidates.append((name, shape))
        if len(shape) in (1, 2) and shape[0] > 0:
            label_candidates.append((name, shape))

    # Prefer common names when multiple candidates exist.
    def _name_priority(n: str) -> int:
        low = n.lower()
        if 'data' in low or 'point' in low or 'xyz' in low:
            return 0
        return 1

    point_candidates.sort(key=lambda x: (_name_priority(x[0]), x[0]))
    label_candidates.sort(key=lambda x: (_name_priority(x[0]), x[0]))

    if not point_candidates:
        raise ValueError(f"No point dataset found. Discovered datasets: {dataset_items}")

    # Match labels to points by sample dimension.
    point_name, point_shape = point_candidates[0]
    n_samples = point_shape[0]

    label_name = None
    for cand_name, cand_shape in label_candidates:
        # 1D labels: (N,), 2D labels: (N,1) or (N,k)
        if cand_shape[0] == n_samples:
            label_name = cand_name
            break

    if label_name is None:
        raise ValueError(
            f"No compatible label dataset for points '{point_name}' shape={point_shape}. "
            f"Discovered datasets: {dataset_items}"
        )

    data = np.asarray(h5f[point_name])
    labels = np.asarray(h5f[label_name]).reshape(-1)

    if data.ndim != 3 or data.shape[-1] < 3:
        raise ValueError(f"Unexpected point tensor shape: {data.shape}")
    if len(labels) != data.shape[0]:
        raise ValueError(f"Label count mismatch: labels={len(labels)}, samples={data.shape[0]}")
    return data[:, :, :3], labels.astype(int)


def _convert_scanobjectnn_h5_to_off(scan_root: Path) -> Path:
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - import-time environment issue
        raise RuntimeError(
            "h5py is required to convert ScanObjectNN h5 to OFF. Install with: pip install h5py"
        ) from exc

    h5_files = _find_h5_files(scan_root)
    if not h5_files:
        raise RuntimeError(f"No .h5 files found under {scan_root}")

    off_root = scan_root / 'OFF'
    converted = 0

    skipped = 0

    for h5_path in h5_files:
        split = _guess_split_from_filename(h5_path.name)
        print(f"Converting ScanObjectNN file: {h5_path.name} (split={split})")
        try:
            with h5py.File(h5_path, 'r') as h5f:
                points, labels = _load_h5_data_labels(h5f)
        except Exception as exc:
            skipped += 1
            print(f"Skipping unsupported/empty h5 file {h5_path.name}: {exc}")
            continue

        for idx in range(points.shape[0]):
            cls_id = int(labels[idx])
            cls_name = f"class_{cls_id:02d}"
            out_file = off_root / cls_name / split / f"{h5_path.stem}_{idx:06d}.off"
            if out_file.exists():
                continue
            _write_off_point_cloud(out_file, points[idx])
            converted += 1

    if converted == 0 and not _contains_off_files(off_root):
        raise RuntimeError("ScanObjectNN conversion produced no OFF files")

    print(f"ScanObjectNN OFF conversion complete. New files: {converted}, skipped h5 files: {skipped}")
    return off_root


def _download_file(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, open(dst, 'wb') as out:
        shutil.copyfileobj(response, out)


def _download_with_fallback(urls: list[str], dst: Path):
    last_err = None
    for url in urls:
        try:
            print(f"Downloading: {url}")
            _download_file(url, dst)
            return
        except Exception as exc:
            last_err = exc
            print(f"Download failed from {url}: {exc}")
    if last_err is not None:
        raise last_err


def _extract_zip(zip_path: Path, extract_to: Path):
    print(f"Extracting: {zip_path} -> {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)


def _bootstrap_modelnet40() -> bool:
    data_dir = _project_data_dir()
    zip_path = data_dir / 'ModelNet40.zip'
    _download_with_fallback(MODELNET40_URLS, zip_path)
    _extract_zip(zip_path, data_dir)
    return True


def _bootstrap_scanobjectnn() -> bool:
    data_dir = _project_data_dir()
    zip_path = data_dir / 'scanobjectnn_h5_files.zip'
    _download_with_fallback([SCANOBJECTNN_URL], zip_path)
    target = data_dir / 'ScanObjectNN'
    _extract_zip(zip_path, target)
    _convert_scanobjectnn_h5_to_off(target)
    return True


def ensure_dataset_available(dataset: str, data_root: str | None, auto_download: bool = True) -> Path | None:
    """
    Resolve dataset root if present; optionally download missing supported datasets.

    Auto-download is supported for:
    - modelnet40
    - scanobjectnn
    """
    root = _resolve_existing_root(dataset, data_root)
    if dataset == 'scanobjectnn' and root is not None and not _contains_off_files(root):
        if auto_download:
            print("ScanObjectNN found without OFF files. Converting h5 to OFF...")
            root = _convert_scanobjectnn_h5_to_off(root)
        else:
            return None

    if root is not None:
        return root

    if data_root:
        return None

    if not auto_download:
        return None

    if dataset == 'modelnet40':
        print("ModelNet40 dataset not found locally. Attempting download...")
        _bootstrap_modelnet40()
    elif dataset == 'scanobjectnn':
        print("ScanObjectNN dataset not found locally. Attempting download...")
        _bootstrap_scanobjectnn()
    else:
        return None

    return _resolve_existing_root(dataset, None)