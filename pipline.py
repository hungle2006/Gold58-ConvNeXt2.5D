#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSNA Knee MRI 2026 - CLEAN BASELINE WITH FIXED 58-GOLD VALIDATION
=================================================================

Purpose
-------
This is a deliberately simple and reproducible baseline rewritten from the
user's larger research pipeline.

Key protocol
------------
1) Read official competition train.csv.
2) Detect the studies with all 12 official binary labels available.
   By default this MUST be 58 studies. These UIDs are NEVER used for training.
3) Read the user's 4,407-study pseudo/manual label CSV.
4) Train on pseudo/manual labels after removing the 58 gold UIDs -> expected
   4,349 training studies.
5) Validation uses ONLY the official labels from the 58 gold studies.
6) Split unit is StudyInstanceUID. Never split by Series or slice.
7) MRI sampling follows a simple Raptor-style baseline:

       Sagittal fluid-sensitive      18 centers
       Sagittal non-fluid            14 centers
       Coronal fluid-sensitive       12 centers
       Coronal non-fluid              8 centers
       Axial                          12 centers

   Centers are sampled uniformly from 6% to 94% of each selected Series.
8) Every selected center becomes a 2.5D clip [i-1, i, i+1].
9) ConvNeXtV2-Tiny encodes each clip.
10) Study representation = concat(mean clip feature, max clip feature).
11) One linear head predicts the 12 findings.
12) Loss = masked BCEWithLogits with train-derived positive weights.
13) Model selection = macro ROC-AUC on the fixed 58 gold studies only.

Existing train cache is reused directly from persistent_cache_dir; no train DICOM
re-decoding is performed when the cache is available.

This file intentionally DOES NOT use:
- label-aware attention
- label-token transformer
- GeM
- depth transformer
- ranking loss
- uncertainty weighting
- random 90/10 validation

Those should be added later one at a time after this baseline is established.
"""

from __future__ import annotations

import os
import gc
import json
import math
import random
import pickle
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom

try:
    import cv2
except ImportError:
    cv2 = None

from sklearn.metrics import roc_auc_score

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import timm
except ImportError as exc:
    raise ImportError("timm is required") from exc

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    # Kaggle competition directory.
    data_root: str = "/kaggle/input/competitions/rsna-knee-abnormalities-detection"

    # The 4,407-study pseudo/manual labels supplied by the user.
    pseudo_train_csv: str = (
        "/kaggle/input/datasets/leminhhung0101/label-train/"
        "train_4407_full_labeled_manual_reviewed.csv"
    )

    # IMPORTANT: this is the OFFICIAL competition file, not the pseudo CSV.
    official_train_csv: str = "train.csv"
    train_series_csv: str = "train_series.csv"
    test_csv: str = "test.csv"
    test_series_csv: str = "test_series.csv"
    sample_submission_csv: str = "sample_submission.csv"

    train_series_dir: str = "train_series"
    test_series_dir: str = "test_series"

    output_dir: str = "/kaggle/working/rsna_knee_baseline_gold58"
    # Local writable cache is only used for a tiny reconstructed index and,
    # optionally, TEST cache if no persistent test cache exists.
    cache_dir: str = "/kaggle/tmp/rsna_knee_baseline_gold58_cache"

    # EXISTING cache from the user's original pipeline. Expected layout:
    #   <root>/index/train_series_index.pkl
    #   <root>/processed/train_series/<StudyUID>/<SeriesUID>.npy
    #   <root>/processed/train_series/<StudyUID>/<SeriesUID>.pkl
    persistent_cache_dir: str = (
        "/kaggle/input/datasets/leminhhung0101/rsna-knee-dicom-cache"
    )

    # Never rebuild the 4,407-study TRAIN cache when the persistent cache exists.
    reuse_persistent_train_cache: bool = True
    allow_train_dicom_rebuild: bool = False
    # Test cache can be built locally later for submission if it is not present.
    allow_test_dicom_rebuild: bool = True
    min_persistent_series_coverage: float = 0.98

    # Strict guard: fail if official fully-labeled study count is not 58.
    expected_gold_studies: int = 58
    strict_gold_count: bool = True

    # train | submission | inspect
    mode: str = "train"
    make_submission_after_train: bool = True
    resume_checkpoint: str = ""

    # Image / model.
    img_size: int = 224
    backbone_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    pretrained: bool = True
    dropout: float = 0.20
    image_chunk_size: int = 96

    # Raptor-style fixed slot sampling.
    sample_low: float = 0.06
    sample_high: float = 0.94

    # Optimization.
    seed: int = 2026
    epochs: int = 12
    batch_size: int = 2
    grad_accum_steps: int = 2
    num_workers: int = 4
    index_workers: int = 4
    lr_backbone: float = 2e-5
    lr_head: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    grad_clip: float = 1.0
    amp: bool = True

    # Class imbalance. n_neg / n_pos is clipped to this value.
    max_pos_weight: float = 20.0

    # Conservative 2.5D augmentation. No horizontal flip.
    use_augmentation: bool = True
    aug_rotation_deg: float = 7.0
    aug_translate: float = 0.03
    aug_scale_min: float = 0.95
    aug_scale_max: float = 1.05
    aug_brightness: float = 0.06
    aug_contrast: float = 0.08
    aug_noise_std: float = 0.01

    # DICOM intensity clipping.
    percentile_low: float = 1.0
    percentile_high: float = 99.0

    # IMPORTANT: match the EXISTING cache produced by the pasted pipeline.
    # Original cache v3 stores per-slice z-scored values clipped to [-5, 5]
    # and quantized to uint8 [0,255]. We decode back to [-5,5] at load time.
    cache_version: int = 3
    cache_dtype: str = "uint8"
    mmap_cache: bool = True
    worker_series_cache_size: int = 8

    # Device.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = CFG()
DEVICE = torch.device(CFG.device)

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
N_LABELS = len(LABELS)
PLANES = ["Sagittal", "Coronal", "Axial"]

# (slot_name, anatomical_plane, desired Fluid_Sensitive value or None, n_centers)
SLOTS = [
    ("SAG_FLUID", "Sagittal", 1, 18),
    ("SAG_NONFLUID", "Sagittal", 0, 14),
    ("COR_FLUID", "Coronal", 1, 12),
    ("COR_NONFLUID", "Coronal", 0, 8),
    ("AXIAL", "Axial", None, 12),
]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


# ============================================================
# REPRODUCIBILITY / DDP
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def ddp_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return enabled, rank, local_rank, world_size


def is_main_process():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


# ============================================================
# TABLES: FIXED 58-GOLD VALIDATION
# ============================================================

def clean_uid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "StudyInstanceUID" in df.columns:
        df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str).str.strip()
    return df


def normalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve underscore / case variants to the canonical label names."""
    df = df.copy()
    canon = {
        str(c).strip().lower().replace("_", " ").replace("-", " "): c
        for c in df.columns
    }
    for label in LABELS:
        key = label.lower().replace("_", " ").replace("-", " ")
        if label not in df.columns and key in canon:
            df[label] = df[canon[key]]
    return df


def find_pseudo_csv() -> Path:
    p = Path(CFG.pseudo_train_csv)
    if p.is_file():
        return p
    matches = list(Path("/kaggle/input").rglob(p.name))
    if not matches:
        raise FileNotFoundError(f"Pseudo/manual label CSV not found: {p}")
    matches.sort(key=lambda x: len(x.parts))
    return matches[0]


def prepare_fixed_gold_split():
    root = Path(CFG.data_root)
    official_path = root / CFG.official_train_csv
    pseudo_path = find_pseudo_csv()
    train_series_path = root / CFG.train_series_csv
    test_path = root / CFG.test_csv
    test_series_path = root / CFG.test_series_csv
    sample_submission_path = root / CFG.sample_submission_csv

    required = [official_path, train_series_path, test_path, test_series_path, sample_submission_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing competition files:\n" + "\n".join(missing))

    official = normalize_label_columns(clean_uid(pd.read_csv(official_path)))
    pseudo = normalize_label_columns(clean_uid(pd.read_csv(pseudo_path)))
    train_series = clean_uid(pd.read_csv(train_series_path))
    test_df = clean_uid(pd.read_csv(test_path))
    test_series = clean_uid(pd.read_csv(test_series_path))
    sample_submission = clean_uid(pd.read_csv(sample_submission_path))

    for df_name, df in [("official train.csv", official), ("pseudo/manual CSV", pseudo)]:
        missing_cols = [c for c in ["StudyInstanceUID"] + LABELS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{df_name} missing columns: {missing_cols}")

    for label in LABELS:
        official[label] = pd.to_numeric(official[label], errors="coerce")
        pseudo[label] = pd.to_numeric(pseudo[label], errors="coerce")

    # Gold = official rows with all 12 binary targets present.
    official_binary = official[LABELS].isin([0, 1])
    gold_mask = official_binary.all(axis=1)
    gold = official.loc[gold_mask, ["StudyInstanceUID"] + LABELS].copy()
    gold = gold.drop_duplicates("StudyInstanceUID", keep="last").reset_index(drop=True)

    if CFG.strict_gold_count and len(gold) != CFG.expected_gold_studies:
        raise RuntimeError(
            f"Expected exactly {CFG.expected_gold_studies} fully-labeled official studies, "
            f"but found {len(gold)}. Check that CFG.data_root points to the competition "
            "and that official_train_csv='train.csv'."
        )

    gold_ids = set(gold["StudyInstanceUID"].astype(str))

    # Training labels come ONLY from the user's pseudo/manual CSV, but gold UIDs
    # are removed completely to prevent leakage.
    train = pseudo.loc[
        ~pseudo["StudyInstanceUID"].astype(str).isin(gold_ids),
        ["StudyInstanceUID"] + LABELS,
    ].copy()

    # Keep rows with at least one valid pseudo label; mask any non-binary cell.
    for label in LABELS:
        bad = ~train[label].isin([0, 1])
        train.loc[bad, label] = np.nan
    train = train.loc[train[LABELS].notna().any(axis=1)].drop_duplicates(
        "StudyInstanceUID", keep="last"
    ).reset_index(drop=True)

    # Validation targets are ALWAYS overwritten by official values.
    val = gold.copy()

    overlap = set(train["StudyInstanceUID"]).intersection(set(val["StudyInstanceUID"]))
    if overlap:
        raise RuntimeError(f"Gold leakage into training: {len(overlap)} studies")

    Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)
    train.to_csv(Path(CFG.output_dir) / "train_pseudo_excluding_gold58.csv", index=False)
    val.to_csv(Path(CFG.output_dir) / "valid_gold58_official.csv", index=False)

    diagnostics = {
        "official_train_csv": str(official_path),
        "pseudo_train_csv": str(pseudo_path),
        "n_official_rows": int(len(official)),
        "n_gold_valid": int(len(val)),
        "n_pseudo_train_after_gold_exclusion": int(len(train)),
        "gold_train_overlap": 0,
    }
    with open(Path(CFG.output_dir) / "split_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print("\nFIXED GOLD SPLIT")
    print("=" * 80)
    print(f"Official fully-labeled validation: {len(val):,}")
    print(f"Pseudo/manual training after exclusion: {len(train):,}")
    print(f"Overlap: {len(overlap)}")
    if len(val) == 58 and len(pseudo) == 4407:
        print(f"Expected 4407 - 58 = 4349; actual train = {len(train)}")
    print("=" * 80)

    return root, train, val, train_series, test_df, test_series, sample_submission


# ============================================================
# DICOM PREPROCESS / CACHE
# ============================================================

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def sort_dicom_paths(paths: List[str]) -> List[str]:
    """Sort by physical slice position; fall back to InstanceNumber."""
    records = []
    for p in paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            ipp = getattr(ds, "ImagePositionPatient", None)
            iop = getattr(ds, "ImageOrientationPatient", None)
            instance = int(getattr(ds, "InstanceNumber", 0) or 0)
            if ipp is not None and iop is not None and len(ipp) >= 3 and len(iop) >= 6:
                row = np.asarray(iop[:3], dtype=np.float64)
                col = np.asarray(iop[3:6], dtype=np.float64)
                normal = np.cross(row, col)
                pos = float(np.dot(np.asarray(ipp[:3], dtype=np.float64), normal))
                records.append((0, pos, instance, p))
            else:
                records.append((1, instance, 0, p))
        except Exception:
            records.append((2, 0, 0, p))
    records.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return [x[3] for x in records]


def read_dicom_pixels(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]

    arr = arr * _safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
    arr = arr + _safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = np.percentile(arr, [CFG.percentile_low, CFG.percentile_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return arr.astype(np.float32, copy=False)


def resize_slice(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape == (size, size):
        return img.astype(np.float32, copy=False)
    if cv2 is not None:
        interp = cv2.INTER_AREA if min(img.shape[:2]) >= size else cv2.INTER_LINEAR
        return cv2.resize(img, (size, size), interpolation=interp).astype(np.float32)
    t = torch.from_numpy(img)[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def preprocess_slice(path: str) -> np.ndarray:
    """Fallback preprocessing that MATCHES the user's existing cache v3.

    Existing cache protocol:
        DICOM -> percentile window -> resize -> per-slice z-score
        -> clip [-5,5] -> uint8 [0,255]

    Training normally will NOT execute this for train_series because the
    persistent cache is reused directly. It is kept for optional test-cache
    construction and for reproducibility.
    """
    img = resize_slice(read_dicom_pixels(path), CFG.img_size)
    mean = float(img.mean())
    std = float(img.std())
    img = (img - mean) / (std + 1e-6)
    img = np.clip(img, -5.0, 5.0)
    encoded = (img + 5.0) / 10.0 * 255.0
    return np.rint(encoded).astype(np.uint8)


def _find_dicom_paths(folder: Path) -> List[str]:
    if not folder.exists():
        return []
    paths = [str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"]
    if not paths:
        paths = [str(p) for p in folder.rglob("*.dcm") if p.is_file()]
    return paths


def _series_cache_paths(series_dir: str, study_uid: str, series_uid: str):
    base = Path(CFG.cache_dir) / "processed" / series_dir / study_uid
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{series_uid}.npy", base / f"{series_uid}.pkl"


def _index_path(cache_root: str, series_dir: str) -> Path:
    return Path(cache_root) / "index" / f"{series_dir}_index.pkl"


def _expected_cache_file(cache_root: str, series_dir: str, study_uid: str, series_uid: str) -> Path:
    return Path(cache_root) / "processed" / series_dir / str(study_uid) / f"{series_uid}.npy"


def _resolve_existing_meta_path(cache_root: str, series_dir: str, study_uid: str, series_uid: str) -> Path:
    return Path(cache_root) / "processed" / series_dir / str(study_uid) / f"{series_uid}.pkl"


def _preprocess_one_series(task):
    study_uid, series_uid, plane, fluid, fat, root, series_dir = task
    folder = Path(root) / series_dir / study_uid / series_uid
    npy_path, meta_path = _series_cache_paths(series_dir, study_uid, series_uid)

    if npy_path.exists() and meta_path.exists():
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            if meta.get("cache_version") == CFG.cache_version:
                return meta
        except Exception:
            pass

    paths = sort_dicom_paths(_find_dicom_paths(folder))
    if not paths:
        return None

    slices, valid_paths = [], []
    for p in paths:
        try:
            slices.append(preprocess_slice(p))
            valid_paths.append(p)
        except Exception:
            continue
    if not slices:
        return None

    arr = np.stack(slices, axis=0)
    tmp_npy = npy_path.with_name(npy_path.stem + ".tmp.npy")
    tmp_meta = meta_path.with_name(meta_path.stem + ".tmp.pkl")
    np.save(tmp_npy, arr, allow_pickle=False)

    meta = {
        "cache_version": CFG.cache_version,
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "plane": plane,
        "fluid": int(fluid),
        "fat": int(fat),
        "n_slices": int(arr.shape[0]),
        "cache_file": str(npy_path),
        "paths": valid_paths,
    }
    with open(tmp_meta, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_npy, npy_path)
    os.replace(tmp_meta, meta_path)
    return meta


def normalize_series_table(series_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_uid(series_df)
    df["SeriesInstanceUID"] = df["SeriesInstanceUID"].astype(str).str.strip()
    df["Anatomical_Plane"] = df["Anatomical_Plane"].astype(str).str.strip().str.title()
    df["Fluid_Sensitive"] = pd.to_numeric(df["Fluid_Sensitive"], errors="coerce").fillna(0).astype(int)
    df["Fat_Suppression"] = pd.to_numeric(df["Fat_Suppression"], errors="coerce").fillna(0).astype(int)
    return df


def _rewrite_index_paths(index: Dict[str, List[dict]], cache_root: str, series_dir: str, series_df=None):
    """Point index records at the mounted persistent cache.

    The original index may contain paths from a different Kaggle session, so
    absolute paths stored inside the pickle must NOT be trusted. We reconstruct
    them from StudyInstanceUID / SeriesInstanceUID and the visible cache layout.
    """
    meta_lookup = {}
    if series_df is not None:
        df = normalize_series_table(series_df)
        for r in df.itertuples(index=False):
            meta_lookup[(str(r.StudyInstanceUID), str(r.SeriesInstanceUID))] = (
                str(r.Anatomical_Plane), int(r.Fluid_Sensitive), int(r.Fat_Suppression)
            )

    rewritten: Dict[str, List[dict]] = {}
    for uid, items in index.items():
        uid = str(uid)
        out_items = []
        for raw in items:
            m = dict(raw)
            study = str(m.get("StudyInstanceUID", uid))
            series = str(m.get("SeriesInstanceUID", ""))
            if not series:
                continue

            # Prefer explicit relative path from the original cache metadata.
            rel = m.get("cache_relpath", "")
            if rel:
                cache_file = Path(cache_root) / rel
            else:
                cache_file = _expected_cache_file(cache_root, series_dir, study, series)

            if not cache_file.is_file():
                continue

            m["StudyInstanceUID"] = study
            m["SeriesInstanceUID"] = series
            m["cache_file"] = str(cache_file)

            # Fill/normalize metadata from train_series.csv if necessary.
            if (study, series) in meta_lookup:
                plane, fluid, fat = meta_lookup[(study, series)]
                m["plane"] = str(m.get("plane", plane) or plane).title()
                m["fluid"] = int(m.get("fluid", fluid))
                m["fat"] = int(m.get("fat", fat))
            else:
                m["plane"] = str(m.get("plane", "Sagittal")).title()
                m["fluid"] = int(m.get("fluid", 0))
                m["fat"] = int(m.get("fat", 0))

            # n_slices should exist in the original pkl; recover cheaply if not.
            if int(m.get("n_slices", 0) or 0) <= 0:
                try:
                    a = np.load(cache_file, mmap_mode="r", allow_pickle=False)
                    m["n_slices"] = int(a.shape[0])
                    del a
                except Exception:
                    continue

            out_items.append(m)
        if out_items:
            out_items.sort(key=lambda x: x["SeriesInstanceUID"])
            rewritten[uid] = out_items
    return rewritten


def build_index_from_existing_processed(series_df: pd.DataFrame, cache_root: str, series_dir: str):
    """Reconstruct an index from existing .npy/.pkl files WITHOUT reading DICOM."""
    df = normalize_series_table(series_df)
    index: Dict[str, List[dict]] = {}
    found = 0

    for r in tqdm(df.itertuples(index=False), total=len(df), desc=f"index existing {series_dir}"):
        study = str(r.StudyInstanceUID)
        series = str(r.SeriesInstanceUID)
        npy_path = _expected_cache_file(cache_root, series_dir, study, series)
        if not npy_path.is_file():
            continue

        meta = {}
        pkl_path = _resolve_existing_meta_path(cache_root, series_dir, study, series)
        if pkl_path.is_file():
            try:
                with open(pkl_path, "rb") as f:
                    meta = dict(pickle.load(f))
            except Exception:
                meta = {}

        n_slices = int(meta.get("n_slices", 0) or 0)
        if n_slices <= 0:
            try:
                a = np.load(npy_path, mmap_mode="r", allow_pickle=False)
                n_slices = int(a.shape[0])
                del a
            except Exception:
                continue

        rec = {
            **meta,
            "cache_version": int(meta.get("cache_version", CFG.cache_version)),
            "StudyInstanceUID": study,
            "SeriesInstanceUID": series,
            "plane": str(r.Anatomical_Plane).title(),
            "fluid": int(r.Fluid_Sensitive),
            "fat": int(r.Fat_Suppression),
            "n_slices": n_slices,
            "cache_file": str(npy_path),
        }
        index.setdefault(study, []).append(rec)
        found += 1

    for uid in index:
        index[uid].sort(key=lambda x: x["SeriesInstanceUID"])

    print(
        f"[EXISTING CACHE] {series_dir}: {found:,}/{len(df):,} series "
        f"({found / max(len(df),1):.2%}), {len(index):,} studies"
    )
    return index


def load_persistent_index(series_df: pd.DataFrame, series_dir: str):
    """Load the user's existing cache/index. No DICOM decoding occurs here."""
    cache_root = Path(CFG.persistent_cache_dir)
    if not cache_root.is_dir():
        return None

    # Exact layout shown by the user / produced by the original pipeline.
    candidates = [
        cache_root / "index" / f"{series_dir}_index.pkl",
        cache_root / f"{series_dir}_index.pkl",
        cache_root / "index" / f"{series_dir}_index_v{CFG.cache_version}.pkl",
    ]

    for p in candidates:
        if not p.is_file():
            continue
        try:
            with open(p, "rb") as f:
                raw = pickle.load(f)
            index = _rewrite_index_paths(raw, str(cache_root), series_dir, series_df)
            if index:
                print(f"[PERSISTENT INDEX] loaded: {p}")
                return index
        except Exception as exc:
            warnings.warn(f"Could not load persistent index {p}: {exc}")

    # Index missing/incompatible: rebuild only the lightweight index from .npy.
    processed = cache_root / "processed" / series_dir
    if processed.is_dir():
        print(f"[PERSISTENT INDEX] rebuilding metadata index from existing NPY files: {processed}")
        return build_index_from_existing_processed(series_df, str(cache_root), series_dir)

    return None


def _cache_coverage(index: Dict[str, List[dict]], series_df: pd.DataFrame):
    expected_series = len(series_df)
    found_series = sum(len(v) for v in index.values())
    expected_studies = series_df["StudyInstanceUID"].astype(str).nunique()
    found_studies = len(index)
    return found_series, expected_series, found_studies, expected_studies


def build_series_index(series_df: pd.DataFrame, root: Path, series_dir: str):
    """Fallback DICOM preprocessing into local writable cache."""
    Path(CFG.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(CFG.cache_dir, "index").mkdir(parents=True, exist_ok=True)
    index_path = _index_path(CFG.cache_dir, series_dir)

    if index_path.is_file():
        try:
            with open(index_path, "rb") as f:
                return _rewrite_index_paths(pickle.load(f), CFG.cache_dir, series_dir, series_df)
        except Exception:
            pass

    df = normalize_series_table(series_df)
    tasks = [
        (
            str(r.StudyInstanceUID), str(r.SeriesInstanceUID),
            str(r.Anatomical_Plane), int(r.Fluid_Sensitive), int(r.Fat_Suppression),
            str(root), series_dir,
        )
        for r in df.itertuples(index=False)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=max(1, CFG.index_workers)) as ex:
        futures = [ex.submit(_preprocess_one_series, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"cache {series_dir}"):
            try:
                meta = fut.result()
            except Exception as exc:
                warnings.warn(f"DICOM worker failed: {exc}")
                continue
            if meta is not None:
                results.append(meta)

    index: Dict[str, List[dict]] = {}
    for meta in results:
        index.setdefault(meta["StudyInstanceUID"], []).append(meta)
    for uid in index:
        index[uid].sort(key=lambda x: x["SeriesInstanceUID"])

    with open(index_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    return index


def load_or_build_series_index(series_df, root, series_dir, rank=0):
    """Prefer the mounted persistent cache; only decode DICOM when explicitly allowed."""
    is_train = series_dir == CFG.train_series_dir

    # Rank 0 resolves persistent index/reconstruction and writes a tiny local copy.
    local_index = _index_path(CFG.cache_dir, series_dir)
    if rank == 0:
        Path(local_index.parent).mkdir(parents=True, exist_ok=True)

        persistent = load_persistent_index(series_df, series_dir)
        if persistent is not None:
            found_s, exp_s, found_st, exp_st = _cache_coverage(persistent, series_df)
            ratio = found_s / max(exp_s, 1)
            print(
                f"[CACHE COVERAGE] {series_dir}: series={found_s:,}/{exp_s:,} ({ratio:.2%}), "
                f"studies={found_st:,}/{exp_st:,}"
            )
            if is_train and ratio < CFG.min_persistent_series_coverage:
                raise RuntimeError(
                    f"Persistent train cache coverage {ratio:.2%} is below "
                    f"minimum {CFG.min_persistent_series_coverage:.2%}."
                )
            # Save only the metadata index locally. NPY data remain in /kaggle/input.
            with open(local_index, "wb") as f:
                pickle.dump(persistent, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            allow = CFG.allow_train_dicom_rebuild if is_train else CFG.allow_test_dicom_rebuild
            if not allow:
                raise FileNotFoundError(
                    f"Existing persistent cache not found for {series_dir}. Expected under "
                    f"{CFG.persistent_cache_dir}/processed/{series_dir}/ and/or index/. "
                    "DICOM rebuild is disabled."
                )
            print(f"[CACHE] No persistent {series_dir} cache found; building local DICOM cache.")
            built = build_series_index(series_df, root, series_dir)
            with open(local_index, "wb") as f:
                pickle.dump(built, f, protocol=pickle.HIGHEST_PROTOCOL)

    barrier()

    if not local_index.is_file():
        raise FileNotFoundError(f"Resolved cache index not found: {local_index}")
    with open(local_index, "rb") as f:
        return pickle.load(f)


def load_series_array(meta: dict):
    path = Path(meta.get("cache_file", ""))
    if not path.is_file():
        return None
    try:
        return np.load(path, mmap_mode="r" if CFG.mmap_cache else None, allow_pickle=False)
    except Exception:
        return None


# ============================================================
# FIXED SLOT / SLICE SAMPLING
# ============================================================

def choose_slot_series(series_list: List[dict]) -> List[Tuple[str, dict, int]]:
    """Pick at most one Series for each of the 5 baseline slots.

    Missing slots stay missing; we do NOT duplicate another protocol into them.
    """
    chosen = []
    used_series = set()

    for slot_name, plane, fluid_req, n_centers in SLOTS:
        candidates = []
        for s in series_list:
            if str(s.get("plane", "")).title() != plane:
                continue
            if fluid_req is not None and int(s.get("fluid", 0)) != fluid_req:
                continue
            sid = str(s.get("SeriesInstanceUID", ""))
            if sid in used_series:
                continue
            # Prefer more complete stacks; use fat suppression as a weak tie-break.
            candidates.append((int(s.get("n_slices", 0)), int(s.get("fat", 0)), s))

        if not candidates:
            continue
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        s = candidates[0][2]
        used_series.add(str(s.get("SeriesInstanceUID", "")))
        chosen.append((slot_name, s, n_centers))

    return chosen


def uniform_centers(n_slices: int, k: int) -> List[int]:
    n_slices = int(n_slices)
    if n_slices <= 0 or k <= 0:
        return []
    if n_slices == 1:
        return [0]

    lo = int(round(CFG.sample_low * (n_slices - 1)))
    hi = int(round(CFG.sample_high * (n_slices - 1)))
    lo = max(0, min(lo, n_slices - 1))
    hi = max(lo, min(hi, n_slices - 1))

    centers = np.rint(np.linspace(lo, hi, num=k)).astype(np.int64)
    centers = np.clip(centers, 0, n_slices - 1)

    # Avoid overweighting repeated indices in very short Series.
    centers = np.unique(centers)
    return centers.tolist()


def load_2p5d_from_array(arr, center: int) -> np.ndarray:
    """Decode 3 slices from the ORIGINAL cache-v3 representation.

    Existing uint8 cache: [0,255] represents standardized [-5,5].
    """
    if arr is None or len(arr) == 0:
        return np.zeros((3, CFG.img_size, CFG.img_size), np.float32)
    center = int(np.clip(center, 0, len(arr) - 1))
    idx = [max(0, center - 1), center, min(len(arr) - 1, center + 1)]
    x = np.asarray(arr[idx], dtype=np.float32)
    if CFG.cache_dtype == "uint8":
        x = x / 255.0 * 10.0 - 5.0
    return x


def augment_2p5d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32).copy()
    h, w = x.shape[-2:]
    angle = float(np.random.uniform(-CFG.aug_rotation_deg, CFG.aug_rotation_deg))
    scale = float(np.random.uniform(CFG.aug_scale_min, CFG.aug_scale_max))
    tx = float(np.random.uniform(-CFG.aug_translate, CFG.aug_translate) * w)
    ty = float(np.random.uniform(-CFG.aug_translate, CFG.aug_translate) * h)

    if cv2 is not None:
        M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        x = np.stack([
            cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            for c in x
        ]).astype(np.float32)

    if CFG.aug_contrast > 0:
        x *= 1.0 + float(np.random.uniform(-CFG.aug_contrast, CFG.aug_contrast))
    if CFG.aug_brightness > 0:
        x += float(np.random.uniform(-CFG.aug_brightness, CFG.aug_brightness))
    if CFG.aug_noise_std > 0:
        x += np.random.normal(0, CFG.aug_noise_std, x.shape).astype(np.float32)
    return np.clip(x, -5.0, 5.0)


# ============================================================
# DATASET / COLLATE
# ============================================================

class KneeBaselineDataset(Dataset):
    def __init__(self, studies: pd.DataFrame, series_index: Dict[str, List[dict]], train: bool):
        self.studies = studies.reset_index(drop=True).copy()
        self.series_index = series_index
        self.train = train
        self._series_cache = OrderedDict()
        self.study_slots = {
            str(uid): choose_slot_series(series_index.get(str(uid), []))
            for uid in self.studies["StudyInstanceUID"].astype(str)
        }

    def __len__(self):
        return len(self.studies)

    def _get_arr(self, meta):
        key = meta.get("cache_file", "")
        if key in self._series_cache:
            arr = self._series_cache.pop(key)
            self._series_cache[key] = arr
            return arr
        arr = load_series_array(meta)
        if arr is not None:
            self._series_cache[key] = arr
            while len(self._series_cache) > CFG.worker_series_cache_size:
                self._series_cache.popitem(last=False)
        return arr

    def __getitem__(self, idx):
        row = self.studies.iloc[idx]
        uid = str(row["StudyInstanceUID"])
        clips = []

        for slot_name, meta, n_centers in self.study_slots.get(uid, []):
            arr = self._get_arr(meta)
            if arr is None:
                continue
            for center in uniform_centers(len(arr), n_centers):
                x = load_2p5d_from_array(arr, center)
                if self.train and CFG.use_augmentation:
                    x = augment_2p5d(x)
                clips.append(x)

        if not clips:
            clips = [np.zeros((3, CFG.img_size, CFG.img_size), np.float32)]

        x = torch.from_numpy(np.stack(clips).astype(np.float32, copy=False))
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

        y = np.full(N_LABELS, np.nan, np.float32)
        for j, label in enumerate(LABELS):
            if label in row.index and pd.notna(row[label]):
                y[j] = float(row[label])

        return {
            "images": x,
            "target": torch.from_numpy(y),
            "study_uid": uid,
        }


def collate_studies(batch):
    images = torch.cat([b["images"] for b in batch], dim=0)
    offsets = [0]
    for b in batch:
        offsets.append(offsets[-1] + b["images"].shape[0])
    return {
        "images": images,
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "target": torch.stack([b["target"] for b in batch]),
        "study_uid": [b["study_uid"] for b in batch],
    }


# ============================================================
# CLEAN BASELINE MODEL
# ============================================================

class ConvNeXtMeanMaxMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG.backbone_name,
            pretrained=CFG.pretrained,
            num_classes=0,
            global_pool="avg",
        )
        d = int(self.backbone.num_features)
        self.norm = nn.LayerNorm(d * 2)
        self.drop = nn.Dropout(CFG.dropout)
        self.classifier = nn.Linear(d * 2, N_LABELS)

    def encode_clips(self, images):
        device = next(self.parameters()).device
        out = []
        chunk = max(1, int(CFG.image_chunk_size))
        for a in range(0, len(images), chunk):
            x = images[a:a + chunk].to(device, non_blocking=True)
            out.append(self.backbone(x))
        return torch.cat(out, dim=0)

    def forward(self, batch):
        feat = self.encode_clips(batch["images"])
        offsets = batch["offsets"]
        study_feat = []
        for i in range(len(offsets) - 1):
            a, b = int(offsets[i]), int(offsets[i + 1])
            f = feat[a:b]
            mean = f.mean(dim=0)
            mx = f.max(dim=0).values
            study_feat.append(torch.cat([mean, mx], dim=0))
        z = torch.stack(study_feat, dim=0)
        return self.classifier(self.drop(self.norm(z)))


# ============================================================
# LOSS / METRIC
# ============================================================

def compute_pos_weight(train_df: pd.DataFrame) -> torch.Tensor:
    weights = []
    for label in LABELS:
        y = pd.to_numeric(train_df[label], errors="coerce")
        y = y[y.isin([0, 1])]
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        w = neg / max(pos, 1.0)
        weights.append(float(np.clip(w, 1.0, CFG.max_pos_weight)))
    return torch.tensor(weights, dtype=torch.float32)


def masked_bce(logits, targets, pos_weight):
    mask = torch.isfinite(targets)
    safe_targets = torch.nan_to_num(targets, nan=0.0)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        safe_targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def macro_auc(targets: np.ndarray, probs: np.ndarray):
    per_label = {}
    values = []
    for j, label in enumerate(LABELS):
        y = targets[:, j]
        p = probs[:, j]
        mask = np.isfinite(y)
        y, p = y[mask], p[mask]
        if len(y) > 0 and len(np.unique(y)) == 2:
            auc = float(roc_auc_score(y, p))
            values.append(auc)
            per_label[label] = auc
        else:
            per_label[label] = float("nan")
    return float(np.mean(values)) if values else float("nan"), per_label


# ============================================================
# OPTIMIZER / SCHEDULER
# ============================================================

def create_optimizer(model):
    m = unwrap_model(model)
    head_params = list(m.norm.parameters()) + list(m.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in m.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": CFG.lr_backbone},
            {"params": head_params, "lr": CFG.lr_head},
        ],
        weight_decay=CFG.weight_decay,
    )


def make_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def autocast_context():
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=CFG.amp and DEVICE.type == "cuda",
    )


def move_batch(batch):
    return {
        "images": batch["images"],  # kept on CPU; model moves chunks to GPU
        "offsets": batch["offsets"],
        "target": batch["target"].to(DEVICE, non_blocking=True),
        "study_uid": batch["study_uid"],
    }


# ============================================================
# TRAIN / VALIDATE
# ============================================================

def optimizer_step(model, optimizer, scheduler, scaler):
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()


def train_one_epoch(model, loader, optimizer, scheduler, scaler, pos_weight):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, n = 0.0, 0

    for step, raw in enumerate(tqdm(loader, desc="train", leave=False)):
        batch = move_batch(raw)
        with autocast_context():
            logits = model({"images": batch["images"], "offsets": batch["offsets"]})
            loss = masked_bce(logits, batch["target"], pos_weight)
            scaled = loss / CFG.grad_accum_steps
        scaler.scale(scaled).backward()

        if (step + 1) % CFG.grad_accum_steps == 0:
            optimizer_step(model, optimizer, scheduler, scaler)

        total += float(loss.detach())
        n += 1

    if len(loader) % CFG.grad_accum_steps != 0:
        optimizer_step(model, optimizer, scheduler, scaler)
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    targets_all, probs_all, uids_all = [], [], []

    for raw in tqdm(loader, desc="valid", leave=False):
        batch = move_batch(raw)
        with autocast_context():
            logits = model({"images": batch["images"], "offsets": batch["offsets"]})
        probs_all.append(torch.sigmoid(logits).float().cpu().numpy())
        targets_all.append(raw["target"].numpy())
        uids_all.extend(raw["study_uid"])

    targets = np.concatenate(targets_all)
    probs = np.concatenate(probs_all)

    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, (targets, probs, uids_all))
        targets = np.concatenate([x[0] for x in gathered])
        probs = np.concatenate([x[1] for x in gathered])
        uids_all = [u for x in gathered for u in x[2]]

        # Remove DistributedSampler padding duplicates by Study UID.
        keep, seen = [], set()
        for i, uid in enumerate(uids_all):
            if uid not in seen:
                seen.add(uid)
                keep.append(i)
        targets, probs = targets[keep], probs[keep]
        uids_all = [uids_all[i] for i in keep]

    score, per_label = macro_auc(targets, probs)
    return score, per_label, targets, probs, uids_all


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_auc, path):
    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "best_auc": float(best_auc),
        "cfg": asdict(CFG),
        "labels": LABELS,
        "protocol": "fixed_gold58_uniform_5slot_2p5d_convnext_meanmax",
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", -1)) + 1, float(ckpt.get("best_auc", -np.inf))


# ============================================================
# SUBMISSION
# ============================================================

@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs_all, uids_all = [], []
    for raw in tqdm(loader, desc="predict", leave=False):
        batch = move_batch(raw)
        with autocast_context():
            logits = model({"images": batch["images"], "offsets": batch["offsets"]})
        probs_all.append(torch.sigmoid(logits).float().cpu().numpy())
        uids_all.extend(raw["study_uid"])
    return uids_all, np.concatenate(probs_all)


def make_submission(root, test_df, test_series, sample_submission, rank=0):
    if not is_main_process():
        return
    best_path = Path(CFG.output_dir) / "best_model.pth"
    if not best_path.is_file():
        raise FileNotFoundError(best_path)

    # Build test cache on main process only.
    test_index = load_or_build_series_index(test_series, root, CFG.test_series_dir, rank=rank)
    test_studies = test_df[["StudyInstanceUID"]].copy()
    for label in LABELS:
        test_studies[label] = np.nan

    ds = KneeBaselineDataset(test_studies, test_index, train=False)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=CFG.num_workers,
        pin_memory=DEVICE.type == "cuda", collate_fn=collate_studies,
    )

    model = ConvNeXtMeanMaxMIL().to(DEVICE)
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    uids, probs = predict(model, loader)

    pred = pd.DataFrame(probs, columns=LABELS)
    pred.insert(0, "StudyInstanceUID", uids)

    # Respect sample_submission ordering and columns.
    sub = sample_submission[["StudyInstanceUID"]].merge(pred, on="StudyInstanceUID", how="left")
    for label in LABELS:
        sub[label] = sub[label].fillna(0.5)
    out = Path(CFG.output_dir) / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"Submission: {out} | shape={sub.shape}")


# ============================================================
# MAIN
# ============================================================

def main():
    ddp_enabled, rank, local_rank, world_size = ddp_setup()
    global DEVICE
    DEVICE = torch.device(f"cuda:{local_rank}" if ddp_enabled else CFG.device)

    seed_everything(CFG.seed + rank)
    Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)
    Path(CFG.cache_dir).mkdir(parents=True, exist_ok=True)

    if is_main_process():
        print("=" * 95)
        print("RSNA KNEE BASELINE | FIXED 58 GOLD VALIDATION")
        print("=" * 95)
        print(json.dumps(asdict(CFG), indent=2))

    root, train_df, val_df, train_series, test_df, test_series, sample_submission = prepare_fixed_gold_split()

    if CFG.mode.lower() == "inspect":
        if is_main_process():
            print("\nTRAIN:", train_df.shape)
            print(train_df.head())
            print("\nVALID GOLD:", val_df.shape)
            print(val_df.head())
        return

    if CFG.mode.lower() == "submission":
        make_submission(root, test_df, test_series, sample_submission, rank=rank)
        return

    # One rank builds cache, everybody then reads the same index.
    series_index = load_or_build_series_index(train_series, root, CFG.train_series_dir, rank=rank)

    if is_main_process():
        missing_tr = sum(uid not in series_index for uid in train_df["StudyInstanceUID"].astype(str))
        missing_va = sum(uid not in series_index for uid in val_df["StudyInstanceUID"].astype(str))
        print(f"Readable DICOM coverage missing: train={missing_tr}, valid={missing_va}")

        # Sampling diagnostics.
        counts = []
        slot_counts = {s[0]: 0 for s in SLOTS}
        for uid in val_df["StudyInstanceUID"].astype(str):
            slots = choose_slot_series(series_index.get(uid, []))
            counts.append(sum(len(uniform_centers(int(m['n_slices']), k)) for _, m, k in slots))
            for name, _, _ in slots:
                slot_counts[name] += 1
        print("Gold58 slot coverage:", slot_counts)
        if counts:
            print(f"Gold58 clips/study mean={np.mean(counts):.1f}, min={np.min(counts)}, max={np.max(counts)}")

    train_ds = KneeBaselineDataset(train_df, series_index, train=True)
    val_ds = KneeBaselineDataset(val_df, series_index, train=False)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    ) if ddp_enabled else None
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    ) if ddp_enabled else None

    common = dict(
        num_workers=CFG.num_workers,
        pin_memory=DEVICE.type == "cuda",
        collate_fn=collate_studies,
        persistent_workers=CFG.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size,
        shuffle=train_sampler is None, sampler=train_sampler, **common
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, sampler=val_sampler, **common
    )

    base_model = ConvNeXtMeanMaxMIL().to(DEVICE)
    model = base_model
    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    pos_weight = compute_pos_weight(train_df).to(DEVICE)
    if is_main_process():
        print("Positive weights:")
        for label, w in zip(LABELS, pos_weight.cpu().tolist()):
            print(f"  {label:20s}: {w:.3f}")
        print(f"Parameters: {sum(p.numel() for p in base_model.parameters()):,}")

    optimizer = create_optimizer(model)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / CFG.grad_accum_steps))
    total_steps = steps_per_epoch * CFG.epochs
    warmup_steps = int(steps_per_epoch * CFG.warmup_epochs)
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=CFG.amp and DEVICE.type == "cuda")

    start_epoch, best_auc = 0, -np.inf
    if CFG.resume_checkpoint:
        start_epoch, best_auc = load_checkpoint(
            CFG.resume_checkpoint, base_model, optimizer, scheduler, scaler
        )

    history = []
    for epoch in range(start_epoch, CFG.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, pos_weight
        )
        val_auc, per_label, targets, probs, uids = validate(model, val_loader)

        if is_main_process():
            print(f"\nEpoch {epoch + 1:02d}/{CFG.epochs} | loss={train_loss:.5f} | GOLD58 macro-AUC={val_auc:.6f}")
            for label in LABELS:
                print(f"  {label:20s}: {per_label[label]:.6f}" if np.isfinite(per_label[label]) else f"  {label:20s}: NaN")

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "gold58_macro_auc": val_auc,
                **{f"auc_{k}": v for k, v in per_label.items()},
            })
            pd.DataFrame(history).to_csv(Path(CFG.output_dir) / "history.csv", index=False)

            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(
                    base_model, optimizer, scheduler, scaler, epoch, best_auc,
                    Path(CFG.output_dir) / "best_model.pth",
                )
                pred_df = pd.DataFrame(probs, columns=LABELS)
                pred_df.insert(0, "StudyInstanceUID", uids)
                pred_df.to_csv(Path(CFG.output_dir) / "best_gold58_predictions.csv", index=False)
                print(f"NEW BEST GOLD58 AUC = {best_auc:.6f}")

            save_checkpoint(
                base_model, optimizer, scheduler, scaler, epoch, best_auc,
                Path(CFG.output_dir) / "last_checkpoint.pth",
            )

        if ddp_enabled:
            t = torch.tensor([best_auc], device=DEVICE, dtype=torch.float64)
            dist.broadcast(t, src=0)
            best_auc = float(t.item())
            barrier()

    if is_main_process():
        metrics = {
            "best_gold58_macro_auc": best_auc,
            "n_train": len(train_df),
            "n_valid_gold": len(val_df),
            "protocol": "fixed_gold58_uniform_5slot_2p5d_convnext_meanmax",
            "cfg": asdict(CFG),
        }
        with open(Path(CFG.output_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nTRAINING FINISHED")
        print(f"Best GOLD58 macro-AUC: {best_auc:.6f}")

    # Finish DDP before rank-0-only submission/cache work. This avoids a
    # distributed barrier deadlock inside test-cache construction.
    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    if CFG.make_submission_after_train and rank == 0:
        make_submission(root, test_df, test_series, sample_submission, rank=0)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
