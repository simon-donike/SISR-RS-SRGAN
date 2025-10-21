import os
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional, Iterator
import math
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio as rio
from rasterio.windows import Window
from rasterio.errors import RasterioIOError
import random
import matplotlib.pyplot as plt
from ...utils.spectral_helpers import normalise_10k
import re
from pathlib import Path


def _percentile_stretch(x: np.ndarray, pmin=2, pmax=98) -> np.ndarray:
    """
    Per-channel percentile stretch to [0,1].
    x: (C,H,W) array
    """
    y = x.copy().astype(np.float32)
    C = y.shape[0]
    for c in range(C):
        lo = np.percentile(y[c], pmin)
        hi = np.percentile(y[c], pmax)
        if hi <= lo:
            hi = lo + 1e-6
        y[c] = (y[c] - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)

def _make_rgb(x: np.ndarray, idx_rgb: tuple[int,int,int]) -> np.ndarray:
    """
    x: (C,H,W), idx_rgb=(iR,iG,iB); returns (H,W,3) float in [0,1]
    """
    x3 = x[list(idx_rgb), ...]                # (3,H,W)
    x3 = _percentile_stretch(x3, 2, 98)       # simple per-band stretch
    return np.transpose(x3, (1, 2, 0))        # (H,W,3)


def _centered_window(rcwh, target_hw, img_hw):
    """Return a Window of size target_hw centered on rcwh, clamped to image bounds."""
    r0, c0, h, w = rcwh
    th, tw = target_hw
    H, W = img_hw
    # center of original window
    cr = r0 + h // 2
    cc = c0 + w // 2
    # new top-left
    nr0 = int(round(cr - th / 2))
    nc0 = int(round(cc - tw / 2))
    # clamp to image
    nr0 = max(0, min(nr0, max(0, H - th)))
    nc0 = max(0, min(nc0, max(0, W - tw)))
    return Window(col_off=nc0, row_off=nr0, width=tw, height=th)

def _read_centered_chip(path: str, rcwh, target_hw=(512, 512), pad_value=None, dtype="float32"):
    """Read a centered chip of size target_hw; pad if read returns smaller block (edge)."""
    th, tw = target_hw
    with rio.open(path) as ds:
        H, W = ds.height, ds.width
        wanted = _centered_window(rcwh, target_hw, (H, W))
        # Try read; if at border, rasterio will still return (th, tw) if within bounds;
        # If using boundless=True you might get smaller shapes. We'll be explicit:
        arr = ds.read(1, window=wanted, boundless=True, fill_value=pad_value if pad_value is not None else 0)
        # Ensure exact (th, tw) by padding (just in case)
        hh, ww = arr.shape
        if hh != th or ww != tw:
            ph = max(0, th - hh)
            pw = max(0, tw - ww)
            # pad bottom/right
            fill = pad_value if pad_value is not None else 0
            arr = np.pad(arr, ((0, ph), (0, pw)), mode="constant", constant_values=fill)
        return arr.astype(dtype)



def _iter_safe_band_files(root: Path,
                          exts=(".jp2", ".tif", ".tiff")) -> Iterator[Path]:
    """
    Yield all band rasters from S2 .SAFE products under root.
    Looks under .../GRANULE/*/IMG_DATA* recursively.
    """
    for safe in root.rglob("*.SAFE"):
        # Typical S2 layout: <PRODUCT>.SAFE/GRANULE/<granule_id>/IMG_DATA(/R10m|R20m|R60m)/*.jp2
        granule_dir = safe / "GRANULE"
        if not granule_dir.exists():
            continue
        for p in granule_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                # Heuristic: restrict to IMG_DATA subtrees
                if "IMG_DATA" in str(p):
                    yield p.resolve()


def _grid_windows(h: int, w: int, chip: Tuple[int, int], stride: Tuple[int, int]) -> List[Window]:
    """Generate a grid of rasterio.Windows over an array of size (h, w)."""
    ch, cw = chip
    sh, sw = stride
    if ch <= 0 or cw <= 0:
        raise ValueError("chip size must be > 0")
    if sh <= 0 or sw <= 0:
        raise ValueError("stride must be > 0")

    windows = []
    # Ensure last window covers the right/bottom edge if not perfectly divisible
    rows = list(range(0, max(1, h - ch + 1), sh))
    cols = list(range(0, max(1, w - cw + 1), sw))
    if rows[-1] + ch < h:
        rows.append(h - ch)
    if cols[-1] + cw < w:
        cols.append(w - cw)

    for r0 in rows:
        for c0 in cols:
            windows.append(Window(col_off=int(c0), row_off=int(r0),
                                  width=int(cw), height=int(ch)))
    return windows


_BAND_RE = re.compile(r"_(B(?:8A|[0-9]{2}))(_(?P<res>10m|20m|60m))?\.", re.IGNORECASE)

def _infer_band_name(p: Path) -> str:
    s = p.name  # filename
    m = _BAND_RE.search(s)
    if m:
        band = m.group(1).upper()  # B05, B8A, B11...
        res = m.group("res")
        if not res:
            # try infer from folder path e.g. .../IMG_DATA/R20m/...
            m2 = re.search(r"/R(10m|20m|60m)/", str(p).replace("\\", "/"), re.IGNORECASE)
            res = m2.group(1) if m2 else None
        if res:
            return f"{band}_{res.lower()}"
        else:
            # default to 20m if not found and file is under R20m
            if "IMG_DATA/R20m" in str(p).replace("\\", "/"):
                return f"{band}_20m"
            return band
    # fallback: original stem
    return p.stem


class S2SAFEWindowIndexBuilder:
    """
    Scans a root folder of .SAFE products, compiles:
      - a file catalog (list of band rasters)
      - a per-file list of raster windows (chip, stride)
    Optionally persists to a JSON manifest.
    """
    def __init__(
        self,
        root: str | Path,
        chip_hw: Tuple[int, int] = (256, 256),
        stride_hw: Tuple[int, int] = (256, 256),
        band_glob_filters: Optional[List[str]] = None,  # e.g., ["*B02*.jp2", "*B03*.jp2", "*B04*.jp2"]
        check_open: bool = True,
        manifest_json: Optional[str | Path] = None,
        skip_if_exists: bool = True,
    ):
        self.root = Path(root)
        self.chip_hw = tuple(map(int, chip_hw))
        self.stride_hw = tuple(map(int, stride_hw))
        self.band_glob_filters = band_glob_filters
        self.check_open = check_open
        self.manifest_json = Path(manifest_json) if manifest_json else None
        self.skip_if_exists = skip_if_exists

        self.files: List[Dict] = []     # [{'path': str, 'band': str, 'shape': [h,w], 'dtype': str, 'transform': [...]}]
        self.windows: List[Dict] = []   # [{'path': str, 'band': str, 'window': [r0,c0,h,w]}]

    def _filter_with_globs(self, paths: List[Path]) -> List[Path]:
        if not self.band_glob_filters:
            return paths
        out = []
        import fnmatch
        for p in paths:
            s = str(p)
            if any(fnmatch.fnmatch(s, pat) for pat in self.band_glob_filters):
                out.append(p)
        return out

    def build(self) -> None:
        # 1) Gather candidates
        candidates = list(_iter_safe_band_files(self.root))
        candidates.sort()
        candidates = self._filter_with_globs(candidates)

        # 2) Make file entries + per-file window grid
        for p in candidates:
            try:
                if self.check_open:
                    with rio.open(p) as ds:
                        h, w = ds.height, ds.width
                        windows = _grid_windows(h, w, chip=self.chip_hw, stride=self.stride_hw)
                        band = _infer_band_name(p)
                        file_entry = {
                            "path": str(p),
                            "band": band,
                            "shape": [h, w],
                            "dtype": str(ds.dtypes[0]),
                            "transform": list(ds.transform),  # Affine -> 6/9-tuple
                            "crs": str(ds.crs) if ds.crs else None,
                            "nodata": ds.nodata,
                        }
                        self.files.append(file_entry)
                        for win in windows:
                            self.windows.append({
                                "path": str(p),
                                "band": band,
                                "window": [int(win.row_off), int(win.col_off), int(win.height), int(win.width)]
                            })
                else:
                    # Cheap path: just record file, defer windowing (not recommended)
                    band = _infer_band_name(p)
                    file_entry = {
                        "path": str(p),
                        "band": band,
                        "shape": None,
                        "dtype": None,
                        "transform": None,
                        "crs": None,
                        "nodata": None,
                    }
                    self.files.append(file_entry)
            except RasterioIOError:
                # Skip unreadable file
                continue

        # 3) persist
        if self.manifest_json:
            # only skip if flag is False and file exists
            if not self.skip_if_exists and self.manifest_json.exists():
                return
            payload = {
                "root": str(self.root),
                "chip_hw": list(self.chip_hw),
                "stride_hw": list(self.stride_hw),
                "files": self.files,
                "windows": self.windows,
            }
            self.manifest_json.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_json, "w") as f:
                json.dump(payload, f, indent=2)

    def load(self) -> None:
        assert self.manifest_json and self.manifest_json.exists(), "Manifest JSON not found."
        with open(self.manifest_json, "r") as f:
            payload = json.load(f)
        self.files = payload.get("files", [])
        self.windows = payload.get("windows", [])


class S2SAFEDataset(Dataset):
    """
    PyTorch dataset that:
      - Uses a prebuilt window manifest (from S2SAFEWindowIndexBuilder)
      - Reads chips from single-band files, optionally stacks multiple bands per sample
    You can pass either:
      - manifest_json=<path>, or
      - files & windows lists directly (already computed)
    """
    def __init__(
        self,
        phase: str,
        manifest_json: Optional[str | Path] = None,
        files: Optional[List[Dict]] = None,
        windows: Optional[List[Dict]] = None,
        *,
        group_by: Optional[str] = None,
        group_regex: Optional[str] = None,
        bands_keep: Optional[List[str]] = None,
        band_order: Optional[List[str]] = None,
        dtype: str = "float32",
        scale: Optional[float] = None,
        hr_size: Tuple[int, int] = (512, 512),   # <— new (HR chip size)
        sr_factor: int = 4,                      # <— new (SR scale)
        antialias: bool = True,                  # <— optional, recommended
        split_seed: int = 42,
        val_fraction: float = 0.1,   # ignored if val_count is set
        val_count: Optional[int] = None,
    ):
        if manifest_json:
            with open(manifest_json, "r") as f:
                payload = json.load(f)
            files = payload["files"]
            windows = payload["windows"]

        assert files is not None and windows is not None, "Provide manifest_json or (files, windows)."
        self.files = files
        self.windows = windows
        self.dtype = dtype
        self.scale = scale
        self.hr_size = tuple(map(int, hr_size))
        self.sr_factor = int(sr_factor)
        if self.sr_factor <= 0:
            raise ValueError("sr_factor must be >= 1")

        # bad data handling
        self.bad_frac_thresh = 0.55
        self.black_eps = 1e-6
        self.max_bad_retries = 10

        # ensure divisibility
        if self.hr_size[0] % self.sr_factor != 0 or self.hr_size[1] % self.sr_factor != 0:
            raise ValueError(f"hr_size {self.hr_size} must be divisible by sr_factor {self.sr_factor}")

        self.lr_size = (self.hr_size[0] // self.sr_factor, self.hr_size[1] // self.sr_factor)
        self.antialias = bool(antialias)

        self.bands_keep = list(bands_keep) if bands_keep else None  # keep order!
        self.band_order = list(band_order) if band_order else self.bands_keep

         # filter windows if bands_keep provided
        if self.bands_keep:
            keep_set = set(self.bands_keep)
            self.windows = [w for w in self.windows if w["band"] in keep_set]

        # Optional grouping: build keys to stack multiple bands into one sample
        self.group_by = group_by
        self.group_regex = group_regex

        # Build a fast lookup from path -> file info
        self._fileinfo: Dict[str, Dict] = {f["path"]: f for f in self.files}

        # If bands_keep is set, filter windows to only those bands
        if self.bands_keep:
            self.windows = [w for w in self.windows if w["band"] in self.bands_keep]

        # If grouping, collapse windows by (group_key, window coords) to form multi-band samples
        if self.group_by:
            self._re = re.compile(self.group_regex) if self.group_regex else None
            self.samples = self._build_grouped_samples()
            # enforce desired order on each sample
            if self.band_order:
                order_map = {b:i for i,b in enumerate(self.band_order)}
                for s in self.samples:
                    idx = sorted(range(len(s["bands"])), key=lambda i: order_map.get(s["bands"][i], 1e9))
                    s["bands"] = [s["bands"][i] for i in idx]
                    s["paths"] = [s["paths"][i] for i in idx]
        else:
            self.samples = [{"paths":[w["path"]], "bands":[w["band"]], "window":w["window"]} for w in self.windows]

               # --- REPLACE your old split block with this ---
        if phase not in ("train", "val"):
            raise ValueError(f"phase must be 'train' or 'val', got {phase!r}")

        n = len(self.samples)
        if n == 0:
            self.phase = phase
            return

        # decide how many go to val
        if val_count is not None:
            k_val = max(0, min(int(val_count), n))
        else:
            k_val = max(1, min(int(round(n * float(val_fraction))), n-1))

        # deterministic shuffle of indices
        idxs = list(range(n))
        rng = random.Random(int(split_seed))
        rng.shuffle(idxs)

        val_set = set(idxs[:k_val])
        if phase == "val":
            self.samples = [self.samples[i] for i in idxs[:k_val]]
        else:  # train
            self.samples = [self.samples[i] for i in idxs[k_val:]]

        self.phase = phase


    def _extract_group_key(self, path: str) -> str:
        """
        Create a grouping key from the file path. If group_regex is set,
        use the first match group; otherwise, use the parent GRANULE folder.
        """
        p = Path(path)
        if self._re:
            m = self._re.search(str(p))
            if not m:
                return str(p.parent)
            return m.group(1) if m.groups() else m.group(0)
        # default: up to GRANULE/<id>
        parts = str(p).split(os.sep)
        if "GRANULE" in parts:
            i = parts.index("GRANULE")
            return os.sep.join(parts[: i + 2])  # .../GRANULE/<granule_id>
        return str(p.parent)

    def _build_grouped_samples(self) -> List[Dict]:
        # bucket windows by (group_key, window-rcwh)
        buckets: Dict[Tuple[str, Tuple[int,int,int,int]], Dict] = {}
        for w in self.windows:
            key = self._extract_group_key(w["path"])
            rcwh = tuple(w["window"])
            k = (key, rcwh)
            b = buckets.get(k)
            if b is None:
                b = {"paths": [], "bands": [], "window": rcwh}
                buckets[k] = b
            b["paths"].append(w["path"])
            b["bands"].append(w["band"])
        # Optionally sort bands in a stable order
        out = list(buckets.values())
        for s in out:
            order = np.argsort(s["bands"])
            s["paths"] = [s["paths"][i] for i in order]
            s["bands"] = [s["bands"][i] for i in order]
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def _read_chip(self, path: str, rcwh: List[int]) -> np.ndarray:
        r0, c0, h, w = rcwh
        with rio.open(path) as ds:
            win = Window(col_off=c0, row_off=r0, width=w, height=h)
            arr = ds.read(1, window=win, boundless=False)
        if self.scale is not None:
            arr = arr.astype(self.dtype) * self.scale
        else:
            arr = arr.astype(self.dtype)
        return arr

    @staticmethod
    def _bad_pixel_ratio(hr: torch.Tensor, *, nodata: float | None, black_eps: float) -> float:
        """
        hr: (C,H,W) torch float tensor (not normalized required; NaNs preserved).
        A pixel is invalid if ANY band is NaN/Inf OR equals nodata OR ALL bands are near-black (<=eps).
        Returns fraction in [0,1].
        """
        # NaN/Inf mask per-pixel
        finite_mask = torch.isfinite(hr).all(dim=0)              # (H,W) True if all bands finite
        # nodata mask (if provided)
        if nodata is not None:
            nodata_mask = (hr == nodata).any(dim=0)             # any band hits nodata
        else:
            nodata_mask = torch.zeros_like(finite_mask, dtype=torch.bool)
        # near-black across ALL bands
        black_mask = (hr.abs() <= black_eps).all(dim=0)

        invalid = (~finite_mask) | nodata_mask | black_mask
        return invalid.float().mean().item()
    
    
    def __getitem__(self, idx: int):
        s = self.samples[idx]
        rcwh = s["window"]  # (r0,c0,h,w) from manifest
        target_hi = self.hr_size
        target_lo = self.lr_size

        # use file's nodata for padding if available (first band as reference)
        nodata = None
        fi = self._fileinfo.get(s["paths"][0])
        if fi:
            nodata = fi.get("nodata", None)

        # read centered HR chips, force to hr_size
        hi_list = []
        for p in s["paths"]:
            chip = _read_centered_chip(p, rcwh, target_hw=target_hi, pad_value=nodata, dtype=self.dtype)
            hi_list.append(chip)
        hi = np.stack(hi_list, axis=0)  # (C, H_hr, W_hr)

        if self.scale is not None:
            hi = hi.astype(self.dtype) * self.scale
        else:
            hi = hi.astype(self.dtype)

        # downsample to LR using specified factor
        hi_t = torch.from_numpy(hi)  # (C,H,W)
        lo_t = F.interpolate(
            hi_t.unsqueeze(0),
            size=target_lo,
            mode="bilinear",
            align_corners=False,
            antialias=self.antialias,   # safe on newer torch; set False if on old version
        ).squeeze(0)

        # --- bad patch check (on raw values) ---
        bad_ratio = self._bad_pixel_ratio(hi_t, nodata=nodata, black_eps=self.black_eps)
        if bad_ratio >= self.bad_frac_thresh and self.phase == "train":
            # re-draw another sample (up to max retries)
            for _ in range(self.max_bad_retries):
                new_idx = random.randrange(len(self))
                if new_idx != idx:
                    return self.__getitem__(new_idx)
            # if all retries failed, proceed with original


        hr = hi_t.contiguous()
        lr = lo_t.contiguous()

        # normalise to 0-1 per band
        lr = normalise_10k(lr,stage="norm")
        hr = normalise_10k(hr,stage="norm")

        return lr, hr
    
    
    def _save_examples(self, out_png: str, *, idx: int | None = None, seed: int | None = None):
        # ... unchanged except titles reflect dynamic sizes ...
        rng = random.Random(seed)
        if idx is None:
            if len(self) == 0:
                raise RuntimeError("Dataset is empty.")
            idx = rng.randrange(len(self))
        lr, hr = self[idx]
        C = hr.shape[0]
        if C < 3:
            raise ValueError(f"Need at least 3 channels for RGB, got {C}.")
        idx_rgb = tuple(sorted(rng.sample(range(C), 3)))
        hr_np = hr.detach().cpu().numpy()
        lr_np = lr.detach().cpu().numpy()
        hr_rgb = _make_rgb(hr_np, idx_rgb)
        lr_rgb = _make_rgb(lr_np, idx_rgb)
        r,g,b = idx_rgb
        map_str = f"R=ch{r} G=ch{g} B=ch{b}"

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=120, constrained_layout=True)
        axes[0].imshow(hr_rgb); axes[0].set_title(f"HR {self.hr_size[0]}×{self.hr_size[1]}\n{map_str}"); axes[0].axis("off")
        axes[1].imshow(lr_rgb); axes[1].set_title(f"LR {self.lr_size[0]}×{self.lr_size[1]}\n(×{self.sr_factor})"); axes[1].axis("off")
        fig.suptitle(f"Sample idx={idx}", fontsize=11)
        fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":

    builder = S2SAFEWindowIndexBuilder(
        root="/data3/S2_20m",
        chip_hw=(512, 512),
        stride_hw=(512, 512),
        band_glob_filters=[
            "*B05*_20m*.jp2","*B06*_20m*.jp2","*B07*_20m*.jp2",
            "*B8A*_20m*.jp2",
            "*B11*_20m*.jp2","*B12*_20m*.jp2",
        ],
        manifest_json="/data3/S2_20m/s2_safe_manifest_20m.json",
        skip_if_exists=True,  # force overwrite
    )
    builder.build()
    print(f"Found {len(builder.files)} files and {len(builder.windows)} windows")

    # 2) multi-band samples grouped by GRANULE and same window, keep 20m
    desired_20m_order = ["B05_20m","B06_20m","B07_20m","B8A_20m","B11_20m","B12_20m"]
    ds_20m = S2SAFEDataset(
        phase="train",
        manifest_json="/data3/S2_20m/s2_safe_manifest_20m.json",
        group_by="granule",
        group_regex=r".*?/GRANULE/([^/]+)/IMG_DATA/.*",
        bands_keep=desired_20m_order,
        band_order=desired_20m_order,
        dtype="float32",
        hr_size=(512, 512),   # keep HR chip size
        sr_factor=8,          # ← now 8×
        antialias=True,
    )
    ds_20m_val = S2SAFEDataset(
        phase="val",
        manifest_json="/data3/S2_20m/s2_safe_manifest_20m.json",
        group_by="granule",
        group_regex=r".*?/GRANULE/([^/]+)/IMG_DATA/.*",
        bands_keep=desired_20m_order,
        band_order=desired_20m_order,
        dtype="float32",
        hr_size=(512, 512),   # keep HR chip size
        sr_factor=8,          # ← now 8×
        antialias=True,
    )

    print(len(ds_20m), len(ds_20m_val   ))

    # 3) Iterate
    x = ds_20m[1]
    lr,hr = x

    # 4) Save example
    ds_20m._save_examples("/data1/simon/GitHub/Remote-Sensing-SRGAN/other/example.png", seed=42)   # random sample
