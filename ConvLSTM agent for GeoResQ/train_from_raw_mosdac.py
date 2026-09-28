"""
train_from_raw_mosdac.py
=========================================================================
End-to-end pipeline: raw multi-band satellite + terrain data on disk ->
reprojected/aligned tensors -> PyTorch ConvLSTM-UNet training loop.

WHAT THIS SCRIPT DOES (matches the pipeline spec it was written against)
-------------------------------------------------------------------------
1. Directory traversal   : walks ROOT_DIR (default "Desktop/project/training/"),
                            treating every immediate subfolder as one
                            historical flood event.
2. Multi-band discovery  : inside each event folder, finds the raw
                            HDF5/NetCDF files for bands TIR1, TIR2, WV, VIS,
                            SWIR, MIR, plus one CartoDEM GeoTIFF.
3. Reprojection           : uses rasterio + pyproj to reproject every band
                            into the same Mercator (EPSG:3857) grid as the
                            CartoDEM, resampling with bilinear interpolation
                            so a 1km VIS pixel and a 4-8km TIR/WV pixel both
                            land on the DEM's exact pixel grid.
4. Tensorization          : stacks the aligned bands into PyTorch tensors of
                            shape [C, H, W] (per timestamp) and [T, C, H, W]
                            (across a whole event's timestamps).
5. Sequential training    : loops over event folders one at a time, builds
                            their tensors, and runs them through a standard
                            PyTorch (not Lightning) training loop.

IMPORTANT - THIS SCRIPT MAKES ASSUMPTIONS YOU WILL LIKELY NEED TO ADAPT
-------------------------------------------------------------------------
Raw satellite HDF5/NetCDF products differ a lot between missions and
processing levels, and you didn't give me your exact file schema, so this
script is built to handle the three cases that cover almost every raw
INSAT/MOSDAC-style product, and clearly flags where YOU plug in your
product's real attribute/dataset names:

  Tier A - "GDAL-readable" files: GeoTIFF, or NetCDF/HDF5 that already carry
            a proper CRS + transform (or GDAL subdatasets). Read directly
            with rasterio. No changes usually needed.
  Tier B - Raw HDF5 with 4 corner lat/lons + a known uniform pixel size in
            its attributes, but no embedded CRS (common for sectorized
            fixed-grid products). We build the affine transform ourselves
            with pyproj + rasterio and reproject from there.
            -> Edit HDF5_CORNER_ATTRS / HDF5_DATA_KEY below to match your
               file's actual attribute/dataset names.
  Tier C - Raw HDF5 with full per-pixel 2D Latitude/Longitude swath arrays
            (true raw L1B geometry, no simple grid). We build Ground
            Control Points (GCPs) from the lat/lon arrays and let
            rasterio.warp do a GCP-based reprojection.
            -> Edit HDF5_LATLON_KEYS below to match your file's actual
               latitude/longitude dataset paths.

Search order per file: Tier A is tried first (cheapest / most robust); if
the file has no embedded georeferencing we fall back to Tier B, then Tier C.

Run this file from inside the ConvLSTM-UNet-master repo root (it imports
models/india_convlstm_unet.py from there).

    python train_from_raw_mosdac.py --root "Desktop/project/training" --epochs 20

Extra deps beyond the repo's requirements.txt:  rasterio, pyproj, h5py
    pip install rasterio pyproj h5py
"""

import argparse
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

import h5py
from pyproj import Transformer

# The model backbone lives in the repo this script is dropped into.
from models.india_convlstm_unet import IndiaConvLSTM_UNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mosdac_pipeline")


# =========================================================================
# 1. CONFIGURATION  -  the knobs you are most likely to need to adjust
# =========================================================================

# The six dynamic (per-timestamp) satellite bands, in the fixed channel
# order they will be stacked in. `pattern` is how we recognize which raw
# file in an event folder belongs to which band (case-insensitive substring
# match against the filename).
DYNAMIC_BANDS: List[str] = ["TIR1", "TIR2", "WV", "VIS", "SWIR", "MIR"]
BAND_FILENAME_PATTERNS: Dict[str, str] = {b: b for b in DYNAMIC_BANDS}

# CartoDEM / label file recognition (case-insensitive substring match,
# restricted to GeoTIFF-like extensions).
DEM_FILENAME_HINTS = ["cartodem", "dem"]
LABEL_FILENAME_HINTS = ["flood_extent", "flood_label", "flood_mask", "ground_truth", "label"]

# Raw satellite files: which extensions we treat as "a band file to read".
RAW_BAND_EXTENSIONS = {".h5", ".hdf", ".hdf5", ".nc", ".nc4"}
TERRAIN_EXTENSIONS = {".tif", ".tiff"}

# Timestamp token used to group same-acquisition-time files across bands,
# and to sort an event's frames into chronological order. Adjust the regex
# and strptime format to match your actual filenames, e.g. for
# "3DIMG_15JUL2023_0330_L1B_TIR1.h5" you'd use r"(\d{2}[A-Z]{3}\d{4}_\d{4})"
# and format "%d%b%Y_%H%M".
TIMESTAMP_REGEX = re.compile(r"(\d{8}_\d{4})")
TIMESTAMP_FORMAT = "%Y%m%d_%H%M"

# --- Tier B: corner-coordinate HDF5 attributes (edit to match your files) --
HDF5_CORNER_ATTRS = {
    "left_lon": "left_longitude",
    "right_lon": "right_longitude",
    "upper_lat": "upper_latitude",
    "lower_lat": "lower_latitude",
}
HDF5_DATA_KEY_TEMPLATE = "{band}"  # e.g. dataset "/TIR1" inside the h5 file

# --- Tier C: per-pixel lat/lon swath dataset paths (edit to match yours) --
HDF5_LATLON_KEYS = ("Latitude", "Longitude")
GCP_GRID_STEP = 32  # sample every Nth pixel for GCPs (speed/accuracy tradeoff)

# Target Mercator CRS for the whole pipeline (Web/Pseudo-Mercator).
TARGET_CRS = CRS.from_epsg(3857)

# Per-band physical value ranges used to min-max normalize into [0, 1]
# before feeding the network. These are placeholder brightness-temperature /
# reflectance ranges typical of INSAT-class imagers -- recalibrate them
# against your product's actual calibrated units.
BAND_NORM_RANGE = {
    "TIR1": (180.0, 330.0),   # Kelvin
    "TIR2": (180.0, 330.0),   # Kelvin
    "WV":   (200.0, 270.0),   # Kelvin
    "VIS":  (0.0, 100.0),     # % reflectance
    "SWIR": (0.0, 100.0),     # % reflectance
    "MIR":  (180.0, 330.0),   # Kelvin
}
DEM_NORM_RANGE = (0.0, 9000.0)  # meters; covers sea level to Himalayan peaks


# =========================================================================
# 2. LOW-LEVEL RASTER I/O - turning one raw file into (array, geolocation)
# =========================================================================

class BandGeolocation:
    """Carries exactly one of (crs + transform) OR (gcps + crs), describing
    how a raw band's pixel grid maps onto the earth. Passed straight into
    rasterio.warp.reproject()."""

    def __init__(self, crs=None, transform=None, gcps=None):
        self.crs = crs
        self.transform = transform
        self.gcps = gcps


def _try_tier_a_gdal_read(filepath: Path, band_name: str) -> Optional[Tuple[np.ndarray, BandGeolocation]]:
    """Tier A: file (or one of its GDAL subdatasets) already carries a
    proper CRS + transform. Works for GeoTIFF and many GDAL-readable
    NetCDF/HDF5 products out of the box."""
    try:
        with rasterio.open(str(filepath)) as ds:
            if ds.crs is not None and ds.transform is not None and not ds.transform.is_identity:
                arr = ds.read(1).astype("float32")
                return arr, BandGeolocation(crs=ds.crs, transform=ds.transform)

            # Multi-variable container (NetCDF/HDF5) exposes GDAL subdatasets
            # instead of being a raster itself -- find the one matching our band.
            if ds.subdatasets:
                target = next((sd for sd in ds.subdatasets if band_name.lower() in sd.lower()), None)
                target = target or ds.subdatasets[0]
                with rasterio.open(target) as sub:
                    if sub.crs is not None and sub.transform is not None:
                        arr = sub.read(1).astype("float32")
                        return arr, BandGeolocation(crs=sub.crs, transform=sub.transform)
    except rasterio.errors.RasterioIOError:
        pass
    return None


def _try_tier_b_corner_attrs(filepath: Path, band_name: str) -> Optional[Tuple[np.ndarray, BandGeolocation]]:
    """Tier B: raw HDF5 with 4 corner lat/lons + a uniform pixel grid in its
    global attributes, but no embedded CRS. We build the source affine
    transform ourselves (pyproj handles any coordinate sanity-checking you
    add; rasterio.warp does the actual reprojection later)."""
    try:
        with h5py.File(filepath, "r") as f:
            attrs = f.attrs
            required = HDF5_CORNER_ATTRS.values()
            if not all(k in attrs for k in required):
                return None
            data_key = HDF5_DATA_KEY_TEMPLATE.format(band=band_name)
            if data_key not in f:
                return None

            arr = f[data_key][()].astype("float32")
            h, w = arr.shape
            left = float(attrs[HDF5_CORNER_ATTRS["left_lon"]])
            right = float(attrs[HDF5_CORNER_ATTRS["right_lon"]])
            upper = float(attrs[HDF5_CORNER_ATTRS["upper_lat"]])
            lower = float(attrs[HDF5_CORNER_ATTRS["lower_lat"]])

            px_w = (right - left) / w
            px_h = (upper - lower) / h  # positive; transform below negates it
            transform = Affine(px_w, 0.0, left, 0.0, -px_h, upper)
            return arr, BandGeolocation(crs=CRS.from_epsg(4326), transform=transform)
    except (OSError, KeyError):
        return None


def _build_gcps_from_latlon(lat2d: np.ndarray, lon2d: np.ndarray, step: int) -> List[GroundControlPoint]:
    """Sample a coarse control-point grid from full-resolution swath lat/lon
    arrays; rasterio only needs a modest number of GCPs to fit an accurate
    warp, and using all of them would be needlessly slow."""
    h, w = lat2d.shape
    step = max(1, min(step, min(h, w) - 1))
    gcps = []
    for row in range(0, h, step):
        for col in range(0, w, step):
            gcps.append(GroundControlPoint(row=row, col=col, x=float(lon2d[row, col]), y=float(lat2d[row, col])))
    # Always include the far corner so the warp covers the full extent.
    gcps.append(GroundControlPoint(row=h - 1, col=w - 1, x=float(lon2d[-1, -1]), y=float(lat2d[-1, -1])))
    return gcps


def _try_tier_c_swath_latlon(filepath: Path, band_name: str) -> Optional[Tuple[np.ndarray, BandGeolocation]]:
    """Tier C: raw HDF5 carries full per-pixel 2D Latitude/Longitude arrays
    (true satellite viewing-geometry geolocation, no simple regular grid).
    We derive GCPs and let rasterio.warp do a proper GCP-based reprojection."""
    try:
        with h5py.File(filepath, "r") as f:
            data_key = HDF5_DATA_KEY_TEMPLATE.format(band=band_name)
            lat_key, lon_key = HDF5_LATLON_KEYS
            if data_key not in f or lat_key not in f or lon_key not in f:
                return None
            arr = f[data_key][()].astype("float32")
            lat2d = f[lat_key][()].astype("float64")
            lon2d = f[lon_key][()].astype("float64")
        gcps = _build_gcps_from_latlon(lat2d, lon2d, GCP_GRID_STEP)
        return arr, BandGeolocation(crs=CRS.from_epsg(4326), gcps=gcps)
    except (OSError, KeyError):
        return None


def read_band_array(filepath: Path, band_name: str) -> Tuple[np.ndarray, BandGeolocation]:
    """Read one raw band file, trying Tier A -> B -> C in order. Raises
    RuntimeError if none of them can make sense of the file (i.e. you need
    to add a fourth branch tailored to your exact product)."""
    for reader in (_try_tier_a_gdal_read, _try_tier_b_corner_attrs, _try_tier_c_swath_latlon):
        result = reader(filepath, band_name)
        if result is not None:
            return result
    raise RuntimeError(
        f"Could not geolocate '{filepath.name}' for band '{band_name}' using any known "
        "reader tier. Check HDF5_CORNER_ATTRS / HDF5_DATA_KEY_TEMPLATE / HDF5_LATLON_KEYS "
        "against this file's actual internal structure (e.g. `h5dump -H` it)."
    )


# =========================================================================
# 3. REPROJECTION - align every band onto the CartoDEM's Mercator grid
# =========================================================================

class TemplateGrid:
    """The reference pixel grid every band gets resampled onto: the
    CartoDEM's own CRS/transform/shape, reprojected into TARGET_CRS."""

    def __init__(self, crs: CRS, transform: Affine, width: int, height: int):
        self.crs = crs
        self.transform = transform
        self.width = width
        self.height = height


def load_dem_as_template(dem_path: Path) -> Tuple[np.ndarray, TemplateGrid]:
    """Reads the CartoDEM GeoTIFF and reprojects it (if needed) into
    TARGET_CRS. Its resulting grid becomes the template every other band is
    resampled onto, so the DEM itself defines H, W, transform and CRS for
    the whole event."""
    with rasterio.open(str(dem_path)) as src:
        if src.crs == TARGET_CRS:
            dem = src.read(1).astype("float32")
            template = TemplateGrid(src.crs, src.transform, src.width, src.height)
            return dem, template

        # Reproject DEM itself into Mercator first, keeping its native
        # resolution as closely as possible (pyproj informs the CRS
        # transform math rasterio.warp uses under the hood).
        from rasterio.warp import calculate_default_transform

        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds
        )
        dem = np.zeros((dst_h, dst_w), dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dem,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
        )
        template = TemplateGrid(TARGET_CRS, dst_transform, dst_w, dst_h)
        return dem, template


def align_band_to_template(
    arr: np.ndarray, geoloc: BandGeolocation, template: TemplateGrid, resampling=Resampling.bilinear
) -> np.ndarray:
    """Reproject+resample one band onto the exact CartoDEM pixel grid, using
    bilinear interpolation so coarser 4-8km TIR/WV pixels and finer 1km
    VIS/SWIR pixels both land correctly on the DEM's grid."""
    dst = np.zeros((template.height, template.width), dtype="float32")
    reproject(
        source=arr,
        destination=dst,
        src_transform=geoloc.transform,
        src_crs=geoloc.crs,
        gcps=geoloc.gcps,
        dst_transform=template.transform,
        dst_crs=template.crs,
        resampling=resampling,
    )
    return dst


def normalize(arr: np.ndarray, value_range: Tuple[float, float]) -> np.ndarray:
    """Clip to the physically plausible range then min-max scale to [0, 1].
    Keeps every channel on a comparable scale for the network."""
    lo, hi = value_range
    clipped = np.clip(arr, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype("float32")


# =========================================================================
# 4. PER-EVENT FILE DISCOVERY
# =========================================================================

def discover_event_folders(root_dir: Path) -> List[Path]:
    """Every immediate subfolder of root_dir is one historical flood event."""
    if not root_dir.exists():
        raise FileNotFoundError(f"Training root directory not found: {root_dir}")
    events = sorted(p for p in root_dir.iterdir() if p.is_dir())
    log.info("Found %d historical event folder(s) under %s", len(events), root_dir)
    return events


def _find_single_file(event_dir: Path, hints: List[str], extensions: set) -> Optional[Path]:
    candidates = [
        p for p in event_dir.iterdir()
        if p.suffix.lower() in extensions and any(h in p.name.lower() for h in hints)
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        log.warning("Multiple candidate files matched %s in %s; using the first: %s", hints, event_dir, candidates[0])
    return candidates[0]


def _extract_timestamp(filename: str) -> str:
    """Returns a sortable timestamp token for grouping/ordering frames. Falls
    back to the raw regex match (lexicographic sort) if strptime parsing
    fails, so discovery still works while you're still tuning the format."""
    match = TIMESTAMP_REGEX.search(filename)
    if not match:
        return filename  # no timestamp found -> treat whole filename as the key (single-frame event)
    token = match.group(1)
    try:
        datetime.strptime(token, TIMESTAMP_FORMAT)
    except ValueError:
        log.warning("Timestamp token '%s' in '%s' didn't match TIMESTAMP_FORMAT; sorting lexicographically.", token, filename)
    return token


def group_band_files_by_timestamp(event_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Scans one event folder and groups raw band files by acquisition
    timestamp, e.g. {"20230715_0330": {"TIR1": Path(...), "VIS": Path(...), ...}}.
    Only timestamps with ALL six bands present are kept, so every frame in
    the resulting sequence has a full, consistent channel stack."""
    by_timestamp: Dict[str, Dict[str, Path]] = defaultdict(dict)

    for f in event_dir.iterdir():
        if f.suffix.lower() not in RAW_BAND_EXTENSIONS:
            continue
        band = next((b for b, pat in BAND_FILENAME_PATTERNS.items() if pat.lower() in f.name.lower()), None)
        if band is None:
            continue  # not a recognized band file (could be metadata, QC, etc.)
        ts = _extract_timestamp(f.name)
        by_timestamp[ts][band] = f

    complete = {ts: bands for ts, bands in by_timestamp.items() if set(bands) == set(DYNAMIC_BANDS)}
    incomplete = set(by_timestamp) - set(complete)
    if incomplete:
        log.warning(
            "%s: dropping %d timestamp(s) with missing bands (found %s expected all of %s)",
            event_dir.name, len(incomplete), {ts: sorted(by_timestamp[ts]) for ts in incomplete}, DYNAMIC_BANDS,
        )
    return complete


# =========================================================================
# 5. TENSOR BUILDING - raw files for one event -> stacked tensors
# =========================================================================

def build_event_tensors(event_dir: Path) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """Builds one event's tensors straight from its raw files:
        dynamic_seq : [T, len(DYNAMIC_BANDS), H, W]  float32, values in [0,1]
        static_dem  : [1, H, W]                       float32, values in [0,1]
        target      : [1, H, W] or None (see LABEL_FILENAME_HINTS) if no
                      ground-truth flood mask was found for this event
    Returns None (and logs why) if the event can't be used at all, e.g. no
    CartoDEM or no complete-band timestamps.
    """
    dem_path = _find_single_file(event_dir, DEM_FILENAME_HINTS, TERRAIN_EXTENSIONS)
    if dem_path is None:
        log.warning("%s: no CartoDEM GeoTIFF found (looked for %s) -- skipping event.", event_dir.name, DEM_FILENAME_HINTS)
        return None

    dem_arr, template = load_dem_as_template(dem_path)
    dem_norm = normalize(dem_arr, DEM_NORM_RANGE)

    timestamp_groups = group_band_files_by_timestamp(event_dir)
    if not timestamp_groups:
        log.warning("%s: no timestamp had all %d bands present -- skipping event.", event_dir.name, len(DYNAMIC_BANDS))
        return None

    frames = []
    for ts in sorted(timestamp_groups):  # chronological order -> the ConvLSTM's time axis
        band_paths = timestamp_groups[ts]
        channels = []
        for band in DYNAMIC_BANDS:  # fixed channel order every frame
            arr, geoloc = read_band_array(band_paths[band], band)
            aligned = align_band_to_template(arr, geoloc, template)
            channels.append(normalize(aligned, BAND_NORM_RANGE[band]))
        frames.append(np.stack(channels, axis=0))  # [C, H, W] for this timestamp

    dynamic_seq = torch.from_numpy(np.stack(frames, axis=0))          # [T, C, H, W]
    static_dem = torch.from_numpy(dem_norm[np.newaxis, ...])          # [1, H, W]

    label_path = _find_single_file(event_dir, LABEL_FILENAME_HINTS, TERRAIN_EXTENSIONS)
    target = None
    if label_path is not None:
        with rasterio.open(str(label_path)) as lsrc:
            label_arr = lsrc.read(1).astype("float32")
            label_geoloc = BandGeolocation(crs=lsrc.crs, transform=lsrc.transform)
        label_aligned = align_band_to_template(label_arr, label_geoloc, template, resampling=Resampling.nearest)
        target = torch.from_numpy((label_aligned > 0).astype("float32")[np.newaxis, ...])  # [1, H, W]
    else:
        log.warning(
            "%s: no ground-truth flood label found (looked for %s) -- event has usable "
            "inputs but no target, it will be skipped during training.",
            event_dir.name, LABEL_FILENAME_HINTS,
        )

    log.info("%s: built tensors dynamic_seq=%s static_dem=%s target=%s",
              event_dir.name, tuple(dynamic_seq.shape), tuple(static_dem.shape),
              tuple(target.shape) if target is not None else None)
    return dynamic_seq, static_dem, target


# =========================================================================
# 6. DATASET
# =========================================================================

class FloodEventDataset(Dataset):
    """One PyTorch Dataset item == one full historical flood event: its
    [T,C,H,W] satellite sequence, [1,H,W] DEM, and [1,H,W] flood-extent
    target. Events without a usable DEM/band-set/label are dropped up front
    (see build_event_tensors) so every item returned here is trainable.

    Tensors are built once (raw files -> numpy -> torch) at construction
    time and cached in memory, since re-running the rasterio reprojection on
    every epoch would be needlessly slow -- swap this for lazy per-`__getitem__`
    loading if your events are too large/numerous to fit in RAM at once."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.event_names: List[str] = []
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for event_dir in discover_event_folders(root_dir):
            built = build_event_tensors(event_dir)
            if built is None:
                continue
            dynamic_seq, static_dem, target = built
            if target is None:
                continue  # no ground truth -> can't supervise this event
            self.event_names.append(event_dir.name)
            self.samples.append((dynamic_seq, static_dem, target))

        if not self.samples:
            raise RuntimeError(
                f"No trainable events found under {root_dir}. Every event was skipped "
                "-- check the warnings above (missing DEM, incomplete bands, or missing labels)."
            )
        log.info("Dataset ready: %d trainable event(s): %s", len(self.samples), self.event_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =========================================================================
# 7. TRAINING LOOP - sequential, one historical event at a time
# =========================================================================

class FocalLoss(nn.Module):
    """Focal BCE loss -- helps with the heavy class imbalance typical of
    flood-extent masks (mostly non-flooded pixels)."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


def train(root_dir: Path, epochs: int, lr: float, device: torch.device, checkpoint_path: Path) -> None:
    # --- Step 1: raw files -> tensors for every usable historical event ---
    dataset = FloodEventDataset(root_dir)
    # batch_size=1: each training step is exactly one historical event, per
    # the "loop through each event folder and train on it" spec. Increase
    # if you pad/crop events to a common H,W and want to batch several.
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- Step 2: build the PyTorch model -------------------------------
    model = IndiaConvLSTM_UNet(
        dynamic_channels=len(DYNAMIC_BANDS), static_channels=1, n_classes=1, base_channels=32
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    # --- Step 3: sequential training loop over historical events -------
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, (dynamic_seq, static_dem, target) in enumerate(loader, start=1):
            dynamic_seq = dynamic_seq.to(device)  # [B=1, T, C, H, W]
            static_dem = static_dem.to(device)    # [B=1, 1, H, W]
            target = target.to(device)            # [B=1, 1, H, W]

            optimizer.zero_grad()
            prediction = model(dynamic_seq, static_dem)  # [B=1, 1, H, W], sigmoid probabilities
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            log.info("epoch %d/%d step %d/%d [%s] loss=%.4f",
                      epoch, epochs, step, len(loader), dataset.event_names[step - 1], loss.item())

        log.info("epoch %d/%d complete - mean loss=%.4f", epoch, epochs, epoch_loss / len(loader))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    log.info("Training complete. Weights saved to %s", checkpoint_path)


# =========================================================================
# 8. ENTRY POINT
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train the flood nowcasting ConvLSTM-UNet on raw MOSDAC-style data.")
    parser.add_argument("--root", type=str, default=str(Path.home() / "Desktop" / "project" / "training"),
                         help="Root directory containing one subfolder per historical flood event.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/flood_nowcast_latest.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    train(Path(args.root), epochs=args.epochs, lr=args.lr, device=device, checkpoint_path=Path(args.checkpoint))


if __name__ == "__main__":
    main()
