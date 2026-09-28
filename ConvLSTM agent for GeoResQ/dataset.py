import os
import re
import glob
import random
from datetime import datetime
from collections import defaultdict
import numpy as np
import h5py
import rasterio
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.enums import Resampling
from affine import Affine
from scipy.spatial import cKDTree
from pyproj import Transformer
import torch
from torch.utils.data import Dataset

DEM_NODATA_SENTINELS = (-9999, -32768)

BAND_KEYS = {
    "tir": "IMG_TIR1",
    "vis": "IMG_VIS",
    "wv": "IMG_WV",
}

MOSDAC_PROJ4 = (
    "+proj=merc +lat_ts=0 +lon_0=75 +x_0=0 +y_0=0 "
    "+a=6378137 +b=6356752.3142 +units=m +no_defs"
)

def parse_mosdac_time(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'(\d{2}[A-Z]{3}\d{4}_\d{4})', filename.upper())
    if match:
        try:
            return datetime.strptime(match.group(1), "%d%b%Y_%H%M")
        except ValueError:
            pass
    return datetime.fromtimestamp(os.path.getmtime(filepath))

def worker_init_fn(worker_id):
    base_seed = torch.initial_seed() % (2 ** 32)
    random.seed(base_seed + worker_id)
    np.random.seed(base_seed + worker_id)

class MOSDACDataset(Dataset):
    def __init__(self, data_dir, dem_path, patch_size=128, seq_length=4,
                 max_align_distance_deg=0.1, norm_sample_size=200, random_crop=True,
                 bbox=None, dem_downsample_factor=1):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.seq_length = seq_length
        self.max_align_distance_deg = max_align_distance_deg
        self.random_crop = random_crop
        self.bbox = bbox

        print(f"Loading DEM template from: {dem_path}")
        with rasterio.open(dem_path) as src:
            if bbox is not None:
                min_lon, min_lat, max_lon, max_lat = bbox
                window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
                out_height = max(1, round(window.height / dem_downsample_factor))
                out_width = max(1, round(window.width / dem_downsample_factor))
                self.dem_data = src.read(
                    1, window=window,
                    out_shape=(out_height, out_width),
                    resampling=Resampling.average,
                ).astype(np.float32)
                self.dem_meta = src.meta.copy()
                win_transform = window_transform(window, src.transform)
                scale_x = window.width / out_width
                scale_y = window.height / out_height
                self.dem_meta['transform'] = win_transform * Affine.scale(scale_x, scale_y)
                self.dem_meta['height'], self.dem_meta['width'] = self.dem_data.shape
            else:
                self.dem_data = src.read(1).astype(np.float32)
                self.dem_meta = src.meta

        nodata_vals = set(DEM_NODATA_SENTINELS)
        if self.dem_meta.get('nodata') is not None:
            nodata_vals.add(self.dem_meta['nodata'])
        for nd in nodata_vals:
            self.dem_data[self.dem_data == nd] = np.nan

        dem_min = np.nanmin(self.dem_data)
        dem_max = np.nanmax(self.dem_data)
        self.dem_data = (self.dem_data - dem_min) / (dem_max - dem_min + 1e-8)
        self.dem_data = np.nan_to_num(self.dem_data, nan=0.0)

        all_files = glob.glob(os.path.join(data_dir, "**", "*.h5"), recursive=True)
        if not all_files:
            raise ValueError(f"No .h5 files found under {data_dir}.")

        event_folders = defaultdict(list)
        for f in all_files:
            event_folders[os.path.dirname(f)].append(f)

        first_folder = next(iter(event_folders))
        sample_h5 = sorted(event_folders[first_folder], key=parse_mosdac_time)[0]

        with h5py.File(sample_h5, 'r') as f:
            x_coords = f['X'][:]
            y_coords = f['Y'][:]

        transformer = Transformer.from_crs(MOSDAC_PROJ4, "EPSG:4326", always_xy=True)
        xx, yy = np.meshgrid(x_coords, y_coords)
        sat_lons, sat_lats = transformer.transform(xx, yy)
        self.sat_shape = sat_lats.shape

        valid_files = set(all_files)
        self.sequences = []
        for folder, files in event_folders.items():
            files = [f for f in files if f in valid_files]
            files = sorted(files, key=parse_mosdac_time)
            if len(files) >= seq_length + 1:
                for i in range(len(files) - seq_length):
                    self.sequences.append(files[i: i + seq_length + 1])

        valid_mask = (sat_lats >= -90) & (sat_lats <= 90) & (sat_lons >= -180) & (sat_lons <= 180)
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            margin = self.max_align_distance_deg
            valid_mask &= (
                (sat_lons >= min_lon - margin) & (sat_lons <= max_lon + margin) &
                (sat_lats >= min_lat - margin) & (sat_lats <= max_lat + margin)
            )
        
        # ==========================================
        # 🚀 FULL CACHING SYSTEM (ALIGNMENT + STATS)
        # ==========================================
        cache_filename = f"spatial_cache_ds{dem_downsample_factor}.npz"
        if bbox is not None:
            cache_filename = f"spatial_cache_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_ds{dem_downsample_factor}.npz"

        band_keys_list = list(BAND_KEYS.keys())

        if os.path.exists(cache_filename):
            print(f"⚡ INSTANT LOAD: Found pre-computed cache ({cache_filename})")
            cache = np.load(cache_filename)
            self.align_full_indices_2d = cache['align_full_indices_2d']
            self.align_invalid_2d = cache['align_invalid_2d']
            
            self.band_stats = {}
            for idx, b_name in enumerate(band_keys_list):
                self.band_stats[b_name] = (float(cache['band_los'][idx]), float(cache['band_his'][idx]))
        else:
            print("⏳ Computing spatial alignment (happens only once)...")
            source_points = np.column_stack((sat_lons[valid_mask], sat_lats[valid_mask]))
            
            height, width = self.dem_data.shape
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            target_lons, target_lats = rasterio.transform.xy(self.dem_meta['transform'], rows, cols)
            target_points = np.column_stack((np.array(target_lons).ravel(), np.array(target_lats).ravel()))

            tree = cKDTree(source_points)
            dist, idx = tree.query(target_points, distance_upper_bound=self.max_align_distance_deg, workers=-1)

            align_valid_1d = np.isfinite(dist)
            align_indices = np.where(align_valid_1d, idx, 0)
            valid_mask_1d = valid_mask.ravel()

            sat_flat_idx = np.flatnonzero(valid_mask_1d)
            self.align_full_indices_2d = np.where(
                align_valid_1d, sat_flat_idx[align_indices], 0
            ).reshape(self.dem_data.shape)
            self.align_invalid_2d = (~align_valid_1d).reshape(self.dem_data.shape)

            print("⏳ Estimating global band stats by scanning sample H5 files...")
            self.band_stats = self._estimate_global_band_stats(valid_files, sample_size=norm_sample_size)
            
            band_los = np.array([self.band_stats[k][0] for k in band_keys_list])
            band_his = np.array([self.band_stats[k][1] for k in band_keys_list])

            print(f"💾 Saving complete cache to '{cache_filename}'.")
            np.savez_compressed(cache_filename, 
                                align_full_indices_2d=self.align_full_indices_2d, 
                                align_invalid_2d=self.align_invalid_2d,
                                band_los=band_los,
                                band_his=band_his)
        # ==========================================

    def _estimate_global_band_stats(self, valid_files, sample_size):
        files = list(valid_files)
        sample = random.sample(files, min(sample_size, len(files)))
        accum = {k: [] for k in BAND_KEYS}
        for fp in sample:
            with h5py.File(fp, 'r') as f:
                for band, key in BAND_KEYS.items():
                    if key in f:
                        arr = np.squeeze(f[key][:]).astype(np.float32)
                        arr = arr[np.isfinite(arr)]
                        if arr.size:
                            if arr.size > 20000:
                                arr = np.random.choice(arr, 20000, replace=False)
                            accum[band].append(arr)
        stats = {}
        for band, chunks in accum.items():
            if chunks:
                all_vals = np.concatenate(chunks)
                lo, hi = np.percentile(all_vals, [1, 99])
            else:
                lo, hi = 0.0, 1.0
            stats[band] = (float(lo), float(hi))
        return stats

    def __len__(self):
        return len(self.sequences)

    def normalize_band(self, array, band_name):
        lo, hi = self.band_stats[band_name]
        if hi - lo <= 1e-8:
            return np.zeros_like(array)
        return np.clip((array - lo) / (hi - lo), 0.0, 1.0)

    def extract_bands(self, h5_path, crop_y, crop_x):
        ps = self.patch_size
        idx_patch = self.align_full_indices_2d[crop_y:crop_y + ps, crop_x:crop_x + ps]
        invalid_patch = self.align_invalid_2d[crop_y:crop_y + ps, crop_x:crop_x + ps]
        valid_mask = ~invalid_patch

        if not valid_mask.any():
            return np.zeros((3, ps, ps), dtype=np.float32)

        valid_idx = idx_patch[valid_mask]
        sat_r = valid_idx // self.sat_shape[1]
        sat_c = valid_idx % self.sat_shape[1]

        min_r, max_r = sat_r.min(), sat_r.max()
        min_c, max_c = sat_c.min(), sat_c.max()

        local_r = sat_r - min_r
        local_c = sat_c - min_c

        def read_and_normalize(f, band_name):
            key = BAND_KEYS[band_name]
            if key not in f:
                window = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=np.float32)
            else:
                ds = f[key]
                if len(ds.shape) == 3:
                    window = ds[0, min_r:max_r + 1, min_c:max_c + 1].astype(np.float32)
                else:
                    window = ds[min_r:max_r + 1, min_c:max_c + 1].astype(np.float32)
            return self.normalize_band(window, band_name)

        with h5py.File(h5_path, 'r') as f:
            tir_win = read_and_normalize(f, "tir")
            vis_win = read_and_normalize(f, "vis")
            wv_win = read_and_normalize(f, "wv")

        out_stacked = np.zeros((3, ps, ps), dtype=np.float32)
        out_stacked[0][valid_mask] = tir_win[local_r, local_c]
        out_stacked[1][valid_mask] = vis_win[local_r, local_c]
        out_stacked[2][valid_mask] = wv_win[local_r, local_c]

        return np.nan_to_num(out_stacked, nan=0.0)

    def _sample_crop_origin(self):
        h, w = self.dem_data.shape
        max_y = h - self.patch_size
        max_x = w - self.patch_size
        if self.random_crop:
            return random.randint(0, max_y), random.randint(0, max_x)
        return max_y // 2, max_x // 2

    def __getitem__(self, idx):
        seq_files = self.sequences[idx]
        crop_y, crop_x = self._sample_crop_origin()

        x_seq = [self.extract_bands(f, crop_y, crop_x) for f in seq_files[:-1]]
        y_target = self.extract_bands(seq_files[-1], crop_y, crop_x)
        dem_patch = self.dem_data[crop_y:crop_y + self.patch_size, crop_x:crop_x + self.patch_size]

        x_seq = np.array(x_seq)
        y_target = np.array(y_target)
        dem_patch = dem_patch[np.newaxis, ...]

        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(dem_patch, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32),
        )