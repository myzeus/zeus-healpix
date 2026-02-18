"""HEALPix grid utilities for reading and working with HEALPix Earth observation data.

This module provides utilities to convert between the HEALPix cube format
(12 faces x nside x nside) and the standard 1D healpy format that xdggs
expects, plus ArrayLake access for remote datasets.

Example:
    from zeus_healpix import HealPixGrid

    # Create grid and open remote data
    grid = HealPixGrid(nside=64)
    ds = grid.open_arraylake("zeus-ai/earthnet2", "goes16/ir/2024")

    # Convert cube to 1D for xdggs
    ds_xdggs = grid.to_xdggs(ds)

    # Convert to lat/lon
    latlon = grid.to_latlon(ds["C13"].isel(time=0))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import healpy as hp

import os
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class HealPixGrid:
    """HEALPix grid handler with xdggs and ArrayLake compatibility.

    Supports both the cube format (face, y, x) and the standard
    healpy 1D format that xdggs expects. Can read data from local zarr
    stores or ArrayLake repositories.

    Attributes:
        nside: Number of pixels per side of each face. Must be power of 2.
        nested: Whether to use NESTED (True) or RING (False) ordering.
    """

    nside: int
    nested: bool = True

    # Derived attributes
    npix: int = field(init=False)
    _idx_cache: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.npix = hp.nside2npix(self.nside)

        if self.nside & (self.nside - 1) != 0:
            raise ValueError(f"nside must be power of 2, got {self.nside}")

    @property
    def resolution_deg(self) -> float:
        """Resolution in degrees."""
        return hp.nside2resol(self.nside, arcmin=False) * 180 / np.pi

    # =========================================================================
    # ArrayLake access methods
    # =========================================================================

    @staticmethod
    def _get_arraylake_store(
        repo: str,
        token: str | None = None,
        branch: str = "main",
    ):
        """Create an ArrayLake readonly store.

        Args:
            repo: ArrayLake repository name (e.g., "earthmind/earthnet2").
            token: ArrayLake API token. If None, uses ARRAYLAKE_API_KEY env var.
            branch: Branch name to read from.

        Returns:
            Tuple of (arraylake_repo, store) for the readonly session.
        """
        try:
            import arraylake
        except ImportError:
            raise ImportError(
                "arraylake is required for ArrayLake access. "
                "Install it with: pip install zeus-healpix[arraylake]"
            )
        if token is None:
            token = os.environ.get("ARRAYLAKE_API_KEY")
        
        client = arraylake.Client(token=token)
        al_repo = client.get_repo(repo)
        store = al_repo.readonly_session(branch).store
        return al_repo, store

    def open_arraylake(
        self,
        repo: str,
        group: str | None = None,
        token: str | None = None,
        branch: str = "main",
    ) -> xr.Dataset:
        """Open a dataset from ArrayLake.

        Connects to an ArrayLake repository, opens the specified group as a
        lazy xarray Dataset, and auto-configures nside from the data.

        Args:
            repo: ArrayLake repository name (e.g., "earthmind/earthnet2").
            group: Zarr group path within the repo (e.g., "goes16/ir/2024").
            token: ArrayLake API token. If None, uses ARRAYLAKE_API_KEY env var.
            branch: Branch name to read from.

        Returns:
            Lazy xarray Dataset.

        Example:
            grid = HealPixGrid(nside=64)
            ds = grid.open_arraylake("earthmind/earthnet2", "goes16/ir/2024")
        """
        _, store = self._get_arraylake_store(repo, token=token, branch=branch)
        ds = xr.open_zarr(
            store, group=group, decode_timedelta=True, consolidated=False
        )

        # Auto-configure grid from data
        if "y" in ds.dims:
            self.nside = ds.sizes["y"]
            self.npix = 12 * self.nside**2
            self._idx_cache = None

        return ds

    @staticmethod
    def list_groups(
        repo: str,
        token: str | None = None,
        branch: str = "main",
        prefix: str | None = None,
    ) -> list[str]:
        """List available zarr groups in an ArrayLake repository.

        Recursively traverses the zarr group hierarchy and returns all
        group paths that contain data arrays (leaf groups).

        Args:
            repo: ArrayLake repository name (e.g., "earthmind/earthnet2").
            token: ArrayLake API token. If None, uses ARRAYLAKE_API_KEY env var.
            branch: Branch name to read from.
            prefix: Optional prefix filter (e.g., "goes16" to list only GOES groups).

        Returns:
            List of group path strings.

        Example:
            groups = HealPixGrid.list_groups("earthmind/earthnet2")
            # ['goes16/ir/2024', 'goes16/ir/2025', 'era5/temperature/2024', ...]
        """
        import zarr

        _, store = HealPixGrid._get_arraylake_store(
            repo, token=token, branch=branch
        )
        root = zarr.open_group(store, mode="r")

        groups = []

        def _traverse(group, path=""):
            children = list(group.groups())
            if not children:
                # Leaf group — has arrays but no subgroups
                if path:
                    groups.append(path)
                return
            for name, child in children:
                child_path = f"{path}/{name}" if path else name
                _traverse(child, child_path)

        _traverse(root)

        if prefix:
            groups = [g for g in groups if g.startswith(prefix)]

        return sorted(groups)

    # =========================================================================
    # Cube <-> 1D conversion methods
    # =========================================================================

    def cube_to_1d(self, cube: np.ndarray) -> np.ndarray:
        """Convert 3D cube (face, y, x) to 1D healpy array.

        Args:
            cube: Array of shape (..., 12, nside, nside).

        Returns:
            Array of shape (..., npix).
        """
        idxs = self._get_cube_to_1d_indices()

        shape_prefix = cube.shape[:-3]
        cube_flat = cube.reshape(*shape_prefix, -1)

        result = np.empty((*shape_prefix, self.npix), dtype=cube.dtype)
        result[..., idxs.flatten()] = cube_flat

        return result

    def cube_to_1d_xarray(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert xarray Dataset from cube format to 1D healpy format.

        Args:
            ds: Dataset with dimensions (face, y, x).

        Returns:
            Dataset with dimension (cell_ids,) suitable for xdggs.
        """
        import healpy as hp

        cell_ids = np.arange(self.npix)

        lon, lat = hp.pix2ang(self.nside, cell_ids, lonlat=True, nest=self.nested)
        lon[lon > 180] -= 360

        new_data_vars = {}
        for var_name, var in ds.data_vars.items():
            if "face" in var.dims and "y" in var.dims and "x" in var.dims:
                other_dims = [d for d in var.dims if d not in ("face", "y", "x")]
                var_t = var.transpose(*other_dims, "face", "y", "x")
                values_1d = self.cube_to_1d(var_t.values)

                new_dims = other_dims + ["cell_ids"]
                new_data_vars[var_name] = xr.DataArray(
                    values_1d,
                    dims=new_dims,
                    coords={
                        d: var.coords[d] for d in other_dims if d in var.coords
                    },
                )
            else:
                new_data_vars[var_name] = var

        ds_1d = xr.Dataset(
            new_data_vars,
            coords={
                "cell_ids": cell_ids,
                "latitude": ("cell_ids", lat),
                "longitude": ("cell_ids", lon),
            },
        )

        ds_1d.attrs["healpix_nside"] = self.nside
        ds_1d.attrs["healpix_nested"] = self.nested
        ds_1d.attrs["healpix_order"] = "nested" if self.nested else "ring"

        return ds_1d

    def _1d_to_cube(self, arr_1d: np.ndarray) -> np.ndarray:
        """Convert 1D healpy array to 3D cube (face, y, x).

        Args:
            arr_1d: Array of shape (..., npix).

        Returns:
            Array of shape (..., 12, nside, nside).
        """
        shape_prefix = arr_1d.shape[:-1]
        idxs = self._get_cube_to_1d_indices()

        cube_flat = arr_1d[..., idxs.flatten()]
        cube = cube_flat.reshape(*shape_prefix, 12, self.nside, self.nside)

        return cube

    def _get_cube_to_1d_indices(self) -> np.ndarray:
        """Get cached index mapping from cube to 1D.

        Returns:
            Array of shape (12, nside, nside) containing healpy indices.
        """
        if self._idx_cache is not None:
            return self._idx_cache

        f, y, x = np.meshgrid(
            np.arange(12),
            np.arange(self.nside),
            np.arange(self.nside),
            indexing="ij",
        )

        idxs = self._fyx_to_healpix_idx(f, y, x)
        self._idx_cache = idxs

        return idxs

    def _fyx_to_healpix_idx(
        self, f: np.ndarray, y: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """Convert face, y, x coordinates to healpy NESTED index.

        Args:
            f: Face indices (0-11).
            y: Y coordinates within face.
            x: X coordinates within face.

        Returns:
            HEALPix NESTED indices.
        """
        n_bits = int(np.log2(self.nside))

        result = np.zeros_like(y, dtype=np.int64)

        for i in range(n_bits):
            shift = n_bits - 1 - i
            y_bit = (y >> shift) & 1
            x_bit = (x >> shift) & 1

            result |= y_bit << (2 * shift + 1)  # y to odd bit positions
            result |= x_bit << (2 * shift)  # x to even bit positions

        return result + f * self.nside**2

    def _healpix_idx_to_fyx(self, pix_idx: int) -> tuple[int, int, int]:
        """Convert single HEALPix index to (face, y, x).

        Args:
            pix_idx: HEALPix NESTED index.

        Returns:
            Tuple of (face, y, x).
        """
        face = pix_idx // (self.nside**2)
        local_idx = pix_idx % (self.nside**2)
        n_bits = int(np.log2(self.nside))

        y = 0
        x = 0
        for i in range(n_bits):
            y += ((local_idx >> (2 * i + 1)) & 1) << i
            x += ((local_idx >> (2 * i)) & 1) << i

        return (face, y, x)

    def _healpix_idx_to_fyx_vectorized(
        self, pix_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert HEALPix indices to (face, y, x) arrays.

        Args:
            pix_indices: Array of HEALPix NESTED indices.

        Returns:
            Tuple of (face, y, x) arrays.
        """
        face = pix_indices // self.nside**2
        local_idx = pix_indices % self.nside**2

        n_bits = int(np.log2(self.nside))

        y = np.zeros_like(local_idx, dtype=np.int64)
        x = np.zeros_like(local_idx, dtype=np.int64)

        for i in range(n_bits):
            y_mask = 1 << (2 * i + 1)
            y += ((local_idx & y_mask) >> (2 * i + 1)) << i
            x_mask = 1 << (2 * i)
            x += ((local_idx & x_mask) >> (2 * i)) << i

        return face, y, x

    # =========================================================================
    # High-level conversion methods
    # =========================================================================

    def to_xdggs(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert Dataset to xdggs-compatible format.

        Converts the cube format (face, y, x) to 1D (cell_ids) format.

        Args:
            ds: Dataset with cube format (face, y, x dimensions).

        Returns:
            Dataset with cell_ids dimension, ready for xdggs.

        Example:
            grid = HealPixGrid(nside=64)
            ds_xdggs = grid.to_xdggs(ds_cube)

            # Now can use with xdggs
            import xdggs
            ds_xdggs = ds_xdggs.dggs.decode()
        """
        return self.cube_to_1d_xarray(ds)

    def read_zarr(
        self,
        path: str | Path,
        format: Literal["auto", "cube", "1d"] = "auto",
        **kwargs,
    ) -> xr.Dataset:
        """Read HEALPix data from Zarr with format auto-detection.

        Args:
            path: Path to Zarr store.
            format: Format to read as. "auto" detects from dimensions.
            **kwargs: Additional arguments to xr.open_zarr.

        Returns:
            Dataset in detected format.
        """
        ds = xr.open_zarr(path, **kwargs)

        if format == "auto":
            if "cell_ids" in ds.dims:
                format = "1d"
            elif "face" in ds.dims:
                format = "cube"
            else:
                raise ValueError(
                    f"Cannot auto-detect format. Dims: {list(ds.dims)}. "
                    "Expected 'cell_ids' or 'face' dimension."
                )

        if format == "cube" and "y" in ds.dims:
            self.nside = ds.sizes["y"]
            self.npix = 12 * self.nside**2
            self._idx_cache = None

        return ds

    def to_latlon(
        self,
        data: xr.DataArray | np.ndarray,
        resolution: float | None = None,
    ) -> xr.DataArray:
        """Convert HEALPix cube data to lat/lon grid.

        Takes data in cube format (face, y, x) and reprojects it to a
        regular lat/lon grid using the reproject library.

        Args:
            data: DataArray or array with (face, y, x) as last 3 dims.
                For DataArray, can have additional leading dims (time, band, etc.)
            resolution: Output resolution in degrees. If None, uses native
                HEALPix resolution.

        Returns:
            DataArray with (lat, lon) dimensions.

        Example:
            grid = HealPixGrid(nside=64)
            ds = xr.open_zarr("healpix_data.zarr")
            latlon = grid.to_latlon(ds["Rad"].isel(time=0, band=0))
        """
        try:
            import reproject as rp
            from astropy import wcs
        except ImportError:
            raise ImportError(
                "reproject and astropy are required for lat/lon conversion. "
                "Install them with: pip install zeus-healpix[latlon]"
            )

        import healpy as hp

        if resolution is None:
            resolution = hp.nside2resol(self.nside) * 180 / np.pi

        lats_grid = np.linspace(
            -90 + resolution / 2, 90 - resolution / 2, int(180 / resolution)
        )
        lons_grid = np.linspace(
            -180 + resolution / 2, 180 - resolution / 2, int(360 / resolution)
        )
        n_lats = len(lats_grid)
        n_lons = len(lons_grid)

        wcs_dict = {
            "CTYPE1": "RA---CAR",
            "CUNIT1": "deg",
            "CDELT1": resolution,
            "CRPIX1": (n_lons + 1) / 2,
            "CRVAL1": 0,
            "NAXIS1": n_lons,
            "CTYPE2": "DEC--CAR",
            "CUNIT2": "deg",
            "CDELT2": resolution,
            "CRPIX2": (n_lats + 1) / 2,
            "CRVAL2": 0.0,
            "NAXIS2": n_lats,
        }
        wcs_output = wcs.WCS(wcs_dict)

        if isinstance(data, xr.DataArray):
            arr = data.values
            other_dims = [d for d in data.dims if d not in ("face", "y", "x")]
            other_coords = {
                d: data.coords[d].values
                for d in other_dims
                if d in data.coords
            }
        else:
            arr = data
            other_dims = []
            other_coords = {}

        def _reproject_single(cube_3d: np.ndarray) -> np.ndarray:
            """Reproject single (face, y, x) array to (lat, lon)."""
            hpx_1d = self.cube_to_1d(cube_3d)

            ll_2d, ll_mask = rp.reproject_from_healpix(
                input_data=(hpx_1d, "icrs"),
                output_projection=wcs_output,
                shape_out=(n_lats, n_lons),
                nested=self.nested,
                order="nearest-neighbor",
            )

            ll_2d[ll_mask == 0] = np.nan
            return ll_2d

        if arr.ndim == 3:
            result = _reproject_single(arr)
        else:
            shape_prefix = arr.shape[:-3]
            result = np.empty((*shape_prefix, n_lats, n_lons), dtype=arr.dtype)

            for idx in np.ndindex(shape_prefix):
                result[idx] = _reproject_single(arr[idx])

        dims_out = other_dims + ["lat", "lon"]
        coords_out = {**other_coords, "lat": lats_grid, "lon": lons_grid}

        return xr.DataArray(result, dims=dims_out, coords=coords_out)


def detect_healpix_format(ds: xr.Dataset) -> Literal["cube", "1d", "unknown"]:
    """Detect the HEALPix format of a dataset.

    Args:
        ds: Dataset to analyze.

    Returns:
        "cube" if (face, y, x) format, "1d" if cell_ids format, else "unknown".
    """
    if "cell_ids" in ds.dims:
        return "1d"
    elif "face" in ds.dims and "y" in ds.dims and "x" in ds.dims:
        return "cube"
    return "unknown"


def export_for_sharing(
    ds: xr.Dataset,
    output_path: str | Path | None = None,
) -> xr.Dataset:
    """Export HEALPix data in xdggs-compatible format for external sharing.

    Converts cube format (face, y, x) to 1D format (cell_ids) that can be
    used with xdggs and healpy.

    Args:
        ds: Dataset in cube format with (face, y, x) dimensions.
        output_path: Optional path to write Zarr output.

    Returns:
        Dataset in 1D format with cell_ids dimension.

    Example:
        from zeus_healpix import export_for_sharing

        ds_xdggs = export_for_sharing(ds_cube, "shared_data.zarr")
    """
    nside = ds.sizes.get("y", 64)
    grid = HealPixGrid(nside=nside)

    ds_xdggs = grid.to_xdggs(ds)

    if output_path:
        ds_xdggs.to_zarr(output_path)

    return ds_xdggs
