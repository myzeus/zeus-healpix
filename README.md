# zeus-healpix

Read and work with HEALPix Earth observation data from ArrayLake.

`zeus-healpix` provides a minimal Python interface for loading HEALPix-gridded satellite and weather data stored in [ArrayLake](https://docs.earthmover.io/arraylake), converting between cube and 1D formats, and reprojecting to lat/lon grids.

## Installation

```bash
pip install zeus-healpix
```

With optional dependencies:

```bash
pip install zeus-healpix[arraylake]   # ArrayLake access
pip install zeus-healpix[latlon]      # Lat/lon reprojection (reproject + astropy)
pip install zeus-healpix[all]         # Everything
```

## Authentication

ArrayLake access requires an API token. Set it as an environment variable:

```bash
export ARRAYLAKE_API_KEY="your-token-here"
```

Or pass it directly:

```python
ds = grid.open_arraylake("zeus-ai/mirs-healpix", "mirs/n21/img/nside512/2024", token="...")
```


## Quick Start

### Open data from ArrayLake

```python
from zeus_healpix import HealPixGrid

grid = HealPixGrid(nside=512)

# List available datasets
groups = HealPixGrid.list_groups("zeus-ai/mirs-healpix")
# ['mirs/n20/img/nside512/2023', 'mirs/n20/img/nside512/2024', 'mirs/n20/img/nside512/2025', ...]

# Open a dataset (lazy loading)
ds = grid.open_arraylake("zeus-ai/mirs-healpix", "mirs/n21/img/nside512/2024")
```

`open_arraylake` auto-configures `nside` from the data, so you don't need to know the resolution in advance.

### Convert to xdggs format

```python
# Cube (face, y, x) -> 1D (cell_ids)
ds_1d = grid.to_xdggs(ds)

# Works with xdggs
import xdggs
ds_1d = ds_1d.dggs.decode()
```

### Reproject to lat/lon

```python
# Convert a single variable to a regular lat/lon grid
latlon = grid.to_latlon(ds["brightness_temperature"].isel(time=10, channel=1), resolution=0.25)

# latlon is an xarray DataArray with (lat, lon) dimensions
latlon.plot()
```

### Read local Zarr files

```python
grid = HealPixGrid(nside=64)
ds = grid.read_zarr("path/to/data.zarr")  # auto-detects cube vs 1D format
```

### Export for sharing

```python
from zeus_healpix import export_for_sharing

# Convert cube-format data to 1D and optionally write to disk
ds_1d = export_for_sharing(ds, output_path="shared_data.zarr")
```

## API Reference

### `HealPixGrid(nside, nested=True)`

| Method | Description |
|---|---|
| `open_arraylake(repo, group, token, branch)` | Open dataset from ArrayLake |
| `list_groups(repo, token, branch, prefix)` | List zarr groups in an ArrayLake repo (static) |
| `read_zarr(path, format)` | Read local zarr with format auto-detection |
| `to_xdggs(ds)` | Convert cube Dataset to 1D (cell_ids) format |
| `to_latlon(data, resolution)` | Reproject cube data to lat/lon grid |
| `cube_to_1d(cube)` | Convert numpy cube array to 1D healpy array |
| `cube_to_1d_xarray(ds)` | Convert xarray Dataset from cube to 1D |

| Property | Description |
|---|---|
| `nside` | Pixels per face side |
| `npix` | Total pixel count (12 * nside^2) |
| `nested` | NESTED ordering flag |
| `resolution_deg` | Resolution in degrees |

### Top-level functions

| Function | Description |
|---|---|
| `detect_healpix_format(ds)` | Returns `"cube"`, `"1d"`, or `"unknown"` |
| `export_for_sharing(ds, output_path)` | Convert cube to 1D and optionally write |


## HEALPix Formats

This package works with two HEALPix representations:

- **Cube format** `(face, y, x)` -- 12 faces of `nside x nside` pixels. This is the internal storage format used in ArrayLake.
- **1D format** `(cell_ids,)` -- flat array of `12 * nside^2` pixels in healpy NESTED ordering. Compatible with [xdggs](https://github.com/xarray-contrib/xdggs) and [healpy](https://healpy.readthedocs.io/).

## License

MIT
