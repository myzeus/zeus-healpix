"""Unit tests for HealPixGrid (no ArrayLake needed)."""

import numpy as np
import pytest
import xarray as xr

from zeus_healpix import HealPixGrid, detect_healpix_format, export_for_sharing


class TestHealPixGridInit:
    def test_create_default(self):
        grid = HealPixGrid(nside=64)
        assert grid.nside == 64
        assert grid.nested is True
        assert grid.npix == 12 * 64 * 64

    def test_create_various_nside(self):
        for nside in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            grid = HealPixGrid(nside=nside)
            assert grid.npix == 12 * nside * nside

    def test_invalid_nside(self):
        with pytest.raises(ValueError, match="power of 2"):
            HealPixGrid(nside=3)

    def test_resolution_deg(self):
        grid = HealPixGrid(nside=64)
        res = grid.resolution_deg
        assert isinstance(res, float)
        assert 0 < res < 10


class TestCubeTo1D:
    def test_basic_roundtrip(self):
        """cube_to_1d -> _1d_to_cube should recover original data."""
        grid = HealPixGrid(nside=4)
        cube = np.random.rand(12, 4, 4).astype(np.float32)

        arr_1d = grid.cube_to_1d(cube)
        assert arr_1d.shape == (grid.npix,)

        recovered = grid._1d_to_cube(arr_1d)
        assert recovered.shape == (12, 4, 4)
        np.testing.assert_allclose(recovered, cube, rtol=1e-6)

    def test_roundtrip_various_nside(self):
        for nside in [1, 2, 4, 8, 16]:
            grid = HealPixGrid(nside=nside)
            cube = np.random.rand(12, nside, nside).astype(np.float64)

            arr_1d = grid.cube_to_1d(cube)
            recovered = grid._1d_to_cube(arr_1d)
            np.testing.assert_allclose(recovered, cube, rtol=1e-10)

    def test_batch_dims(self):
        """Test with leading batch dimensions."""
        grid = HealPixGrid(nside=4)
        cube = np.random.rand(3, 5, 12, 4, 4)

        arr_1d = grid.cube_to_1d(cube)
        assert arr_1d.shape == (3, 5, grid.npix)

        recovered = grid._1d_to_cube(arr_1d)
        assert recovered.shape == (3, 5, 12, 4, 4)
        np.testing.assert_allclose(recovered, cube, rtol=1e-10)

    def test_index_cache(self):
        """Index cache should be populated after first call."""
        grid = HealPixGrid(nside=4)
        assert grid._idx_cache is None

        cube = np.ones((12, 4, 4))
        grid.cube_to_1d(cube)
        assert grid._idx_cache is not None

    def test_all_indices_covered(self):
        """Every healpy pixel should be assigned exactly once."""
        grid = HealPixGrid(nside=8)
        idxs = grid._get_cube_to_1d_indices()
        flat = idxs.flatten()
        assert len(flat) == grid.npix
        assert len(np.unique(flat)) == grid.npix
        assert flat.min() == 0
        assert flat.max() == grid.npix - 1


class TestCubeTo1DXarray:
    def _make_cube_dataset(self, nside=4):
        """Create a synthetic cube-format xarray Dataset."""
        data = np.random.rand(12, nside, nside).astype(np.float32)
        return xr.Dataset(
            {"temperature": (["face", "y", "x"], data)},
            coords={
                "face": np.arange(12),
                "y": np.arange(nside),
                "x": np.arange(nside),
            },
        )

    def test_basic(self):
        grid = HealPixGrid(nside=4)
        ds = self._make_cube_dataset(nside=4)

        ds_1d = grid.cube_to_1d_xarray(ds)
        assert "cell_ids" in ds_1d.dims
        assert ds_1d.sizes["cell_ids"] == grid.npix
        assert "latitude" in ds_1d.coords
        assert "longitude" in ds_1d.coords
        assert "temperature" in ds_1d.data_vars

    def test_with_extra_dims(self):
        """Test with time dimension in addition to spatial."""
        nside = 4
        grid = HealPixGrid(nside=nside)

        data = np.random.rand(3, 12, nside, nside).astype(np.float32)
        ds = xr.Dataset(
            {"temperature": (["time", "face", "y", "x"], data)},
            coords={
                "time": [0, 1, 2],
                "face": np.arange(12),
                "y": np.arange(nside),
                "x": np.arange(nside),
            },
        )

        ds_1d = grid.cube_to_1d_xarray(ds)
        assert ds_1d["temperature"].dims == ("time", "cell_ids")
        assert ds_1d.sizes["time"] == 3
        assert ds_1d.sizes["cell_ids"] == grid.npix

    def test_metadata_preserved(self):
        grid = HealPixGrid(nside=4)
        ds = self._make_cube_dataset(nside=4)

        ds_1d = grid.cube_to_1d_xarray(ds)
        assert ds_1d.attrs["healpix_nside"] == 4
        assert ds_1d.attrs["healpix_nested"] is True

    def test_non_spatial_var_preserved(self):
        """Variables without spatial dims should pass through."""
        nside = 4
        grid = HealPixGrid(nside=nside)

        ds = xr.Dataset(
            {
                "temperature": (
                    ["face", "y", "x"],
                    np.random.rand(12, nside, nside),
                ),
                "scalar": ([], 42.0),
            },
            coords={
                "face": np.arange(12),
                "y": np.arange(nside),
                "x": np.arange(nside),
            },
        )

        ds_1d = grid.cube_to_1d_xarray(ds)
        assert "scalar" in ds_1d.data_vars
        assert float(ds_1d["scalar"]) == 42.0


class TestToXdggs:
    def test_converts_cube_to_1d(self):
        nside = 4
        grid = HealPixGrid(nside=nside)

        ds = xr.Dataset(
            {"var": (["face", "y", "x"], np.ones((12, nside, nside)))},
            coords={
                "face": np.arange(12),
                "y": np.arange(nside),
                "x": np.arange(nside),
            },
        )

        ds_1d = grid.to_xdggs(ds)
        assert "cell_ids" in ds_1d.dims


class TestDetectHealpixFormat:
    def test_cube_format(self):
        ds = xr.Dataset(
            {"var": (["face", "y", "x"], np.zeros((12, 4, 4)))},
            coords={
                "face": np.arange(12),
                "y": np.arange(4),
                "x": np.arange(4),
            },
        )
        assert detect_healpix_format(ds) == "cube"

    def test_1d_format(self):
        ds = xr.Dataset(
            {"var": (["cell_ids"], np.zeros(192))},
            coords={"cell_ids": np.arange(192)},
        )
        assert detect_healpix_format(ds) == "1d"

    def test_unknown_format(self):
        ds = xr.Dataset(
            {"var": (["lat", "lon"], np.zeros((10, 20)))},
        )
        assert detect_healpix_format(ds) == "unknown"


class TestExportForSharing:
    def test_basic_export(self):
        nside = 4
        ds = xr.Dataset(
            {"temperature": (["face", "y", "x"], np.random.rand(12, nside, nside))},
            coords={
                "face": np.arange(12),
                "y": np.arange(nside),
                "x": np.arange(nside),
            },
        )

        ds_1d = export_for_sharing(ds)
        assert "cell_ids" in ds_1d.dims
        assert ds_1d.sizes["cell_ids"] == 12 * nside * nside

    def test_export_to_file(self, tmp_path):
        nside = 4
        ds = xr.Dataset(
            {"temperature": (["face", "y", "x"], np.random.rand(12, nside, nside))},
            coords={
                "face": np.arange(12),
                "y": np.arange(nside),
                "x": np.arange(nside),
            },
        )

        out = tmp_path / "test.zarr"
        ds_1d = export_for_sharing(ds, output_path=str(out))
        assert out.exists()

        ds_read = xr.open_zarr(str(out))
        assert "cell_ids" in ds_read.dims


class TestToLatlon:
    def test_basic(self):
        """Test to_latlon produces correct output shape."""
        nside = 4
        grid = HealPixGrid(nside=nside)

        data = np.random.rand(12, nside, nside).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=["face", "y", "x"],
            coords={
                "face": np.arange(12),
                "y": np.arange(nside),
                "x": np.arange(nside),
            },
        )

        result = grid.to_latlon(da, resolution=5.0)
        assert "lat" in result.dims
        assert "lon" in result.dims
        assert result.sizes["lat"] == 36  # 180 / 5.0
        assert result.sizes["lon"] == 72  # 360 / 5.0

    def test_numpy_input(self):
        """Test to_latlon with raw numpy array."""
        nside = 4
        grid = HealPixGrid(nside=nside)

        data = np.random.rand(12, nside, nside).astype(np.float32)
        result = grid.to_latlon(data, resolution=5.0)
        assert result.sizes["lat"] == 36
        assert result.sizes["lon"] == 72

    def test_with_batch_dims(self):
        """Test to_latlon with leading batch dimensions."""
        nside = 4
        grid = HealPixGrid(nside=nside)

        data = np.random.rand(2, 12, nside, nside).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=["time", "face", "y", "x"],
            coords={
                "time": [0, 1],
                "face": np.arange(12),
                "y": np.arange(nside),
                "x": np.arange(nside),
            },
        )

        result = grid.to_latlon(da, resolution=5.0)
        assert result.dims == ("time", "lat", "lon")
        assert result.sizes["time"] == 2


class TestIndexMapping:
    def test_fyx_to_healpix_roundtrip(self):
        """Single pixel: fyx -> healpix -> fyx should roundtrip."""
        grid = HealPixGrid(nside=8)

        for face in [0, 5, 11]:
            for y in [0, 3, 7]:
                for x in [0, 4, 7]:
                    f_arr = np.array([face])
                    y_arr = np.array([y])
                    x_arr = np.array([x])

                    idx = grid._fyx_to_healpix_idx(f_arr, y_arr, x_arr)
                    f2, y2, x2 = grid._healpix_idx_to_fyx(int(idx[0]))

                    assert f2 == face
                    assert y2 == y
                    assert x2 == x

    def test_vectorized_matches_scalar(self):
        """Vectorized fyx->healpix should match scalar version."""
        grid = HealPixGrid(nside=8)

        for pix in [0, 100, grid.npix - 1]:
            f1, y1, x1 = grid._healpix_idx_to_fyx(pix)
            f_arr, y_arr, x_arr = grid._healpix_idx_to_fyx_vectorized(
                np.array([pix])
            )
            assert f_arr[0] == f1
            assert y_arr[0] == y1
            assert x_arr[0] == x1


class TestReadZarr:
    def test_cube_format(self, tmp_path):
        nside = 4
        ds = xr.Dataset(
            {"var": (["face", "y", "x"], np.random.rand(12, nside, nside))},
            coords={
                "face": np.arange(12),
                "y": np.arange(nside),
                "x": np.arange(nside),
            },
        )
        path = str(tmp_path / "cube.zarr")
        ds.to_zarr(path)

        grid = HealPixGrid(nside=1)  # Will be overwritten
        ds_read = grid.read_zarr(path)
        assert grid.nside == nside
        assert "face" in ds_read.dims

    def test_1d_format(self, tmp_path):
        nside = 4
        npix = 12 * nside * nside
        ds = xr.Dataset(
            {"var": (["cell_ids"], np.random.rand(npix))},
            coords={"cell_ids": np.arange(npix)},
        )
        path = str(tmp_path / "1d.zarr")
        ds.to_zarr(path)

        grid = HealPixGrid(nside=nside)
        ds_read = grid.read_zarr(path)
        assert "cell_ids" in ds_read.dims

    def test_auto_detect_error(self, tmp_path):
        ds = xr.Dataset(
            {"var": (["lat", "lon"], np.zeros((10, 20)))},
        )
        path = str(tmp_path / "latlon.zarr")
        ds.to_zarr(path)

        grid = HealPixGrid(nside=4)
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            grid.read_zarr(path)
