"""Integration tests for ArrayLake access (requires ARRAYLAKE_API_KEY)."""

import os

import numpy as np
import pytest

from zeus_healpix import HealPixGrid

REPO = "earthmind/earthnet2"

pytestmark = pytest.mark.skipif(
    not os.environ.get("ARRAYLAKE_API_KEY"),
    reason="ARRAYLAKE_API_KEY not set",
)


class TestListGroups:
    def test_returns_non_empty(self):
        groups = HealPixGrid.list_groups(REPO)
        assert isinstance(groups, list)
        assert len(groups) > 0

    def test_prefix_filter(self):
        groups = HealPixGrid.list_groups(REPO, prefix="goes16")
        assert all(g.startswith("goes16") for g in groups)

    def test_sorted(self):
        groups = HealPixGrid.list_groups(REPO)
        assert groups == sorted(groups)


class TestOpenArraylake:
    @pytest.fixture
    def grid_and_ds(self):
        """Open a known dataset from ArrayLake."""
        groups = HealPixGrid.list_groups(REPO)
        assert len(groups) > 0, "No groups found in repo"
        group = groups[0]

        grid = HealPixGrid(nside=64)
        ds = grid.open_arraylake(REPO, group)
        return grid, ds, group

    def test_returns_dataset(self, grid_and_ds):
        import xarray as xr

        grid, ds, _ = grid_and_ds
        assert isinstance(ds, xr.Dataset)

    def test_nside_auto_configured(self, grid_and_ds):
        grid, ds, _ = grid_and_ds
        if "y" in ds.dims:
            assert grid.nside == ds.sizes["y"]
            assert grid.npix == 12 * grid.nside**2

    def test_has_spatial_dims(self, grid_and_ds):
        grid, ds, _ = grid_and_ds
        # Should have either cube or 1d format
        has_cube = "face" in ds.dims and "y" in ds.dims and "x" in ds.dims
        has_1d = "cell_ids" in ds.dims
        assert has_cube or has_1d


class TestEndToEnd:
    @pytest.fixture
    def cube_dataset(self):
        """Find and open a cube-format dataset."""
        groups = HealPixGrid.list_groups(REPO)
        grid = HealPixGrid(nside=64)

        for group in groups:
            ds = grid.open_arraylake(REPO, group)
            if "face" in ds.dims and "y" in ds.dims:
                return grid, ds
        pytest.skip("No cube-format dataset found in repo")

    def test_to_xdggs(self, cube_dataset):
        grid, ds = cube_dataset
        ds_1d = grid.to_xdggs(ds)
        assert "cell_ids" in ds_1d.dims
        assert ds_1d.sizes["cell_ids"] == grid.npix

    def test_to_latlon(self, cube_dataset):
        grid, ds = cube_dataset

        # Pick the first data variable and select a single spatial slice
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]

        # Reduce to (face, y, x) by selecting first index of extra dims
        for dim in da.dims:
            if dim not in ("face", "y", "x"):
                da = da.isel({dim: 0})

        result = grid.to_latlon(da, resolution=5.0)
        assert "lat" in result.dims
        assert "lon" in result.dims
        assert result.sizes["lat"] == 36
        assert result.sizes["lon"] == 72
