"""Gridded daily climate forcing for the model domain (DEM grid, NetCDF).

Builds the meteorological forcing listed in section A of
``DATA_REQUIREMENTS.md`` from two DWD reanalysis products and writes one
NetCDF per variable, co-registered to the DEM grid (CRS, extent, resolution)
so it lines up cell-for-cell with the topography, soil, and land-use inputs.

Sources
-------
HYRAS (daily, EPSG:3035 ETRS89-LAEA, ~1 km) — the primary source, one NetCDF
per calendar year per variable:
    tas -> tavg, tasmin -> tmin, tasmax -> tmax, pr -> prec,
    hurs -> hurs, rsds -> rad, evpot(FAO ET0) -> et0

HOSTRADA (hourly, EPSG:3034 ETRS89-LCC, ~1 km) — used for the two channels
HYRAS does not publish as daily grids, aggregated hour->day here:
    sfcWind -> wind, ps -> psurf

The FAO grass-reference ET0 series (``grids_germany_daily_evaporation_fao_*``)
is on its own projection — DHDN / 3-degree Gauss-Kruger zone 3 (EPSG:31467),
not the standard HYRAS ETRS89-LAEA grid — so it carries a per-variable CRS
override in :data:`CLIMATE_VARS`. Standard HYRAS carries the projection in a
``crs`` variable and HOSTRADA likewise; both also ship a pair of 2-D
``lat``/``lon`` auxiliary coordinates. The auxiliary coordinates are dropped and
the projected 1-D ``x``/``y`` (``X``/``Y`` in HOSTRADA) are used as the spatial
dims for reprojection.

HYRAS variables are not all stamped at the same hour (temperature at 00:00,
precipitation at 06:00 for the hydrological day), so each series' time
coordinate is floored to the calendar day before slicing and writing, keeping
every variable aligned on a common daily index.

Regridding
----------
Each raw field is clipped to the DEM extent (with a small buffer so
reprojection has source pixels past the edges), then reprojected and resampled
onto the DEM grid with :func:`rioxarray.raster_array.RasterArray.reproject_match`.
Continuous fields use bilinear resampling by default. By default the output
matches the DEM cell-for-cell, which *upsamples* ~1 km climate onto the (~30 m)
DEM grid: it adds no sub-kilometre detail and daily files are large (one domain
timestep is height*width*4 bytes). Pass ``target_resolution`` to keep the DEM's
CRS and extent but write a coarser (e.g. ~1 km) grid instead, avoiding the
upsampling. Files are written per calendar year and zlib-compressed.

Output
------
One file per variable, ``<out_dir>/<var>/<var>.nc``, with dims ``(time, y, x)``
on the DEM grid, ``x``/``y`` in degrees (EPSG:4326) and a ``time`` (daily)
coordinate spanning the whole requested period. Each calendar year is regridded
to a temporary part and the parts are concatenated along time into this single
file. Derived vapour pressure ``vp`` [kPa] can be computed from ``hurs`` +
``tmin``/``tmax``.

Examples
--------
>>> get_climate(
...     reference_path="docs/examples/dem_latlon.tif",
...     start="2015-10-01",
...     end="2016-08-31",
...     variables=["tavg", "tmin", "tmax", "prec", "rad", "hurs",
...                "et0", "wind", "psurf"],
...     out_dir="data/interim/climate",
... )
>>> # then the derived actual vapour pressure channel:
>>> compute_vapour_pressure("data/interim/climate")
"""

from __future__ import annotations

import warnings
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

# Native grid CRS of each raw product. Both share metre-scale X/Y ranges, so a
# wrong-CRS clip silently yields empty/NaN output — the CRS must be written
# explicitly before any spatial operation.
HYRAS_CRS = "EPSG:3035"  # ETRS89-LAEA
HOSTRADA_CRS = "EPSG:3034"  # ETRS89-LCC

# Per-feature source configuration.
#   source   : "hyras" (yearly files) or "hostrada" (monthly hourly files)
#   raw_var  : variable name inside the NetCDF
#   dir      : subdirectory under the product root
#   hourly   : hour->day aggregation for HOSTRADA ("mean"/"sum"); None for HYRAS
#   resample : spatial resampling onto the DEM grid
#   crs      : native grid CRS override (else the source default is used)
CLIMATE_VARS: dict[str, dict] = {
    "tavg": {"source": "hyras", "raw_var": "tas", "dir": "TAS", "hourly": None,
             "resample": "bilinear"},
    "tmin": {"source": "hyras", "raw_var": "tasmin", "dir": "TASMIN", "hourly": None,
             "resample": "bilinear"},
    "tmax": {"source": "hyras", "raw_var": "tasmax", "dir": "TASMAX", "hourly": None,
             "resample": "bilinear"},
    "prec": {"source": "hyras", "raw_var": "pr", "dir": "PR", "hourly": None,
             "resample": "bilinear"},
    "hurs": {"source": "hyras", "raw_var": "hurs", "dir": "HURS", "hourly": None,
             "resample": "bilinear"},
    "rad": {"source": "hyras", "raw_var": "rsds", "dir": "RSDS", "hourly": None,
            "resample": "bilinear"},
    "et0": {"source": "hyras", "raw_var": "evpot", "dir": "ET0", "hourly": None,
            "resample": "bilinear", "crs": "EPSG:31467"},
    "wind": {"source": "hostrada", "raw_var": "sfcWind", "dir": "WSPD",
             "hourly": "mean", "resample": "bilinear"},
    "psurf": {"source": "hostrada", "raw_var": "ps", "dir": "PRESSURE_SURFACE",
              "hourly": "mean", "resample": "bilinear"},
}

# The FAO ET0 series stores its values under two internal names across years
# (``et0`` for 1961-1990 and 2021+, ``eta_fao`` for 1991-2020).
ET0_NC_VAR_CANDIDATES = ("et0", "eta_fao")

_RESAMPLING = {
    "bilinear": Resampling.bilinear,
    "nearest": Resampling.nearest,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}


def _src_crs(cfg: dict) -> str:
    """Native grid CRS for a variable: the per-variable ``crs`` override if
    present, else the source product default."""
    if "crs" in cfg:
        return cfg["crs"]
    return HYRAS_CRS if cfg["source"] == "hyras" else HOSTRADA_CRS


def _detect_spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    """Return the projected ``(x_dim, y_dim)`` names (HYRAS ``x``/``y`` or
    HOSTRADA ``X``/``Y``)."""
    for xd, yd in (("x", "y"), ("X", "Y")):
        if xd in da.dims and yd in da.dims:
            return xd, yd
    raise KeyError(f"No projected x/y dims found in {da.dims}")


def _hyras_file(root: str, cfg: dict, year: int) -> Path | None:
    """Highest-version HYRAS NetCDF for a ``(variable, year)`` pair.

    Standard HYRAS files are ``<var>_hyras_<n>_<year>_v*_de.nc``; the FAO ET0
    series uses ``grids_germany_daily_evaporation_fao_<year>_v*.nc``. Multiple
    version suffixes may coexist, so the lexicographically largest match (the
    latest version) is chosen.
    """
    var_dir = Path(root) / cfg["dir"]
    if cfg["raw_var"] == "evpot":
        matches = sorted(var_dir.glob(f"grids_germany_daily_evaporation_fao_{year}_v*.nc"))
    else:
        matches = sorted(var_dir.glob(f"{cfg['raw_var']}_hyras_*_{year}_v*_de.nc"))
    return matches[-1] if matches else None


def _hostrada_files(root: str, cfg: dict, year: int) -> list[Path]:
    """All monthly HOSTRADA NetCDFs for a ``(variable, year)`` pair."""
    pattern = (
        f"{root}/{cfg['dir']}/"
        f"{cfg['raw_var']}_1hr_HOSTRADA-v1-0_BE_gn_{year}*-{year}*.nc"
    )
    return sorted(Path(p) for p in glob(pattern))


def _reference_extent(reference_path: str, src_crs: str, buffer_ratio: float):
    """DEM bounds transformed into ``src_crs`` and buffered, as
    ``(minx, miny, maxx, maxy)`` for clipping the raw grid before reprojection."""
    ref = rioxarray.open_rasterio(reference_path)
    ref_bounds = ref.rio.bounds()
    ref_crs = ref.rio.crs
    minx, miny, maxx, maxy = transform_bounds(
        ref_crs, src_crs, *ref_bounds, densify_pts=21
    )
    bx = buffer_ratio * (maxx - minx)
    by = buffer_ratio * (maxy - miny)
    return (minx - bx, miny - by, maxx + bx, maxy + by)


def _prep_grid(da: xr.DataArray, src_crs: str, extent) -> xr.DataArray:
    """Attach ``src_crs``, drop 2-D lat/lon aux coords, and clip to ``extent``
    (in ``src_crs``) so the subsequent reprojection stays cheap."""
    xd, yd = _detect_spatial_dims(da)
    drop = [c for c in ("lat", "lon", "crs") if c in da.coords]
    da = da.drop_vars(drop, errors="ignore")
    da = da.rio.set_spatial_dims(x_dim=xd, y_dim=yd).rio.write_crs(src_crs)
    minx, miny, maxx, maxy = extent
    da = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    return da


def _load_year(
    var: str,
    cfg: dict,
    year: int,
    hyras_root: str,
    hostrada_root: str,
    extent,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> xr.DataArray | None:
    """Load one calendar year of a variable as a daily ``(time, y, x)`` array
    on its native grid, clipped to the DEM extent. HOSTRADA hourly data is
    aggregated to daily here, and every series' time coordinate is floored to
    the calendar day so variables stay aligned despite differing hour stamps.

    ``start_ts``/``end_ts`` bound which HOSTRADA monthly files and hourly steps
    are actually read, so out-of-range months are never resampled."""
    src_crs = _src_crs(cfg)
    if cfg["source"] == "hyras":
        path = _hyras_file(hyras_root, cfg, year)
        if path is None:
            warnings.warn(f"Missing HYRAS file for {var} {year} under {cfg['dir']}/")
            return None
        with xr.open_dataset(path) as ds:
            candidates = (
                ET0_NC_VAR_CANDIDATES
                if cfg["raw_var"] == "evpot"
                else (cfg["raw_var"],)
            )
            name = next((c for c in candidates if c in ds.data_vars), None)
            if name is None:
                raise KeyError(
                    f"None of {candidates} in {path} (have {list(ds.data_vars)})"
                )
            da = _prep_grid(ds[name].load(), src_crs, extent)
        return da.assign_coords(time=da["time"].dt.floor("D"))

    # HOSTRADA: for each monthly hourly file, clip to the DEM window and slice
    # to the requested dates *before* the hour->day resample, so only the small
    # in-window subset is ever aggregated. The last day's 23:00 stamp lies just
    # under end+1day, so the closed slice keeps the final day whole.
    files = _hostrada_files(hostrada_root, cfg, year)
    if not files:
        warnings.warn(f"No HOSTRADA files for {var} {year} under {cfg['dir']}/")
        return None
    hi = end_ts + pd.Timedelta(days=1)
    daily_months: list[xr.DataArray] = []
    for fp in files:
        with xr.open_dataset(fp) as ds:
            da = _prep_grid(ds[cfg["raw_var"]], src_crs, extent)
            da = da.sel(time=slice(start_ts, hi))
            if da.sizes.get("time", 0) == 0:
                continue
            resampler = da.load().resample(time="1D")
            daily = resampler.sum() if cfg["hourly"] == "sum" else resampler.mean()
            daily_months.append(daily)
    if not daily_months:
        return None
    da = xr.concat(daily_months, dim="time").sortby("time")
    # resample/concat drop the rio CRS; re-attach before reprojection.
    xd, yd = _detect_spatial_dims(da)
    da = da.rio.set_spatial_dims(x_dim=xd, y_dim=yd).rio.write_crs(src_crs)
    return da.assign_coords(time=da["time"].dt.floor("D"))


def _to_dem_grid(da: xr.DataArray, ref, resample: str) -> xr.DataArray:
    """Reproject/resample a native-grid daily array onto the DEM grid."""
    out = da.rio.reproject_match(ref, resampling=_RESAMPLING[resample])
    # Each source ships its own scalar grid-mapping variable (``crs`` for HYRAS,
    # ``transverse_mercator`` for ET0) that survives the reprojection and shadows
    # rioxarray's canonical ``spatial_ref``, so the CRS fails to resolve on
    # reload. Drop any inherited scalar coord and re-write the CRS cleanly.
    stale = [c for c in out.coords if c not in ("x", "y", "time") and out[c].ndim == 0]
    out = out.drop_vars(stale, errors="ignore")
    out.attrs.pop("grid_mapping", None)
    # Force the canonical ``spatial_ref`` grid-mapping name; a coord inheriting
    # the source name (``crs``) does not resolve back to a CRS on reload.
    return out.rio.write_crs(ref.rio.crs, grid_mapping_name="spatial_ref")


def _write_nc(da: xr.DataArray, var: str, out_path: Path) -> None:
    """Write a ``(time, y, x)`` array to a zlib-compressed NetCDF on the DEM
    grid. Dask-backed arrays stream to disk, so this stays memory-bounded when
    writing a multi-year concatenation."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da = da.rename(var).astype("float32")
    # Drop source fill/nodata attrs and stale encoding so neither collides with
    # the encoding we set here.
    for key in ("_FillValue", "missing_value"):
        da.attrs.pop(key, None)
    da.encoding = {}
    ds = da.to_dataset()
    encoding = {
        var: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
    }
    ds.to_netcdf(out_path, encoding=encoding)


def _combine_parts(part_paths: list[Path], var: str, out_path: Path) -> None:
    """Concatenate per-year part files into one NetCDF along ``time``.

    A single part is just renamed; multiple parts are opened lazily
    (chunked by time) and streamed to disk so peak memory stays near one
    time-chunk rather than the whole record."""
    if len(part_paths) == 1:
        part_paths[0].replace(out_path)
        return
    with xr.open_mfdataset(
        sorted(part_paths),
        combine="by_coords",
        decode_coords="all",
        chunks={"time": 30},
    ) as ds:
        _write_nc(ds[var].sortby("time"), var, out_path)
    for part in part_paths:
        part.unlink(missing_ok=True)


def get_climate(
    reference_path: str,
    start: str,
    end: str,
    variables: list[str] | None = None,
    out_dir: str = "data/interim/climate",
    hyras_root: str = "/data01/FDS/muduchuru/Atmos/DWD/HYRAS",
    hostrada_root: str = "/data01/FDS/muduchuru/Atmos/DWD/HOSTRADA",
    target_resolution: float | None = None,
    buffer_ratio: float = 0.1,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Build gridded daily climate forcing on the DEM grid for a time period.

    For each requested variable, every calendar year overlapping ``[start, end]``
    is clipped to the DEM extent, reprojected and resampled onto the target grid
    (DEM CRS + extent), and sliced to the requested dates; the years are then
    concatenated along time into one file per variable,
    ``<out_dir>/<var>/<var>.nc``.

    Parameters
    ----------
    reference_path : str
        Path to the DEM raster defining the target CRS and extent. Its native
        resolution is used unless ``target_resolution`` overrides it.
    start, end : str
        Inclusive date bounds, e.g. ``"2015-10-01"`` / ``"2016-08-31"``. A crop
        season crossing a calendar-year boundary is fully supported.
    variables : list of str, optional
        Subset of :data:`CLIMATE_VARS` to process. Defaults to all.
    out_dir : str, optional
        Root output directory. Defaults to ``data/interim/climate``.
    hyras_root, hostrada_root : str, optional
        Roots of the raw DWD products.
    target_resolution : float, optional
        Output cell size in the DEM CRS units (degrees for EPSG:4326). By
        default (``None``) the output matches the DEM grid cell-for-cell, which
        *upsamples* ~1 km climate to the DEM resolution. Set a coarser value
        (e.g. ``0.01`` deg ~ 1 km) to keep the DEM's CRS and extent but a
        lighter grid — avoiding the upsampling and the large file sizes it
        implies. The grid is still DEM-aligned, so climate, DEM, soil, and land
        use stay co-registered; only the climate cell size differs.
    buffer_ratio : float, optional
        Fraction to expand the DEM bbox on each side before clipping the raw
        grid, so bilinear resampling has source pixels past the target edges.
        Default ``0.1`` (10%).
    overwrite : bool, optional
        Re-write outputs that already exist. Default ``False``.

    Returns
    -------
    dict of {str: list of Path}
        Mapping of variable name to the list of NetCDF files written.
    """
    variables = variables or list(CLIMATE_VARS)
    unknown = set(variables) - set(CLIMATE_VARS)
    if unknown:
        raise ValueError(f"Unknown climate variables: {sorted(unknown)}")

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError(f"start ({start}) is after end ({end})")
    years = range(start_ts.year, end_ts.year + 1)

    # Target grid: the DEM itself, or a coarser grid on the DEM's CRS + extent.
    ref = rioxarray.open_rasterio(reference_path).squeeze()
    if target_resolution is not None:
        if target_resolution <= 0:
            raise ValueError(f"target_resolution must be > 0, got {target_resolution}")
        ref = ref.rio.reproject(ref.rio.crs, resolution=target_resolution)
    out_root = Path(out_dir)
    written: dict[str, list[Path]] = {}

    for var in variables:
        cfg = CLIMATE_VARS[var]
        extent = _reference_extent(reference_path, _src_crs(cfg), buffer_ratio)
        written[var] = []
        out_path = out_root / var / f"{var}.nc"
        if out_path.exists() and not overwrite:
            print(f"Skip {var} (exists)")
            written[var].append(out_path)
            continue

        # Each calendar year is regridded and written to a temporary part, then
        # the parts are concatenated along time into the single per-variable
        # file. Peak memory stays at one year (or one time-chunk when combining).
        parts: list[Path] = []
        for year in years:
            da = _load_year(
                var, cfg, year, hyras_root, hostrada_root, extent,
                start_ts, end_ts,
            )
            if da is None:
                continue

            # Slice to the requested dates within this year before regridding.
            da = da.sel(time=slice(start_ts, end_ts))
            if da.sizes.get("time", 0) == 0:
                continue

            da = _to_dem_grid(da, ref, cfg["resample"])
            part_path = out_path.parent / f".{var}_{year}.part.nc"
            _write_nc(da, var, part_path)
            parts.append(part_path)
            print(
                f"  {var} {year}: {da.sizes['time']} days, "
                f"{da.sizes[da.rio.y_dim]}x{da.sizes[da.rio.x_dim]}"
            )

        if not parts:
            print(f"No data for {var} over {start}..{end}")
            continue

        _combine_parts(parts, var, out_path)
        written[var].append(out_path)
        print(f"Wrote {out_path} ({len(parts)} year(s))")

    return written


def _sat_vp_kpa(t_c: xr.DataArray) -> xr.DataArray:
    """FAO-56 saturation vapour pressure [kPa] from temperature [degC]."""
    return 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))


def compute_vapour_pressure(
    climate_dir: str,
    overwrite: bool = False,
) -> Path | None:
    """Derive actual vapour pressure ``vp`` [kPa] on the DEM grid.

    Uses the FAO-56 convention of averaging saturation vapour pressure at
    ``tmin`` and ``tmax`` (the e_s curve is non-linear, so ``tavg`` would
    underestimate it), then scaling by relative humidity::

        es = (es(tmin) + es(tmax)) / 2
        vp = (hurs / 100) * es

    Reads the single-file NetCDFs written by :func:`get_climate` for ``tmin``,
    ``tmax``, and ``hurs`` and writes ``<climate_dir>/vp/vp.nc``.

    Returns
    -------
    Path or None
        The NetCDF written, or ``None`` if inputs are missing.
    """
    root = Path(climate_dir)
    out_path = root / "vp" / "vp.nc"
    if out_path.exists() and not overwrite:
        print(f"Skip vp (exists)")
        return out_path

    paths = {v: root / v / f"{v}.nc" for v in ("tmin", "tmax", "hurs")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        warnings.warn(f"Cannot compute vp, missing: {missing}")
        return None

    with xr.open_dataset(paths["tmin"], decode_coords="all") as dmin, \
            xr.open_dataset(paths["tmax"], decode_coords="all") as dmax, \
            xr.open_dataset(paths["hurs"], decode_coords="all") as dhurs:
        es = (_sat_vp_kpa(dmin["tmin"]) + _sat_vp_kpa(dmax["tmax"])) / 2.0
        vp = (dhurs["hurs"] / 100.0) * es
        _write_nc(vp, "vp", out_path)
    print(f"Wrote {out_path}")
    return out_path
