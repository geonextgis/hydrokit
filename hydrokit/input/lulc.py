import numpy as np
import planetary_computer
import pystac_client
import rasterio as rio
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
from odc.stac import load
from rasterio.warp import Resampling, reproject


ESA_WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def get_lulc(
    bbox=None,
    reference_path=None,
    out_path=None,
    year=2021,
    nodata_value=0,
    buffer_ratio=None,
):
    """
    Download ESA WorldCover land-use / land-cover data from Planetary Computer.

    Data is fetched from the ``esa-worldcover`` STAC collection (10 m native
    resolution). If ``reference_path`` is provided the output is reprojected
    and resampled onto that raster's grid using nearest-neighbor resampling
    (appropriate for categorical data). Otherwise the native 10 m grid in
    EPSG:4326 is returned.

    Parameters
    ----------
    bbox : list, optional
        Bounding box as ``[minx, miny, maxx, maxy]`` in EPSG:4326.
        Required if ``reference_path`` is not provided.
    reference_path : str, optional
        Path to a reference raster (typically the DEM) defining the target
        CRS, transform, and shape. If given, ``bbox`` is derived from the
        reference bounds and the LULC is aligned to this grid.
    out_path : str, optional
        Path to save the LULC as GeoTIFF. If ``None``, data is not saved.
    year : int, optional
        WorldCover release year. Accepts ``2020`` (v100) or ``2021`` (v200).
        Defaults to ``2021``.
    nodata_value : int, optional
        NoData value to write to the output GeoTIFF. Defaults to ``0``.
    buffer_ratio : float, optional
        Fraction to expand the search / download bbox on each side before
        querying WorldCover (e.g. ``0.1`` = 10% on every edge). Prevents
        edge extrapolation when reprojecting onto the reference grid: the
        source raster extends past the target extent so nearest-neighbor
        resampling always has real pixels to sample. Only affects what is
        downloaded — the output grid still matches ``reference_path``
        exactly. Default: ``None`` (no buffer).

    Returns
    -------
    tuple of (np.ndarray, dict)
        The LULC array and a metadata dictionary with keys ``crs``,
        ``transform``, ``bounds``, ``resolution``, ``width``, ``height``,
        ``nodata``, and ``dtype``.

    Examples
    --------
    >>> bbox = [10.5, 51.5, 11.5, 52.0]
    >>> lulc, meta = get_lulc(bbox=bbox, out_path="lulc.tif")
    >>> lulc, meta = get_lulc(
    ...     reference_path="dem_latlon.tif",
    ...     out_path="lulc_aligned.tif",
    ...     buffer_ratio=0.1,
    ... )
    """
    if year not in (2020, 2021):
        raise ValueError(f"year must be 2020 or 2021, got {year}")

    if bbox is None and reference_path is None:
        raise ValueError("Either bbox or reference_path must be provided.")

    # Derive bbox from reference raster if needed
    if reference_path is not None:
        with rio.open(reference_path) as src:
            ref_meta = src.meta.copy()
            ref_bounds = src.bounds
            ref_crs = src.crs

        # WorldCover STAC search expects EPSG:4326 bounds
        if ref_crs.to_epsg() != 4326:
            from rasterio.warp import transform_bounds

            search_bbox = list(
                transform_bounds(ref_crs, "EPSG:4326", *ref_bounds, densify_pts=21)
            )
        else:
            search_bbox = [
                ref_bounds.left,
                ref_bounds.bottom,
                ref_bounds.right,
                ref_bounds.top,
            ]
        if bbox is None:
            bbox = search_bbox
    else:
        search_bbox = list(bbox)

    # Expand the download bbox so reprojection has data past the target edges.
    # The output grid (reference_path) is unchanged — only the source extent grows.
    if buffer_ratio:
        minx, miny, maxx, maxy = search_bbox
        buffer_x = buffer_ratio * (maxx - minx)
        buffer_y = buffer_ratio * (maxy - miny)
        search_bbox = [
            minx - buffer_x,
            miny - buffer_y,
            maxx + buffer_x,
            maxy + buffer_y,
        ]

    # Open the Planetary Computer STAC catalog
    API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/"
    catalog = pystac_client.Client.open(API_URL)

    # Filter by release year via datetime range
    # (2020 corresponds to product v1.0.0, 2021 to v2.0.0)
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=search_bbox,
        datetime=f"{year}-01-01/{year}-12-31",
    )

    items = list(search.get_items())
    print(f"Found {len(items)} WorldCover tile(s) for year {year}")

    if not items:
        raise RuntimeError(
            f"No ESA WorldCover items found for year {year} over bbox {search_bbox}."
        )

    signed_items = [planetary_computer.sign(item) for item in items]

    # Load the LULC using ODC STAC (map band contains the class values)
    ds = load(signed_items, bbox=search_bbox, bands=["map"])
    lulc_array = ds["map"].isel(time=0)

    # Align to reference grid if requested
    if reference_path is not None:
        src_data = lulc_array.values
        src_transform = lulc_array.rio.transform()
        src_crs = lulc_array.rio.crs

        dst_data = np.full(
            (ref_meta["height"], ref_meta["width"]),
            nodata_value,
            dtype=np.uint8,
        )

        reproject(
            source=src_data.astype(np.uint8),
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            src_nodata=0,
            dst_nodata=nodata_value,
            resampling=Resampling.nearest,
        )

        metadata = {
            "crs": ref_meta["crs"],
            "transform": ref_meta["transform"],
            "bounds": ref_bounds,
            "resolution": (
                ref_meta["transform"].a,
                -ref_meta["transform"].e,
            ),
            "width": ref_meta["width"],
            "height": ref_meta["height"],
            "nodata": nodata_value,
            "dtype": np.uint8,
        }

        if out_path:
            out_meta = ref_meta.copy()
            out_meta.update(
                {"count": 1, "dtype": "uint8", "nodata": nodata_value}
            )
            with rio.open(out_path, "w", **out_meta) as dst:
                dst.write(dst_data, 1)
            print(f"LULC saved to {out_path}")

        return dst_data, metadata

    # Native-resolution path
    metadata = {
        "crs": lulc_array.rio.crs,
        "transform": lulc_array.rio.transform(),
        "bounds": lulc_array.rio.bounds(),
        "resolution": lulc_array.rio.resolution(),
        "width": lulc_array.rio.width,
        "height": lulc_array.rio.height,
        "nodata": nodata_value,
        "dtype": lulc_array.dtype,
    }

    if out_path:
        lulc_array = lulc_array.rio.write_nodata(nodata_value)
        lulc_array.rio.to_raster(out_path)
        print(f"LULC saved to {out_path}")

    return lulc_array, metadata
