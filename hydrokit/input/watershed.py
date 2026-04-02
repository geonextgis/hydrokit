import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rioxarray
import xarray as xr
from odc.stac import load
from pysheds.grid import Grid
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import box, mapping


def get_dem(bbox, out_path=None, nodata_value=-32768):
    """
    Download and load DEM data from Copernicus (30m resolution).

    Parameters
    ----------
    bbox : list
        Bounding box as [minx, miny, maxx, maxy]
    out_path : str, optional
        Path to save the DEM as GeoTIFF. If None, data is not saved.
    nodata_value : int or float, optional
        The NoData value to write to the output GeoTIFF. Defaults to -32768.

    Returns
    -------
    xarray.Dataset
        Dataset containing the DEM data with spatial coordinates

    Examples
    --------
    >>> bbox = [-180, -90, 180, 90]
    >>> ds = get_dem(bbox, out_path="dem.tif")
    """
    # Open the Planetary Computer STAC catalog
    API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/"
    catalog = pystac_client.Client.open(API_URL)

    # Search for Copernicus DEM tiles covering the bounding box
    search = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox, max_items=10)

    items = list(search.get_items())
    print(f"Found {len(items)} items")

    # Sign items for Planetary Computer access (enables URL access)
    signed_items = [planetary_computer.sign(item) for item in items]

    # Load the DEM using ODC STAC
    # Resolution: 30m (native Copernicus DEM resolution)
    # Band: "data" contains the elevation values
    ds = load(signed_items, bbox=bbox, bands=["data"])
    dem_array = ds["data"].isel(time=0)

    # Save to file if output path is specified
    if out_path:
        dem_array = dem_array.rio.write_nodata(nodata_value)
        dem_array.rio.to_raster(out_path)
        print(f"DEM saved to {out_path}")

    return dem_array


def calculate_flow_metrics(dem_path, fdir_out, acc_out, nodata=-32768):
    """
    Calculate flow direction and accumulation from a DEM and save as GeoTIFFs.

    Parameters:
    - dem_path: path to DEM raster (tif)
    - fdir_out: output path for flow direction tiff
    - acc_out: output path for flow accumulation tiff
    - nodata: nodata value in DEM
    """
    # Open DEM with rasterio to get metadata
    with rasterio.open(dem_path) as src:
        dem_array = src.read(1)
        meta = src.meta.copy()
        crs = src.crs
        transform = src.transform

    # Initialize Grid
    grid = Grid.from_raster(dem_path)

    # Read DEM into pysheds (ensuring nodata)
    dem = grid.read_raster(dem_path, nodata=nodata)

    # Hydrological conditioning
    print("Filling pits...")
    dem_filled = grid.fill_pits(dem)

    print("Filling depressions...")
    dem_filled = grid.fill_depressions(dem_filled)

    print("Resolving flats...")
    dem_inflated = grid.resolve_flats(dem_filled)

    # Flow direction
    print("Calculating flow direction...")
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(dem_inflated, dirmap=dirmap)

    # Flow accumulation
    print("Calculating flow accumulation...")
    acc = grid.accumulation(fdir, dirmap=dirmap)

    # Save as GeoTIFFs
    meta.update(dtype=rasterio.float32, nodata=nodata, count=1)

    print(f"Saving flow direction to {fdir_out}...")
    with rasterio.open(fdir_out, "w", **meta) as dst:
        dst.write(fdir.astype(np.float32), 1)

    print(f"Saving flow accumulation to {acc_out}...")
    with rasterio.open(acc_out, "w", **meta) as dst:
        dst.write(acc.astype(np.float32), 1)

    print("Processing complete!")

    return fdir, acc
