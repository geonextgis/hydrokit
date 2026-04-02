import geopandas as gpd
import planetary_computer
import pystac_client
import rasterio
import rioxarray
import xarray as xr
from odc.stac import load
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import box, mapping


def get_dem(bbox, out_path=None):
    """
    Download and load DEM data from Copernicus (30m resolution).

    Parameters
    ----------
    bbox : list
        Bounding box as [minx, miny, maxx, maxy]
    out_path : str, optional
        Path to save the DEM as GeoTIFF. If None, data is not saved.

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

    # Save to file if output path is specified
    if out_path:
        ds["data"].isel(time=0).rio.to_raster(out_path)
        print(f"DEM saved to {out_path}")

    return ds
