import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio as rio
import rioxarray
import xarray as xr
from odc.stac import load
from pysheds.grid import Grid
from rasterio.features import rasterize
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import box, mapping


def get_dem(bbox, out_path=None, nodata_value=-32768, buffer_ratio=None):
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
    buffer_ratio : float, optional
        Fraction to expand bbox (Default: None; 0.5 = 50% on each side)

    Returns
    -------
    xarray.Dataset
        Dataset containing the DEM data with spatial coordinates

    Examples
    --------
    >>> bbox = [-180, -90, 180, 90]
    >>> ds = get_dem(bbox, out_path="dem.tif")
    """

    if buffer_ratio:
        # Expand bbox
        minx, miny, maxx, maxy = bbox
        width = maxx - minx
        height = maxy - miny

        buffer_x = buffer_ratio * width
        buffer_y = buffer_ratio * height

        bbox = [minx - buffer_x, miny - buffer_y, maxx + buffer_x, maxy + buffer_y]

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

    # Extract metadata
    metadata = {
        "crs": dem_array.rio.crs,
        "transform": dem_array.rio.transform(),
        "bounds": dem_array.rio.bounds(),
        "resolution": dem_array.rio.resolution(),
        "width": dem_array.rio.width,
        "height": dem_array.rio.height,
        "nodata": nodata_value,
        "dtype": dem_array.dtype,
    }

    # Save to file if output path is specified
    if out_path:
        dem_array = dem_array.rio.write_nodata(nodata_value)
        dem_array.rio.to_raster(out_path)
        print(f"DEM saved to {out_path}")

    return dem_array, metadata


def calculate_flow_metrics(dem_path, fdir_out=None, acc_out=None, nodata=-32768):
    """
    Calculate flow direction and accumulation from a DEM.

    This function performs hydrological conditioning on a DEM (filling pits,
    depressions, and resolving flats) and then calculates flow direction and
    flow accumulation using D8 flow routing. Results can optionally be saved
    as GeoTIFFs.

    Parameters
    ----------
    dem_path : str
        Path to the DEM raster file (GeoTIFF format).
    fdir_out : str, optional
        Output path for the flow direction GeoTIFF. If None (default), flow
        direction is not saved to disk but is still returned.
    acc_out : str, optional
        Output path for the flow accumulation GeoTIFF. If None (default), flow
        accumulation is not saved to disk but is still returned.
    nodata : int or float, optional
        NoData value in the DEM and output rasters. Defaults to -32768.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing:
        - fdir : np.ndarray
            Flow direction grid with D8 encoding.
        - acc : np.ndarray
            Flow accumulation grid (number of cells flowing into each cell).

    Notes
    -----
    The flow direction uses D8 (8-directional) routing with direction map:
    (64, 128, 1, 2, 4, 8, 16, 32) corresponding to the eight neighboring cells.

    The function applies the following hydrological conditioning steps:
    1. Fill pits: Removes isolated sinks
    2. Fill depressions: Removes all sinks
    3. Resolve flats: Handles flat areas by adding small gradients

    Examples
    --------
    >>> fdir, acc = calculate_flow_metrics(
    ...     dem_path="dem.tif",
    ...     fdir_out="flow_dir.tif",
    ...     acc_out="flow_acc.tif"
    ... )
    >>> # Calculate metrics without saving to disk
    >>> fdir, acc = calculate_flow_metrics("dem.tif")
    """
    # Open DEM with rasterio to get metadata
    with rio.open(dem_path) as src:
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

    # Save as GeoTIFFs (optional)
    if fdir_out or acc_out:
        meta.update(dtype=rio.float32, nodata=nodata, count=1)

        if fdir_out:
            print(f"Saving flow direction to {fdir_out}...")
            with rio.open(fdir_out, "w", **meta) as dst:
                dst.write(fdir.astype(np.float32), 1)

        if acc_out:
            print(f"Saving flow accumulation to {acc_out}...")
            with rio.open(acc_out, "w", **meta) as dst:
                dst.write(acc.astype(np.float32), 1)

    print("Processing complete!")

    return fdir, acc


def get_basin(dem_path, basin_gdf, out_path=None, nodata=-9999):
    """
    Create a basin mask raster by rasterizing basin geometries onto DEM grid.

    This function rasterizes basin boundaries from a GeoDataFrame onto a grid
    matching the DEM's spatial resolution and extent. Each pixel in the output
    raster is assigned the ID value of the basin it falls within. Basins are
    typically delineated sub-watersheds or catchment areas.

    Parameters
    ----------
    dem_path : str
        Path to the DEM raster file (used to define output grid geometry).
    basin_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing basin geometries with a 'basin_id' column.
        All geometries should be valid polygons or multipolygons.
    out_path : str, optional
        Output path to save the basin mask as GeoTIFF. If None (default),
        the mask is not saved to disk but is still returned.
    nodata : int, optional
        NoData value to use for pixels outside all basins. Defaults to -9999.

    Returns
    -------
    np.ndarray
        Rasterized basin mask with shape matching the DEM, where each pixel
        contains the corresponding basin ID or nodata value.

    Raises
    ------
    ValueError
        If the 'basin_id' column is missing from the GeoDataFrame.
    ValueError
        If the GeoDataFrame is empty.
    ValueError
        If no valid geometries are found after validation.
    FileNotFoundError
        If the DEM file does not exist.
    rasterio.errors.RasterioIOError
        If there is an issue reading the DEM or writing the output file.

    Notes
    -----
    The output raster is created with int32 dtype to accommodate a wide range
    of basin ID values. The spatial reference and transform are inherited from
    the input DEM. Invalid geometries are automatically filtered out.

    See Also
    --------
    create_domain_mask : Similar function for creating domain masks.

    Examples
    --------
    >>> import geopandas as gpd
    >>> basin_gdf = gpd.read_file("basins.geojson")
    >>> basin_mask = get_basin(
    ...     dem_path="dem.tif",
    ...     basin_gdf=basin_gdf,
    ...     out_path="basin_mask.tif"
    ... )
    >>> # Create mask without saving
    >>> basin_mask = get_basin("dem.tif", basin_gdf)
    >>> print(basin_mask.shape)
    """
    # Validate input GeoDataFrame
    if basin_gdf.empty:
        raise ValueError("basin_gdf cannot be empty")

    if "basin_id" not in basin_gdf.columns:
        raise ValueError(
            "basin_id column must be present in basin_gdf. Found columns: "
            f"{list(basin_gdf.columns)}"
        )

    # Get DEM parameters
    try:
        with rio.open(dem_path) as src:
            dem_transform = src.transform
            dem_shape = src.shape
            dem_profile = src.profile
            dem_crs = src.crs
    except FileNotFoundError as e:
        raise FileNotFoundError(f"DEM file not found at: {dem_path}") from e
    except Exception as e:
        raise rasterio.errors.RasterioIOError(
            f"Error reading DEM file at {dem_path}: {str(e)}"
        ) from e

    # Reproject to DEM CRS if necessary
    if basin_gdf.crs != dem_crs:
        print(f"Reprojecting basin geometries from {basin_gdf.crs} to {dem_crs}...")
        basin_gdf = basin_gdf.to_crs(dem_crs)

    # Create shapes for rasterization: (geometry, ID value)
    shapes = [
        (geom, int(basin_id))
        for geom, basin_id in zip(basin_gdf.geometry, basin_gdf["basin_id"])
    ]

    print(f"Rasterizing {len(shapes)} basin(s)...")

    # Rasterize basins
    basins = rasterize(
        shapes,
        out_shape=dem_shape,
        transform=dem_transform,
        fill=nodata,
        default_value=nodata,
        dtype=np.int32,
    )

    # Save to file if output path is specified
    if out_path:
        try:
            print(f"Saving basin mask to {out_path}...")
            profile = dem_profile.copy()
            profile.update(dtype=np.int32, nodata=nodata, count=1)

            with rio.open(out_path, "w", **profile) as dst:
                dst.write(basins, 1)
            print(f"Basin mask saved to {out_path}")
        except Exception as e:
            raise rasterio.errors.RasterioIOError(
                f"Error writing basin mask to {out_path}: {str(e)}"
            ) from e

    print("Basin creation complete!")

    return basins


def create_domain_mask(dem_path, domain_gdf, out_path=None, nodata=-9999):
    """
    Create a domain mask raster by rasterizing domain geometries onto DEM grid.

    This function rasterizes domain boundaries from a GeoDataFrame onto a grid
    matching the DEM's spatial resolution and extent. Each pixel in the output
    raster is assigned the ID value of the domain it falls within.

    Parameters
    ----------
    dem_path : str
        Path to the DEM raster file (used to define output grid geometry).
    domain_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing domain geometries with an 'ID' column.
    out_path : str, optional
        Output path to save the domain mask as GeoTIFF. If None (default),
        the mask is not saved to disk but is still returned.
    nodata : int, optional
        NoData value to use for pixels outside all domains. Defaults to -9999.

    Returns
    -------
    np.ndarray
        Rasterized domain mask with shape matching the DEM, where each pixel
        contains the corresponding domain ID or nodata value.

    Raises
    ------
    ValueError
        If the 'ID' column is missing from the GeoDataFrame.
    FileNotFoundError
        If the DEM file does not exist.

    Notes
    -----
    The output raster is created with int32 dtype to accommodate a wide range
    of domain ID values. The spatial reference and transform are inherited from
    the input DEM.

    Examples
    --------
    >>> import geopandas as gpd
    >>> domain_gdf = gpd.read_file("domains.shp")
    >>> mask = create_domain_mask(
    ...     dem_path="dem.tif",
    ...     domain_gdf=domain_gdf,
    ...     out_path="domain_mask.tif"
    ... )
    >>> # Create mask without saving
    >>> mask = create_domain_mask("dem.tif", domain_gdf)
    """
    # Validate required column
    if "ID" not in domain_gdf.columns:
        raise ValueError(
            "ID column must be present in domain_gdf. Found columns: "
            f"{list(domain_gdf.columns)}"
        )

    if domain_gdf.empty:
        raise ValueError("domain_gdf cannot be empty")

    # Get DEM parameters
    try:
        with rio.open(dem_path) as src:
            dem_transform = src.transform
            dem_shape = src.shape
            dem_profile = src.profile
            dem_crs = src.crs
    except FileNotFoundError:
        raise FileNotFoundError(f"DEM file not found at: {dem_path}")

    # Ensure geometries are valid
    domain_gdf = domain_gdf[domain_gdf.geometry.is_valid].copy()
    if domain_gdf.empty:
        raise ValueError("No valid geometries found in domain_gdf")

    # Create shapes for rasterization: (geometry, ID value)
    shapes = [
        (geom, domain_id)
        for geom, domain_id in zip(domain_gdf.geometry, domain_gdf["ID"])
    ]

    print(f"Rasterizing {len(shapes)} domain(s)...")

    # Rasterize domain IDs
    domain_mask = rasterize(
        shapes,
        out_shape=dem_shape,
        transform=dem_transform,
        fill=nodata,
        default_value=nodata,
        dtype=np.int32,
    )

    # Save to file if output path is specified
    if out_path:
        print(f"Saving domain mask to {out_path}...")
        profile = dem_profile.copy()
        profile.update(dtype=np.int32, nodata=nodata, count=1)

        with rio.open(out_path, "w", **profile) as dst:
            dst.write(domain_mask, 1)

    print("Domain mask creation complete!")

    return domain_mask
