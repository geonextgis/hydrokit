import geopandas as gpd
import numpy as np
from shapely.geometry import box


def create_grids(roi, lat_resolution=0.25, lon_resolution=0.15):
    """
    Create a rectangular grid over the ROI with separate resolutions for latitude and longitude.

    Parameters:
    - roi: GeoDataFrame of the region of interest (must have a CRS)
    - lat_resolution: grid size in latitude degrees
    - lon_resolution: grid size in longitude degrees

    Returns:
    - GeoDataFrame with grid cells intersecting the ROI
    """

    # Get bounding box
    minx, miny, maxx, maxy = roi.total_bounds

    # Snap bounds to grid (avoids floating point drift)
    minx = np.floor(minx / lon_resolution) * lon_resolution
    maxx = np.ceil(maxx / lon_resolution) * lon_resolution
    miny = np.floor(miny / lat_resolution) * lat_resolution
    maxy = np.ceil(maxy / lat_resolution) * lat_resolution

    # Generate grid edges
    lon_edges = np.arange(minx, maxx, lon_resolution)
    lat_edges = np.arange(miny, maxy, lat_resolution)

    polygons = []
    x_idx = []
    y_idx = []

    for i, x in enumerate(lon_edges):
        for j, y in enumerate(lat_edges):
            polygons.append(box(x, y, x + lon_resolution, y + lat_resolution))
            x_idx.append(i)
            y_idx.append(j)

    # Create GeoDataFrame
    grid = gpd.GeoDataFrame(
        {"ID": np.arange(len(polygons)), "X": x_idx, "Y": y_idx, "geometry": polygons},
        crs=roi.crs,
    )

    # Keep only grids that intersect ROI
    roi_union = roi.unary_union
    grid_roi = grid[grid.intersects(roi_union)].reset_index(drop=True)

    return grid_roi


def create_domain(gdf, lat_buffer=0.05, lon_buffer=0.05):
    """
    Create rectangular buffers around each geometry in a GeoDataFrame
    while preserving all attributes, with separate buffers for latitude and longitude.

    Parameters:
    - gdf: input GeoDataFrame (grid cells)
    - lat_buffer: expansion distance in latitude
    - lon_buffer: expansion distance in longitude

    Returns:
    - GeoDataFrame with expanded rectangles
    """
    new_geoms = []

    for geom in gdf.geometry:
        minx, miny, maxx, maxy = geom.bounds

        # Expand bounds separately for lat and lon
        minx -= lon_buffer
        maxx += lon_buffer
        miny -= lat_buffer
        maxy += lat_buffer

        new_geoms.append(box(minx, miny, maxx, maxy))

    out_gdf = gdf.copy()
    out_gdf["geometry"] = new_geoms

    return out_gdf
