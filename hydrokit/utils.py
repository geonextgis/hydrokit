import geopandas as gpd
import numpy as np
from shapely.geometry import box


def create_grids(roi, resolution=0.25):
    assert roi.crs.to_string() == "EPSG:4326", "ROI must be in EPSG:4326."

    # Get bounding box
    minx, miny, maxx, maxy = roi.total_bounds

    # Create grid coordinates
    lon = np.arange(minx, maxx + resolution, resolution)
    lat = np.arange(miny, maxy + resolution, resolution)

    # Create polygons and store grid indices
    polygons = []
    x_idx = []
    y_idx = []
    idx = 0
    for i, x in enumerate(lon):
        for j, y in enumerate(lat):
            polygons.append(box(x, y, x + resolution, y + resolution))
            x_idx.append(i)
            y_idx.append(j)
            idx += 1

    # Create GeoDataFrame
    grid = gpd.GeoDataFrame(
        {
            "ID": np.arange(1, len(polygons) + 1),
            "X": x_idx,
            "Y": y_idx,
            "geometry": polygons,
        },
        crs="EPSG:4326",
    )

    # Keep only grids that intersect ROI
    grid_roi = gpd.sjoin(grid, roi, predicate="intersects", how="inner")

    # Remove join columns from ROI if present
    grid_roi = grid_roi[["ID", "X", "Y", "geometry"]].reset_index(drop=True)

    return grid_roi
