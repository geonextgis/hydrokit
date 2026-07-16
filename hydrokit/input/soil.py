import os

import numpy as np
import rasterio as rio
from rasterio.warp import Resampling, reproject
from soilgrids import SoilGrids

from hydrokit.pedotransfer import (
    Bubbling_Pressure_Maidment92,
    Ksat_Saxton2006,
    Lambda_Saxton_2006,
    Psisat_Saxton2006,
    Residual_Water_Content_Maidment92,
    Theta_33_Saxton2006,
    Theta_1500_Saxton2006,
    ThetaS_Saxton2006,
)

MAP_SERVICES = {
    "bdod": {
        "name": "Bulk density",
        "link": "https://maps.isric.org/mapserv?map=/map/bdod.map",
        "units": "cg/cm3",
    },
    "cec": {
        "name": "Citation exchange capacity at ph7",
        "link": "https://maps.isric.org/mapserv?map=/map/cec.map",
        "units": "mmol(c)/kg",
    },
    "cfvo": {
        "name": "Coarse fragments volumetric",
        "link": "https://maps.isric.org/mapserv?map=/map/cfvo.map",
        "units": "cm3/dm3 (vol‰)",
    },
    "clay": {
        "name": "Clay content",
        "link": "https://maps.isric.org/mapserv?map=/map/clay.map",
        "units": "g/kg",
    },
    "nitrogen": {
        "name": "Nitrogen",
        "link": "https://maps.isric.org/mapserv?map=/map/nitrogen.map",
        "units": "cg/kg",
    },
    "phh2o": {
        "name": "Soil pH in H2O",
        "link": "https://maps.isric.org/mapserv?map=/map/phh2o.map",
        "units": "pH*10",
    },
    "sand": {
        "name": "Sand content",
        "link": "https://maps.isric.org/mapserv?map=/map/sand.map",
        "units": "g/kg",
    },
    "silt": {
        "name": "Silt content",
        "link": "https://maps.isric.org/mapserv?map=/map/silt.map",
        "units": "g/kg",
    },
    "soc": {
        "name": "Soil organic carbon content",
        "link": "https://maps.isric.org/mapserv?map=/map/soc.map",
        "units": "dg/kg",
    },
    "ocs": {
        "name": "Soil organic carbon stock",
        "link": "https://maps.isric.org/mapserv?map=/map/ocs.map",
        "units": "t/ha",
    },
    "ocd": {
        "name": "Organic carbon densities",
        "link": "https://maps.isric.org/mapserv?map=/map/ocd.map",
        "units": "hg/dm3",
    },
    "wrb": {
        "name": "World Reference Base (WRB) classes and probabilities",
        "link": "https://maps.isric.org/mapserv?map=/map/wrb.map",
        "units": "none",
    },
    # --- NEWLY ADDED WATER RETENTION VARIABLES ---
    "wv0010": {
        "name": "Volumetric water content at 10 kPa",
        "link": "https://maps.isric.org/mapserv?map=/map/wv0010.map",
        "units": "cm3/dm3 (vol‰)",
    },
    "wv0033": {
        "name": "Volumetric water content at 33 kPa (Field Capacity)",
        "link": "https://maps.isric.org/mapserv?map=/map/wv0033.map",
        "units": "cm3/dm3 (vol‰)",
    },
    "wv1500": {
        "name": "Volumetric water content at 1500 kPa (Wilting Point)",
        "link": "https://maps.isric.org/mapserv?map=/map/wv1500.map",
        "units": "cm3/dm3 (vol‰)",
    },
}


def align_to_base(src_path, base_meta, dst_path):
    """
    Reproject and align a raster to match the grid of a reference raster.

    Parameters
    ----------
    src_path : str
        Path to the source raster file to be reprojected.
    base_meta : dict
        Metadata of the reference raster (e.g., from ``rasterio.open().meta``),
        including ``height``, ``width``, ``transform``, and ``crs``.
    dst_path : str
        Output path where the aligned raster will be saved.
    """
    with rio.open(src_path) as src:
        src_data = src.read(1)

        # Get source nodata
        src_nodata = src.nodata if src.nodata is not None else -9999

        # Define destination nodata
        dst_nodata = base_meta.get("nodata", -9999)

        # Initialize destination with nodata
        dst_data = np.full(
            (base_meta["height"], base_meta["width"]), dst_nodata, dtype=src_data.dtype
        )

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=base_meta["transform"],
            dst_crs=base_meta["crs"],
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
            resampling=Resampling.bilinear,
        )

        out_meta = base_meta.copy()
        out_meta.update({"count": 1, "nodata": dst_nodata})

        with rio.open(dst_path, "w", **out_meta) as dst:
            dst.write(dst_data, 1)


def get_soil(reference_path, out_dir="soil"):
    """
    Download and align SoilGrids data to match a reference raster grid.

    Parameters
    ----------
    reference_path : str
        Path to the reference raster used to define the target grid
        (CRS, extent, resolution).
    out_dir : str, optional
        Directory where output soil rasters will be saved.
        Created automatically if it does not exist. Default is ``"soil"``.
    """
    os.makedirs(out_dir, exist_ok=True)

    with rio.open(reference_path) as src:
        base_meta = src.meta.copy()
        width = src.width
        height = src.height
        bounds = src.bounds

    west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top

    variables = ["sand", "silt", "clay", "soc"]
    depths = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

    SoilGrids.MAP_SERVICES = MAP_SERVICES
    soil_grids = SoilGrids()

    for var in variables:
        for depth in depths:
            coverage_id = f"{var}_{depth}_mean"
            out_file = os.path.join(out_dir, f"{coverage_id}.tif")

            print(f"Downloading: {coverage_id} -> {out_file}")

            soil_grids.get_coverage_data(
                service_id=var,
                coverage_id=coverage_id,
                west=west,
                south=south,
                east=east,
                north=north,
                crs="urn:ogc:def:crs:EPSG::4326",
                width=width,
                height=height,
                output=out_file,
            )

            align_to_base(out_file, base_meta, out_file)

    print(f"Done. Soil layers saved to '{out_dir}/'")


class SoilWorkspace:
    """
    Build derived soil-parameter rasters from downloaded SoilGrids layers.

    The workspace expects the raw SoilGrids files produced by :func:`get_soil`
    inside ``soil_dir`` (one file per variable and depth, named
    ``{var}_{depth}_mean.tif``). It reads them, converts to conventional units,
    applies the pedotransfer functions in :mod:`hydrokit.pedotransfer`, and
    writes the derived layers in a per-variable subdirectory layout that
    mirrors ``docs/examples/reference/``::

        out_dir/
            sand/sand_latlon_2.5cm.tif
            clay/clay_latlon_2.5cm.tif
            thetas/thetas_latlon_2.5cm.tif
            ...

    Parameters
    ----------
    reference_path : str
        Path to the reference raster (typically the DEM) defining the target
        grid. All outputs inherit its CRS, transform, shape and nodata value.
    soil_dir : str
        Directory containing the raw SoilGrids tifs produced by
        :func:`get_soil`.
    """

    # SoilGrids standard depth intervals mapped to their mid-depth labels.
    DEPTH_MAP = {
        "0-5cm": "2.5cm",
        "5-15cm": "10.0cm",
        "15-30cm": "22.5cm",
        "30-60cm": "45.0cm",
        "60-100cm": "80.0cm",
        "100-200cm": "150.0cm",
    }

    # Raw variables that :func:`get_soil` downloads and that :meth:`build`
    # consumes to derive the hydraulic parameters.
    RAW_VARIABLES = (
        "sand",
        "silt",
        "clay",
        "soc",
    )

    # Derived variables written to ``out_dir``.
    DERIVED_VARIABLES = (
        "sand",
        "silt",
        "clay",
        "om",
        "theta33",
        "theta1500",
        "thetas",
        "thetar",
        "ksat",
        "psisat",
        "lambda",
        "bb",
        "hb",
        "qtz",
        "dsat",
        "texture_class",
    )

    def __init__(self, reference_path, soil_dir):
        self.reference_path = reference_path
        self.soil_dir = soil_dir

        with rio.open(reference_path) as src:
            self.base_meta = src.meta.copy()
        self.nodata = self.base_meta.get("nodata", -9999)

    # ------------------------------------------------------------------ IO

    def _load_raw(self, var, depth):
        """Read a SoilGrids raster; nodata pixels are returned as NaN."""
        path = os.path.join(self.soil_dir, f"{var}_{depth}_mean.tif")
        with rio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            src_nodata = src.nodata
        if src_nodata is not None:
            arr[arr == src_nodata] = np.nan
        return arr

    def _write_tif(self, arr, name, center_depth, out_dir):
        """Write ``arr`` as ``out_dir/name/name_latlon_{center_depth}.tif``."""
        var_dir = os.path.join(out_dir, name)
        os.makedirs(var_dir, exist_ok=True)
        out_path = os.path.join(var_dir, f"{name}_latlon_{center_depth}.tif")

        data = np.asarray(arr, dtype=np.float32)
        data = np.where(np.isfinite(data), data, self.nodata).astype(np.float32)

        meta = self.base_meta.copy()
        meta.update({"dtype": "float32", "count": 1, "nodata": self.nodata})

        with rio.open(out_path, "w", **meta) as dst:
            dst.write(data, 1)

    # --------------------------------------------------------- classifier

    @staticmethod
    def _usda_texture_class(sand_pct, silt_pct, clay_pct):
        """
        USDA 12-class soil texture classification from sand/silt/clay
        percentages (values 0-100). Returns a float32 array with classes
        1-12; pixels that fail every rule return NaN.
        """
        s = np.asarray(sand_pct, dtype=np.float32)
        si = np.asarray(silt_pct, dtype=np.float32)
        c = np.asarray(clay_pct, dtype=np.float32)

        t = np.zeros(s.shape, dtype=np.int32)

        def assign(code, cond):
            nonlocal t
            t = np.where((t == 0) & cond, code, t)

        # Order matters: walk the USDA triangle from coarse to fine.
        assign(1, (si + 1.5 * c) < 15)  # sand
        assign(2, ((si + 1.5 * c) >= 15) & ((si + 2 * c) < 30))  # loamy sand
        assign(
            3,
            (
                ((c >= 7) & (c < 20) & (s > 52) & ((si + 2 * c) >= 30))
                | ((c < 7) & (si < 50) & ((si + 2 * c) >= 30))
            ),
        )  # sandy loam
        assign(4, (c >= 7) & (c < 27) & (si >= 28) & (si < 50) & (s <= 52))  # loam
        assign(
            5,
            ((si >= 50) & (c >= 12) & (c < 27)) | ((si >= 50) & (si < 80) & (c < 12)),
        )  # silt loam
        assign(6, (si >= 80) & (c < 12))  # silt
        assign(7, (c >= 20) & (c < 35) & (si < 28) & (s > 45))  # sandy clay loam
        assign(8, (c >= 27) & (c < 40) & (s > 20) & (s <= 45))  # clay loam
        assign(9, (c >= 27) & (c < 40) & (s <= 20))  # silty clay loam
        assign(10, (c >= 35) & (s > 45))  # sandy clay
        assign(11, (c >= 40) & (si >= 40))  # silty clay
        assign(12, (c >= 40) & (s <= 45) & (si < 40))  # clay

        valid = np.isfinite(s) & np.isfinite(si) & np.isfinite(c) & (t > 0)
        out = t.astype(np.float32)
        out[~valid] = np.nan
        return out

    # --------------------------------------------------------------- main

    def build(self, out_dir="soil"):
        """
        Generate the derived soil layers for every depth in :attr:`DEPTH_MAP`.

        Parameters
        ----------
        out_dir : str, optional
            Directory that will receive the per-variable subfolders. Created
            automatically. Default is ``"soil"``.
        """
        os.makedirs(out_dir, exist_ok=True)

        for depth, center_depth in self.DEPTH_MAP.items():
            print(f"Processing depth {depth} -> {center_depth}")

            # Read raw SoilGrids layers (native units).
            sand_gkg = self._load_raw("sand", depth)  # g/kg
            silt_gkg = self._load_raw("silt", depth)  # g/kg
            clay_gkg = self._load_raw("clay", depth)  # g/kg
            soc_dgkg = self._load_raw("soc", depth)  # dg/kg

            # Convert to the units expected by the pedotransfer functions.
            # Saxton 2006 takes sand/clay as fractions and OM as percent.
            sand_frac = sand_gkg / 1000.0
            silt_frac = silt_gkg / 1000.0
            clay_frac = clay_gkg / 1000.0
            soc_pct = soc_dgkg / 100.0  # dg/kg -> %
            om_pct = 1.724 * soc_pct  # Van Bemmelen factor

            # --- Saxton 2006 --------------------------------------------
            theta33 = Theta_33_Saxton2006(sand_frac, clay_frac, om_pct)
            theta1500 = Theta_1500_Saxton2006(sand_frac, clay_frac, om_pct)
            thetas = ThetaS_Saxton2006(sand_frac, clay_frac, om_pct)
            ksat = Ksat_Saxton2006(sand_frac, clay_frac, om_pct)  # mm/hr
            lam = Lambda_Saxton_2006(sand_frac, clay_frac, om_pct)
            psisat_kpa = Psisat_Saxton2006(sand_frac, clay_frac, om_pct)
            psisat = psisat_kpa * 0.10197  # kPa -> m

            bb = 1.0 / lam  # Clapp-Hornberger b

            # --- Maidment 1992 (Rawls-Brakensiek) -----------------------
            # Clay and sand arguments are expected in percent; phi is a fraction.
            sand_p = sand_frac * 100.0
            clay_p = clay_frac * 100.0
            silt_p = silt_frac * 100.0

            thetar = Residual_Water_Content_Maidment92(thetas, clay_p, sand_p)
            hb_cm = Bubbling_Pressure_Maidment92(thetas, clay_p, sand_p)
            hb = hb_cm / 100.0  # cm -> m

            # --- Auxiliary parameters -----------------------------------
            # Quartz fraction is commonly approximated by the sand fraction,
            # which dominates the mineralogy of sand-sized particles.
            qtz = sand_frac.copy()
            # Satiated water content ~= saturated water content for most soils.
            dsat = thetas.copy()

            texture_class = self._usda_texture_class(sand_p, silt_p, clay_p)

            # Outputs follow the convention of docs/examples/reference: the
            # textural fractions and organic matter are stored in percent,
            # while hydraulic parameters stay in their computed units.
            outputs = {
                "sand": sand_p,
                "silt": silt_p,
                "clay": clay_p,
                "om": om_pct,
                "theta33": theta33,
                "theta1500": theta1500,
                "thetas": thetas,
                "thetar": thetar,
                "ksat": ksat,
                "psisat": psisat,
                "lambda": lam,
                "bb": bb,
                "hb": hb,
                "qtz": qtz,
                "dsat": dsat,
                "texture_class": texture_class,
            }

            for name, arr in outputs.items():
                self._write_tif(arr, name, center_depth, out_dir)

        print(f"Derived soil layers saved to '{out_dir}/'")


def prepare_soil(reference_path, raw_dir="soil_raw", out_dir="soil"):
    """
    End-to-end soil preparation: download SoilGrids layers aligned to the
    reference grid, then derive the hydraulic and textural parameters.

    Parameters
    ----------
    reference_path : str
        Path to the reference raster (typically the DEM) defining the target
        grid.
    raw_dir : str, optional
        Directory for the raw aligned SoilGrids downloads. Default
        ``"soil_raw"``.
    out_dir : str, optional
        Directory for the derived per-variable subfolders. Default ``"soil"``.
    """
    get_soil(reference_path, out_dir=raw_dir)
    SoilWorkspace(reference_path, raw_dir).build(out_dir=out_dir)
