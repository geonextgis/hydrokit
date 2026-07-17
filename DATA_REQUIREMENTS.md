# Data Requirements Checklist — Full HydroPy ↔ TorchCrop Coupling

Complete list of input data/parameters needed from real measurements (or
established databases) to drive the full coupling. Tick each item as obtained.

Legend for **Status now** in the current Hasselfelde test setup:
`SYNTHETIC` = fabricated placeholder, must be replaced · `REAL` = real data in
place · `PRESET` = from a calibrated library (measure only if calibrating) ·
`MISSING` = not yet provided.

---

## A. Meteorological forcing — daily time series
Drives HydroPy's hydrology **and** all 8 of TorchCrop's weather channels. Use the
**same source** for both models; co-register to the model grid / DEM.

- [ ] Air temperature **mean** [°C/K] — HydroPy `TSurf`, TorchCrop `davtmp` — *station / ERA5* — Status now: `SYNTHETIC`
- [ ] Air temperature **min** & **max** [°C] — TorchCrop `tmin`,`tmax` (phenology, heat stress) — Status now: `SYNTHETIC` (from Brandenburg sample)
- [ ] **Precipitation** [mm d⁻¹] — HydroPy `Precip`, TorchCrop `rain` — *gauge / radar / ERA5* — Status now: `SYNTHETIC`
- [ ] **Global shortwave radiation** [MJ m⁻² d⁻¹] — TorchCrop `irrad` (photosynthesis) + PET — *pyranometer / ERA5* — Status now: `SYNTHETIC`
- [ ] **Vapour pressure** (or RH + T) [kPa] — TorchCrop `vp`, PET — Status now: `SYNTHETIC`
- [ ] **Wind speed** at 2 m [m s⁻¹] — TorchCrop `wind`, PET — Status now: `SYNTHETIC`
- [ ] Surface pressure [kPa] — PET (Penman) — Status now: `MISSING`
- [ ] **PET** [mm d⁻¹] — HydroPy `PET` — *derive from the above (Penman-Monteith; see `hydropy/util/gen_forc_era5.bash`)* — Status now: `SYNTHETIC`
- [ ] (If computing PET from scratch) longwave radiation, specific humidity, albedo

## B. Topography
- [ ] **High-resolution DEM** [1–30 m] — coupling: subgrid cells, Step B/C flow routing, slope — *LiDAR / photogrammetry / Copernicus DEM* — Status now: `REAL` (dem_latlon.tif, Harz)
- [ ] Cell mean elevation `srftopo`, subgrid slope `slope_avg`, orographic std `topo_std` [0.5°] — HydroPy — *upscaled from DEM* — Status now: `REAL` (parameter file)
- [ ] Flow direction / routing targets `rout_lat`,`rout_lon` [0.5°] — HydroPy river routing (Step A) — *from DEM* — Status now: `REAL` (parameter file)

## C. Land cover / land use
- [ ] **Land-use map** (crop/forest/water/bare/urban) [subgrid 10–30 m] — coupling: distributed subgrid balance, crop-cell placement — *Sentinel-2 / ESA WorldCover / CORINE / ATKIS* — Status now: `SYNTHETIC` (landuse_fake.tif, elevation-derived)
- [ ] Vegetation fraction `fveg`, lake `flake`, wetland `fwetl`, glacier [0.5°] — HydroPy (evap, interception, lake budget) — Status now: `REAL` (parameter file)
- [ ] **LAI** monthly climatology [0.5°] — HydroPy — *MODIS / Copernicus* — Status now: `REAL` (parameter file)

## D. Soil hydraulic properties (per soil unit / horizon)
Soil sampling or texture-based pedotransfer functions.
- [ ] Water-holding capacity `wcap`, plant-available water `wava` [kg m⁻²] — HydroPy — *soil survey / SoilGrids* — Status now: `REAL` (parameter file)
- [ ] Field capacity `wcfc`, wilting point `wcwp`, saturation `wcst`, air-dry `wcad` [m³ m⁻³] — TorchCrop water balance — Status now: `PRESET` (TorchCrop defaults)
- [ ] Saturated conductivity / max percolation `ksub` [mm d⁻¹] — TorchCrop drainage — Status now: `PRESET`
- [ ] Critical air content `crairc` (waterlogging) [m³ m⁻³] — TorchCrop oxygen stress — Status now: `PRESET`
- [ ] Max soil rooting depth `rdmso` [m] — TorchCrop — Status now: `PRESET`
- [ ] Soil texture (sand/silt/clay), bulk density, organic matter — *derive the above* — *lab / SoilGrids* — Status now: `MISSING`
- [ ] Permafrost fraction `perm` [–] — HydroPy (cold regions) — Status now: `REAL` (parameter file)

## E. Crop parameters (LINTUL5 traits — per crop/cultivar)
Normally from calibrated presets (TorchCrop ships them); measure only when
calibrating a specific cultivar via field trials.
- [ ] Phenology: `tbasem`, `teffmx`, `tsumem`, `tsum1`, `tsum2`, vernalization, photoperiod `idsl` — Status now: `PRESET` (wheat)
- [ ] Growth/light: radiation-use efficiency (RUE), extinction coefficient, specific leaf area `slatb`(DVS), initial biomass `tdwi`
- [ ] Partitioning tables vs DVS: `frtb`,`fltb`,`fstb`,`fotb`
- [ ] Roots: `rdi`, `rdmcr`, root-growth rate `rri`
- [ ] Senescence & stress: leaf death rates, drought depletion `depnr`, heat-stress thresholds, waterlogging response
- [ ] Nutrients (only if N/P/K-limited): tissue N-P-K max/min, uptake, fixation

## F. Crop management (farm / field records)
- [ ] **Sowing date** `idpl` (e.g. Oct 1, winter wheat) — TorchCrop — Status now: `REAL` (set to Oct 1)
- [ ] Cultivar / crop type — Status now: `PRESET` (wheat)
- [ ] Irrigation schedule (if any) — TorchCrop `irrigation` — Status now: `MISSING` (none)
- [ ] Fertilization N/P/K amounts & dates — TorchCrop `fertilizer` — Status now: `MISSING` (none)

## G. Site / atmosphere
- [ ] Latitude, altitude [°, m] — both (daylength, ET) — *GPS / DEM* — Status now: `REAL`
- [ ] **CO₂ concentration** [ppm] — TorchCrop (photosynthesis, transpiration) — *Mauna Loa / regional* — Status now: `PRESET` (370 ppm)

## H. Initial conditions (or a spin-up) — CRITICAL
The cold-start dry soil starved the in-loop crop. Provide measured initial states
**or** run a multi-year spin-up before the analysis period.
- [ ] **Initial soil moisture** (root zone) [mm / m³ m⁻³] — seeds HydroPy `rootmoist` **and** TorchCrop `wa`; must be realistic or the crop starves — *soil probe / spin-up* — Status now: `SYNTHETIC` (cold-start = 0)
- [ ] Snow water equivalent [mm] — Status now: `SYNTHETIC` (cold-start = 0)
- [ ] Groundwater / surface-water storage [mm] — baseflow, sustained lateral supply — Status now: `SYNTHETIC` (cold-start = 0)

---

## Priorities

1. **Minimum to run:** A (all 8 met channels) + B (DEM) + C (land use) + D (soil) + F (sowing) + H (initial soil moisture or spin-up).
2. **Highest leverage for the lateral-flow question:** high-res **DEM** + **land-use map** (B, C) and **soil hydraulic properties** (D) — they control downslope routing and whether the crop is water- vs oxygen-limited.
3. **Swap from synthetic first (current placeholders):** real met forcing (A), the land-use map (C), and initial soil moisture / spin-up (H) — these are the `SYNTHETIC` items currently masking the coupling.

## Consistency requirements
- HydroPy and TorchCrop met forcing must come from the **same source** (mismatch corrupts the coupling).
- All spatial inputs co-registered: DEM, land use, soil, and the model grid on a common CRS/extent.
- Forcing period must span the crop calendar (e.g. October sowing → following-summer harvest ⇒ a season crossing a calendar-year boundary).
