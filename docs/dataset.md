# The SPR dataset

The Satellite Precipitation Retrieval (SPR) benchmark dataset consists of paired
satellite observations and corresponding estimates of surface precipitation
rates derived from ground-based precipitation radars. The dataset is derived
from collocations of overpasses of passive microwave (PMW) sensors from the
constellation of the [Global Precipitation Measurements](https://gpm.nasa.gov/)
mission. The PMW observations are augmented with time-resolved, visible and
infrared observations derived from the
[GOES-16](https://www.star.nesdis.noaa.gov/GOES/conus.php?sat=G16) satellite and
the [CPC merged-IR
dataset](https://www.cpc.ncep.noaa.gov/products/global_precip/html/wpage.merged_IR.html)
and a selection of environmental *ancillary data*.

## Data organization

The SPR dataset aims to be easy to use while supporting a wide range of retrieval scenarios. To this end, it is organized into several subsets, each providing access to the data in different formats and spatial geometries and for the various stages of the ML training workflow. The table below summarizes the levels of the hierarchical organization of the data.

```{table} SPR data organization
:name: data_organization

| Configuration name | Possible values               | Significance                                                                                                             |   |
|--------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------|---|
| Reference sensor   | GMI, ATMS                     | The sensor whose overpasses over CONUS form the basis of the dataset                                                     |   |

| Geometry           | on-swath, gridded             | Whether observations are on the native spatial sampling of the sensor or re-gridded to a regular latitude-longitude grid |   |
| Training split     | training, validation, test | Intended stage of the ML training workflow                                                                               |   |
| Format             | spatial, tabular              | Format of the training data: 2D scenes or flattened samples                                                              |   |
| Input or target   | gmi, atms, geo, geo_ir, target | Input and target data samples                                                              |   |
```

### Reference sensor

The SPR dataset comprises two independent sub-datasets: the first one generated from CONUS overpasses of the GPM Microwave Imager (GMI) sensor, and the second one generated from overpasses of the Advanced Technology Microwave Sounder (ATMS). Since GMI and ATMS are different sensors with different viewing geometries and spectral sampling, and their overpasses occur at other times of the day and with different spatial sampling, these two subsets are treated as independent. The reference sensor constitutes the top level of the organization of the SPR dataset.

### Gridded and on-swath geometries

The SPR data are provided on both *on-swath* and *gridded* coordinate systems. Here, on-swath refers to the native, 2D scan pattern of the sensor, which is organized into scans and pixels, wheras *gridded* designates the observations remappted to a regular longitude-latitude grid with a resolution of 0.036 degree.

SPR supports both of these geometries to allow greater flexibility in the design of the retrieval algorithm. Traditionally, many currently operational algorithms operate on single pixels, which makes the on-swath geometry a natural choice. However, a gridded geometry may be a more natural choice for ML-based retrievals, particularly for those combining observations from multiple sensors.


### Data splits

Following machine-learning best practices, the SPR dataset provides separate training, validation, and testing splits. The training and
validation data are extracted from the collocations from 2019-2022. The validation data uses the collocations from each month's 1st, 11th, and 21st days, while the remaining days are assigned to the training data. Collocations from the year 2023 are used for the testing of the retrievals.

### Spatial and tabular data

Finally, the SPR data is provided in both *spatial* and *tabular* format. The spatial data contains the training data organized into spatial scenes of extend 256 x 256 pixels for the gridded geometry and 64 x 64 pixels for the on-swath data. The tabular data contains only pixels with valid precipitation estimates in a single file. Providing the data in both spatial and tabular formats aims to simplify the development of pixel-based retrievals, for which it is desirable to be able to sample from the full set of available training samples.


# Dataset generation

The basis of the SPR dataset is formed by overpasses of the GMI and ATMS sensors
over CONUS. An example of such an overpass is shown in the figure below. The
figure shows an overpass of GMI over CONUS. Panel (a) shows the 89-H channels of
the GMI sensor. Panel (b) and (c) show visible observations from the GOES-16 ABI
and IR-window observations from the CPCIR merged-IR datset. Finally, panel (d)
shows the collocated precipitation measurements from the NOAA
[Multi-Radar/Multi-Sensor System](https://www.nssl.noaa.gov/projects/mrms/).

These ovepasses form the basis for the extraction of the SPR data. For the
generation of the tabular data, all pixels with valid MRMS estimates are extracted
from the scene. The data in spatial format is extracted by extracting random
sub-scenes from the overpass. The scene sizes are 256x256 for the gridded data and
64x64 for the swath-based data. The sub-scenes are allowed to overlap by about 50%, to maximize the available training scenes. The scenes extracted for gridded and on-swath geometries are shown in the figure below using the grey, dashed and grey, dot-dashed lines respectively. Panels (e) and (f) showcase the extracted scenes highlighted in black in panel (b) and (d).

```{figure} /figures/example_scene.png
---
height: 400px
name: example_scene
---
Retrieval input and target data of the SPR benchmark dataset. Panels (a), (b), and (c) show selected, collocated observations from the passive microwave (PMW) observations (Panel (a)) and geostationary visible (Panel (b)) and infrared observations that make up the input data of the SPR data. Panel (d) shows the precipitation-radar-based precipitation estimates that are the retrieval targets. Grey dashed and dash-dotted lines mark the outlines of the training samples extracted from this collocation scene for the gridded and on-swath observations. Black dashed and dash-dotted lines mark the sample training scenes displayed in Panel (e) and (f).
```

## Preprocessing

The SPR dataset is generated in two steps: The first one consists of extracting the collocation scenes from overpasses of GPM sensors over the MRMS domain. Each collocation scene is stored in the on-swath coordinate system following the spatial sampling of the GPM sensor and regridded to a uniform latitude-longitude grid.

The processing flow is illustrated in the figure below. The PMW observations
from the GPM sensor are augmented with ancillary data. MRMS measurements are
matched with the PMW observations by loading all two-minute measurements that
cover the overpass time and interpolating them to the nearest scan time of the
PMW sensor. The MRMS measurements are then downsampled spatially by smoothing
them with a Gaussian filter with a full-width at half-maximum of 0.036 degree
and interpolating them to the 0.036-degree resolution grid used by SPR. During
the downsampling the MRMS precipitation classes are transformed to
precipitation-type fractions by calculating the fractional occurrence of each
class within the Gaussian smoothing kernel.

The MRMS measurements and PMW observations are combined into the gridded collocation scene by interpolating the PMW observations to the regular 0.036-degree grid using nearest-neighbor interpolation. Similarly, the observations from the geostationary sensors are added to the scene by nearest-neighbor interpolation.

To create the on-swath collocations the MRMS measurements as well as the goestationary observations are interpolated to the PMW sensor observations using nearest-neightbor interpolation. 


```{figure} /figures/processing.svg
---
height: 400px
name: data_processing
---
Preprocessing flow for generating the collocation scenes upon which the SPR dataset is based.
```

## Training file generation

The collocation scenes form the basis for the generation for the training files that make up the SPR dataset. The data in tabular format is generated by extract all pixel that have valid MRMS estimates from all scenes separately for the on-swath and gridded geometries.

The training data in spatial format is generated by randomly extracting training scenes from the collocation files. The scenes are required to have at least 75% of valid MRMS pixels and are allowed to overlap by 50%. Scenes of size 256x256 pixels are extract from the gridded collocation files, while scenes of size 64x64 are extracted from the on-swath collocation files.

## Dataset structure

Physically, the SPR dataset is oganized following the hierarchy defined in the table [above](data_organization). Note that the directory tree contains an additional directory ``evaluation``. This folder holds the full collocation scenes, which are used for the general retrieval evaluation, as described in [](evaluation.md). 



Note that the ``ipwgml`` package downloads data as required and local copies of the SPR dataset may thus contain only a subset of the folders below.


````
 spr
 └── gmi
     ├── evaluation
     │   ├── gridded
     │   │   ├── ancillary
     │   │   ├── geo
     │   │   ├── geo_ir
     │   │   ├── gmi
     │   │   └── target
     │   └── on_swath
     │       ├── ancillary
     │       ├── geo
     │       ├── geo_ir
     │       ├── gmi
     │       └── target
     ├── testing
     │   ├── gridded
     │   │   ├── spatial
     │   │   │   ├── ancillary
     │   │   │   ├── geo
     │   │   │   ├── geo_ir
     │   │   │   ├── gmi
     │   │   │   └── target
     │   │   └── tabular
     │   │       ├── ancillary
     │   │       ├── geo
     │   │       ├── geo_ir
     │   │       ├── gmi
     │   │       └── target
     │   └── on_swath
     │       ├── spatial
     │       │   ├── ancillary
     │       │   ├── geo
     │       │   ├── geo_ir
     │       │   ├── gmi
     │       │   └── target
     │       └── tabular
     │           ├── ancillary
     │           ├── geo
     │           ├── geo_ir
     │           ├── gmi
     │           └── target
     ├── training
     │   ├── gridded
     │   │   ├── spatial
     │   │   │   ├── ancillary
     │   │   │   ├── geo
     │   │   │   ├── geo_ir
     │   │   │   ├── gmi
     │   │   │   └── target
     │   │   └── tabular
     │   │       ├── ancillary
     │   │       ├── geo
     │   │       ├── geo_ir
     │   │       ├── gmi
     │   │       └── target
     │   └── on_swath
     │       ├── spatial
     │       │   ├── ancillary
     │       │   ├── geo
     │       │   ├── geo_ir
     │       │   ├── gmi
     │       │   └── target
     │       └── tabular
     │           ├── ancillary
     │           ├── geo
     │           ├── geo_ir
     │           ├── gmi
     │           └── target
     └── validation
         ├── gridded
         │   ├── spatial
         │   │   ├── ancillary
         │   │   ├── geo
         │   │   ├── geo_ir
         │   │   ├── gmi
         │   │   └── target
         │   └── tabular
         │       ├── ancillary
         │       ├── geo
         │       ├── geo_ir
         │       ├── gmi
         │       └── target
         └── on_swath
             ├── spatial
             │   ├── ancillary
             │   ├── geo
             │   ├── geo_ir
             │   ├── pmw
             │   └── target
             └── tabular
                 ├── ancillary
                 ├── geo
                 ├── geo_ir
                 ├── gmi
                 └── target
````
