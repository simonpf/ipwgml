# The Satellite Precipitation Retrieval (SPR) benchmakr dataset created by the International Precipitation Working Group

This documentation describes the ``ipwgml`` Python package, which provides functionality
to access and use the satellite precipitation retrieval (SPR) benchmark dataset
created by the machine-learning working group of the International Precipitation
Working Group (IPWG).

## The SPR benchmark dataset

The SPR benchmark dataset is an ML-ready dataset for satellite precipitation retrievals, i.e., algorithms to estimate surface precipitation rates from satellite imagery. The dataset combines satellite imagery from multiple platforms and sensors with precipitation estimates from gauge-corrected ground radar measurements. The principal purpose of the dataset is to provide a tool for testing and comparing ML-based retrieval techniques. In addition, the ``ipwgml`` package provides a generic interface to assess precipitation estimates from any algorithm against the gauge-corrected ground radar measurements used as target estimates for the SPR benchmark dataset.


```{figure} /figures/example_scene.png
---
height: 400px
name: example_scene
---
Retrieval input and target data of the SPR benchmark dataset. Panels (a), (b), and (c) show selected, collocated observations from the passive microwave (PMW) observations (Panel (a)) and geostationary visible (Panel (b)) and infrared observations that make up the input data of the SPR data. Panel (d) shows the precipitation-radar-based precipitation estimates that are the retrieval targets. Grey dashed, and dash-dotted lines mark the outlines of the training samples extracted from this collocation scene for the gridded and on-swath observations. Black dashed and dash-dotted lines mark the sample training scenes displayed in Panel (e) and (f).
```

### Features

The principal features provided by the SPR dataset and the ``ipwgml`` package are:


1. An ML-ready satellite precipitation retrieval dataset combining passive microwave observations with visible and infrared observations from geostationary satellites with gauge-corrected precipitation estimates from ground-based precipitation radars. In addition to collocated multi-sensor satellite observations, the input data of the SPR benchmark dataset comprises various environmental variables, so-called ancillary data, as well as multiple, multiple time steps from the geostationary sensors, thus providing a comprehensive base for exploring sensor synergies, temporal fusion, and the benefits of ancillary data.
   
2. A generic retrieval evaluation framework that can be used to evaluate any precipitation retrieval against the SPR test data, thus allowing direct comparison of ML-based and conventional retrievals.
