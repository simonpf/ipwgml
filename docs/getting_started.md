# Getting started

This section describes how to get started with the IPWG SPR benchmark dataset.

While all data is currently hosted
[here](https://rain.atmos.colostate.edu/gprof_nn/ipwgml/) and can be downloaded
and used completely independently, the ``ipwgml`` Python package provides
functionality to make assessing and using the data easier.

## Installing the ``ipwgml`` package

The currently recommended way to install the package from the GitHub repository:

```
pip install ipwgml[complete]@git+https://github.com/simonpf/ipwgml
```

> **Note**: The above command installs all dependencies required to run the examples included here. If
this is a concern, use ``pip install git+https://github.com/simonpf/ipwgml`` for a minimal installation.

After successful installation, the ``ipwgml`` command should be available from the command line.

## Data download using the ``ipwgml`` command

The ``ipwgml`` command line interface provides a command to download all or parts of the SPR dataset.


```
ipwgml download --data_path /path/to/store/data --sensors gmi --splits training,validation,testing --geometries gridded --format spatial
```

This will download the gridded SPR training, validation, and testing data. The ``ipwgml download`` command takes the following options:

 - ``--sensors`` A comma-separated lists of the sensors for which to download the data.
 - ``--splits`` A comma-separated lists of the data splits to download. Available options are ``training``, ``validation``, ``testing``, and ``evaluation``.
 - ``--geometries`` A comma-separated lists of the data geometries. Available options are ``gridded`` for gridded
   observations and ``on_swath`` for the data on the PMW swath.
 - ``--inputs`` A comma-separated list of the input data to download. Available options are ``ancillary``, ``geo``, ``geo_ir`` and ``pmw``.
 - ``--formats`` A comma-separated list of the data formats to download. Available options are ``spatial`` for 2D
   training scenes and ``tabular`` for tabular data.
   
```{note}
While using the ``ipwgml`` command to download the data is one way to obtain the SPR dataset, all ``ipwgml`` functionality provides an option to automatically download required data. Therefore, a manual download is not strictly required if you intend to use the data solely through the ``ipwgml`` package.
```
   
## Listing available files

 The ``iwpgml list`` command can be used to list the files on the local machine
 that ``ipwgml`` is aware of. After a successful download, it should show a
 table listing relative locations of each dataset and how many files it
 comprises.
 
## Configuring the data path

The ``ipwgml`` package expects data to be located in a path called the ``ipwgml`` data path.
``ipwgml`` does its best to keep track of the data path between subsequent
invocations to avoid downloading data multiple times.

After a fresh install, the ``data_path`` points to the current working directory.
To set an explicit ``data_path``, you can use the ``ipwgml config
set_data_path`` command. This will create a ``ipwgml`` configuration file in the
current user's configuration directory, which will allow the setting to persist
for subsequent use of the ``ipwgml`` package. A configuration file storing the
``data_path`` will also be created when the ``ipwgml download`` command is
invoked with the ``--data_path`` option.

Alternatively, the ``data_path`` can be set using the ``IPWGML_DATA_PATH`` environment
variable. The path in ``IPWGML_DATA_PATH`` will overwrite the setting in the user's configuration
file.

The ``ipwgml config show`` command can be used to find out the value of the ``ipwgml`` data path
and how it is derived:

```
ipwgml config show
```

