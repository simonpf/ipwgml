"""
ipwgml.config
=============

Provides functionality to manage the local configuration of the ipwgml package.

The local configuration of the ``ipwgml`` package contains only a single entry
named 'data_path' and specifies the location at which the training and testing
data is stored.

In order to conserve the ``ipwgml`` configuration between sessions, ``ipwgml`` will
create a ``config.toml`` in the user's configuration folder. By default, the 'data_path'
 will be read from this file. However, the 'data_path' read from the configuration file
will be overwritten by the ``IPWGML_DATA_PATH`` environment variable.
"""
import logging
from pathlib import Path
import os

import appdirs
import rich
import toml


LOGGER = logging.getLogger(__name__)


CONFIG_DIR = Path(appdirs.user_config_dir("ipwgml", "ipwg"))


def get_data_path() -> Path:
    """
    Get the root of the IPWGML data path.

    The ipwgml data path is determined as follows:
        1. In the absence of any configuration, it defaults to the current working directory.
        2. If a ipwgml config file exists and it contains a 'data_path' entry, this will replace
           the current working directory determined in the previous step.
        3. Finally, if the 'IPWGML_DATA_PATH' environment variable is set, it will overwrite the
           settings from the config file.

    Return:
        A Path object pointing to the root of the ipwgml data path.
    """
    # Default value is current working directory.
    data_path = Path(os.getcwd())

    # If config file exists, try to parse 'data_path' from it.
    config_file = CONFIG_DIR / "config.toml"
    if config_file.exists():
        try:
            config = toml.loads(open(config_file, "r").read())
        except Exception:
            LOGGER.exception(
                "Encountered an error when trying the read the config file located as %s",
                config_file
            )
        new_data_path = config.get("data_path", None)
        if new_data_path is None:
            LOGGER.warning(
                "ipwgml config file exists at %s but it doesn't contain a 'data_path' entry."
            )
        else:
            data_path = Path(new_data_path)

    # Finally, check if environment variable is set.
    new_data_path = os.environ.get("IPWGML_DATA_PATH", None)
    if new_data_path is not None:
        data_path = Path(new_data_path)

    return data_path


def set_data_path(path: str | Path) -> None:
    """
    Set data path and write data path to 'ipwgml' config file.

    Args:
        path: A string or Path or path object specifying the ipwgml data path.
    """
    CONFIG_DIR.mkdir(exist_ok=True, parents=True)
    config_file = CONFIG_DIR / "config.toml"
    config = {"data_path": str(path)}
    with open(config_file, "w") as output:
        output.write(toml.dumps(config))

def show() -> None:
    """
    Display configuration information.
    """
    current_data_path = str(get_data_path())

    config_file = CONFIG_DIR / "config.toml"
    if config_file.exists():
        config_file = str(config_file)
    else:
        config_file = "None"

    ipwgml_data_path = os.environ.get("IPWGML_DATA_PATH")
    if ipwgml_data_path is None:
        ipwgml_data_path = "None"

    rich.print(
        f"""
[bold red]ipwgml config [/bold red]

Current data path: [bold red]{current_data_path}[/bold red]
Config file:       {config_file}
IPWGML_DATA_PATH:  {ipwgml_data_path}
        """
    )
