"""
ipwgml.cli
==========

Provides a command-line interface (CLI) for managing the ipwgml configuration and
downloading data.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List

import click
import rich
from rich.table import Table

from ipwgml.config import (
    show,
    set_data_path,
    get_data_path
)
from ipwgml import data
import ipwgml.logging


LOGGER = logging.getLogger(__name__)


@click.group
def ipwgml():
    """Command line interface for the 'ipwgml' package."""


#
# ipwgml config
#

@ipwgml.group()
def config():
    """
    Configure the ipwgml package for the current user.
    """

config.command(show)

@config.command(name="set_data_path")
@click.argument("path")
def set_data_path(path: str):
    """Set the ipwgml data path."""
    from ipwgml.config import set_data_path
    set_data_path(path)

#
# ipwgml download
#


ipwgml.add_command(data.cli, name="download")


def flatten(dict_or_list: List[Path] | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(dict_or_list, list):
        return len(dict_or_list)
    if isinstance(dict_or_list, dict):
        flattened = {}
        for name, value in dict_or_list.items():
            value_flat = flatten(value)
            if isinstance(value_flat, int):
                flattened[name] = value_flat
            else:
                for name_flat, files in value_flat.items():
                    flattened[name + "/" + name_flat] = files
        return flattened


@ipwgml.command(name="list_files")
def list_files():
    """
    List locally available ipwgml file.
    """
    current_data_path = str(get_data_path())

    rich.print(f"""
Data path: {current_data_path}
    """)

    table = Table(title="ipwgml files")
    table.add_column("Relative path", justify="left")
    table.add_column("# files", justify="right")

    files = flatten(data.list_local_files())
    for rel_path, n_files in files.items():
        table.add_row(str(rel_path), str(n_files))

    rich.print(table)
