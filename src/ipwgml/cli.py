"""
ipwgml.cli
==========

Provides a command-line interface (CLI) for managing the ipwgml configuration and
downloading data.
"""
import click
from ipwgml.config import (
    show,
    set_data_path
)

@click.group
def ipwgml():
    """Command line interface for the 'ipwgml' package."""


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
