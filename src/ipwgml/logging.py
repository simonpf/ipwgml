"""
ipwgml.logging
==============

Configures the logging for the ipwgml package.
"""
import logging

from rich.logging import RichHandler
from rich.console import Console

_LOG_LEVEL = "INFO"
_CONSOLE = Console()
_HANDLER = RichHandler(console=_CONSOLE)

# The parent logger for the module.
LOGGER = logging.getLogger("ipwgml")
logging.basicConfig(level=_LOG_LEVEL, force=True, handlers=[_HANDLER])

def get_console():
    """
    Return the console to use for live logging.
    """
    return _CONSOLE
