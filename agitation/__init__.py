"""Module containing the CLI entrypoint."""

import click

from agitation.data_manager import manage
from agitation.inference import inference_cli


@click.group()
def cli():
    """Set main entrypoint."""


cli.add_command(manage)
cli.add_command(inference_cli)
