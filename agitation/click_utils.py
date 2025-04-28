"""Module defining simple click helpers"""

import click


def usage_fail(message: str):
    """Helper function to raise and format error consistently

    Args:
        message (str): Error message

    Raises:
        click.UsageError: Signal problem with user input
    """
    raise click.UsageError(click.style(message, fg="red", bold=True))
