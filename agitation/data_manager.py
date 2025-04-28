"""Module defining functions to download and delete model script."""

import hashlib
import os
import shutil

import click
import requests

from agitation import config


class MD5MismatchError(Exception):
    """Custom exception for MD5 mismatch."""


def calculate_md5(filename: str, chunk_size=8192) -> str:
    """Compute MD5 hash of a given file.

    Args:
        filename (str): path to file
        chunk_size (int, optional): size of chunk to read. Defaults to 8192.

    Returns:
        str: file's MD5 hash
    """
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_model():
    """Download the torchscript model from Zenodo."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    r = requests.get(config.ZENODO_URL, timeout=10)
    with open(config.MODEL_PATH, "wb") as file:
        file.write(r.content)

    if calculate_md5(config.MODEL_PATH) != config.ZENODO_MD5:
        raise MD5MismatchError("Downloaded model MD5 did not match...")


def remove_data():
    """Remove all stored data."""
    shutil.rmtree(config.DATA_DIR)


@click.group()
def manage():
    """Download model and manage data."""


def check():
    """Check model files and download if missing."""
    if os.path.exists(config.MODEL_PATH):
        click.secho(f"Model weights available at :{config.MODEL_PATH}")

        if calculate_md5(config.MODEL_PATH) != config.ZENODO_MD5:
            click.secho(
                "MD5 do not match, the model file seems to be corrupted !",
                fg="red",
                bold=True,
            )
            click.confirm("Do you want to re-download the model ? ", abort=True)
            os.remove(config.MODEL_PATH)
        else:
            click.secho("MD5 matches, everything is setup !", fg="green", bold=True)
            return

    click.secho("Downloading weights...")
    download_model()
    click.secho(f"Model weights available at :{config.MODEL_PATH}")
    click.secho("Everything is setup !", fg="green", bold=True)


@manage.command("check")
def cli_check_model():
    """Check model files and download if missing.

    (Click decorated version)
    """
    check()


@manage.command("delete")
def cli_delete():
    """Delete Agitation data directory containing the model."""
    click.confirm(
        f"Confirm you want to delete the folder : {config.DATA_DIR} ", abort=True
    )
    click.secho("Deleting data...")
    remove_data()
    click.secho("Done !", fg="green", bold=True)
