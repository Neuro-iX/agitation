"""Module defining the motion inference functions."""

import glob
import os
import re
from typing import Callable

import click
import pandas as pd
import torch
import tqdm
from monai.data.dataset import Dataset
from torch.utils.data import DataLoader

from agitation import config
from agitation.click_utils import usage_fail
from agitation.data_manager import check
from agitation.processing import LoadVolume, SoftLabelToPred


def estimate_motion_dl(dl: DataLoader, cuda: int | None = None) -> pd.DataFrame:
    """Estimate motion using a dataloader.

    Args:
        dl (DataLoader): Dataloader containing MRI
        cuda (int | None): GPU's id. Default to None for cpu inference

    Returns:
        pd.DataFrame
    """
    if cuda is None:
        model = torch.jit.load(config.MODEL_PATH)
    else:
        model = torch.jit.load(config.MODEL_PATH, map_location=f"cuda:{cuda}")

    converter = SoftLabelToPred()

    motion_res = []
    with torch.inference_mode():
        for batch in tqdm.tqdm(dl):
            data = batch.pop("data")
            if cuda is not None:
                data = data.cuda(cuda)
            motions = model(data).cpu()
            motions = converter(motions)

            df = pd.DataFrame.from_dict(batch)
            df["motion"] = motions
            motion_res.append(df)

    return pd.concat(motion_res)


def get_sub(path: str) -> str | None:
    """Find subject identifier in a path.

    Args:
        path (str): path to analyze

    Returns:
        str | None: Identifier (sub-<label>)
    """
    match = re.match(r".*(sub-[\dA-Za-z]*).*", path)
    if match:
        return match.groups()[0]
    return None


def get_ses(path: str) -> str | None:
    """Find session identifier in a path.

    Args:
        path (str): path to analyze

    Returns:
        str | None: Identifier (ses-<label>)
    """
    match = re.match(r".*(ses-[\dA-Za-z]*).*", path)
    if match:
        return match.groups()[0]
    return None


def paths_to_list(paths: list[str]) -> list[dict[str, str | None]]:
    """Create a list of dictionnary describing each path element.

    Args:
        paths (list[str]): list of paths

    Returns:
        list[dict[str,str]]: A list of dictionnary with fields :
          `data`, `sub` and `ses` (optionnal)
        Used to define a dataset
    """
    data = []
    for volume in paths:
        point = {"data": volume, "sub": get_sub(volume)}
        ses = get_ses(volume)
        if ses is not None:
            point["ses"] = ses
        data.append(point)

    return data


def has_session(path_to_bids: str) -> bool:
    """Check for session existence in a bids dataset.

    Args:
        path_to_bids str: Path to bids root.

    Returns:
        bool: True if has session layer
    """
    return (
        next(glob.iglob(os.path.join(path_to_bids, "sub-*", "ses-*")), None) is not None
    )


def bids_to_list(path_to_bids: str) -> list[dict[str, str | None]]:
    """Create a list of dictionnary describing each element of bids dataset.

    Args:
       path_to_bids (str): path to bids datasset

    Returns:
        list[dict[str,str]]: A list of dictionnary with fields :
          `data`, `sub` and `ses` (optionnal)
        Used to define a dataset
    """
    if has_session(path_to_bids):
        t1_volumes = glob.glob(
            os.path.join(path_to_bids, "sub-*", "ses-*", "anat", "*T1w.nii*")
        )
    else:
        t1_volumes = glob.glob(os.path.join(path_to_bids, "sub-*", "anat", "*T1w.nii*"))

    return paths_to_list(t1_volumes)


def clinica_to_list(path_to_clinica: str) -> list[dict[str, str | None]]:
    """Create a list of dictionnary describing each element of clinica dataset.

    Args:
       path_to_clinica (str): path to clinica datasset

    Returns:
        list[dict[str,str]]: A list of dictionnary with fields :
          `data`, `sub` and `ses` (optionnal)
        Used to define a dataset
    """
    t1_volumes = glob.glob(
        os.path.join(
            path_to_clinica, "subjects", "sub-*", "ses-*", "t1_linear", "*T1w.nii*"
        )
    )

    return paths_to_list(t1_volumes)


def detect_t1_linear(path_to_ds: str) -> bool:
    """Detect presence of t1_linear folders.

    Args:
        path_to_ds (str): path to dataset to test

    Returns:
        bool: True if t1_linear folder found
    """
    return (
        next(
            glob.iglob(
                os.path.join(path_to_ds, "subjects", "sub-*", "ses-*", "t1_linear")
            ),
            None,
        )
        is not None
    )


def detect_anat(path_to_ds) -> bool:
    """Detect presence of anat folders.

    Args:
        path_to_ds (str): path to dataset to test

    Returns:
        bool: True if anat folder found
    """
    if has_session(path_to_ds):
        return (
            next(glob.iglob(os.path.join(path_to_ds, "sub-*", "ses-*", "anat")), None)
            is not None
        )
    return next(glob.iglob(os.path.join(path_to_ds, "sub-*", "anat")), None) is not None


@click.command("inference")
@click.option(
    "-d",
    "--dataset",
    help="Path to dataset root, can be BIDS or CAPS (Clinica)",
    default=None,
)
@click.option(
    "-f",
    "--file",
    help="File defining cases to process (CSV)",
    default=None,
)
@click.option("-g", "--gpu", help="Toggle GPU inference", is_flag=True)
@click.option("--cuda", help="Specify GPU id to use, if using GPU", default=0)
@click.option(
    "-o", "--output", help="Path to output csv file", default="./motion_scores.csv"
)
def inference_cli(dataset: str, file: str, gpu: bool, cuda: int, output: str):
    """Process a dataset using our motion quantification model."""
    if dataset is not None and file is not None:
        usage_fail("Only one data source can be provided (`dataset` OR `file`).")

    if dataset is not None:
        convert_func: Callable[[str], list[dict[str, str | None]]] | None = None
        if detect_t1_linear(dataset):
            convert_func = clinica_to_list
            click.secho("Clinica's t1-linear processed data detected.")
        elif detect_anat(dataset):
            convert_func = bids_to_list
            click.secho(
                (
                    "Warning : This datset does not seem to have been processed by Clinica's "
                    "t1-linear pipeline"
                ),
                bold=True,
                fg="yellow",
            )
            click.secho(
                "Our model has only been trained and tested on volume processed with this pipeline."
            )
            click.confirm("Do you wish to continue ?", abort=True)

        if convert_func is not None:
            ds = Dataset(convert_func(dataset), transform=LoadVolume())
        else:
            usage_fail(
                "Provided dataset cannot be interpreted as BIDS or CAPS"
                " (no anat or t1_linear folders)",
            )
    elif file is not None:
        subjects = pd.read_csv(file)
        if "data" not in subjects.columns:
            usage_fail(
                "Provided csv does not include a `data` column.",
            )
        ds = Dataset(subjects.to_dict("records"), transform=LoadVolume())
    else:
        raise usage_fail("Missing `dataset` OR `file` arguments.")

    check()

    click.secho("Begin inference pipeline...")
    dl = DataLoader(ds, batch_size=6)
    df = estimate_motion_dl(dl, cuda=(cuda if gpu else None))
    click.secho("Finished inference")
    df.to_csv(output)
    click.secho(f"Result store at : {output}", bold=True, fg="green")
