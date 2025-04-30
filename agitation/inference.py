"""Module defining the motion inference functions."""

import glob
import os
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
from agitation.dataset import (
    bids_to_list,
    clinica_to_list,
    detect_anat,
    detect_t1_linear,
)
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


@click.command("dataset")
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
def dataset_cli(dataset: str, file: str, gpu: bool, cuda: int, output: str):
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
                    "Warning : This dataset does not seem to have been processed by Clinica's "
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
        usage_fail("Missing `dataset` OR `file` arguments.")

    check()

    click.secho("Begin inference pipeline...")
    dl = DataLoader(ds, batch_size=6)
    df = estimate_motion_dl(dl, cuda=(cuda if gpu else None))
    click.secho("Finished inference")
    df.to_csv(output)
    click.secho(f"Result store at : {output}", bold=True, fg="green")


@click.command("inference")
@click.option(
    "--bids_dir",
    help="Path to dataset root",
)
@click.option(
    "--subject_id",
    help="Id of subject to process (sub-<label>)",
)
@click.option(
    "--session_id",
    help="Id of session to process (ses-<label>)",
    default=None,
)
@click.option("-g", "--gpu", help="Toggle GPU inference", is_flag=True)
@click.option("--cuda", help="Specify GPU id to use, if using GPU", default=0)
@click.option("-o", "--output_dir", help="Path to output csv dir")
def inference_cli(
    bids_dir: str,
    subject_id: str,
    session_id: str,
    gpu: bool,
    cuda: int,
    output_dir: str,
):
    """Compute motion for a single subject, intended for usage with Boutiques and nipoppy"""
    click.secho(f"Processing subject : {subject_id}, session : {session_id}")
    check()
    if gpu:
        model = torch.jit.load(config.MODEL_PATH, map_location=f"cuda:{cuda}")
    else:
        model = torch.jit.load(config.MODEL_PATH)

    converter = SoftLabelToPred()
    load_volume = LoadVolume()

    glob_template = os.path.join(bids_dir, subject_id)
    output_file = subject_id
    if session_id is not None:
        glob_template = os.path.join(glob_template, session_id)
        output_file = f"{output_file}_{session_id}"
    glob_template = os.path.join(glob_template, "anat", "*_T1w.nii.gz")
    output_file = os.path.join(output_dir, f"{output_file}_motion.csv")

    t1_path = glob.glob(glob_template)

    if len(t1_path) == 0:
        usage_fail(f"No T1w volume found. Searching for : {glob_template}")
    elif len(t1_path) > 1:
        usage_fail(f"More than one T1w volume found. Searching for : {glob_template}")

    t1_path = t1_path[0]
    click.secho(f"T1w file found : {t1_path}", bold=True, fg="green")
    click.secho("Starting inference")

    with torch.inference_mode():
        data = load_volume({"data": t1_path})["data"]
        data = data.unsqueeze(0)
        if gpu:
            data = data.cuda(cuda)
        motions = model(data).cpu()
        motions = converter(motions)

        df = pd.DataFrame.from_dict(
            {
                "subject_id": [subject_id],
                "session_id": [session_id],
                "path_to_t1w": [t1_path],
                "motion": [motions[0].item()],
            }
        )
        df.to_csv(output_file)
    click.secho("Finished inference")
    click.secho(f"Result store at : {output_file}", bold=True, fg="green")
