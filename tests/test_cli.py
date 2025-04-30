"""Test commands defined in our cli tool."""

import os
import shutil

import pandas as pd
from click.testing import CliRunner

from agitation import cli, config

runner = CliRunner()


def test_manage_check_no_model():
    """Test `check` command when there is no model."""
    response = runner.invoke(cli, ["manage", "check"])

    assert "Everything is setup !" in response.output
    assert response.exit_code == 0
    assert os.path.exists(config.MODEL_PATH)


def test_manage_check_existing_valid_model():
    """Test `check` command when there is already a model."""
    runner.invoke(cli, ["manage", "check"])
    response = runner.invoke(cli, ["manage", "check"])

    assert "MD5 matches, everything is setup !" in response.output
    assert response.exit_code == 0
    assert os.path.exists(config.MODEL_PATH)


def test_manage_check_existing_broken_model():
    """Test `check` command when there is already a corrupted model."""
    runner.invoke(cli, ["manage", "check"])

    # Corrupt model data
    with open(config.MODEL_PATH, "r+b") as f:
        f.truncate(100)

    response = runner.invoke(cli, ["manage", "check"], input="y\n")

    assert "MD5 do not match, the model file seems to be corrupted !" in response.output
    assert "Everything is setup !" in response.output
    assert response.exit_code == 0
    assert os.path.exists(config.MODEL_PATH)


def test_manage_delete():
    """Test `delete` command."""
    runner.invoke(cli, ["manage", "check"])

    response = runner.invoke(cli, ["manage", "delete"], input="y\n")

    assert "Deleting data..." in response.output
    assert "Done !" in response.output
    assert response.exit_code == 0
    assert not os.path.exists(config.MODEL_PATH)


def test_dataset_cli_clinica():
    """Test `dataset` command on clinica dataset."""
    response = runner.invoke(
        cli,
        [
            "dataset",
            "-d",
            "tests/data/clinica",
            "-g",
            "-o",
            "./tests/data/tmp_motion_out.csv",
        ],
    )

    assert "Clinica's t1-linear processed data detected." in response.output
    assert "Begin inference pipeline..." in response.output
    assert "Finished inference" in response.output
    assert "Result store at : ./tests/data/tmp_motion_out.csv" in response.output
    assert os.path.exists("tests/data/tmp_motion_out.csv")
    df = pd.read_csv("tests/data/tmp_motion_out.csv", index_col=0)
    assert len(df) == 3
    assert set(df.columns) == set(["sub", "ses", "motion"])


def test_dataset_cli_bids_sub_ses():
    """Test `dataset` command on bids dataset."""
    response = runner.invoke(
        cli,
        [
            "dataset",
            "-d",
            "tests/data/bids_sub_ses",
            "-g",
            "-o",
            "./tests/data/tmp_motion_out.csv",
        ],
        input="y\n",
    )

    assert (
        "Warning : This dataset does not seem to have been processed"
        " by Clinica's t1-linear pipeline" in response.output
    )
    assert "Begin inference pipeline..." in response.output
    assert "Finished inference" in response.output
    assert "Result store at : ./tests/data/tmp_motion_out.csv" in response.output
    assert os.path.exists("tests/data/tmp_motion_out.csv")
    df = pd.read_csv("tests/data/tmp_motion_out.csv", index_col=0)
    assert len(df) == 3
    assert set(df.columns) == set(["sub", "ses", "motion"])

    os.remove("tests/data/tmp_motion_out.csv")


def test_dataset_cli_bids_sub():
    """Test `dataset` command on sub-only bids dataset."""
    response = runner.invoke(
        cli,
        [
            "dataset",
            "-d",
            "tests/data/bids_sub",
            "-g",
            "-o",
            "./tests/data/tmp_motion_out.csv",
        ],
        input="y\n",
    )

    assert (
        "Warning : This dataset does not seem to have been processed"
        " by Clinica's t1-linear pipeline" in response.output
    )
    assert "Begin inference pipeline..." in response.output
    assert "Finished inference" in response.output
    assert "Result store at : ./tests/data/tmp_motion_out.csv" in response.output
    assert os.path.exists("tests/data/tmp_motion_out.csv")
    df = pd.read_csv("tests/data/tmp_motion_out.csv", index_col=0)
    assert len(df) == 2
    assert set(df.columns) == set(["sub", "motion"])

    os.remove("tests/data/tmp_motion_out.csv")


def test_dataset_cli_file():
    """Test `dataset` command on csv file."""
    response = runner.invoke(
        cli,
        [
            "dataset",
            "-f",
            "tests/data/test_dataset_file.csv",
            "-o",
            "./tests/data/tmp_motion_out.csv",
        ],
    )

    assert "Begin inference pipeline..." in response.output
    assert "Finished inference" in response.output
    assert "Result store at : ./tests/data/tmp_motion_out.csv" in response.output
    assert os.path.exists("tests/data/tmp_motion_out.csv")
    df = pd.read_csv("tests/data/tmp_motion_out.csv", index_col=0)
    assert len(df) == 2
    assert set(df.columns) == set(["test_field2", "test_field1", "motion"])
    os.remove("tests/data/tmp_motion_out.csv")


def test_dataset_cli_clinica_cpu():
    """Test `dataset` command on clinica dataset using cpu."""
    response = runner.invoke(
        cli,
        [
            "dataset",
            "-d",
            "tests/data/clinica",
            "-o",
            "./tests/data/tmp_motion_out.csv",
        ],
    )

    assert "Clinica's t1-linear processed data detected." in response.output
    assert "Begin inference pipeline..." in response.output
    assert "Finished inference" in response.output
    assert "Result store at : ./tests/data/tmp_motion_out.csv" in response.output
    assert os.path.exists("tests/data/tmp_motion_out.csv")
    df = pd.read_csv("tests/data/tmp_motion_out.csv", index_col=0)
    assert len(df) == 3
    assert set(df.columns) == set(["sub", "ses", "motion"])
    os.remove("tests/data/tmp_motion_out.csv")


def test_dataset_cli_no_data_source():
    """Test `dataset` command without data source raises error"""
    result = runner.invoke(cli, ["dataset"])

    assert result.exit_code != 0
    assert "Missing `dataset` OR `file` arguments." in result.output


def test_dataset_cli_two_data_sources():
    """Test `dataset` command with two data sources raises error."""
    result = runner.invoke(
        cli,
        ["dataset", "-d", "random", "-f", "ramndom"],
    )

    assert result.exit_code != 0
    assert (
        "Only one data source can be provided (`dataset` OR `file`)." in result.output
    )


def test_dataset_cli_bad_dataset():
    """Test `dataset` command with wrong dataset raises error."""
    result = runner.invoke(
        cli,
        ["dataset", "-d", "tests/data/clinicano_t1_linear"],
    )

    assert result.exit_code != 0
    assert "Provided dataset cannot be interpreted as BIDS or CAPS" in result.output


def test_dataset_cli_bad_file():
    """Test `dataset` command with wrong file raises error."""
    result = runner.invoke(
        cli,
        ["dataset", "-f", "tests/data/test_dataset_file_nodata.csv"],
    )

    assert result.exit_code != 0
    assert "Provided csv does not include a `data` column." in result.output


def test_inference():
    """Test `inference` command."""
    if os.path.exists("tests/tmp_output"):
        shutil.rmtree("tests/tmp_output")
    os.makedirs("tests/tmp_output")

    result = runner.invoke(
        cli,
        [
            "inference",
            "--bids_dir",
            "tests/data/bids_sub_ses",
            "--subject_id",
            "sub-000103",
            "--session_id",
            "ses-standard",
            "--output_dir",
            "tests/tmp_output",
        ],
    )

    assert result.exit_code == 0
    assert "Processing subject : sub-000103, session : ses-standard" in result.output
    assert (
        "T1w file found : tests/data/bids_sub_ses/sub-000103/ses-standard/"
        "anat/sub-000103_ses-standard_T1w.nii.gz" in result.output
    )
    assert "Starting inference" in result.output
    assert (
        "Result store at : tests/tmp_output/sub-000103_ses-standard_motion.csv"
        in result.output
    )
    assert os.path.exists("tests/tmp_output/sub-000103_ses-standard_motion.csv")
    output = pd.read_csv(
        "tests/tmp_output/sub-000103_ses-standard_motion.csv", index_col=0
    )
    assert len(output) == 1
    assert set(output.columns) == set(
        ["subject_id", "session_id", "path_to_t1w", "motion"]
    )

    shutil.rmtree("tests/tmp_output")


def test_inference_sub_only():
    """Test `inference` command."""
    if os.path.exists("tests/tmp_output"):
        shutil.rmtree("tests/tmp_output")
    os.makedirs("tests/tmp_output")

    result = runner.invoke(
        cli,
        [
            "inference",
            "--bids_dir",
            "tests/data/bids_sub",
            "--subject_id",
            "sub-000103",
            "--output_dir",
            "tests/tmp_output",
        ],
    )

    assert result.exit_code == 0
    assert "Processing subject : sub-000103, session : None" in result.output
    assert (
        "T1w file found : tests/data/bids_sub/sub-000103/anat/sub-000103_acq-standard_T1w.nii.gz"
        in result.output
    )
    assert "Starting inference" in result.output
    assert "Result store at : tests/tmp_output/sub-000103_motion.csv" in result.output
    assert os.path.exists("tests/tmp_output/sub-000103_motion.csv")
    output = pd.read_csv("tests/tmp_output/sub-000103_motion.csv", index_col=0)
    assert len(output) == 1
    assert set(output.columns) == set(
        ["subject_id", "session_id", "path_to_t1w", "motion"]
    )

    shutil.rmtree("tests/tmp_output")


def test_inference_gpu():
    """Test `inference` command."""
    if os.path.exists("tests/tmp_output"):
        shutil.rmtree("tests/tmp_output")
    os.makedirs("tests/tmp_output")

    result = runner.invoke(
        cli,
        [
            "inference",
            "--bids_dir",
            "tests/data/bids_sub_ses",
            "--subject_id",
            "sub-000103",
            "--session_id",
            "ses-standard",
            "--output_dir",
            "tests/tmp_output",
            "-g",
        ],
    )

    assert result.exit_code == 0
    assert "Processing subject : sub-000103, session : ses-standard" in result.output
    assert (
        "T1w file found : tests/data/bids_sub_ses/sub-000103/ses-standard/"
        "anat/sub-000103_ses-standard_T1w.nii.gz" in result.output
    )
    assert "Starting inference" in result.output
    assert (
        "Result store at : tests/tmp_output/sub-000103_ses-standard_motion.csv"
        in result.output
    )
    assert os.path.exists("tests/tmp_output/sub-000103_ses-standard_motion.csv")
    output = pd.read_csv(
        "tests/tmp_output/sub-000103_ses-standard_motion.csv", index_col=0
    )
    assert len(output) == 1
    assert set(output.columns) == set(
        ["subject_id", "session_id", "path_to_t1w", "motion"]
    )

    shutil.rmtree("tests/tmp_output")


def test_inference_cli_two_volumes():
    """Test `dataset` command with wrong file raises error."""
    result = runner.invoke(
        cli,
        [
            "inference",
            "--bids_dir",
            "tests/data/bids_sub_ses_two_t1",
            "--subject_id",
            "sub-000103",
            "--session_id",
            "ses-headmotion2",
            "--output_dir",
            "tests/tmp_output",
            "-g",
        ],
    )
    assert result.exit_code != 0
    assert (
        "More than one T1w volume found. Searching for : "
        "tests/data/bids_sub_ses_two_t1/sub-000103/ses-headmotion2/anat/*_T1w.nii.gz"
        in result.output
    )


def test_inference_cli_no_volumes():
    """Test `dataset` command with wrong file raises error."""
    result = runner.invoke(
        cli,
        [
            "inference",
            "--bids_dir",
            "tests/data/bids_sub_ses_noanat",
            "--subject_id",
            "sub-000103",
            "--session_id",
            "ses-headmotion2",
            "--output_dir",
            "tests/tmp_output",
            "-g",
        ],
    )
    assert result.exit_code != 0
    assert (
        "No T1w volume found. Searching for : tests/data/"
        "bids_sub_ses_noanat/sub-000103/ses-headmotion2/anat/*_T1w.nii.gz"
        in result.output
    )
