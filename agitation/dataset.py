"""Module defining interactions with datasets"""

import glob
import os
import re


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
