"""
This package defines basic ConfigSpace related utility functions.
"""
from typing import Optional, Union, List

import importlib
import os
import sys
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration
from deepcave.config import Config
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool3x3"]
nasbench201_params = ['op_0', 'op_1', 'op_2', 'op_3', 'op_4', 'op_5']


def parse_config(filename: Optional[str] = None) -> Config:
    """
    Parses the config given the filename. Both relative and absolute paths are possible.

    Parameters
    ----------
    filename : Optional[str], optional
        Location of the config. Must be a python file.
        By default None (default configuration will be used).

    Note
    ----
    The python file must contain a class named ``Config`` and inherit ``deepcave.config.Config``.

    Returns
    -------
    Config
        Either the default config (if no filename is given) or the config parsed from the given
        filename.

    Raises
    ------
    RuntimeError
        If config class could not be loaded.
    """
    config = Config()
    if filename is not None and filename != "default":
        try:
            p = Path(filename)

            # Absolute path
            if filename.startswith("/") or filename.startswith("~"):
                path = p.parent
                script_dir = path.stem
                module_name = p.stem
            else:
                path = Path(os.getcwd()) / p.parent

            script_dir = path.stem  # That's the path without the script name
            module_name = p.stem  # That's the script name without the extension

            # Now we add to sys path
            sys.path.append(str(path))

            module = importlib.import_module(f"{script_dir}.{module_name}")
            config = module.Config()

        except Exception:
            raise RuntimeError(f"Could not load class Config from {p}.")

    return config


def configure_nasbench201():
    """
    Creates the ConfigSpace for NAS-Bench-201

    :return: ConfigSpace object for the NAS-Bench-201 search space
    """
    cs = CS.ConfigurationSpace()
    op_0 = CSH.CategoricalHyperparameter(nasbench201_params[0], choices=OP_NAMES)
    op_1 = CSH.CategoricalHyperparameter(nasbench201_params[1], choices=OP_NAMES)
    op_2 = CSH.CategoricalHyperparameter(nasbench201_params[2], choices=OP_NAMES)
    op_3 = CSH.CategoricalHyperparameter(nasbench201_params[3], choices=OP_NAMES)
    op_4 = CSH.CategoricalHyperparameter(nasbench201_params[4], choices=OP_NAMES)
    op_5 = CSH.CategoricalHyperparameter(nasbench201_params[5], choices=OP_NAMES)

    cs.add_hyperparameters([op_0, op_1, op_2, op_3, op_4, op_5])
    return cs


def configuration2op_indices(config):
    """
    Given a NAS-Bench-201 configuration return operation indices for search space

    :param config: a sample NAS-Bench-201 configuration sampled from the ConfigSpace
    :return: operation indices
    """
    print(config)
    op_indices = np.ones(len(nasbench201_params)) * -1
    for idx, param in enumerate(nasbench201_params):
        op_indices[idx] = OP_NAMES.index(config[param])
    return op_indices.astype(int)


def op_indices2config(op_indices: Union[List[Union[int, str]], str]) -> Configuration:
    """
    Returns a configuration for nasbech201 configuration space, given operation indices

    :param op_indices: Iterable of operation indices
    :return: The configuration object corresponding to the op_indices
    """
    if isinstance(op_indices, str):
        op_indices = list(op_indices)

    cs = configure_nasbench201()

    values = {nasbench201_params[idx]: OP_NAMES[int(value)] for idx, value in enumerate(op_indices)}
    print(values)
    config = Configuration(configuration_space=cs, values=values)
    config.is_valid_configuration()

    return config

