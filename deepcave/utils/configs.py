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


def _nasbench201_parameters():
    """
    Get nasbench201 specific parameters. Returns the parameters needed to convert between ConfigSpace and nasbench201.

    :return: OP_NAMES: list, nasbench201_params: list
    """
    OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool3x3"]
    nasbench201_params = ['op_0', 'op_1', 'op_2', 'op_3', 'op_4', 'op_5']
    return OP_NAMES, nasbench201_params


def optimal_nasbench201_performance():
    """
    Returns the best possible performance that can be reached with the trained architectures on nasbench201.
    This includes validation, and test accuracy for all datasets that are defined for this benchmark. Namely, cifar10,
    cifar100, and Imagenet.

    Example:
        To get the best possible validation accuracy on cifar10 with nasbench201 use the key 'cifar10_val_acc' with the
        dictionary returned by this function

    :return: dictionary of optimal results
    """
    # The following optimal results are the mean optimal results, so it is not a good upper limit for accuracy
    # nasbenc201_optimal_results = {
    #     "cifar10_val_acc": 91.61, "cifar10_test_acc": 94.37,
    #     "cifar100_val_acc": 73.49, "cifar100_test_acc": 73.51,
    #     "imgnet_val_acc": 46.77, "imgnet_test_acc": 47.31,
    # }
    nasbenc201_optimal_results = {
        "cifar10_val_acc": 100, "cifar10_test_acc": 100,
        "cifar100_val_acc": 100, "cifar100_test_acc": 100,
        "imgnet_val_acc": 100, "imgnet_test_acc": 100,
    }
    return nasbenc201_optimal_results


def configure_nasbench201():
    """
    Creates the ConfigSpace for NAS-Bench-201

    :return: ConfigSpace object for the NAS-Bench-201 search space
    """
    OP_NAMES, nasbench201_params = _nasbench201_parameters()
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
    OP_NAMES, nasbench201_params = _nasbench201_parameters()

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
    OP_NAMES, nasbench201_params = _nasbench201_parameters()

    if isinstance(op_indices, str):
        op_indices = list(op_indices)

    cs = configure_nasbench201()

    values = {nasbench201_params[idx]: OP_NAMES[int(value)] for idx, value in enumerate(op_indices)}
    config = Configuration(configuration_space=cs, values=values)
    config.is_valid_configuration()

    return config
