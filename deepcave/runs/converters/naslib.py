import json
from pathlib import Path

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import Configuration

from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash

# TODO: Convert this to a ConfigurationAdapter Class
OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool3x3"]
nasbench201_params = ['op_0', 'op_1', 'op_2', 'op_3', 'op_4', 'op_5']
nasbenc201_optimal_results = {
    "cifar10_val_acc": 91.61, "cifar10_test_acc": 94.37,
    "cifar100_val_acc": 73.49, "cifar100_test_acc": 73.51,
    "imgnet_val_acc": 46.77, "imgnet_test_acc": 47.31
}

from typing import List, Union


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
    # print(values)
    config = Configuration(configuration_space=cs, values=values)
    config.is_valid_configuration()

    return config


configspace_dict = {"nasbench201": configure_nasbench201}


class NASLibRun(Run):
    prefix = "NASLib"
    _initial_order = 2

    @property
    def hash(self):
        if self.path is None:
            return ""

        # Use hash of errors.json as id
        return file_to_hash(self.path / "errors.json")

    @property
    def latest_change(self):
        if self.path is None:
            return 0

        return Path(self.path / "errors.json").stat().st_mtime

    @classmethod
    def from_path(cls, path):
        path = Path(path)

        with (path / "errors.json").open() as json_file:
            json_text = json_file.read(-1)
            config, errors_dict = json.loads(json_text)

        search_space = config['search_space']

        configspace = configspace_dict[search_space]()

        obj1 = Objective("Train regret", lower=0, upper=100)
        obj2 = Objective("Validation regret", lower=0, upper=100)
        obj3 = Objective("Test regret", lower=0, upper=100)
        obj4 = Objective("Train time", lower=0)
        objectives = [obj1, obj2, obj3, obj4]

        config.update(config.pop('search'))

        run = NASLibRun(name=path.stem, configspace=configspace, objectives=objectives, meta=config)

        # We have to set the path manually
        run._path = path

        start_time = 0.0
        end_time = 0.0
        # TODO: Use the best performance of the search space instead of 100
        for index in range(config['epochs']):
            train_regret = 100 - float(errors_dict['train_acc'][index])
            valid_regret = 100 - float(errors_dict['valid_acc'][index])
            test_regret = 100 - float(errors_dict['test_acc'][index])
            train_time = float(errors_dict['train_time'][index])
            runtime = float(errors_dict['runtime'][index])
            op_indices = errors_dict['configs'][index]

            config = op_indices2config(op_indices).get_dictionary()
            end_time = start_time + (train_time + runtime)

            # The ignored parameters
            status = Status.SUCCESS
            budget = -1
            origin = "none"
            additional_info = {}

            run.add(
                costs=[train_regret, valid_regret, test_regret, train_time],
                config=config,
                budget=budget,
                start_time=start_time,
                end_time=end_time,
                status=status,
                origin=origin,
                additional=additional_info,
            )

            start_time = end_time
        return run
