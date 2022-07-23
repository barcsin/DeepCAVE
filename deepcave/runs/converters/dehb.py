"""
This package defines converters for the outputs of some optimizers, that then can be interpreted by DeepCAVE.
"""
import pickle
from pathlib import Path

import ConfigSpace
import ConfigSpace.util
import numpy as np

from deepcave.runs.run import Run
from deepcave import Objective
from deepcave.utils.hash import file_to_hash


class DEHBRun(Run):
    prefix = 'dehb'
    _initial_order = 1

    @property
    def hash(self):
        """
        Returns a unique hash for the run (e.g. hashing the trial history).

        :return:
        """
        if self.path is None:
            return ""

        return file_to_hash(self.path / "history_dehb.pkl")

    @property
    def latest_change(self):
        """
        Returns when the latest change was.

        :return:
        """
        if self.path is None:
            return 0
        # You can change the file name to the one that your DEHB run generates (this can be controlled with the name
        # parameter in DEHB's run method)
        return Path(self.path / "history_dehb.pkl").stat().st_mtime

    @classmethod
    def from_path(cls, path):
        """
        Read DEHB's outputs and create a DEHBRun instance using this information.

        :param path: The path to the directory storing the outputs of the optimizer in the non-deepcave format
        :return: A Run object from the path.
        """
        path = Path(path)

        # Read the configspace of the search space
        # DEHB does not have a configspace.json by default, so generate one yourself in the output folder
        # This might change in the future
        from ConfigSpace.read_and_write import json as cs_json
        with (path / "configspace.json").open("r") as f:
            configspace = cs_json.read(f.read())

        # Read history, which stores all the relevant data for a single optimization run
        # You can change the file name to the one that your DEHB run generates (this can be controlled with the name
        # parameter in DEHB's run method)
        with open(path / "history_dehb.pkl", "rb") as f:
            history = pickle.load(f)

        # Define objective of the optimization, this is needed for DeepCAVE
        obj1 = Objective("Train loss", lower=0, upper=100)
        obj2 = Objective("Validation loss", lower=0, upper=100)
        obj3 = Objective("Test loss", lower=0, upper=100)
        obj4 = Objective("Train regret", lower=0, upper=100)
        obj5 = Objective("Validation regret", lower=0, upper=100)
        obj6 = Objective("Test regret", lower=0, upper=100)
        obj7 = Objective("Train time", lower=0)
        objectives = [obj1, obj2, obj3, obj4, obj5, obj6, obj7]

        # Create the run, which will store all the optimization steps a.k.a. trials
        run = DEHBRun(path.stem, configspace=configspace, objectives=objectives, meta={})
        # Remember to set the path of the Run manually
        run._path = path

        start_time = 0
        # A single step taken by the optimizer results in several important information that is stored in history,
        # like the picked architecture and its evaluated performance with additional information.
        # Let's loop through the history of the optimization run to extract this information and add it one-by-one
        # to the run object we just defined
        for result in history:
            # DEHB represents a configuration as a list of continuous values in the range (0, 1), called vector
            dehb_vector = result[0]
            valid_regret = result[1]
            train_time = result[2]
            budget = int(result[3])
            info = result[4]
            train_loss = info['train_loss']
            valid_loss = info['valid_loss']
            test_loss = info['test_loss']
            train_regret = info['train_regret']
            test_regret = info['test_regret']

            config = cls.vector_to_configspace(dehb_vector, configspace)
            # Simulate train time
            end_time = start_time + train_time

            run.add(costs=[train_loss,
                           valid_loss,
                           test_loss,
                           train_regret,
                           valid_regret,
                           test_regret,
                           train_time],
                    config=config,
                    budget=budget,
                    start_time=start_time,
                    end_time=end_time)

            start_time = end_time
        return run

    @classmethod
    def vector_to_configspace(cls, vector: np.array, cs: ConfigSpace) -> ConfigSpace.Configuration:
        """Converts numpy array to ConfigSpace object

        Works when cs is a ConfigSpace object and the input vector is in the domain [0, 1].

        Notes:
            Code copied from the DEHB project
            https://github.com/automl/DEHB/blob/master/dehb/optimizers/de.py
        """
        # creates a ConfigSpace object dict with all hyperparameters present, the inactive too
        new_config = ConfigSpace.util.impute_inactive_values(
            cs.sample_configuration()
        ).get_dictionary()
        # iterates over all hyperparameters and normalizes each based on its type
        for i, hyper in enumerate(cs.get_hyperparameters()):
            if type(hyper) == ConfigSpace.OrdinalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1 / len(hyper.sequence))
                param_value = hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]
            elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1 / len(hyper.choices))
                param_value = hyper.choices[np.where((vector[i] < ranges) == False)[0][-1]]
            else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
                # rescaling continuous values
                if hyper.log:
                    log_range = np.log(hyper.upper) - np.log(hyper.lower)
                    param_value = np.exp(np.log(hyper.lower) + vector[i] * log_range)
                else:
                    param_value = hyper.lower + (hyper.upper - hyper.lower) * vector[i]
                if type(hyper) == ConfigSpace.UniformIntegerHyperparameter:
                    param_value = int(np.round(param_value))  # converting to discrete (int)
                else:
                    param_value = float(param_value)
            new_config[hyper.name] = param_value
        # the mapping from unit hypercube to the actual config space may lead to illegal
        # configurations based on conditions defined, which need to be deactivated/removed
        new_config = ConfigSpace.util.deactivate_inactive_hyperparameters(
            configuration=new_config, configuration_space=cs
        )
        return new_config
