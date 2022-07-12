"""
This package defines converters for the outputs of some optimizers, that then can be interpreted by DeepCAVE.
"""
import os
import pickle
from pathlib import Path
from deepcave.runs.run import Run
from deepcave import Objective
from deepcave.utils.configs import op_indices2config
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
        # Find the name of the history file under the path
        file_name = str([x for x in os.listdir(self.path) if str(x).startswith("hist") and str(x).endswith(".pkl")][0])
        return file_to_hash(self.path / file_name)

    @property
    def latest_change(self):
        """
        Returns when the latest change was.

        :return:
        """
        if self.path is None:
            return 0
        # Find the name of the history file under the path
        file_name = str([x for x in os.listdir(self.path) if str(x).startswith("hist") and str(x).endswith(".pkl")][0])
        return Path(self.path / file_name).stat().st_mtime

    @classmethod
    def from_path(cls, path):
        """
        Read DEHB's outputs and create a DEHBRun instance using this information.

        :param path: The path to the directory storing the outputs of the optimizer in the non-deepcave format
        :return: A Run object from the path.
        """
        path = Path(path)

        # Find the name of the history file under the path
        file_name = str([x for x in os.listdir(path) if str(x).startswith("hist") and str(x).endswith(".pkl")][0])

        # Read the configspace of the search space
        from ConfigSpace.read_and_write import json as cs_json
        with (path / "configspace.json").open("r") as f:
            configspace = cs_json.read(f.read())

        # Read history, which stores all the relevant data for a single optimization run
        with open(path / file_name, "rb") as f:
            history = pickle.load(f)

        # Define objective of the optimization, this is needed for DeepCAVE
        objective = Objective("regret", lower=0, upper=100)

        # Create the run, which will store all the optimization steps a.k.a. trials
        run = DEHBRun(path.stem, configspace=configspace, objectives=objective, meta={})
        # Remember to set the path of the Run manually
        run._path = path

        start_time = 0
        # A single step taken by the optimizer results in several important information that is stored in history,
        # like the picked architecture and its evaluated performance with additional information.
        # Let's loop through the history of the optimization run to extract this information and add it one-by-one
        # to the run object we just defined
        for result in history:
            # Because the DEHB representation of configuration is a list of continues values, we will use the
            # NAS-Bench-201 representation instead, which is a discrete version of it, called operation indices
            config_dehb = result[0]
            regret = result[1]
            train_time = result[2]
            budget = int(result[3])
            info = result[4]
            # Get the operation indices and convert them to configspace objects
            config = op_indices2config(config_dehb)
            # simulate train time
            end_time = start_time + train_time

            run.add(costs=regret,
                    config=config,
                    budget=budget,
                    start_time=start_time,
                    end_time=end_time,
                    additional=info)

            start_time = end_time
        return run

