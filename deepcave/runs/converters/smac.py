import json
from pathlib import Path

import numpy as np

from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class SMACRun(Run):
    prefix = "SMAC"
    _initial_order = 2

    @property
    def hash(self):
        if self.path is None:
            return ""

        # Use hash of history.json as id
        return file_to_hash(self.path / "runhistory.json")

    @property
    def latest_change(self):
        if self.path is None:
            return 0

        return Path(self.path / "runhistory.json").stat().st_mtime

    @classmethod
    def from_path(cls, path):
        """
        Based on working_dir/run_name/*, return a new trials object.
        """
        path = Path(path)

        # Read configspace
        from ConfigSpace.read_and_write import json as cs_json

        with (path / "configspace.json").open("r") as f:
            configspace = cs_json.read(f.read())

        # Read objectives
        # We have to define it ourselves, because we don't know the type of the objective
        # Only lock lower
        # objective1 = Objective("Cost", lower=0)
        # objective2 = Objective("Time", lower=0)
        obj1 = Objective("Train loss", lower=0)
        obj2 = Objective("Validation loss", lower=0)
        obj3 = Objective("Test loss", lower=0)
        obj4 = Objective("Train regret", lower=0, upper=100)
        obj5 = Objective("Validation regret", lower=0, upper=100)
        obj6 = Objective("Test regret", lower=0, upper=100)
        obj7 = Objective("Train time", lower=0)
        objectives = [obj1, obj2, obj3, obj4, obj5, obj6, obj7]

        # Read meta
        # Everything else is ignored
        ignore = ["train_inst_fn", "pcs_fn", "execdir"]

        meta = {}
        with (path / "scenario.txt").open() as f:
            for line in f.readlines():
                items = line.split(" = ")
                arg = items[0]
                value = items[1]

                # Remove \n
                value = value.replace("\n", "")

                if arg not in ignore:
                    meta[arg] = value

        # Let's create a new run object
        run = SMACRun(
            name=path.stem, configspace=configspace, objectives=objectives, meta=meta
        )

        # We have to set the path manually
        run._path = path

        # Iterate over the runhistory
        with (path / "runhistory.json").open() as json_file:
            all_data = json.load(json_file)
            data = all_data["data"]
            config_origins = all_data["config_origins"]
            configs = all_data["configs"]

        with (path / "run_history.json").open() as json_file:
            listObj = json.load(json_file)
            for instance in listObj:
                train_regret = instance['train_acc']
                valid_regret = instance['val_acc']
                test_regret = instance['test_acc']
                train_loss = instance['train_loss']
                valid_loss = instance['val_loss']
                test_loss = instance['test_loss']
                train_time = instance['train_time']
                budget = instance['budget']

        instance_ids = []

        first_starttime = None
        seeds = []
        for (config_id, instance_id, seed, budget), (
            cost,
            time,
            status,
            starttime,
            endtime,
            additional_info,
        ) in data:
            if instance_id not in instance_ids:
                instance_ids += [instance_id]

            if len(instance_ids) > 1:
                raise RuntimeError("Instances are not supported.")

            config_id = str(config_id)
            config = configs[config_id]

            if seed not in seeds:
                seeds.append(seed)

            if len(seeds) > 1:
                raise RuntimeError("Multiple seeds are not supported.")

            if first_starttime is None:
                first_starttime = starttime

            starttime = starttime - first_starttime
            endtime = endtime - first_starttime

            status = status["__enum__"]

            if "SUCCESS" in status:
                status = Status.SUCCESS
            elif "TIMEOUT" in status:
                status = Status.TIMEOUT
            elif "ABORT" in status:
                status = Status.ABORTED
            elif "MEMOUT" in status:
                status = Status.MEMORYOUT
            elif "RUNNING" in status:
                continue
            else:
                status = Status.CRASHED

            if status != Status.SUCCESS:
                # We don't want cost included which are failed
                cost = None
                time = None
            else:
                time = endtime - starttime

            # Round budget
            budget = round(budget)
            endtime = starttime + train_time

            run.add(
                costs=[train_loss,
                           valid_loss,
                           test_loss,
                           train_regret,
                           valid_regret,
                           test_regret,
                           train_time],
                config=config,
                budget=budget,
                start_time=starttime,
                end_time=endtime,
                status=status,
                origin=config_origins[config_id],
                additional=additional_info,
            )

            starttime = endtime

        return run
