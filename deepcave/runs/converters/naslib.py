import json
from pathlib import Path

from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.configs import op_indices2config, configure_nasbench201
from deepcave.utils.hash import file_to_hash


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
        configspace_dict = {"nasbench201": configure_nasbench201}

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
