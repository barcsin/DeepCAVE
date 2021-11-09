from abc import abstractmethod
import os
import glob
from typing import Dict, Type, Any

from src.runs.run import Run


class Converter:

    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_available_run_names(self, working_dir) -> list:
        """
        Lists the run names in working_dir.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_run_id(self, working_dir, run_name) -> str:
        """
        The id from the files in the current working_dir/run_id/*. For example, history.json could be read and hashed.
        Idea behind: If id changed, then we have to update cached trials.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_run(self, working_dir, run_name) -> Run:
        """
        Based on working_dir/run_name/*, return a new trials object.
        """

        raise NotImplementedError()