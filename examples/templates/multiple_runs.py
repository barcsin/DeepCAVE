"""
Multiple Runs
^^^^^^^^^^^^^

"""

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from typing import List
from dash import dcc, html
from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.utils.layout import get_select_options
from deepcave.utils.logs import get_logger
from deepcave.runs import AbstractRun, check_equality

logger = get_logger(__name__)


class MultipleRuns(DynamicPlugin):
    id = "multiple_runs"
    name = "Multiple Runs"
    description = ""

    def check_runs_compatibility(self, runs: List[AbstractRun]) -> None:
        # Check if the selected runs in general share some common attributes
        check_equality(runs, objectives=True, budgets=True)

        # Set some attributes here
        run = runs[0]

        budgets = run.get_budgets(human=True)
        self.budget_options = get_select_options(budgets, range(len(budgets)))
        objective_names = run.get_objective_names()
        self.objective_options = get_select_options(objective_names)

    @staticmethod
    def get_input_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Objective"),
                    dbc.Select(
                        id=register("objective", ["options", "value"]),
                        placeholder="Select objective ...",
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Budget"),
                    dbc.Select(
                        id=register("budget", ["options", "value"]),
                        placeholder="Select budget ...",
                    ),
                ]
            ),
        ]

    def load_inputs(self):
        return {
            "objective": {
                "options": self.objective_options,
                "value": self.objective_options[0]["value"],
            },
            "budget": {
                "options": self.budget_options,
                "value": self.budget_options[0]["value"],
            },
        }

    @staticmethod
    def process(run, inputs):
        budget_id = inputs["budget"]["value"]
        budget = run.get_budget(int(budget_id))
        objective = run.get_objective(inputs["objective"]["value"])

        test = 1

        # Make sure the output is serializable
        return {
            "test": test,
        }

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure")),
        ]

    def load_outputs(self, inputs, outputs, runs):
        for run_name, run in runs.items():
            # Get the data from `process`
            result = outputs[run_name]["test"]

            # Do something here
            # ...

        return [go.Figure()]