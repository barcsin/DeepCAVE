import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant
from dash import dcc, html

from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.utils.compression import deserialize, serialize
from deepcave.utils.data_structures import update_dict
from deepcave.utils.layout import (
    get_checklist_options,
    get_select_options,
    get_slider_marks,
)
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_hovertext_from_config, get_hyperparameter_ticks
from deepcave.runs import Status

logger = get_logger(__name__)


class CCube(DynamicPlugin):
    id = "ccube"
    name = "Configuration Cube"
    icon = "fas fa-cube"
    activate_run_selection = True

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
                ],
            ),
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Number of Configurations"),
                    dcc.Slider(
                        id=register("n_configs", ["min", "max", "marks", "value"]), step=None
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Hyperparameters"),
                    dbc.Checklist(
                        id=register("hyperparameters", ["options", "value"]), inline=True
                    ),
                ]
            ),
        ]

    def load_inputs(self):
        return {
            "objective": {"options": get_select_options(), "value": None},
            "budget": {"options": get_select_options(), "value": None},
            "n_configs": {"min": 0, "max": 0, "marks": get_slider_marks(), "value": 0},
            "hyperparameters": {"options": get_checklist_options(), "value": []},
        }

    def load_dependency_inputs(self, previous_inputs, inputs, selected_run=None):
        # Prepare objetives
        objective_names = selected_run.get_objective_names()
        objective_options = get_select_options(objective_names)
        objective_value = inputs["objective"]["value"]

        # Prepare budgets
        budgets = selected_run.get_budgets(human=True)
        budget_options = get_select_options(budgets, range(len(budgets)))

        # Prepare others
        hp_names = selected_run.configspace.get_hyperparameter_names()

        # Get selected values
        n_configs_value = inputs["n_configs"]["value"]

        # Pre-set values
        if objective_value is None:
            objective_value = objective_names[0]
            budget_value = budget_options[-1]["value"]
        else:
            budget_value = inputs["budget"]["value"]

        budget = selected_run.get_budget(int(budget_value))
        configs = selected_run.get_configs(budget=budget)
        if n_configs_value == 0:
            n_configs_value = len(configs) - 1
        else:
            if n_configs_value > len(configs) - 1:
                n_configs_value = len(configs) - 1

        new_inputs = {
            "objective": {
                "options": objective_options,
                "value": objective_value,
            },
            "budget": {
                "options": budget_options,
                "value": budget_value,
            },
            "n_configs": {
                "min": 0,
                "max": len(configs) - 1,
                "marks": get_slider_marks(list(range(len(configs)))),
                "value": n_configs_value,
            },
            "hyperparameters": {
                "options": get_select_options(hp_names),
            },
        }

        # We merge the new inputs with the previous inputs
        update_dict(inputs, new_inputs)

        # Restrict to three hyperparameters
        selected = inputs["hyperparameters"]["value"]
        n_selected = len(selected)
        if n_selected > 3:
            del selected[0]

        inputs["hyperparameters"]["value"] = selected

        return inputs

    @staticmethod
    def process(run, inputs):
        budget_id = inputs["budget"]["value"]
        budget = run.get_budget(int(budget_id))
        objective = run.get_objective(inputs["objective"]["value"])

        df = run.get_encoded_data(
            objectives=objective, budget=budget, statuses=Status.SUCCESS, include_config_ids=True
        )
        return {"df": serialize(df)}

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure"), style={"height": "50vh"}),
        ]

    def load_outputs(self, inputs, outputs, run):
        df = deserialize(outputs["df"], dtype=pd.DataFrame)
        hp_names = inputs["hyperparameters"]["value"]
        n_configs = inputs["n_configs"]["value"]
        objective_name = inputs["objective"]["value"]

        # Limit to n_configs
        idx = [str(i) for i in range(n_configs, len(df))]
        df = df.drop(idx)

        costs = df[objective_name].values.tolist()
        config_ids = df["config_id"].values.tolist()
        data = []

        # Specify layout kwargs
        layout_kwargs = {}
        if n_configs > 0 and len(hp_names) > 0:
            for i, (hp_name, axis_name) in enumerate(zip(hp_names, ["xaxis", "yaxis", "zaxis"])):
                hp = run.configspace.get_hyperparameter(hp_name)
                values = df[hp_name].values.tolist()

                tickvals, ticktext = get_hyperparameter_ticks(hp, ticks=4, include_nan=True)
                layout_kwargs[axis_name] = {
                    "tickvals": tickvals,
                    "ticktext": ticktext,
                    "title": hp_name,
                }
                data.append(values)

        # Specify scatter kwargs
        scatter_kwargs = {
            "mode": "markers",
            "marker": {
                "size": 5,
                "color": costs,
                "colorbar": {"thickness": 30, "title": objective_name},
            },
            "hovertext": [get_hovertext_from_config(run, config_id) for config_id in config_ids],
            "meta": {"colorbar": costs},
            "hoverinfo": "text",
        }

        if len(data) == 3:
            trace = go.Scatter3d(x=data[0], y=data[1], z=data[2], **scatter_kwargs)
            layout = go.Layout({"scene": {**layout_kwargs}})
        else:
            if len(data) == 1:
                trace = go.Scatter(x=data[0], y=[0 for _ in range(len(data[0]))], **scatter_kwargs)
            elif len(data) == 2:
                trace = go.Scatter(x=data[0], y=data[1], **scatter_kwargs)
            else:
                trace = go.Scatter(x=[], y=[])
            layout = go.Layout(**layout_kwargs)

        fig = go.Figure(data=trace, layout=layout)
        return fig