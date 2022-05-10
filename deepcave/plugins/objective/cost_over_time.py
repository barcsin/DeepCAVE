from typing import Dict, List, Union

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html

from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun, check_equality
from deepcave.utils.layout import get_radio_options, get_select_options
from deepcave.utils.styled_plotty import get_color, get_hovertext_from_config


class CostOverTime(DynamicPlugin):
    id = "cost_over_time"
    name = "Cost Over Time"
    icon = "fas fa-chart-line"

    def check_runs_compatibility(self, runs: List[AbstractRun]) -> None:
        check_equality(runs, objectives=True, budgets=True)

        # Set some attributes here
        run = runs[0]

        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        self.objective_options = get_select_options(objective_names, objective_ids)

        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        self.budget_options = get_select_options(budgets, budget_ids)

    @staticmethod
    def get_input_layout(register):
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Objective"),
                            dbc.Select(
                                id=register("objective_id", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Budget"),
                            dbc.Select(
                                id=register("budget_id", ["value", "options"], type=int),
                                placeholder="Select budget ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
            ),
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("X-Axis"),
                    dbc.RadioItems(id=register("xaxis", ["value", "options"])),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Display ..."),
                    dbc.RadioItems(id=register("display", ["value", "options"])),
                ],
                className="",
            ),
        ]

    def load_inputs(self):
        display_labels = ["Runs", "Groups"]
        display_values = ["runs", "groups"]
        display_options = get_radio_options(display_labels, display_values)

        return {
            "objective_id": {
                "options": self.objective_options,
                "value": self.objective_options[0]["value"],
            },
            "budget_id": {
                "options": self.budget_options,
                "value": self.budget_options[0]["value"],
            },
            "xaxis": {
                "options": [
                    {"label": "Time", "value": "times"},
                    {"label": "Time (logarithmic)", "value": "times_log"},
                    {"label": "Number of evaluated configurations", "value": "configs"},
                ],
                "value": "times",
            },
            "display": {
                "options": display_options,
                "value": display_values[0],
            },
        }

    @staticmethod
    def process(run, inputs) -> Dict[str, List[Union[float, str]]]:
        budget = run.get_budget(inputs["budget_id"])
        objective = run.get_objective(inputs["objective_id"])

        times, costs_mean, costs_std, ids, config_ids = run.get_trajectory(
            objective=objective, budget=budget
        )

        return {
            "times": times,
            "costs_mean": costs_mean,
            "costs_std": costs_std,
            "ids": ids,
            "config_ids": config_ids,
        }

    @staticmethod
    def get_output_layout(register):
        return dcc.Graph(register("graph", "figure"))

    @staticmethod
    def load_outputs(runs, inputs, outputs):
        traces = []
        for idx, run in enumerate(runs):
            objective = run.get_objective(inputs["objective_id"])
            config_ids = outputs[run.id]["config_ids"]
            x = outputs[run.id]["times"]
            if inputs["xaxis"] == "configs":
                x = outputs[run.id]["ids"]

            if inputs["display"] == "runs":
                if run.prefix == "group":
                    continue
            elif inputs["display"] == "groups":
                # Prefix could be not only run but also the name of the converter
                if run.prefix != "group":
                    continue
            else:
                raise RuntimeError("Unknown display option.")

            y = np.array(outputs[run.id]["costs_mean"])
            y_err = np.array(outputs[run.id]["costs_std"])
            y_upper = list(y + y_err)
            y_lower = list(y - y_err)
            y = list(y)

            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    name=run.name,
                    line_shape="hv",
                    line=dict(color=get_color(idx)),
                    hovertext=[
                        get_hovertext_from_config(run, config_id) for config_id in config_ids
                    ],
                    hoverinfo="text",
                )
            )

            traces.append(
                go.Scatter(
                    x=x,
                    y=y_upper,
                    line=dict(color=get_color(idx, 0)),
                    line_shape="hv",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            traces.append(
                go.Scatter(
                    x=x,
                    y=y_lower,
                    fill="tonexty",
                    fillcolor=get_color(idx, 0.2),
                    line=dict(color=get_color(idx, 0)),
                    line_shape="hv",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        type = None
        if inputs["xaxis"] == "times_log":
            type = "log"

        xaxis_label = "Wallclock time [s]"
        if inputs["xaxis"] == "configs":
            xaxis_label = "Number of evaluated configurations"

        layout = go.Layout(
            xaxis=dict(title=xaxis_label, type=type),
            yaxis=dict(title=objective["name"]),
        )

        return go.Figure(data=traces, layout=layout)