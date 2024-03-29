# Copyright 2021-2024 Avaiga Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from algos import split, fit, score, predict, plot
from sklearn.datasets import make_moons
from taipy import Config

dataset = make_moons(noise=0.3, random_state=0)


def configure():
    X_cfg = Config.configure_data_node("X")
    y_cfg = Config.configure_data_node("y")
    X_train_cfg = Config.configure_data_node("X_train")
    X_test_cfg = Config.configure_data_node("X_test")
    y_train_cfg = Config.configure_data_node("y_train")
    y_test_cfg = Config.configure_data_node("y_test")
    split_cfg = Config.configure_task(
        "split",
        function=split,
        input=[X_cfg, y_cfg],
        output=[X_train_cfg, X_test_cfg, y_train_cfg, y_test_cfg],
        skippable=True,
    )
    alpha_cfg = Config.configure_data_node("alpha", default_data=1.0)
    model_cfg = Config.configure_data_node("model")
    fit_cfg = Config.configure_task(
        "fit",
        function=fit,
        input=[X_train_cfg, y_train_cfg, alpha_cfg],
        output=model_cfg,
        skippable=True,
    )
    accuracy_cfg = Config.configure_data_node("accuracy")
    score_cfg = Config.configure_task(
        "score",
        function=score,
        input=[model_cfg, X_test_cfg, y_test_cfg],
        output=accuracy_cfg,
        skippable=True,
    )
    y_pred_cfg = Config.configure_data_node("y_pred")
    predict_cfg = Config.configure_task(
        "predict",
        function=predict,
        input=[model_cfg, X_test_cfg],
        output=y_pred_cfg,
        skippable=True,
    )
    fig_cfg = Config.configure_data_node("fig")
    plot_cfg = Config.configure_task(
        "plot",
        function=plot,
        input=[X_test_cfg, y_test_cfg, y_pred_cfg, model_cfg],
        output=fig_cfg,
        skippable=True,
    )
    scenario_cfg = Config.configure_scenario(
        "scenario_configuration",
        task_configs=[split_cfg, fit_cfg, score_cfg, predict_cfg, plot_cfg],
    )
    return scenario_cfg
