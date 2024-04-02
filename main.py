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

from config.config import configure
from pages.root import root, selected_scenario

import taipy as tp
from taipy import Core, Gui

import pandas as pd
from sklearn.datasets import make_moons


def on_change(state, var, val):
    if var == "selected_data_node" and val:
        state["scenario"].manage_data_node_partial(state)


pages = {
    "/": root,
}


if __name__ == "__main__":
    # Instantiate, configure and run the Core
    core = Core()
    default_scenario_cfg = configure()
    core.run()

    for alpha in [0.1, 0.316, 1, 3.16, 10]:
        scenario = tp.create_scenario(default_scenario_cfg, name=f"alpha {alpha:.2f}")
        dataset = make_moons(noise=0.3, random_state=0)
        scenario.y.write(pd.Series(dataset[1], name="y"))
        scenario.alpha.write(alpha)
        tp.submit(scenario)

    # Instantiate, configure and run the GUI
    gui = Gui(pages=pages)
    data_node_partial = gui.add_partial("")
    gui.run(dark_mode=False, title="📊Varying regularization in Multi-layer Perceptron")
