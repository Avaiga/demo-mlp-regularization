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


from taipy.gui import Markdown, notify
import pandas as pd
from sklearn.datasets import make_moons

selected_scenario = None
selected_data_node = None
content = ""
figure = None


selected_alpha = 0.5
visual_alpha = 0.5
visual_accuracy = None


def on_change(state, var_name, var_value):
    if var_name == "selected_scenario":
        state.visual_alpha = state.selected_scenario.alpha.read()
        state.selected_alpha = state.visual_alpha
        state.figure = state.selected_scenario.fig.read()
        state.visual_accuracy = state.selected_scenario.accuracy.read()
    return state


def write_alpha(state):
    state.selected_alpha = float(state.selected_alpha)
    state.visual_alpha = state.selected_alpha
    state.selected_scenario.alpha.write(state.selected_alpha)
    state.selected_scenario.label = f"Alpha: {state.selected_alpha}"
    notify(state, "info", f"Alpha set to {state.selected_alpha}")
    dataset = make_moons(noise=0.3, random_state=0)
    state.selected_scenario.X.write(pd.DataFrame(dataset[0], columns=["x1", "x2"]))
    state.selected_scenario.y.write(pd.Series(dataset[1], name="y"))


def notify_on_submission(state, submitable, details):
    if details["submission_status"] == "COMPLETED":
        notify(state, "success", "Submission completed!")
    elif details["submission_status"] == "FAILED":
        notify(state, "error", "Submission failed!")
    else:
        notify(state, "info", "In progress...")


root_page = """
<|container|

# 📊 Varying regularization in Multi-layer Perceptron

<intro_card|card|

## A comparison of different values for regularization parameter **alpha** on synthetic datasets

Alpha is a parameter for regularization term, aka penalty term, that combats overfitting by constraining the size of the weights. Increasing alpha may fix high variance (a sign of overfitting) by encouraging smaller weights, resulting in a decision boundary plot that appears with lesser curvatures. Similarly, decreasing alpha may fix high bias (a sign of underfitting) by encouraging larger weights, potentially resulting in a more complicated decision boundary.

<br/>

Learn more in the <a href="https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html#sphx-glr-auto-examples-neural-networks-plot-mlp-alpha-py" target="_blank">scikit-learn docs</a> 


<br/>

|intro_card>

<br/>

<scenario_card|card|

## Run Model 🤖

### 1. Create a scenario or select an existing one:

<|layout|columns=1 2|
<|{selected_scenario}|scenario_selector|>

<|part|render={selected_scenario}|
<|layout|columns=1 2|
#### **alpha:** <|{visual_alpha}|text|raw|>

#### **accuracy:** <|{visual_accuracy}|text|raw|>
|>
<|chart|figure={figure}|>
|>
|>

### 2. Change alpha:

<|{selected_alpha}|input|on_action=write_alpha|label=Press Enter to set alpha|>

### 3. Submit:

<|{selected_scenario}|scenario|not expandable|expanded|on_submission_change=notify_on_submission|show_cycle=False|show_properties=False|show_sequences=False|show_tags=False|>

|scenario_card>

<br/>

|>
"""

root = Markdown(root_page)
