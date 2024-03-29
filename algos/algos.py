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

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import plotly.express as px


def split(X, y):
    return train_test_split(X, y, test_size=0.4, random_state=42)


def fit(X_train, y_train, alpha):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            solver="lbfgs",
            alpha=alpha,
            random_state=1,
            max_iter=2000,
            early_stopping=True,
            hidden_layer_sizes=[10, 10],
        ),
    ).fit(X_train, y_train)


def score(model, X_test, y_test):
    return model.score(X_test, y_test)


def predict(model, X_test):
    return pd.DataFrame(model.predict(X_test), columns=["y_pred"])


def plot(X_test, y_test, y_pred, model):
    # X_test: (40, 2), y_test (40,), y_pred (40,), model: MLPClassifier

    # Plot the points using X_test and y_test, plot the decision boundary using y_pred
    x1, x2 = X_test.iloc[:, 0], X_test.iloc[:, 1]
    fig = px.scatter(x=x1, y=x2, color=y_test)
    fig.update_traces(marker={"size": 15})
    x1_range = np.linspace(x1.min(), x1.max(), 100)
    x2_range = np.linspace(x2.min(), x2.max(), 100)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    y_pred_mesh = model.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)
    fig.add_contour(x=x1_range, y=x2_range, z=y_pred_mesh, showscale=False, opacity=0.2)
    return fig
