import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


df = pd.read_csv('extras\\spain_energy_market.csv')
df_2 = df.pivot_table(index=['datetime'], columns='name', values='value').reset_index()

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(df_2.drop(columns=['datetime']).dropna(axis=1))

target_name = "Precio mercado SPOT Diario ESP"

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop([target_name], axis=1)
test_x = test.drop([target_name], axis=1)
train_y = train[[target_name]]
test_y = test[[target_name]]
normalize = True

with mlflow.start_run():

    lr = LinearRegression(normalize)

    mlflow.log_param("Model_type", type(lr))
    mlflow.log_param("normalize", normalize)
    mlflow.log_param("target", target_name)

    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    #
    # # Model registry does not work with file store
    # if tracking_url_type_store != "file":
    #
    #     # Register the model
    #     # There are other ways to use the Model Registry, which depends on the use case,
    #     # please refer to the doc for more information:
    #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
    #     mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
    # else:
    #     mlflow.sklearn.log_model(lr, "model")

