library(mlflow)
library(reshape2)
library(glmnet)
library(carrier)

set.seed(40)
Sys.getenv('MLFLOW_BIN')
Sys.getenv('MLFLOW_PYTHON_BIN')
#Sys.setenv(MLFLOW_BIN = "C:/Users/d.las.heras.montero/Anaconda3/envs/r-mlflow-1.17.0/Scripts/mlflow")
old_path <- Sys.getenv("PATH")

Sys.setenv(PATH = paste("C:/Users/d.las.heras.montero/Anaconda3/envs/r-mlflow-1.17.0/Scripts/", old_path, sep = ":"))

# Read the csv file
data <- read.csv('./data/spain_energy_market.csv', encoding = "UTF-8")
data2 <- dcast(data, datetime ~ name, value.var="value", fun.aggregate=sum)

# Split the data into training and test sets. (0.75, 0.25) split.
sampled <- sample(1:nrow(data2), 0.75 * nrow(data2))
train <- data2[sampled, ]
test <- data2[-sampled, ]

target_name <- "Precio mercado SPOT Diario ESP"
other_columns <- c("datetime")
col_names_data <- colnames(data2)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x <- as.matrix(train[, !(names(train) %in% c(target_name, other_columns))])
test_x <- as.matrix(test[, !(names(train) %in% c(target_name, other_columns))])
train_y <- train[, target_name]
test_y <- test[, target_name]

alpha <- mlflow_param("alpha", 0.5, "numeric")
lambda <- mlflow_param("lambda", 0.5, "numeric")


with(mlflow_start_run(), {
    model <- glmnet(train_x, train_y, alpha = alpha, lambda = lambda, family= "gaussian", standardize = FALSE)
    predictor <- crate(~ glmnet::predict.glmnet(!!model, as.matrix(.x)), !!model)
    predicted <- predictor(test_x)

    rmse <- sqrt(mean((predicted - test_y) ^ 2))
    mae <- mean(abs(predicted - test_y))
    r2 <- as.numeric(cor(predicted, test_y) ^ 2)

    message("Elasticnet model (alpha=", alpha, ", lambda=", lambda, "):")
    message("  RMSE: ", rmse)
    message("  MAE: ", mae)
    message("  R2: ", r2)

    mlflow_log_param("alpha", alpha)
    mlflow_log_param("lambda", lambda)
    mlflow_log_metric("rmse", rmse)
    mlflow_log_metric("r2", r2)
    mlflow_log_metric("mae", mae)

    mlflow_log_model(predictor, "model")
})