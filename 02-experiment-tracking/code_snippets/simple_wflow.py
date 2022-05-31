import mlflow
with mlflow.start_run():
    mlflow.set_tag("dev","carlos")
    mlflow.log_param("train-data-path","./data/green_tripdata_2021-01.csv")
    mlflow.log_param("valid-data-path","./data/green_tripdata_2021-02.csv")
    alpha=0.1
    mlflow.log_param("alpha",alpha)
    lr = Lasso(alpha)
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_val)
    rmse = mean_squared_error(y_val,y_pred,squared=False)
    mlflow.log_metric("rmse",rmse)
