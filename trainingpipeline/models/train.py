import pickle
import pandas as pd
from sklearn.metrics import  root_mean_squared_error
from sklearn.feature_extraction import  DictVectorizer
from sklearn.linear_model import Lasso, Ridge, LinearRegression
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import pathlib
import prefect
from prefect import flow, task
import mlflow
from dagshub import DagshubLogger, dagshub_logger
import dagshub

@task
def read_dataframe(filename):

    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

@task
def prepare_data(df):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']  #'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    
    return df, categorical, numerical

@task
def feature_engineering(df, categorical, numerical):
    dv = DictVectorizer()

    train_dicts = df[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    
    target = 'duration'
    y_train = df[target].values
    
    return X_train, y_train, dv

@task
def find_best_model(X_train, y_train, X_val, y_val):
    
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    
    def objective(params):
        #with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "xgboost")
            mlflow.log_params(params)
            
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, 'validation')],
                early_stopping_rounds=10
            )
            
            #mlflow.xgboost.log_model(booster, artifact_path="model")
            y_pred = booster.predict(valid)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:squarederror',
        'seed': 42
    }
    
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials()
    )
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["seed"] = 42
    best_params["objective"] = "reg:squarederror"
    
    mlflow.log_params(best_params)
    mlflow.set_tags(
        tags={
            "project": "NYC Taxi Time Prediction Project",
            "optimizer_engine": "hyper-opt",
            "model_family": "xgboost",
            "feature_set_version": 1,
        }
    )

    booster = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=100,
        evals=[(valid, 'validation')],
        early_stopping_rounds=10
    )
        
    y_pred = booster.predict(valid)
    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)
    
    return booster, dv, best_params

@task
def train_model(booster, dv, best_params):
    pathlib.Path("models").mkdir(exist_ok=True)
    with open("models/preprocessor.b", "wb") as f_out:
        pickle.dump(dv, f_out)
        
    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

    return best_params

@flow
def main(train_path : str = "../data/green_tripdata_2024-01.parquet", val_path : str = "../data/green_tripdata_2024-02.parquet"):
    
    
    dagshub.init(url="https://dagshub.com/luislopez3105/DagsTry", mlflow=True)
    
    with dagshub_logger() as logger:
        df_train = read_dataframe(train_path)
        df_val = read_dataframe(val_path)
        
        df_train, categorical, numerical = prepare_data(df_train)
        df_val, _, _ = prepare_data(df_val)
        
        X_train, y_train, dv = feature_engineering(df_train, categorical, numerical)
        X_val, y_val, _ = feature_engineering(df_val, categorical, numerical)
        
        booster, dv, best_params = find_best_model(X_train, y_train, X_val, y_val)
        
        # Log data
        logger.log_datasets(
            training_dataset = (X_train, y_train),
            validation_dataset = (X_val, y_val)
        )
        
        best_params = train_model(booster, dv, best_params)
        














