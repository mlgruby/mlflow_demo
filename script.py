
import argparse
import os

import mlflow
from mlflow.sklearn import log_model

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, r2_score

if __name__ =='__main__':

    print('extracting arguments')
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument('--n-estimators', type=int, default=10)
    parser.add_argument('--min-samples-leaf', type=int, default=3)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='boston_train.csv')
    parser.add_argument('--test-file', type=str, default='boston_test.csv')
#     parser.add_argument('--data_table', type=str, default='boston_data')
    parser.add_argument('--features', type=str)  # in this script we ask user to explicitly name features
    parser.add_argument('--target', type=str) # in this script we ask user to explicitly name the target

    args, _ = parser.parse_known_args()

    print('reading data')
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))
#     engine = create_engine('postgresql://mlflow_admin:EdBx6FUjDEgvG5@mlflow-redshift0.cdwamzbulp7n.eu-west-1.redshift.amazonaws.com:5439/dev')
#     data_frame = pd.read_sql_query('SELECT * FROM boston_data;', engine)
#     train_df, test_df = train_test_split(data_frame, test_size=0.25, random_state=42)

    print('building training and testing datasets')
    X_train = train_df[args.features.split()]
    X_test = test_df[args.features.split()]
    y_train = train_df[args.target]
    y_test = test_df[args.target]

    # setting MLFlow tracker 
    uri = "http://internal-a19641f33008a11eaa1590a387f0e3c9-331214759.eu-west-1.elb.amazonaws.com:5000"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("ml-demo")
    
    # train
    print('training model')
    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=-1)

        model.fit(X_train, y_train)

        # print abs error, rmse and r2_score
        print('validating model')
        y_test_predict = model.predict(X_test)
        abs_err = np.abs(y_test_predict - y_test)
#         rmse = (np.sqrt(mean_squared_error(y_test_predict, y_test)))
#         r2 = r2_score(y_test, y_test_predict)
#         print('rmse: {}, r2_score: {}'.format(rmse, r2))
#         mlflow.log_metric('rmse', rmse)
#         mlflow.log_metric('r2', r2)


        # print couple perf metrics
        for q in [10, 50, 90]:
            print('AE-at-' + str(q) + 'th-percentile: '
                  + str(np.percentile(a=abs_err, q=q)))
            mlflow.log_metric('AE-at-' + str(q) + 'th-percentile', np.percentile(a=abs_err, q=q))

        print(args.min_samples_leaf)
        
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        log_model(model, "model")
        mlflow.log_artifcat()