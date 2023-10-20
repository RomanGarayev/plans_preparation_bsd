import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from clickhouse_driver import Client

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from etna.datasets import TSDataset
from etna.transforms import *
from etna.pipeline import Pipeline as etna_pipe
from etna.metrics import MAE

from tqdm import tqdm
import datetime
from workalendar.europe import Russia
import re
from joblib import Parallel, delayed, cpu_count
import gc

import catboost as cb
import optuna

from prepro_utils import *
from upload_utils import *
from modeling_utils import *
from planning_utils import *
from stm_utils import *


if __name__ == '__main__':

    client = Client(host="10.44.102.129",
                    port=9000,
                    user=pd.read_csv('plan_month_fact.txt')['ch_user'].values.tolist()[0],
                    password=pd.read_csv('plan_month_fact.txt')['ch_pass'].values.tolist()[0],
                    settings={'use_numpy': False},
                    database='default')

    res_entire, raw_res_entire, forecast_df_entire = entire_plan(client)
    res_entire.to_csv('res_entire.csv', index=False)
    forecast_df_entire.to_csv('forecast_df_entire.csv', index=False)
    res_stm, raw_res_stm, forecast_df_stm = stm_plan(client, res_entire)
    res_stm.to_csv('res_stm.csv', index=False)
    forecast_df_stm.to_csv('forecast_df_stm.csv', index=False)
