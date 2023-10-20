import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from clickhouse_driver import Client
from prepro_utils import *
from upload_utils import *

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from etna.datasets import TSDataset
from etna.clustering import EuclideanDistance, DTWDistance, DTWClustering
from etna.transforms import *
from etna.pipeline import Pipeline as etna_pipe
from etna.metrics import MAE
from etna.models import SeasonalMovingAverageModel

from tqdm import tqdm

import lightgbm as lgb
import catboost as cb
import optuna
import datetime
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate, cross_val_predict
import re
from workalendar.europe import Russia
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed, cpu_count
import gc

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)

client = Client(host="10.44.102.129",
                port=9000,
                user=pd.read_csv('plan_month_fact.txt')['ch_user'].values.tolist()[0],
                password=pd.read_csv('plan_month_fact.txt')['ch_pass'].values.tolist()[0],
                settings={'use_numpy': False},
                database='default')

plan_month_fact = pd.read_csv('plan_month_fact.txt')['plan_month_fact'].values.tolist()[0]
plan_month_fact, churn_dt, plan_month_table, hmp_upload_dt_start, fill_q_begin, \
min_dt, max_dt, max_dt_wd_holidays, holdout_dt, low_val, low_val_stm, low_val_hmp_oz, \
low_val_hmp_ons, freq, horizon, exog_cols, cat_features, rare_ignore_cols, ms_stat_features, int_cat_cols, n_folds, n_jobs, border_stm_percentage, param_max_stm = get_constants(plan_month_fact)

def gb_multi_comb_parallel(train, test, gb_train_foo, model_name, comb_col, train_cols, y):
    comb_list = train[comb_col].unique()
    number_of_cpu = cpu_count() - 1
    delayed_funcs = [delayed(gb_model)(train[train[comb_col] == i], test[test[comb_col] == i],
                                       gb_train_foo, model_name, comb_col, i, train_cols, y) \
                     for i in tqdm(comb_list)]
    parallel_pool = Parallel(n_jobs=number_of_cpu)
    res = parallel_pool(delayed_funcs)
    res = pd.concat((res)).reset_index(drop=True)
    return res

def gb_model(train, test, gb_train_foo, model_name, comb_col, comb_name, train_cols, y):
    print(comb_name)
    cb_result = pd.DataFrame()
    cur_comb_df = train[train[comb_col] == comb_name]
    cur_comb_df['timestamp'] = pd.to_datetime(cur_comb_df['timestamp'])
    cur_comb_df.to_csv('cur_comb_df.csv', index=False)
    cur_comb_df_train = cur_comb_df[cur_comb_df['timestamp'] < holdout_dt]
    cur_comb_df_train.to_csv('cur_comb_df_train.csv', index=False)
    cur_comb_df_hd = cur_comb_df[cur_comb_df['timestamp'] == holdout_dt]
    cur_comb_model = gb_train_foo(cur_comb_df_train, train_cols, y)
    cb_cv_res = pd.DataFrame(columns=['segment', 'MAE', 'fold_number'])
    for i in range(1, n_folds + 1):
        cur_fold_cv_res = pd.DataFrame(columns=['segment', 'MAE', 'fold_number'])
        border_dt = pd.Timestamp(holdout_dt) - pd.offsets.MonthBegin(i)
        cur_fold_train = cur_comb_df_train[cur_comb_df_train['timestamp'] < border_dt][train_cols]
        cur_fold_test = cur_comb_df_train[cur_comb_df_train['timestamp'] < border_dt][y]
        cur_fold_hd_train = cur_comb_df_train[cur_comb_df_train['timestamp'] == border_dt][train_cols]
        cur_fold_hd_train['segment'] = cur_comb_df_train[cur_comb_df_train['timestamp'] == border_dt]['segment'].values
        cur_fold_hd_train.to_csv('cur_fold_hd_train.csv', index=False)
        cur_fold_hd_test = cur_comb_df_train[cur_comb_df_train['timestamp'] == border_dt][y]
        cur_comb_model.fit(cur_fold_train, cur_fold_test)
        preds = cur_comb_model.predict(cur_fold_hd_train[train_cols])
        preds[preds < 0] = 0
        cur_fold_mae = np.abs(cur_fold_hd_test.tolist() - preds)
        cur_fold_cv_res['MAE'] = cur_fold_mae
        cur_fold_cv_res.to_csv('cur_fold_cv_res.csv', index=False)
        cur_fold_cv_res['segment'] = cur_fold_hd_train['segment'].tolist()
        cur_fold_cv_res['fold_number'] = i
        cb_cv_res = pd.concat((cb_cv_res, cur_fold_cv_res))
    cb_cv_res = cb_cv_res.groupby('segment')['MAE'] \
        .agg(['mean', 'std']).sum(1) \
        .to_frame('MAE').reset_index() \
        .assign(model=model_name)
    cur_comb_model.fit(cur_comb_df_train[train_cols], cur_comb_df_train[y])

    try:
        imp_df = pd.DataFrame({'features': train_cols,
                           'importances': cur_comb_model.feature_importance()}) \
                .sort_values(by='importances', ascending=False)
        print(imp_df)
    except Exception as e:
        print(e)

    cur_comb_preds = cur_comb_model.predict(cur_comb_df_hd[train_cols])
    cur_comb_preds[cur_comb_preds < 0] = 0
    cur_comb_df_hd.to_csv('cur_comb_df_hd_check.csv', index=False)
    cur_hd_preds = pd.DataFrame({'target': cur_comb_preds, 'segment': cur_comb_df_hd['segment'].values})
    cb_cv_res['segment'] = cb_cv_res['segment'].astype(int)
    cb_cv_res = cb_cv_res.join(cur_hd_preds.set_index('segment'), on='segment')
    cur_comb_model.fit(cur_comb_df[train_cols], cur_comb_df[y])
    final_preds = cur_comb_model.predict(test[test[comb_col] == comb_name][train_cols])
    cb_cv_res['final_preds'] = final_preds
    cb_result = pd.concat((cb_result, cb_cv_res))
    cb_result = cb_result.rename(columns={'target': 'hd_preds', 'preds': 'final_preds'})
    return cb_result


def lightgbm_tuner_ring_finish(df, best_max_depth, best_subsample, best_colsample_bytree, train_cols, y, trial):
    params_to_tune = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.30),
        'n_estimators': trial.suggest_int('n_estimators', 350, 1200),
        'verbose': -1,
        'max_depth': best_max_depth,
        'subsample': best_subsample,
        'colsample_bytree': best_colsample_bytree
    }
    model = lgb.LGBMRegressor(**params_to_tune)
    tscv = TimeSeriesSplit(n_splits=3, test_size=df['segment'].nunique())
    cv_score = cross_val_score(model, df[train_cols], df[y],
                               cv=tscv, scoring='neg_mean_absolute_error', n_jobs=n_jobs).mean()
    return np.abs(cv_score)


def model_params_tuner_init(df, n_trials, target_func, train_cols, y):
    print(target_func.__name__, 'tuning_begin')
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, warn_independent_sampling=False),
        direction='minimize')
    study.optimize(lambda trial: target_func(df, train_cols, y, trial), n_trials=n_trials)
    return study.trials_dataframe()


def model_params_tuner_middle(df, n_trials, lr, n_trees, target_func, train_cols, y):
    print(target_func.__name__, 'tuning_begin')
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, warn_independent_sampling=False),
        direction='minimize')
    study.optimize(lambda trial: target_func(df, lr, n_trees, train_cols, y, trial), n_trials=n_trials)
    return study.trials_dataframe()


def get_best_lgb_model(train, train_cols, y):
    n_trials = 5
    train[cat_features] = train[cat_features].astype(int, errors='ignore').astype(str)
    train[int_cat_cols] = train[int_cat_cols].fillna(0).astype(int)

    trial_df_start = model_params_tuner_init(train, n_trials, lightgbm_tuner_ring_start, train_cols, y)
    best_iteration = trial_df_start['value'].idxmin()
    best_lr = trial_df_start.loc[best_iteration]['params_learning_rate']
    best_n_trees = trial_df_start.loc[best_iteration]['params_n_estimators']

    trial_df_middle = model_params_tuner_middle(train, n_trials * 5, best_lr, best_n_trees, lightgbm_tuner_ring_middle,
                                                train_cols, y)
    best_iteration = trial_df_middle['value'].idxmin()
    best_max_depth = trial_df_middle.loc[best_iteration]['params_max_depth']
    best_subsample = trial_df_middle.loc[best_iteration]['params_subsample']
    best_colsample_bytree = trial_df_middle.loc[best_iteration]['params_colsample_bytree']

    trial_df_finish = lightgbm_params_tuner_finish(train, n_trials,
                                                   best_max_depth, best_subsample, best_colsample_bytree,
                                                   lightgbm_tuner_ring_finish, train_cols, y)
    best_iteration = trial_df_finish['value'].idxmin()
    best_lr = trial_df_finish.loc[best_iteration]['params_learning_rate']
    best_n_trees = trial_df_finish.loc[best_iteration]['params_n_estimators']
    final_params_lgb = {
        'verbose': -1,
        'learning_rate': best_lr,
        'n_estimators': int(1.06 * best_n_trees),
        'max_depth': best_max_depth,
        'subsample': best_subsample,
        'colsample_bytree': best_colsample_bytree,
    }
    model = lgb.LGBMRegressor(**final_params_lgb)
    return model


def lightgbm_tuner_ring_start(df, train_cols, y, trial):
    params_to_tune = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.30),
        'n_estimators': trial.suggest_int('n_estimators', 350, 1200),
        'verbose': -1,
    }
    model = lgb.LGBMRegressor(**params_to_tune)
    tscv = TimeSeriesSplit(n_splits=3, test_size=df['segment'].nunique())
    cv_score = cross_val_score(model, df[train_cols], df[y],
                               cv=tscv, scoring='neg_mean_absolute_error', n_jobs=n_jobs).mean()
    return np.abs(cv_score)


def lightgbm_tuner_ring_middle(df, lr, n_trees, train_cols, y, trial):
    params_to_tune = {
        'learning_rate': lr,
        'n_estimators': n_trees,
        'verbose': -1,
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1),
    }
    model = lgb.LGBMRegressor(**params_to_tune)
    tscv = TimeSeriesSplit(n_splits=3, test_size=df['segment'].nunique())
    cv_score = cross_val_score(model, df[train_cols], df[y],
                               cv=tscv, scoring='neg_mean_absolute_error', n_jobs=n_jobs).mean()
    return np.abs(cv_score)


def lightgbm_params_tuner_finish(df, n_trials, best_max_depth, best_subsample, best_feature_fraction, target_func,
                                 train_cols, y):
    print(target_func.__name__, 'tuning_begin')
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, warn_independent_sampling=False),
        direction='minimize')
    study.optimize(
        lambda trial: target_func(df, best_max_depth, best_subsample, best_feature_fraction, train_cols, y, trial),
        n_trials=n_trials)
    return study.trials_dataframe()


def sma_model(df):
    ts = TSDataset.to_dataset(df[['timestamp', 'segment', 'target']])
    ts = TSDataset(df=ts, freq=freq)
    df = ts.to_pandas(flatten=True).fillna(0)
    cur_train = df[df['timestamp'] < holdout_dt]
    cur_hd = df[df['timestamp'] == holdout_dt]
    ts = TSDataset.to_dataset(cur_train[['timestamp', 'segment', 'target']])
    ts = TSDataset(df=ts, freq=freq)

    sma1 = etna_pipe(SeasonalMovingAverageModel(window=1, seasonality=1), [], horizon)
    sma3 = etna_pipe(SeasonalMovingAverageModel(window=3, seasonality=1), [], horizon)
    sma6 = etna_pipe(SeasonalMovingAverageModel(window=6, seasonality=1), [], horizon)
    sma12 = etna_pipe(SeasonalMovingAverageModel(window=12, seasonality=1), [], horizon)

    pipelines_names = ['sma3', 'sma6', 'sma12', 'sma1']
    pipelines = [sma3, sma6, sma12, sma1]
    sma_df = get_cv_metrics(ts, pipelines, pipelines_names, 3).drop('timestamp', axis=1).reset_index(drop=True)
    sma_df = sma_df.drop(sma_df[(sma_df['model'] == 'sma1') & (sma_df['target'] == 0)].index)

    ts = TSDataset.to_dataset(df[['timestamp', 'segment', 'target']])
    ts = TSDataset(df=ts, freq=freq)
    preds = pd.DataFrame()
    for counter, pipeline in enumerate(pipelines):
        pipeline.fit(ts)
        cur_preds = pipeline.forecast().to_pandas(flatten=True)
        cur_preds['model'] = pipelines_names[counter]
        preds = pd.concat((preds, cur_preds))
    preds['segment'] = preds['segment'].astype(int)
    preds = preds.rename(columns={'target': 'final_preds'})
    sma_df = sma_df.rename(columns={'target': 'hd_preds'})
    sma_df = sma_df.join(preds.set_index(['segment', 'model']), on=['segment', 'model']).drop('timestamp', axis=1)
    return sma_df


def get_cv_metrics(ts, pipelines, pipelines_names, n_folds):
    metrics = pd.DataFrame()
    for counter, pipeline in enumerate(pipelines):
        pipeline_metrics = pipeline.backtest(
            ts=ts,
            metrics=[MAE()],
            n_folds=3,
            aggregate_metrics=False,
            mode='expand', n_jobs=n_jobs,
            joblib_params=dict(verbose=0)
        )[0]
        pipeline_metrics = pipeline_metrics.groupby('segment')['MAE'] \
            .agg(['mean', 'std']).sum(1) \
            .to_frame('MAE').reset_index() \
            .assign(model=pipelines_names[counter])
        metrics = pd.concat((metrics, pipeline_metrics))
    preds = pd.DataFrame()
    for counter, pipeline in enumerate(pipelines):
        pipeline.fit(ts)
        cur_preds = pipeline.forecast().to_pandas(flatten=True)
        cur_preds['model'] = pipelines_names[counter]
        preds = pd.concat((preds, cur_preds))
    metrics = metrics.join(preds.set_index(['segment', 'model']), on=['segment', 'model'])
    metrics['MAE'] = metrics['MAE'].astype(int)
    metrics['segment'] = metrics['segment'].astype(int)
    return metrics


def catboost_params_tuner_finish(df, n_trials, l2_leaf_reg, bagging_temp, depth, grow_policy, random_strength,
                                 target_func, train_cols, y):
    print(target_func.__name__, 'tuning_begin')
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, warn_independent_sampling=False),
        direction='minimize')
    study.optimize(
        lambda trial: target_func(df, l2_leaf_reg, bagging_temp, depth, grow_policy, random_strength, train_cols, y,
                                  trial), n_trials=n_trials)
    return study.trials_dataframe()


def catboost_tuner_ring_start(df, train_cols, y, trial):
    params_to_tune = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.30),
        'n_estimators': trial.suggest_int('n_estimators', 300, 800),
        'loss_function': 'MAE',
        'verbose': 0,
        'cat_features': cat_features,
    }
    model = cb.CatBoostRegressor(**params_to_tune)
    tscv = TimeSeriesSplit(n_splits=3, test_size=df['segment'].nunique())
    cv_score = cross_val_score(model, df[train_cols], df[y], error_score='raise',
                               cv=tscv, scoring='neg_mean_absolute_error', n_jobs=n_jobs).mean()
    return np.abs(cv_score)


def catboost_tuner_ring_middle(df, lr, n_trees, train_cols, y, trial):
    params_to_tune = {
        'learning_rate': lr,
        'n_estimators': n_trees,
        'loss_function': 'MAE',
        'verbose': 0,
        'cat_features': cat_features,
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 2, 30),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 1),
        'depth': trial.suggest_int('depth', 4, 8),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'random_strength': trial.suggest_float('random_strength', 0.1, 1),
    }
    model = cb.CatBoostRegressor(**params_to_tune)
    tscv = TimeSeriesSplit(n_splits=3, test_size=df['segment'].nunique())
    cv_score = cross_val_score(model, df[train_cols], df[y],
                               cv=tscv, scoring='neg_mean_absolute_error', n_jobs=n_jobs).mean()
    return np.abs(cv_score)


def catboost_tuner_ring_finish(df, l2_leaf_reg, bagging_temp, depth, grow_policy, random_strength, train_cols, y,
                               trial):
    params_to_tune = {
        'loss_function': 'MAE',
        'verbose': 0,
        'cat_features': cat_features,
        'l2_leaf_reg': l2_leaf_reg,
        'bagging_temperature': bagging_temp,
        'depth': depth,
        'grow_policy': grow_policy,
        'random_strength': random_strength,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.30),
        'n_estimators': trial.suggest_int('n_estimators', 300, 800),
    }
    model = cb.CatBoostRegressor(**params_to_tune)
    tscv = TimeSeriesSplit(n_splits=3, test_size=df['segment'].nunique())
    cv_score = cross_val_score(model, df[train_cols], df[y],
                               cv=tscv, scoring='neg_mean_absolute_error', n_jobs=n_jobs).mean()
    return np.abs(cv_score)


def get_best_cb_model(train, train_cols, y):
    n_trials = 5
    train[cat_features] = train[cat_features].astype(int, errors='ignore').astype(str)
    train[int_cat_cols] = train[int_cat_cols].fillna(0).astype(int)

    trial_df_start = model_params_tuner_init(train, n_trials, catboost_tuner_ring_start, train_cols, y)
    best_iteration = trial_df_start['value'].idxmin()
    best_lr = trial_df_start.loc[best_iteration]['params_learning_rate']
    best_n_trees = trial_df_start.loc[best_iteration]['params_n_estimators']

    trial_df_middle = model_params_tuner_middle(train, n_trials * 5, best_lr, best_n_trees, catboost_tuner_ring_middle,
                                                train_cols, y)
    best_iteration = trial_df_middle['value'].idxmin()
    best_l2_leaf_reg = trial_df_middle.loc[best_iteration]['params_l2_leaf_reg']
    best_bagging_temp = trial_df_middle.loc[best_iteration]['params_bagging_temperature']
    best_depth = trial_df_middle.loc[best_iteration]['params_depth']
    best_grow_policy = trial_df_middle.loc[best_iteration]['params_grow_policy']
    best_random_strength = trial_df_middle.loc[best_iteration]['params_random_strength']

    trial_df_finish = catboost_params_tuner_finish(train, n_trials,
                                                   best_l2_leaf_reg, best_bagging_temp, best_depth, best_grow_policy,
                                                   best_random_strength, catboost_tuner_ring_finish, train_cols, y)
    best_iteration = trial_df_finish['value'].idxmin()
    best_lr = trial_df_finish.loc[best_iteration]['params_learning_rate']
    best_n_trees = trial_df_finish.loc[best_iteration]['params_n_estimators']

    final_params_cb = {
        'loss_function': 'MAE',
        'verbose': 0,
        'cat_features': cat_features,
        'l2_leaf_reg': best_l2_leaf_reg,
        'bagging_temperature': best_bagging_temp,
        'depth': best_depth,
        'grow_policy': 'Lossguide',
        'random_strength': best_random_strength,
        'learning_rate': best_lr,
        'n_estimators': int(1.06 * best_n_trees)
    }
    model = cb.CatBoostRegressor(**final_params_cb)
    return model


def merge_models_preds(models, hd_test, shipments):
    if isinstance(models, tuple):
        models_df = pd.concat(models)
    else:
        models_df = models.copy()
    sma_6 = models_df[models_df['model'] == 'sma6']
    models_df = models_df[~((models_df['final_preds']==0)|(models_df['hd_preds']==0))]
    models_df['rel'] = models_df['hd_preds'] / models_df['final_preds']
    rel_to_drop_ind = models_df[(~(models_df['rel'].between(0.25, 1.75)))&(models_df['model'].isin(['lgb_ps','cb_ms']))].index
    models_df = models_df.drop(rel_to_drop_ind)
    models_df['segment'] = models_df['segment'].astype(int)
    hd_test['segment'] = hd_test['segment'].astype(int)
    models_df = models_df.join(shipments[shipments['timestamp']>hmp_upload_dt_start].groupby('segment')['target'].agg(['min','max']), on='segment')
    min_max_check = (models_df['final_preds'].between(models_df['min'], models_df['max'] * 0.95))& \
                    (models_df['final_preds'].between(models_df['min'], models_df['max'] * 0.95))
    models_df = models_df[min_max_check]
    models_df = models_df.join(hd_test.groupby('segment')['target'].sum(), on='segment', rsuffix='_true').fillna(0)
    models_df['MAE_hd'] = models_df.apply(lambda x: mean_absolute_error([x['target']], [x['hd_preds']]), 1)
    models_df['MAE_ensemble'] = models_df['MAE'] * 0.33 + models_df['MAE_hd'] * 0.66
    models_df = models_df.groupby('segment') \
                         .apply(lambda x: x[x['MAE_ensemble'] == x['MAE_ensemble'].min()][['MAE_hd', 'model', 'hd_preds', 'final_preds']] \
                                           .drop_duplicates(subset=['MAE_hd'])) \
                         .reset_index(level=1, drop=True).reset_index() \
                         .join(hd_test.groupby('segment')['target'].sum(), on='segment') \
                         .fillna(0)
    models_df['score'] = models_df.apply(lambda x: mean_absolute_percentage_error([x['target']], [x['hd_preds']]), 1)
    print('models distribution:', models_df['model'].value_counts(normalize=True),
          'median score', models_df[models_df['target'] > 0]['score'].median(),
          'mean score', models_df[models_df['target'] > 0]['score'].mean(),
          'std score', models_df[models_df['target'] > 0]['score'].std())
    models_df = models_df[['segment', 'final_preds']].rename(columns={'final_preds':'target'}).astype(int)
    to_add = sma_6[sma_6['segment'].isin(list(set(sma_6['segment']).difference(models_df['segment'])))][['segment', 'final_preds']].astype(int)
    to_add.columns = ['segment', 'target']
    models_df = pd.concat((models_df, to_add))
    return models_df

