import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from upload_utils import *

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

import warnings
warnings.filterwarnings("ignore")


def get_train_test_cols(train, test):
    train_cols = train.drop(['timestamp', 'comb', 'target'] + exog_cols, axis=1).columns.tolist()
    test_col = 'target'
    date_cols_encode = ['date_flag_month_number_in_year', 'date_flag_season_number', 'date_flag_year_number']
    train[date_cols_encode] = train[date_cols_encode].astype(int)
    test[date_cols_encode] = test[date_cols_encode].astype(int)
    return train_cols, test_col

def get_constants(plan_month_fact):
    plan_month_fact = pd.to_datetime(plan_month_fact)
    churn_dt = str(plan_month_fact - pd.offsets.MonthBegin(3))
    plan_month_table = str(plan_month_fact - pd.offsets.MonthBegin(1))
    hmp_upload_dt_start = str(plan_month_fact - pd.offsets.MonthBegin(12))
    fill_q_begin = str(plan_month_fact - pd.offsets.MonthBegin(12))
    min_dt = '2017-11-01'
    max_dt = str(plan_month_fact - pd.offsets.MonthEnd(1))
    max_dt_wd_holidays = str(plan_month_fact + pd.offsets.MonthEnd(1))
    holdout_dt = str(plan_month_fact - pd.offsets.MonthBegin(1))
    low_val = 30000
    low_val_stm = 500
    low_val_hmp_oz = 500
    low_val_hmp_ons = 500
    freq = 'MS'
    horizon = 1
    # exog_cols = ['web_order_rel', 'volume', 'weight', 'density', 'med_amn', 'nds', 'position_count']
    # cat_features = ['sale_deptno', 'warehouse_oebs_org_id', 'type_name', 'manager']
    # rare_ignore_cols = ['manager']
    exog_cols = ['web_order_rel', 'volume', 'weight', 'density', 'med_amn', 'nds', 'position_count', 'market_rel']
    cat_features = ['CURRENT_SALE_LOC', 'ZDRAVSITI', 'SUBSEGMENT_TP',
                    'FACT_REG_CAPITAL', 'WORK_WITH_STRONG_MED', 'segment',
                    'LICENSE_TP', 'FACT_FEDERAL', 'manager', 'CURRENT_SOUZFARMA_INET_FLAG',
                    ]
    int_cat_cols = ['VALUE', 'LTK', 'CLAIM_PERIOD', 'TURNOVER', 'PERIOD_SHIPMENT', 'CURRENT_DISCOUNT_2K']
    rare_ignore_cols = ['manager', 'segment']
    ms_stat_features = ['mad_3', 'mean_3', 'std_3', 'minmax_3', 'minmax_mean_rel_3', 'minmax_rel_3',
                        'minmax_mean_rel_3_flag',
                        'min_rel_3', 'max_rel_3', 'mean_32']
    n_jobs = 1
    n_folds = 3
    return str(plan_month_fact), churn_dt, plan_month_table, hmp_upload_dt_start, fill_q_begin, \
        min_dt, max_dt, max_dt_wd_holidays, holdout_dt, low_val, low_val_stm, low_val_hmp_oz, \
        low_val_hmp_ons, freq, horizon, exog_cols, cat_features, rare_ignore_cols, ms_stat_features, int_cat_cols, n_folds, n_jobs

plan_month_fact = '2023-10-01'
plan_month_fact, churn_dt, plan_month_table, hmp_upload_dt_start, fill_q_begin, \
    min_dt, max_dt, max_dt_wd_holidays, holdout_dt, low_val, low_val_stm, low_val_hmp_oz, \
    low_val_hmp_ons, freq, horizon, exog_cols, cat_features, rare_ignore_cols, ms_stat_features, int_cat_cols, n_folds, n_jobs = get_constants(
    plan_month_fact)

def init_prepro_light(shipments, churn_dt, low_val):
    shipments['timestamp'] = pd.to_datetime(shipments['timestamp'])
    churned = shipments.groupby('segment') \
        .filter(lambda x: x[x['timestamp'] >= pd.to_datetime(churn_dt)]['target'].sum() == 0)['segment'].unique()
    df = shipments[~shipments['segment'].isin(churned)]
    rare = set(df.groupby('segment').filter(lambda x: x['timestamp'].nunique() < 6)['segment'].unique())
    low = set(df.groupby('segment') \
              .filter(lambda x: x[x['timestamp'] >= pd.to_datetime(churn_dt)]['target'].max() <= low_val)[
                  'segment'].unique())
    rare = rare.difference(low)
    to_drop = list(rare.union(low))
    df = df[~df['segment'].isin(to_drop)]
    return df, rare, low, churned

def resample_light(df, freq):
    df = df.groupby('segment').apply(lambda x: x.set_index('timestamp').resample(freq).sum()) \
        .drop('segment', axis=1).reset_index()
    return df

def join_holidays_wd_features(df, test):
    dates_dict = {'timestamp': pd.date_range(min_dt, max_dt_wd_holidays),
                  'target': [-1] * len(pd.date_range(min_dt, max_dt_wd_holidays)),
                  'segment': 'tmp'
                  }
    dates_df = pd.DataFrame(data=dates_dict)
    ts = TSDataset.to_dataset(dates_df)
    ts = TSDataset(df=ts, freq='D')
    transforms = [HolidayTransform(iso_code="RUS", out_column="holidays"), ]
    ts.fit_transform(transforms)
    df_holidays = ts.to_pandas(flatten=True)
    df_holidays['holidays'] = df_holidays['holidays'].fillna(0).astype(int)
    df_holidays = df_holidays.resample(on='timestamp', rule=freq)['holidays'].sum()
    country = Russia()
    dates_df['work_day'] = dates_df['timestamp'].apply(lambda x: country.is_working_day(x)).astype(int)
    df_workdays = dates_df[['timestamp', 'work_day']].resample(on='timestamp', rule=freq)['work_day'].sum()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = pd.merge(df, df_holidays, on='timestamp')
    df = pd.merge(df, df_workdays, on='timestamp')
    test = pd.merge(test, df_holidays, on='timestamp')
    test = pd.merge(test, df_workdays, on='timestamp')
    return df, test

def border_clustering(df, test):
    time_borders = [
        ['2017-11-01', '2017-12-01'],
        ['2018-01-01', '2018-12-01'],
        ['2019-01-01', '2019-12-01'],
        ['2020-01-01', '2020-12-01'],
        ['2021-01-01', '2021-12-01'],
        ['2022-01-01', '2022-12-01'],
        ['2023-01-01', '2023-12-01'],
    ]
    df['min_ts'] = df.groupby('segment')['timestamp'].transform('min')
    for counter, t in enumerate(time_borders):
        df.loc[df['min_ts'].between(t[0], t[1]), 'time_border'] = counter
    df = df.join(pd.qcut(df.groupby('segment')['target'].mean(), q=10, labels=['q1', 'q2', 'q3', 'q4', 'q5',
                                                                               'q6', 'q7', 'q8', 'q9', 'q10'
                                                                               ]),
                 on='segment', rsuffix='_border')
    df['comb'] = df['time_border'].astype(int).astype(str) + '_' + df['target_border'].astype(str)
    test = test.join(df[['segment', 'comb']] \
                     .drop_duplicates(subset=['segment'], keep='last') \
                     .set_index('segment')['comb'], on='segment')
    return df.drop(['min_ts', 'time_border', 'target_border'], axis=1), test

def join_cats(shipments, cats, rare_ignore_cols, int_cat_cols):
    shipments['segment'] = shipments['segment'].astype(int)
    cats['segment'] = cats['segment'].astype(int)
    transform_cols = [x for x in cat_features if x not in rare_ignore_cols]
    transform_cols = [x for x in transform_cols if x not in int_cat_cols]
    for col in transform_cols:
        col_vc = cats[col].value_counts(normalize=True)
        rare_cats = col_vc[col_vc < 0.01].index.tolist()
        cats.loc[cats[col].isin(rare_cats), col] = np.nan
    transformer = ColumnTransformer([('cat', SimpleImputer(strategy='most_frequent'), transform_cols)])
    new_cols = transformer.fit_transform(cats[transform_cols])
    copy_cats = cats[transform_cols].copy()
    copy_cats[transform_cols] = new_cols
    copy_cats[int_cat_cols] = cats[int_cat_cols].fillna(0).astype(int)
    copy_cats[[i for i in rare_ignore_cols if i != 'segment']] = cats[[i for i in rare_ignore_cols if i != 'segment']].values
    copy_cats['segment'] = cats['segment'].values
    diff_count = len(set(copy_cats['segment']).difference(shipments['segment']))
    print('in cats directory, not in shipments: {0}'.format(diff_count))
    shipments = pd.merge(shipments, copy_cats, on='segment')
    return shipments

def get_transforms():
    transforms = [
        LagTransform(in_column='target', out_column='lag', lags=[1, 3, 6]),
        MeanTransform(in_column='target', out_column='mean_3', window=3, seasonality=1),
        MeanTransform(in_column='target', out_column='mean_23', window=2, seasonality=3),
        MeanTransform(in_column='target', out_column='mean_32', window=3, seasonality=2),
        MeanTransform(in_column='target', out_column='mean_6', window=6, seasonality=1),
        StdTransform(in_column='target', out_column='std_3', window=3, seasonality=1),
        StdTransform(in_column='target', out_column='std_6', window=6, seasonality=1),
        MinMaxDifferenceTransform(in_column='target', out_column='minmax_3', window=3, seasonality=1),
        MinMaxDifferenceTransform(in_column='target', out_column='minmax_6', window=6, seasonality=1),
        MinMaxDifferenceTransform(in_column='target', out_column='minmax_12', window=3, seasonality=1),
        MinTransform(in_column='target', out_column='min_3', window=3, seasonality=1),
        MinTransform(in_column='target', out_column='min_6', window=6, seasonality=1),
        MaxTransform(in_column='target', out_column='max_3', window=3, seasonality=1),
        MaxTransform(in_column='target', out_column='max_6', window=6, seasonality=1),
        MADTransform(in_column='target', window=3, seasonality=1, out_column='mad_3'),
        DateFlagsTransform(day_number_in_week=False, day_number_in_month=False, day_number_in_year=False,
                           week_number_in_month=False, is_weekend=False,
                           week_number_in_year=False, season_number=True, month_number_in_year=True, year_number=True,
                           out_column='date_flag')
    ]
    for exog in exog_cols:
        transforms.append(LagTransform(in_column=exog, out_column='lag_{0}'.format(exog), lags=[horizon]), )
    return transforms

def etna_features_postprocessing(df):
    df['minmax_mean_rel_3'] = df['minmax_3'] / df['mean_3']
    df['minmax_rel_3'] = df['min_3'] / df['max_3']
    df['minmax_mean_rel_6'] = df['minmax_6'] / df['mean_6']
    df['minmax_rel_6'] = df['min_6'] / df['max_6']
    df['min_rel_3'] = df['min_3'] / df['mean_3']
    df['max_rel_3'] = df['max_3'] / df['mean_3']
    df['min_rel_6'] = df['min_6'] / df['mean_6']
    df['max_rel_6'] = df['max_6'] / df['mean_6']
    df['minmax_rel_6'] = df['min_6'] / df['max_6']
    df['minmax_mean_rel_6_flag'] = (df['minmax_mean_rel_6'] > 1).astype(int)
    df['minmax_mean_rel_3_flag'] = (df['minmax_mean_rel_3'] > 1).astype(int)
    to_drop = ['min_3', 'max_3', 'min_6', 'max_6']
    df = df.drop(to_drop, axis=1)
    return df

def get_etna_features(df, shipments):
    transforms = get_transforms()
    ts = TSDataset.to_dataset(df)
    ts = TSDataset(df=ts, freq=freq)

    ts.fit_transform(transforms)
    train = ts.to_pandas(flatten=True)
    test = ts.make_future(horizon, transforms=transforms).to_pandas(flatten=True)
    test = test.join(train[['segment']].drop_duplicates().dropna().set_index('segment'), on='segment')
    train['segment'] = train['segment'].astype(int)
    test['segment'] = test['segment'].astype(int)
    train = train.join(shipments.groupby('segment')['timestamp'].min(), on='segment', rsuffix='_min')
    train = train.groupby('segment') \
        .apply(lambda x: x[x['timestamp'] >= x['timestamp_min']]) \
        .reset_index(drop=True).drop('timestamp_min', axis=1) \
        .sort_values(by=['timestamp', 'segment'])

    train = etna_features_postprocessing(train)
    test = etna_features_postprocessing(test)
    return train, test

def prepare_boosting_df(df, shipments, cats_df, rare_ignore_cols, int_cat_cols):
    train, test = get_etna_features(df, shipments)
    train, test = join_holidays_wd_features(train, test)
    train = join_cats(train, cats_df, rare_ignore_cols, int_cat_cols)
    test = join_cats(test, cats_df, rare_ignore_cols, int_cat_cols)
    train, test = border_clustering(train, test)
    return train, test

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

