import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from prepro_utils import *
from upload_utils import *
from modeling_utils import *
from planning_utils import *

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

def prepare_stm_forecast(client):
    cards, shipments, cats_df, old_managers_dir, df, rare, low, churned = upload_divide_resample(client, mode='stm_forecast')
    sma_res = sma_model(df)
    hd_test = shipments[shipments['timestamp'].astype('datetime64') == holdout_dt]
    forecast_df = merge_models_preds((sma_res), hd_test, df)
    forecast_df = merge_other_subsets(forecast_df, low, rare, churned, shipments, low_val_stm)
    plans_filially = upload_filial_plans(client, 'stm', plan_month_fact)
    return forecast_df, plans_filially, shipments

def fill_stm_segments(entire_plan, forecast_df):
    to_add_segments = list(set(entire_plan['segment']).difference(forecast_df['segment']))
    to_add_segments_df = pd.DataFrame({'segment': to_add_segments, 'target': entire_plan[entire_plan['segment'].isin(to_add_segments)]['target'].values})
    to_add_segments_df.loc[to_add_segments_df['target'] > 1, 'target'] = low_val_stm
    forecast_df = pd.concat((forecast_df, to_add_segments_df))
    sale_loc_dict = upload_sale_loc_dict(client)
    forecast_df['sale_loc'] = forecast_df['segment'].map(sale_loc_dict).values
    return forecast_df


def prepare_forecast_df_stm(forecast_df, shipments, shipments_stm, churn_dt, fill_q_begin, low_val_stm,
                            border_stm_percentage):
    forecast_df = forecast_df.join(shipments[shipments['timestamp'] > churn_dt].groupby('segment')['target'].sum() // 3,
                                   on='segment',
                                   rsuffix='_entire_mean_3')
    forecast_df['rel_stm'] = forecast_df['target'] / forecast_df['target_entire_mean_3']
    forecast_df.loc[forecast_df['target_entire_mean_3'].isnull(), 'target_entire_mean_3'] = \
        low_val_stm / border_stm_percentage * forecast_df[forecast_df['target_entire_mean_3'].isnull()]['target'].values
    forecast_df['rel_stm'] = forecast_df['rel_stm'].fillna(1)
    stm_shipments_min_max = shipments_stm[shipments_stm['timestamp'] > fill_q_begin].groupby('segment')['target'].agg(
        ['min', 'max'])
    forecast_df = forecast_df.join(stm_shipments_min_max, on='segment')
    forecast_df.loc[forecast_df['min'].isnull(), 'min'] = forecast_df[forecast_df['min'].isnull()]['target'].values
    forecast_df.loc[forecast_df['max'].isnull(), 'max'] = forecast_df[forecast_df['max'].isnull()]['target'].values
    forecast_df.loc[forecast_df['sale_loc'] == 'П-19 Пермь', 'sale_loc'] = 'П-26 Киров'
    forecast_df.loc[forecast_df['sale_loc'] == 'П-02 Волгоград', 'sale_loc'] = 'П-44 Астрахань'
    forecast_df.loc[forecast_df['sale_loc'] == 'ОГП', 'sale_loc'] = 'ОПАС'
    return forecast_df


def plan_preparation_stm_helper(forecast_df, plans_filially, border_stm_percentage, param_max):
    sale_loc = forecast_df['sale_loc'].unique()[0]
    sale_loc_forecast = forecast_df[forecast_df['sale_loc'] == sale_loc]
    print('----------')
    print(sale_loc, 'BEGIN')
    flag = False
    plan_cond = (plans_filially['sale_loc'] == sale_loc) & (plans_filially['plan_month_fact'].astype('datetime64') == plan_month_fact)
    sale_loc_plan = plans_filially[plan_cond]['plan_value_stm'].tolist()[0]
    init_rel = sale_loc_plan / forecast_df['target'].sum()
    print('init_rel:', init_rel)
    if init_rel < 1:
        sale_loc_forecast['target'] *= init_rel
        flag = True
        return sale_loc_forecast, flag
    else:
        sale_loc_forecast, flag = fill_to_border_remain_borders_stm(sale_loc_forecast, border_stm_percentage,
                                                                    sale_loc_plan, param_max)
        return sale_loc_forecast, flag


def plan_preparation_stm(forecast_df, plans_filially, border_stm_percentage, param_max):
    res_dict = dict()
    res_df = pd.DataFrame()
    for sale_loc in forecast_df['sale_loc'].unique():
        sub = forecast_df[forecast_df['sale_loc'] == sale_loc]
        sub, flag = plan_preparation_stm_helper(sub, plans_filially, border_stm_percentage, param_max)
        res_dict[sale_loc] = flag
        if flag:
            res_df = pd.concat((res_df, sub))
    bad_sale_loc = list({k: v for k, v in res_dict.items() if v == False}.keys())
    param_max = 0.02
    for sale_loc in bad_sale_loc:
        for i in range(1, 7):
            sub = forecast_df[forecast_df['sale_loc'] == sale_loc]
            sub, flag = plan_preparation_stm_helper(sub, plans_filially, border_stm_percentage, param_max + i / 50)
            if flag or i == 6:
                print(i, 'DONE')
                res_dict[sale_loc] = flag
                res_df = pd.concat((res_df, sub))
                break
    res_df['rel_stm'] = res_df['target'] / res_df['target_entire_mean_3']
    return res_df[['segment', 'target', 'sale_loc']], res_df


def fill_to_border_remain_borders_stm(cur_sale_loc, border_stm_percentage, sale_loc_plan, param_max):
    possible_fill = cur_sale_loc[cur_sale_loc['rel_stm'] < border_stm_percentage]
    gt_border_part = cur_sale_loc[cur_sale_loc['rel_stm'] >= border_stm_percentage]
    possible_fill['border_coeff'] = border_stm_percentage / possible_fill['rel_stm']
    possible_fill['possible_target'] = possible_fill['border_coeff'] * possible_fill['target']
    possible_fill.loc[possible_fill['possible_target'].isnull(), 'possible_target'] = \
        possible_fill[possible_fill['possible_target'].isnull()]['target'].values
    possible_fill.loc[possible_fill['possible_target'] > possible_fill['max'], 'possible_target'] = \
        possible_fill[possible_fill['possible_target'] > possible_fill['max']]['max'].values
    possible_fill.loc[possible_fill['target'] <= 500, 'rel_stm'] = 1
    cur_sale_loc = pd.concat((possible_fill[gt_border_part.columns.tolist() + ['possible_target']], gt_border_part))
    cur_sale_loc.loc[cur_sale_loc['possible_target'].isnull(), 'possible_target'] = \
        cur_sale_loc[cur_sale_loc['possible_target'].isnull()]['target'].values
    if cur_sale_loc['possible_target'].sum() > sale_loc_plan:
        print('stm fill lt border | approved')
        need_to_fill_diff = sale_loc_plan - cur_sale_loc['target'].sum()
        cur_sale_loc['diff'] = cur_sale_loc['possible_target'] - cur_sale_loc['target']
        cur_sale_loc['diff'] *= need_to_fill_diff / cur_sale_loc['diff'].sum()
        cur_sale_loc['target'] += cur_sale_loc['diff']
        cur_sale_loc['rel_stm'] = cur_sale_loc['target'] / cur_sale_loc['target_entire_mean_3']
        cur_sale_loc = cur_sale_loc.drop('diff', axis=1)
        return cur_sale_loc, True
    else:
        print('stm fill lt border | failed, gt border begin')
        border_stm_percentage_loop = int(border_stm_percentage * 100)
        all_change_ind = []
        for num in range(border_stm_percentage_loop, 97):
            factor = num / 100
            increment = 0.009999
            if factor == border_stm_percentage:
                factor = border_stm_percentage + 0.001
            upper_bound = param_max
            cur_change_ind = cur_sale_loc[(cur_sale_loc['rel_stm'].between(factor, factor + increment)) &
                                          (cur_sale_loc['target'] > 500)].index.tolist()
            all_change_ind.extend(cur_change_ind)
            no_change_ind = cur_sale_loc.index.difference(all_change_ind)
            cur_sale_loc.loc[cur_change_ind, 'rel_stm'] = factor + upper_bound
            cur_sale_loc.loc[no_change_ind, 'rel_stm'] = cur_sale_loc.loc[no_change_ind]['rel_stm'].values
            cur_sale_loc['possible_target'] = cur_sale_loc['rel_stm'] * cur_sale_loc['target_entire_mean_3']
            cur_sale_loc.loc[cur_sale_loc['possible_target'] > cur_sale_loc['max'], 'possible_target'] = \
                cur_sale_loc[cur_sale_loc['possible_target'] > cur_sale_loc['max']]['max'].values
            if cur_sale_loc['possible_target'].sum() > sale_loc_plan:
                print('stm fill gt border | approved on:', num)
                need_to_fill_diff = sale_loc_plan - cur_sale_loc['target'].sum()
                cur_sale_loc['diff'] = cur_sale_loc['possible_target'] - cur_sale_loc['target']
                cur_sale_loc['diff'] *= need_to_fill_diff / cur_sale_loc['diff'].sum()
                cur_sale_loc['target'] += cur_sale_loc['diff']
                cur_sale_loc['rel_stm'] = cur_sale_loc['target'] / cur_sale_loc['target_entire_mean_3']
                cur_sale_loc = cur_sale_loc.drop(['diff', 'possible_target'], axis=1)
                return cur_sale_loc, True
        print('stm fill gt border | failed now is:', cur_sale_loc['possible_target'].sum() // 1e6)
        cur_sale_loc['target'] = cur_sale_loc['possible_target'].values
        cur_sale_loc['rel_stm'] = cur_sale_loc['target'] / cur_sale_loc['target_entire_mean_3']
        cur_sale_loc = cur_sale_loc.drop(['possible_target'], axis=1)
        return cur_sale_loc, False

def stm_plan(client, res_entire):
    forecast_df_stm, plans_filially_stm, shipments_stm = prepare_stm_forecast(client)
    forecast_df_stm = fill_stm_segments(res_entire, forecast_df_stm)
    _, shipments_entire, _, _, _, _, _, _ = upload_divide_resample(client, mode='entire_forecast')
    forecast_df_stm = prepare_forecast_df_stm(forecast_df_stm, shipments_entire, shipments_stm, churn_dt, fill_q_begin, low_val_stm, border_stm_percentage)
    res, raw_res = plan_preparation_stm(forecast_df_stm, plans_filially_stm, border_stm_percentage, param_max)
    return res, raw_res, forecast_df_stm

