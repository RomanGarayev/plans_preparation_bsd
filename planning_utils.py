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

def entire_plan(client):
    cards, shipments, cats_df, old_managers_dir, df, rare, low, churned = upload_divide_resample(client, mode='entire_forecast')
    train, test = prepare_boosting_df(df, shipments, cards, rare_ignore_cols, int_cat_cols)
    train_cols, y = get_train_test_cols(train, test)
    print('train_cols:',  train_cols)
    train['target'] = train['target'].fillna(0)
    boosting_models_info = dict()
    boosting_models_info['lgb_ps'] = [i for i in train_cols if i not in cat_features], 'segment'
    boosting_models_info['lgb_ms'] = [i for i in train_cols if i not in (cat_features + ms_stat_features)], 'comb'
    boosting_models_info['cb_ps'] = train_cols, 'segment'
    boosting_models_info['cb_ms'] = [i for i in train_cols if i not in ms_stat_features], 'comb'
    cb_ms = gb_multi_comb_parallel(train, test, get_best_cb_model, 'cb_ms',
                                    boosting_models_info['cb_ms'][1], boosting_models_info['cb_ms'][0], y)
    lgb_ms = gb_multi_comb_parallel(train, test, get_best_lgb_model, 'lgb_ms',
                                    boosting_models_info['lgb_ms'][1], boosting_models_info['lgb_ms'][0], y)
    sma_res = sma_model(train)
    hd_test = shipments[shipments['timestamp'].astype('datetime64') == holdout_dt]
    forecast_df = merge_models_preds((sma_res, cb_ms, lgb_ms), hd_test, df)
    forecast_df = merge_other_subsets(forecast_df, low, rare, churned, shipments, low_val)
    plans_filially = upload_filial_plans(client, 'entire', plan_month_fact)
    forecast_df, plans = merge_plans_forecast_df(forecast_df, cards, client, plans_filially)
    res, raw_res = all_filials_plan_prep(forecast_df, plans_filially, df, shipments, rare)
    return res, raw_res, forecast_df

def upload_filial_plans(client, plan_type, plan_month_fact):
    query = '''
        select * from CUST.top_level_plans_filially
        where plan_month_fact = toDate('{0}')
    '''.format(plan_month_fact)
    df, cols = client.execute(query, with_column_types=True)
    df = pd.DataFrame(df, columns=[col[0] for col in cols])
    if plan_type == 'entire':
        return df[['sale_loc', 'plan_month_fact', 'plan_value']]
    elif plan_type == 'stm':
        return df[['sale_loc', 'plan_month_fact', 'plan_value_stm']]
    elif plan_type == 'onls':
        return df[['sale_loc', 'plan_month_fact', 'plan_value_oz']]
    elif plan_type == 'oz':
        return df[['sale_loc', 'plan_month_fact', 'plan_value_onls']]

def merge_plans_forecast_df(forecast_df, plans, client, plans_filially):
    # fix turnover, soz relation
    heads_cols = ['TURNOVER', 'SITE_ID_LINK', 'CURRENT_SALE_LOC', 'VALUE']
    plans = plans[heads_cols]
    plans['VALUE'] = plans['VALUE'].astype(int)
    forecast_df = forecast_df.join(plans.set_index('SITE_ID_LINK'), on='segment')
    sale_loc_dict = upload_sale_loc_dict(client)
    forecast_df.loc[forecast_df['CURRENT_SALE_LOC'].isnull(), 'CURRENT_SALE_LOC'] = \
    forecast_df[forecast_df['CURRENT_SALE_LOC'].isnull()]['segment'].map(sale_loc_dict).values
    forecast_df.loc[forecast_df['TURNOVER'].isnull(), 'TURNOVER'] = \
        forecast_df[forecast_df['TURNOVER'].isnull()]['target'].values * 5
    forecast_df.loc[forecast_df['TURNOVER'].isin([0, 1]), 'TURNOVER'] = \
        forecast_df[forecast_df['TURNOVER'].isin([0, 1])]['target'].values * 5
    forecast_df['TURNOVER'] = forecast_df['TURNOVER'].astype(float)
    forecast_df['fact_soz'] = forecast_df['target'] / forecast_df['TURNOVER']
    forecast_df.loc[forecast_df['fact_soz'] > 1, 'TURNOVER'] = \
        forecast_df.loc[forecast_df['fact_soz'] > 1, 'target'].values * 5
    forecast_df['fact_soz'] = forecast_df['target'] / forecast_df['TURNOVER']
    # clip low to 30k, make soz rel = 1 and turnover = 1 and churned to 0, soz rel = 1 and turnover = 0
    forecast_df.loc[forecast_df['target'].between(1, 30000), 'fact_soz'] = 1
    forecast_df.loc[forecast_df['target'].between(1, 30000), 'target'] = 30000
    forecast_df.loc[forecast_df['target'].between(1, 30000), 'TURNOVER'] = 30000
    forecast_df.loc[forecast_df['target'] == 0, 'fact_soz'] = 1
    forecast_df.loc[forecast_df['target'] == 0, 'TURNOVER'] = 0
    # rename columns and drop last plan
    forecast_df = forecast_df.drop(['VALUE'], axis=1)
    plans = plans.drop('VALUE', axis=1)
    plans = plans.join(plans_filially[plans_filially['plan_month_fact'] == plan_month_fact].set_index('sale_loc')['plan_value'],
                       on='CURRENT_SALE_LOC')
    print(forecast_df)
    forecast_df.columns = ['segment', 'target', 'soz', 'sale_loc', 'fact_soz']
    plans.columns = ['soz', 'segment', 'sale_loc', 'plan']
    forecast_df.loc[forecast_df['sale_loc'] == 'П-19 Пермь', 'sale_loc'] = 'П-26 Киров'
    forecast_df.loc[forecast_df['sale_loc'] == 'П-02 Волгоград', 'sale_loc'] = 'П-44 Астрахань'
    forecast_df.loc[forecast_df['sale_loc'] == 'ОГП', 'sale_loc'] = 'ОПАС'
    return forecast_df, plans

def add_border_to_very_low(sale_loc_forecast, sale_loc_plan, param_low):
    df = sale_loc_forecast.copy()
    change_ind = df[df['fact_soz'].between(0, 0.009999)].index
    no_change_ind = df.index.difference(change_ind)
    df.loc[change_ind, 'border_soz'] = df.loc[change_ind]['fact_soz'].values * param_low
    df.loc[no_change_ind, 'border_soz'] = df.loc[no_change_ind]['fact_soz'].values
    df['corrected_plan'] = df['border_soz'] * df['soz']
    df['diff'] = df['corrected_plan'] - df['target']
    print('after low', df['corrected_plan'].sum() // 1e6)
    if df['corrected_plan'].sum() > sale_loc_plan:
        need_to_fill_diff = sale_loc_plan - df['target'].sum()
        df['diff'] *= need_to_fill_diff / df['diff'].sum()
        df['target'] += df['diff']
        df['fact_soz'] = df['target'] / df['soz']
        df = df.drop(['border_soz', 'corrected_plan', 'diff'], axis=1)
        print('low | approved')
        return True, df
    else:
        print('low| failed')
        return False, sale_loc_forecast

def add_border_0_to_5(sale_loc_forecast, sale_loc_plan, param_border_25,
                      param_low, param_5):
    df = sale_loc_forecast.copy()
    border_25_ind = df[df['fact_soz'] < param_border_25].index
    very_low_ind = df[df['fact_soz'].between(0, 0.009999)].index
    change_ind = df[df['fact_soz'].between(0, 0.049999)].index
    no_change_ind = df.index.difference(change_ind)
    df.loc[df['fact_soz'].between(0.01, 0.049999), 'border_soz'] = param_5
    df.loc[very_low_ind, 'border_soz'] = df.loc[very_low_ind]['fact_soz'].values * param_low
    df.loc[no_change_ind, 'border_soz'] = df.loc[no_change_ind]['fact_soz'].values
    df.loc[(df.index.isin(border_25_ind)) & (df['border_soz'] > 0.25), 'border_soz'] = 0.25
    df['corrected_plan'] = df['border_soz'] * df['soz']
    df['diff'] = df['corrected_plan'] - df['target']
    print('0 to 5 after', df['corrected_plan'].sum() // 1e6)
    if df['corrected_plan'].sum() > sale_loc_plan:
        need_to_fill_diff = sale_loc_plan - df['target'].sum()
        df['diff'] *= need_to_fill_diff / df['diff'].sum()
        df['target'] += df['diff']
        df['fact_soz'] = df['target'] / df['soz']
        df = df.drop(['border_soz', 'corrected_plan', 'diff'], axis=1)
        print('0 to 5| approved')
        return True, df
    else:
        print('0 to 5| failed')
        return False, sale_loc_forecast

def add_border_0_to_10(sale_loc_forecast, sale_loc_plan, param_border_25,
                       param_low, param_5, param_10, ):
    df = sale_loc_forecast.copy()
    border_25_ind = df[df['fact_soz'] < param_border_25].index
    very_low_ind = df[df['fact_soz'].between(0, 0.009999)].index
    change_ind = df[df['fact_soz'].between(0, 0.099999)].index
    df.loc[very_low_ind, 'border_soz'] = df.loc[very_low_ind]['fact_soz'].values * param_low
    df.loc[df['fact_soz'].between(0.01, 0.049999), 'border_soz'] = param_5
    df.loc[df['fact_soz'].between(0.05, 0.09999), 'border_soz'] = param_10
    no_change_ind = df.index.difference(change_ind)
    df.loc[no_change_ind, 'border_soz'] = df.loc[no_change_ind]['fact_soz'].values
    df.loc[(df.index.isin(border_25_ind)) & (df['border_soz'] > 0.25), 'border_soz'] = 0.25
    df['corrected_plan'] = df['border_soz'] * df['soz']
    df['diff'] = df['corrected_plan'] - df['target']
    print('0 to 10 after', df['corrected_plan'].sum() // 1e6)
    if df['corrected_plan'].sum() > sale_loc_plan:
        need_to_fill_diff = sale_loc_plan - df['target'].sum()
        df['diff'] *= need_to_fill_diff / df['diff'].sum()
        df['target'] += df['diff']
        df['fact_soz'] = df['target'] / df['soz']
        df = df.drop(['border_soz', 'corrected_plan', 'diff'], axis=1)
        print('0 to 10| approved')
        return True, df
    else:
        print('0 to 10| failed')
        return False, sale_loc_forecast

def add_border_0_to_15(sale_loc_forecast, sale_loc_plan, param_border_25,
                       param_low, param_5, param_10, param_15):
    df = sale_loc_forecast.copy()
    border_25_ind = df[df['fact_soz'] < param_border_25].index
    very_low_ind = df[df['fact_soz'].between(0, 0.009999)].index
    change_ind = df[df['fact_soz'].between(0, 0.149999)].index
    df.loc[very_low_ind, 'border_soz'] = df.loc[very_low_ind]['fact_soz'].values * param_low
    df.loc[df['fact_soz'].between(0.01, 0.049999), 'border_soz'] = param_5
    df.loc[df['fact_soz'].between(0.05, 0.09999), 'border_soz'] = param_10
    df.loc[df['fact_soz'].between(0.1, 0.14999), 'border_soz'] = param_15
    no_change_ind = df.index.difference(change_ind)
    df.loc[no_change_ind, 'border_soz'] = df.loc[no_change_ind]['fact_soz'].values
    df.loc[(df.index.isin(border_25_ind)) & (df['border_soz'] > 0.25), 'border_soz'] = 0.25
    df['corrected_plan'] = df['border_soz'] * df['soz']
    df['diff'] = df['corrected_plan'] - df['target']
    print('0 to 15 after', df['corrected_plan'].sum() // 1e6)
    if df['corrected_plan'].sum() > sale_loc_plan:
        need_to_fill_diff = sale_loc_plan - df['target'].sum()
        df['diff'] *= need_to_fill_diff / df['diff'].sum()
        df['target'] += df['diff']
        df['fact_soz'] = df['target'] / df['soz']
        df = df.drop(['border_soz', 'corrected_plan', 'diff'], axis=1)
        print('0 to 15| approved')
        return True, df
    else:
        print('0 to 15| failed')
        return False, sale_loc_forecast

def add_border_0_to_20(sale_loc_forecast, sale_loc_plan, param_border_25,
                       param_low, param_5, param_10, param_15, param_20):
    df = sale_loc_forecast.copy()
    border_25_ind = df[df['fact_soz'] < param_border_25].index
    very_low_ind = df[df['fact_soz'].between(0, 0.009999)].index
    change_ind = df[df['fact_soz'].between(0, 0.149999)].index
    df.loc[very_low_ind, 'border_soz'] = df.loc[very_low_ind]['fact_soz'].values * param_low
    df.loc[df['fact_soz'].between(0.01, 0.049999), 'border_soz'] = param_5
    df.loc[df['fact_soz'].between(0.05, 0.09999), 'border_soz'] = param_10
    df.loc[df['fact_soz'].between(0.1, 0.14999), 'border_soz'] = param_15
    df.loc[df['fact_soz'].between(0.15, 0.19999), 'border_soz'] = param_20
    no_change_ind = df.index.difference(change_ind)
    df.loc[no_change_ind, 'border_soz'] = df.loc[no_change_ind]['fact_soz'].values
    df.loc[(df.index.isin(border_25_ind)) & (df['border_soz'] > 0.25), 'border_soz'] = 0.25
    df['corrected_plan'] = df['border_soz'] * df['soz']
    df['diff'] = df['corrected_plan'] - df['target']
    print('0 to 20 after', df['corrected_plan'].sum() // 1e6)
    if df['corrected_plan'].sum() > sale_loc_plan:
        need_to_fill_diff = sale_loc_plan - df['target'].sum()
        df['diff'] *= need_to_fill_diff / df['diff'].sum()
        df['target'] += df['diff']
        df['fact_soz'] = df['target'] / df['soz']
        df = df.drop(['border_soz', 'corrected_plan', 'diff'], axis=1)
        print('0 to 20| approved')
        return True, df
    else:
        print('0 to 20| failed')
        return False, sale_loc_forecast

def add_border_0_to_25(sale_loc_forecast, sale_loc_plan, param_border_25,
                       param_low, param_5, param_10, param_15, param_20, param_25):
    df = sale_loc_forecast.copy()
    border_25_ind = df[df['fact_soz'] < param_border_25].index
    very_low_ind = df[df['fact_soz'].between(0, 0.009999)].index
    change_ind = df[df['fact_soz'].between(0, 0.249999)].index
    df.loc[very_low_ind, 'border_soz'] = df.loc[very_low_ind]['fact_soz'].values * param_low
    df.loc[df['fact_soz'].between(0.01, 0.049999), 'border_soz'] = param_5
    df.loc[df['fact_soz'].between(0.05, 0.09999), 'border_soz'] = param_10
    df.loc[df['fact_soz'].between(0.1, 0.14999), 'border_soz'] = param_15
    df.loc[df['fact_soz'].between(0.15, 0.19999), 'border_soz'] = param_20
    df.loc[df['fact_soz'].between(0.2, 0.24999), 'border_soz'] = param_25
    no_change_ind = df.index.difference(change_ind)
    df.loc[no_change_ind, 'border_soz'] = df.loc[no_change_ind]['fact_soz'].values
    df.loc[(df.index.isin(border_25_ind)) & (df['border_soz'] > 0.25), 'border_soz'] = 0.25
    df['corrected_plan'] = df['border_soz'] * df['soz']
    df['diff'] = df['corrected_plan'] - df['target']
    print('0 to 25 after', df['corrected_plan'].sum() // 1e6)
    if df['corrected_plan'].sum() > sale_loc_plan:
        need_to_fill_diff = sale_loc_plan - df['target'].sum()
        df['diff'] *= need_to_fill_diff / df['diff'].sum()
        df['target'] += df['diff']
        df['fact_soz'] = df['target'] / df['soz']
        df = df.drop(['border_soz', 'corrected_plan', 'diff'], axis=1)
        print('0 to 25| approved')
        return True, df
    else:
        print('0 to 25| failed')
        return False, sale_loc_forecast

def add_border_0_to_max(sale_loc_forecast, sale_loc_plan, param_border_25,
                        param_low, param_5, param_10, param_15, param_20, param_25, param_max):
    df = sale_loc_forecast.copy()
    border_25_ind = df[df['fact_soz'] < param_border_25].index
    very_low_ind = df[df['fact_soz'].between(0, 0.009999)].index
    change_ind = df[df['fact_soz'].between(0, 0.249999)].index
    no_change_ind = df.index.difference(change_ind)
    df.loc[very_low_ind, 'border_soz'] = df.loc[very_low_ind]['fact_soz'].values * param_low
    df.loc[df['fact_soz'].between(0.01, 0.049999), 'border_soz'] = param_5
    df.loc[df['fact_soz'].between(0.05, 0.09999), 'border_soz'] = param_10
    df.loc[df['fact_soz'].between(0.1, 0.149999), 'border_soz'] = param_15
    df.loc[df['fact_soz'].between(0.15, 0.19999), 'border_soz'] = param_20
    df.loc[df['fact_soz'].between(0.2, 0.249999), 'border_soz'] = param_25
    df.loc[no_change_ind, 'border_soz'] = df.loc[no_change_ind]['fact_soz'].values
    df.loc[(df.index.isin(border_25_ind)) & (df['border_soz'] > 0.25), 'border_soz'] = 0.25
    df['target'] = df['soz'] * df['border_soz']
    df['fact_soz'] = df['border_soz'].values
    df = df.drop('border_soz', axis=1)
    bad_flag_df = df.copy()
    all_change_ind = []
    for num in range(25, 99):
        factor = num / 100
        increment = 0.009999
        upper_bound = param_max  # 0.015
        if num == 25:
            factor = 0.250001
        cur_change_ind = df[df['fact_soz'].between(factor, factor + increment)].index.tolist()
        all_change_ind.extend(cur_change_ind)
        no_change_ind = df.index.difference(all_change_ind)
        df.loc[cur_change_ind, 'border_soz'] = factor + upper_bound
        df.loc[no_change_ind, 'border_soz'] = df.loc[no_change_ind]['fact_soz'].values
        df['corrected_plan'] = df['border_soz'] * df['soz']
        flag = df['corrected_plan'].sum() > sale_loc_plan
        if flag:
            print('0 to max after:', df['corrected_plan'].sum() // 1e6)
            need_to_fill_diff = sale_loc_plan - df['target'].sum()
            df['diff'] = df['corrected_plan'] - df['target']
            df['diff'] *= need_to_fill_diff / df['diff'].sum()
            df['target'] += df['diff']
            df['fact_soz'] = df['target'] / df['soz']
            df = df.drop(['border_soz', 'corrected_plan', 'diff'], axis=1)
            print('0 to {0}| approved'.format(num))
            return flag, df
    print('0 to {0}| failed, final sum: {1}'.format(num, df['corrected_plan'].sum() // 1e6), )
    df['target'] = df['corrected_plan'].values
    df['fact_soz'] = df['border_soz'].values
    df = df.drop(['border_soz', 'corrected_plan', ], axis=1)
    return flag, df

def add_to_big_init(sale_loc_forecast, sale_loc_plan):
    df = sale_loc_forecast.copy()
    big_ind = df[(df['target'] > 5e6) & (df['fact_soz'] > 0.01)].index
    big_sum = df.loc[big_ind, 'target'].sum()
    residual = sale_loc_plan - df['target'].sum()
    rel = df['target'].sum() / sale_loc_plan
    df.loc[big_ind, 'target'] = df.loc[big_ind, 'target'] / big_sum * residual + df.loc[big_ind, 'target']
    df.loc[big_ind, 'fact_soz'] = df.loc[big_ind, 'target'].values / df.loc[big_ind, 'soz'].values
    print('add big init| approved, after', df['target'].sum() // 1e6)
    flag = True
    return flag, df

def add_to_big_finish(sale_loc_forecast, sale_loc_plan, param):
    df = sale_loc_forecast.copy()
    big_ind = df[(df['target'] > 5e6) & (df['fact_soz'] > 0.01)].index
    big_sum = df.loc[big_ind, 'target'].sum()
    residual = sale_loc_plan - df['target'].sum()
    rel = df['target'].sum() / sale_loc_plan
    if (1 - rel) <= param:
        df.loc[big_ind, 'target'] = df.loc[big_ind, 'target'] / big_sum * residual + df.loc[big_ind, 'target']
        df.loc[big_ind, 'fact_soz'] = df.loc[big_ind, 'target'].values / df.loc[big_ind, 'soz'].values
        print('add big finish| approved, after', df['target'].sum() // 1e6)
        flag = True
        return flag, df
    else:
        print('add big finish| failed conditions')
        flag = False
        return flag, sale_loc_forecast


param_low = 1.1
param_5 = 0.1
param_10 = 0.15
param_15 = 0.17
param_20 = 0.22
param_25 = 0.25
param_max = 0.015
param_big = 0.2


def plan_preparation(forecast_df, sale_loc, plans, df, shipments, rare, param_border_25, param_low, param_5, param_10,
                     param_15, param_20, param_25, param_max, param_big):
    print('----------------------------------------')
    print(sale_loc, 'BEGIN')
    flag = False
    sale_loc_forecast = forecast_df[forecast_df['sale_loc'] == sale_loc]
    big_ind = sale_loc_forecast[(sale_loc_forecast['target'] > 5e6) & (sale_loc_forecast['fact_soz'] > 0.01)].index
    sale_loc_plan = plans[plans['sale_loc'] == sale_loc]['plan_value'].tolist()[0]
    init_relation = sale_loc_plan / sale_loc_forecast['target'].sum()
    init_diff = sale_loc_plan - sale_loc_forecast['target'].sum()
    sale_loc_forecast['fact_soz'] = sale_loc_forecast['target'] / sale_loc_forecast['soz']
    sale_loc_forecast['fact_soz'] = sale_loc_forecast['fact_soz'].fillna(1)
    sale_loc_forecast = sale_loc_forecast.drop_duplicates(subset='segment')
    big_init_rel = init_diff / sale_loc_forecast.loc[big_ind]['target'].sum()
    print('residual to big relation:', big_init_rel)
    if (big_init_rel < 0.24) and (big_init_rel > 0):
        flag, sale_loc_forecast = add_to_big_init(sale_loc_forecast, sale_loc_plan)
        return flag, sale_loc_forecast
    need_to_fill = sale_loc_plan - sale_loc_forecast['target'].sum()
    print('init relation:', np.round(init_relation, 2),
          'need to fill:', int(need_to_fill // 1e6),
          'now:', sale_loc_forecast['target'].sum() // 1e6)
    cur_rel = sale_loc_plan / sale_loc_forecast['target'].sum()
    if cur_rel < 1:
        sale_loc_forecast['target'] *= cur_rel
        flag = True
        return flag, sale_loc_forecast
    else:
        flag, sale_loc_forecast = add_border_to_very_low(sale_loc_forecast, sale_loc_plan,
                                                         param_low)
        if flag:
            return flag, sale_loc_forecast
        flag, sale_loc_forecast = add_border_0_to_5(sale_loc_forecast, sale_loc_plan, param_border_25,
                                                    param_low, param_5)
        if flag:
            return flag, sale_loc_forecast
        flag, sale_loc_forecast = add_border_0_to_10(sale_loc_forecast, sale_loc_plan, param_border_25,
                                                     param_low, param_5, param_10)
        if flag:
            return flag, sale_loc_forecast
        flag, sale_loc_forecast = add_border_0_to_15(sale_loc_forecast, sale_loc_plan, param_border_25,
                                                     param_low, param_5, param_10, param_15)
        if flag:
            return flag, sale_loc_forecast
        flag, sale_loc_forecast = add_border_0_to_20(sale_loc_forecast, sale_loc_plan, param_border_25,
                                                     param_low, param_5, param_10, param_15, param_20)
        if flag:
            return flag, sale_loc_forecast
        flag, sale_loc_forecast = add_border_0_to_25(sale_loc_forecast, sale_loc_plan, param_border_25,
                                                     param_low, param_5, param_10, param_15, param_20, param_25)
        if flag:
            return flag, sale_loc_forecast
        flag, sale_loc_forecast = add_border_0_to_max(sale_loc_forecast, sale_loc_plan, param_border_25,
                                                      param_low, param_5, param_10, param_15, param_20, param_25,
                                                      param_max)
        if flag:
            return flag, sale_loc_forecast
        flag, sale_loc_forecast = add_to_big_finish(sale_loc_forecast, sale_loc_plan, param_big)

        if flag:
            return flag, sale_loc_forecast
        flag, sale_loc_forecast = add_by_q(sale_loc_forecast, sale_loc_plan, df, shipments, rare)

        if flag:
            return flag, sale_loc_forecast
        flag, sale_loc_forecast = add_to_big_finish(sale_loc_forecast, sale_loc_plan, param_big)

        return flag, sale_loc_forecast


def add_by_q(sale_loc_forecast, sale_loc_plan, df, shipments, rare):
    rare_df = shipments[(shipments['segment'].isin(rare)) & (shipments['timestamp'] >= fill_q_begin)][
        ['segment', 'timestamp', 'target']]
    fill_q = df[df['timestamp'] >= fill_q_begin][['segment', 'timestamp', 'target']]
    fill_q = pd.concat((fill_q, rare_df))
    fill_q = fill_q.groupby('segment') \
        .apply(lambda x: x.set_index('timestamp')
               .reindex(pd.date_range(fill_q_begin, plan_month_fact, freq='MS'))) \
        .drop('segment', axis=1).fillna(0) \
        .reset_index().rename(columns={'level_1': 'timestamp'})
    low_segments = sale_loc_forecast[sale_loc_forecast['target'] == 30000].index
    for q in range(30, 80):
        cur_res_df = sale_loc_forecast \
            .join(fill_q.groupby('segment')['target'].quantile(q / 100),
                  on='segment', rsuffix='_q')
        cur_res_df.loc[cur_res_df['target_q'].isnull(), 'target_q'] = \
            cur_res_df[cur_res_df['target_q'].isnull()]['target'].values
        cur_res_df.loc[cur_res_df['target_q'] > cur_res_df['target'], 'corrected_plan'] = \
            cur_res_df[cur_res_df['target_q'] > cur_res_df['target']]['target_q'].values
        cur_res_df.loc[cur_res_df['target_q'] <= cur_res_df['target'], 'corrected_plan'] = \
            cur_res_df[cur_res_df['target_q'] <= cur_res_df['target']]['target'].values
        cur_res_df.loc[cur_res_df['target'] == 0, 'corrected_plan'] = \
            cur_res_df[cur_res_df['target'] == 0]['target'].values
        cur_res_df.loc[low_segments, 'corrected_plan'] = 30000
        cur_res_df['corrected_soz'] = cur_res_df['corrected_plan'] / cur_res_df['soz']
        flag = cur_res_df['corrected_plan'].sum() > sale_loc_plan
        if flag:
            print('q approved after:', cur_res_df['corrected_plan'].sum() // 1e6)
            need_to_fill_diff = sale_loc_plan - cur_res_df['target'].sum()
            cur_res_df['diff'] = cur_res_df['corrected_plan'] - cur_res_df['target']
            cur_res_df['diff'] *= need_to_fill_diff / cur_res_df['diff'].sum()
            cur_res_df['target'] += cur_res_df['diff']
            cur_res_df['fact_soz'] = cur_res_df['target'] / cur_res_df['soz']
            cur_res_df['fact_soz'] = cur_res_df['fact_soz'].fillna(1)
            print('q {0}| approved'.format(q))
            cur_res_df = cur_res_df.drop(['target_q', 'corrected_plan', 'corrected_soz'], axis=1)
            return flag, cur_res_df

    print('q | failed, final sum: {0}'.format(cur_res_df['corrected_plan'].sum() // 1e6), )
    cur_res_df['target'] = cur_res_df['corrected_plan'].values
    cur_res_df['fact_soz'] = cur_res_df['corrected_soz'].values
    cur_res_df['fact_soz'] = cur_res_df['fact_soz'].fillna(1)
    cur_res_df = cur_res_df.drop(['target_q', 'corrected_plan', 'corrected_soz'], axis=1)
    return flag, cur_res_df


def all_filials_plan_prep(forecast_df, plans_filially, df, shipments, rare):
    param_low = 1.1
    param_5 = 0.1
    param_10 = 0.15
    param_15 = 0.2
    param_20 = 0.25
    param_25 = 0.25
    param_max = 0.05
    param_big = 0.15
    param_border_25 = 0.25
    res_dict = dict()
    res_df = pd.DataFrame()

    for sale_loc in forecast_df['sale_loc'].unique():
        flag, sale_loc_forecast = plan_preparation(forecast_df, sale_loc, plans_filially, df, shipments, rare,
                                                   param_border_25,
                                                   param_low, param_5, param_10, param_15,
                                                   param_20, param_25, param_max, param_big)

        res_dict[sale_loc] = flag
        if flag:
            res_df = pd.concat((res_df, sale_loc_forecast))

    bad_sale_loc = list({k: v for k, v in res_dict.items() if v==False}.keys())
    param_max = 0.1
    param_big = 0.18
    for sale_loc in bad_sale_loc:
        for i in range(1, 7):
            flag, sale_loc_forecast = plan_preparation(forecast_df, sale_loc, plans_filially, df, shipments, rare,
                                                       param_border_25,
                                                       param_low + i / 50,
                                                       param_5,
                                                       param_10 + i / 50,
                                                       param_15 + i / 50,
                                                       param_20 + i / 50,
                                                       param_25,
                                                       param_max + i / 50,
                                                       param_big + i / 50, )
            if flag or i == 6:
                print(i, 'DONE')
                res_dict[sale_loc] = flag
                res_df = pd.concat((res_df, sale_loc_forecast))
                break

    return res_df[['segment', 'target', 'sale_loc']], res_df