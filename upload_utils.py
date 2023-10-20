import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from prepro_utils import *

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
    border_stm_percentage = 0.03
    low_val = 30000
    low_val_stm = 500
    low_val_hmp_oz = 500
    low_val_hmp_ons = 500
    freq = 'MS'
    horizon = 1
    param_max_stm = 0.01
    # exog_cols = ['web_order_rel', 'volume', 'weight', 'density', 'med_amn', 'nds', 'position_count']
    # cat_features = ['sale_deptno', 'warehouse_oebs_org_id', 'type_name', 'manager']
    # rare_ignore_cols = ['manager']
    exog_cols = ['web_order_rel', 'volume', 'weight', 'density', 'med_amn', 'nds', 'position_count', 'market_rel']
    cat_features = [ 'CURRENT_SALE_LOC', 'ZDRAVSITI', 'SUBSEGMENT_TP',
                    'FACT_REG_CAPITAL', 'WORK_WITH_STRONG_MED', 'segment',
                    'LICENSE_TP', 'FACT_FEDERAL', 'manager', 'CURRENT_SOUZFARMA_INET_FLAG',
                    ]
    int_cat_cols = ['VALUE', 'LTK', 'CLAIM_PERIOD', 'TURNOVER','PERIOD_SHIPMENT','CURRENT_DISCOUNT_2K']
    rare_ignore_cols = ['manager', 'segment']
    ms_stat_features = ['mad_3', 'mean_3', 'std_3', 'minmax_3', 'minmax_mean_rel_3', 'minmax_rel_3', 'minmax_mean_rel_3_flag',
                        'min_rel_3', 'max_rel_3', 'mean_32' ]
    n_jobs = 1
    n_folds = 3
    return str(plan_month_fact), churn_dt, plan_month_table, hmp_upload_dt_start, fill_q_begin, \
            min_dt, max_dt, max_dt_wd_holidays, holdout_dt, low_val, low_val_stm, low_val_hmp_oz, \
            low_val_hmp_ons, freq, horizon, exog_cols, cat_features, rare_ignore_cols, ms_stat_features, int_cat_cols,  n_folds, n_jobs, border_stm_percentage, param_max_stm

plan_month_fact = pd.read_csv('plan_month_fact.txt')['plan_month_fact'].values.tolist()[0]
plan_month_fact, churn_dt, plan_month_table, hmp_upload_dt_start, fill_q_begin, \
min_dt, max_dt, max_dt_wd_holidays, holdout_dt, low_val, low_val_stm, low_val_hmp_oz, \
low_val_hmp_ons, freq, horizon, exog_cols, cat_features, rare_ignore_cols, ms_stat_features, int_cat_cols, n_folds, n_jobs, border_stm_percentage, param_max_stm = get_constants(plan_month_fact)

def upload_cards(client, plan_month_table):
    query = '''
            select vppl.SITE_ID_LINK as SITE_ID_LINK, vppl.VALUE as VALUE, vppl.PLAN_PERIOD as PLAN_PERIOD ,
            vppl.CUSTOMER_NAME as CUSTOMER_NAME, sc.TURNOVER as TURNOVER, vppl.CURRENT_SALE_LOC as CURRENT_SALE_LOC,
            sc.CUST_BILL_STATUS, dpc.CURRENT_ACTIVITY_CATEGORY , dpc.SITE_STATUS , vppl.CURRENT_SALE_LOC,
            dpc.OPF_SHORT_NAME,dpc.LICENSE_EXP_DATE,dpc.FIRST_SHIPMENT_DATE,
            dpc.FACT_FULL_ADDRESS, dpc.FACT_FEDERAL as FACT_FEDERAL,
            dpc.FACT_REG_NAME, dpc.FACT_REG_TP_FULL_NAME, dpc.LTK as LTK, 
            dpc.FACT_REG_CAPITAL as FACT_REG_CAPITAL, dpc.SUBSEGMENT_TP as SUBSEGMENT_TP,dpc.HISTORY_MANAGER_SECTION_NAME, 
            dpc.HISTORY_MANAGER_FULL_NAME, dpc.CURRENT_MANAGER_FULL_NAME as manager, dpc.RIGLA_SQUARE, 
            dpc.CURRENT_MANAGER_SECTION_NAME,  dpc.CURRENT_GRACE_NAME, dpc.CURRENT_DISCOUNT_2K as CURRENT_DISCOUNT_2K, 
            dpc.FACT_ADDRESS_STREET, dpc.WORK_WITH_STRONG_MED as WORK_WITH_STRONG_MED, dpc.WHOLE_SALE, 
            dpc.CURRENT_SOUZFARMA_INET_FLAG as CURRENT_SOUZFARMA_INET_FLAG,
            dpc.FACT_ADDRESS_HOUSE,dpc.FACT_ADDR_POST_CODE,dpc.CURRENT_AS_NAME,
            dpc.USE_ACTION, dpc.LICENSE_TP as LICENSE_TP, dpc.ZDRAVSITI as ZDRAVSITI, dpc.PERIOD_SHIPMENT as PERIOD_SHIPMENT,
            dpc.CURRENT_ASSOCIATION_AS_NAME, 
            dpc.CLAIM_PERIOD as CLAIM_PERIOD, dpc.LAST_SHIPMENT_DATE as LAST_SHIPMENT_DATE, dpc.WAREHOUSE_PRESENCE,
            dpc.SITE_USE_ID as segment, dpc.BILL_TO_SITE_USE_ID, dpc.SMS_MAILING 
            from CUST.v_plan_partners_links vppl 
            left join `default`.sus_customers sc 
            on vppl.CUSTOMERS = sc.CUSTOMERS 
            left join default.dct_partner_cur dpc 
            on vppl.SITE_ID_LINK = dpc.SITE_USE_ID 
            where vppl.PLAN_PERIOD = toDate('{0}')
            and sc.CUST_BILL_STATUS = 'A'
            and dpc.SITE_STATUS = 'A'
            and dpc.CURRENT_ACTIVITY_CATEGORY = 'Коммерция'
            '''.format(plan_month_table)
    df, cols = client.execute(query, with_column_types=True)
    df = pd.DataFrame(df, columns=[col[0] for col in cols])
    old_managers_dir = df[['manager', 'segment']]
    df['manager'] = df['manager'].apply(lambda x: re.sub(r'[0-9,()]+', '', x).lower())
    df['TURNOVER'] = df.groupby(['SITE_ID_LINK'])['TURNOVER'].transform('mean').values
    df = df.drop_duplicates(subset=['SITE_ID_LINK'], keep='last')
    return df, df[cat_features], old_managers_dir

def init_prepro_light(shipments, churn_dt, low_val):
    shipments['timestamp'] = pd.to_datetime(shipments['timestamp'])
    churned = shipments.groupby('segment')\
            .filter(lambda x: x[x['timestamp']>=pd.to_datetime(churn_dt)]['target'].sum()==0)['segment'].unique()
    df = shipments[~shipments['segment'].isin(churned)]
    rare = set(df.groupby('segment').filter(lambda x: x['timestamp'].nunique()<6)['segment'].unique())
    low = set(df.groupby('segment')\
            .filter(lambda x: x[x['timestamp']>=pd.to_datetime(churn_dt)]['target'].max()<=low_val)['segment'].unique())
    rare = rare.difference(low)
    to_drop = list(rare.union(low))
    df = df[~df['segment'].isin(to_drop)]
    return df, rare, low, churned

def upload_shipments(client, min_dt, max_dt, mode = 'entire_forecast'):
    if mode == 'entire_forecast':
        query = \
        '''
        select 
            site_use_id as segment,
            toStartOfMonth(ordered_day) as timestamp,
            sum(cast(cost_rub as int)) as target,
            avg(volume) as volume,
            avg(weight) as weight,
            volume / weight as density,
            avg(med_amn) as med_amn,
            avg(nds_rate) as nds,
            sum(if(web_order_flag=1, 1, 0))/count(*) as web_order_rel,
            sum(if(dlo_id=1, 1, 0))/count(*) as market_rel,
            count (distinct cd_m) as position_count
        from default.dmt_shipment ds 
        where 1=1
            and timestamp between toDate('{0}') and toDate('{1}')
            and oper_tp_id = 1
            and segment is not NULL
        group by 
            segment, timestamp
        '''.format(min_dt, max_dt,)
    elif mode == 'stm_forecast':
        query = \
        '''
        select 
            site_use_id as segment,
            toStartOfMonth(ordered_day) as timestamp,
            sum(cast(cost_rub as int)) as target
        from 
            default.dmt_shipment ds     
        where 1=1
            and timestamp between toDate('{0}') and toDate('{1}')
            and cast (cd_m as int) in (select
                                       distinct cast ( MEDICINE_CODE as int)
                                       from `default`.dct_medicine_cur dmc 
                                       where 1 = 1
                                            and OWN_TR_MARK_PROTEK = 1)
            and oper_tp_id = 1
            and segment is not NULL
        group by
            segment, timestamp
        '''.format(min_dt, max_dt)
    df, cols = client.execute(query, with_column_types=True)
    df = pd.DataFrame(df, columns=[col[0] for col in cols])
    df['segment'] = df['segment'].astype(int)
    return df

def resample_light(df, freq):
    df = df.groupby('segment').apply(lambda x: x.set_index('timestamp').resample(freq).sum())\
           .drop('segment', axis=1).reset_index()
    return df

def upload_divide_resample(client, mode):
    if mode == 'entire_forecast':
        mode_low_val = low_val
    elif mode == 'stm_forecast':
        mode_low_val = low_val_stm
    shipments = upload_shipments(client, min_dt, max_dt, mode=mode)
    cards, cats_df, old_managers_dir = upload_cards(client, plan_month_table)
    shipments = shipments[shipments['segment'].isin(cards['SITE_ID_LINK'].unique())]
    df, rare, low, churned = init_prepro_light(shipments, churn_dt, mode_low_val)
    df = resample_light(df, freq)
    return cards, shipments, cats_df, old_managers_dir, df, rare, low, churned

def merge_other_subsets(forecast_df, low, rare, churned, shipments, low_val):
    rare_df = shipments[(shipments['segment'].isin(rare))&(shipments['timestamp']>=churn_dt)]\
                .groupby('segment', as_index=False)['target'].mean()
    rare_df.loc[rare_df['target']<low_val, 'target'] = low_val
    low_df = pd.DataFrame({'segment':list(low), 'target': [low_val]*len(low)})
    churned_df = pd.DataFrame({'segment':list(churned), 'target': [0]*len(churned)})
    forecast_df = pd.concat((forecast_df, low_df, churned_df, rare_df))
    forecast_df['segment'] = forecast_df['segment'].astype(int)
    forecast_df = forecast_df.reset_index(drop=True)
    return forecast_df

def upload_sale_loc_dict(client):
    query = '''SELECT anyLast(CURRENT_SALE_LOC) as sale_loc, SITE_USE_ID as segment
    FROM default.dct_partner_cur
    where SITE_USE_ID is not NULL
    GROUP BY SITE_USE_ID'''
    sale_loc_dir, cols = client.execute(query, with_column_types=True)
    sale_loc_dir = pd.DataFrame(sale_loc_dir, columns = [col[0] for col in cols])
    sale_loc_dir['segment'] = sale_loc_dir['segment'].astype(int)
    sale_loc_dict = sale_loc_dir.set_index('segment').to_dict()['sale_loc']
    return sale_loc_dict