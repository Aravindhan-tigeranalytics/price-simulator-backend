import os

from django.db.models.fields import mixins
import scenario_planner
from typing import List
import numpy as np
import pandas as pd
import re
import time
import datetime
import math
import itertools
# import openpyxl
from itertools import chain
from joblib import Parallel, delayed
import statistics 
from pulp import *
import pandas as pd
from itertools import combinations
from utils import constants as CONST
import copy
from optimiser import process as pr
from utils import units_calculation as cal
from optimiser import process as process_calc
import logging.config

from scenario_planner import mixins as mixin
# from optimiser import testdata as value_dict

logging.config.dictConfig(CONST.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

value_dict = {}

def get_promo_wave_values(tpr_discount):
    no_of_promo = 0
    max_promo = 0
    min_promo = 0
    duration_of_waves = []
    waves = []
    min_consecutive_promo = 53
    max_consecutive_promo = 0
    gap = []
    promo_gap = []
    max_length_gap = 0
    min_length_gap = 53
    tot_promo_min = 0
    tot_promo_max = 0

    val = tpr_discount
    for i in range(0, len(val)):

        if val[i]:
            waves.append(val[i])
            if i + 1 == len(val):
                duration_of_waves.append(waves)
                if len(waves) > max_consecutive_promo:
                    max_consecutive_promo = len(waves)
                if len(waves) < min_consecutive_promo:
                    min_consecutive_promo = len(waves)
            no_of_promo = no_of_promo + 1
        else:
            gap.append(val[i])
            if i + 1 == len(val):
                promo_gap.append(gap)
                if len(gap) < min_length_gap:
                    min_length_gap = len(gap)

        if not val[i] and val[(0 if i - 1 < 0 else i - 1)]:
            duration_of_waves.append(waves)
            if len(waves) > max_consecutive_promo:
                max_consecutive_promo = len(waves)
            if len(waves) < min_consecutive_promo:
                min_consecutive_promo = len(waves)
            waves = []
        elif val[i] and not val[(0 if i - 1 < 0 else i - 1)]:
            promo_gap.append(gap)
            if len(gap) < min_length_gap:
                min_length_gap = len(gap)
            gap = []

        if max_promo > val[i]:
            max_promo = val[i]
        if min_promo < val[i]:
            min_promo = val[i]
    tot_promo_max = no_of_promo + 3
    tot_promo_min = no_of_promo - 3
    print (max_consecutive_promo, 'maximum consecutive  promo')
    print (min_consecutive_promo, 'minimum consecutive promo')
    print (min_length_gap, 'promo gap')
    print (tot_promo_min, 'total promo min')
    print (tot_promo_max, 'total promo max')
    print (duration_of_waves, 'duration of waves')
    print (len(duration_of_waves), 'duration of waves length')
    print (no_of_promo, 'no of promo')
    return (
        min_consecutive_promo,
        max_consecutive_promo,
        min_length_gap,
        tot_promo_min,
        tot_promo_max,
        no_of_promo,
        len(duration_of_waves),
        )

class my_dictionary(dict):
    
    # __init__ function

    def __init__(self):
        self = dict()

    # Function to add key:value

    def add(self, key, value):
        self[key] = value

def _update_params(config , request_value):
    '''
    update the optimizer parameter from request
    '''

    MINIMIZE_PARAMS = ['Trade_Expense']

    config['Retailer'] = request_value['account_name']
    config['PPG'] = request_value['product_group']
    config['MARS_TPRS'] =  request_value['mars_tpr']
    config['Co_investment'] = request_value['co_investment']
    config['Objective_metric'] = request_value['objective_function']
    config['Objective'] = 'Minimize' if(request_value['objective_function'] in MINIMIZE_PARAMS) else 'Maximize'
    config['Segment'] = request_value['corporate_segment']
    config['Fin_Pref_Order']=request_value['fin_pref_order']
    config['Promo_Mech']=request_value['promo_mech']
    config['constrain_params']['MAC'] = request_value['param_mac']
    config['constrain_params']['RP'] = request_value['param_rp']
    config['constrain_params']['Trade_Expense'] = request_value['param_trade_expense']
    config['constrain_params']['Units'] = request_value['param_units']
    config['constrain_params']['NSV'] = request_value['param_nsv']
    config['constrain_params']['GSV'] = request_value['param_gsv']
    config['constrain_params']['Sales'] = request_value['param_sales']
    config['constrain_params']['MAC_Perc'] = request_value['param_mac_perc']
    config['constrain_params']['RP_Perc'] = request_value['param_rp_perc']
    config['constrain_params']['min_consecutive_promo'] = request_value['param_min_consecutive_promo']
    config['constrain_params']['max_consecutive_promo'] = request_value['param_max_consecutive_promo']
    config['constrain_params']['promo_gap'] = request_value['param_promo_gap']
    config['constrain_params']['tot_promo_min'] = request_value['param_total_promo_min']
    config['constrain_params']['tot_promo_max'] = request_value['param_total_promo_max']
    config['constrain_params']['compul_no_promo_weeks'] = request_value['param_compulsory_no_promo_weeks']
    config['constrain_params']['compul_promo_weeks'] =request_value['param_compulsory_promo_weeks']
    
    config['config_constrain']['MAC'] = request_value['config_mac']
    config['config_constrain']['RP'] =  request_value['config_rp']
    config['config_constrain']['Trade_Expense'] = request_value['config_trade_expense']
    config['config_constrain']['Units'] = request_value['config_units']
    config['config_constrain']['NSV'] =request_value['config_nsv']
    config['config_constrain']['GSV'] = request_value['config_gsv']
    config['config_constrain']['Sales'] =request_value['config_sales']
    config['config_constrain']['MAC_Perc'] = request_value['config_mac_perc']
    config['config_constrain']['RP_Perc'] = request_value['config_rp_perc']
    config['config_constrain']['min_consecutive_promo'] = request_value['config_min_consecutive_promo']
    config['config_constrain']['max_consecutive_promo'] = request_value['config_max_consecutive_promo']
    config['config_constrain']['promo_gap'] = request_value['config_promo_gap']


def predict_sales(coeffs, data):
    
    predict = 0
    for i in coeffs['names']:
        if i == 'Intercept':
            predict = predict + coeffs[coeffs['names']
                    == i]['model_coefficients'].values
        else:

            predict = predict + data[i] * coeffs[coeffs['names']
                    == i]['model_coefficients'].values
    data['pred_vol'] = predict
    data['Predicted_Volume'] = np.exp(data['pred_vol'])
    return data['Predicted_Volume']


##### Promo wave/slot calculation#######  

def promo_wave_cal(tpr_data):
    tpr_data['Promo_wave'] = 0
    c = 1
    i = 0
    while i <= tpr_data.shape[0] - 1:
        if tpr_data.loc[i, 'tpr_discount_byppg'] > 0:  # ####Also tpr ??since in validation consdered tpr
            tpr_data.loc[i, 'Promo_wave'] = c
            j = i + 1
            if j == tpr_data.shape[0]:
                break
            while (j <= tpr_data.shape[0] - 1) & (tpr_data.loc[j,
                    'tpr_discount_byppg'] > 0):
                tpr_data.loc[j, 'Promo_wave'] = c
                i = j + 1
                j = i
                if j == tpr_data.shape[0]:
                    break
            c = c + 1
        i = i + 1
    return tpr_data['Promo_wave']

def _predict(
    pred_df,
    model_coef,
    var_col='model_coefficient_name',
    coef_col='model_coefficient_value',
    intercept_name='(Intercept)',
    ):
    """Predict Sales.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Dataframe consisting of IDVs
    model_coef : pd.DataFrame
        Dataframe consisting of model coefficient and its value
    var_col : str
        Column containing model variable
    coef_col : str
        Column containing model variables and their coefficient estimate
    intercept_name : str
        Name of intercept variable

    Returns
    -------
    ndarray
    """

    idv_cols = [col for col in model_coef[var_col] if col
                != intercept_name]
    idv_coef = model_coef.loc[model_coef[var_col].isin(idv_cols),
                              coef_col]
    idv_df = pred_df.loc[:, idv_cols]

    intercept = model_coef.loc[~model_coef[var_col].isin(idv_cols),
                               coef_col].to_list()[0]
    prediction = idv_df.values.dot(idv_coef.values) + intercept

    return prediction


def set_baseline_value(
    model_df,
    model_coef,
    var_col='model_coefficient_name',
    coef_col='model_coefficient_value',
    intercept_name='(Intercept)',
    baseline_value=None,
    ):
    """Set baseline value.

    Parameters
    ----------
    model_df : pd.DataFrame
        Dataframe consisting of IDVs
    model_coef : pd.DataFrame
        Dataframe consisting of model variables and their coefficient estimate
    var_col : str
        Column containing model variable
    coef_col : str
        Column containing model variables' coefficient estimate
    intercept_name : str
        Name of intercept variable
    baseline_value : pd.DataFrame
        Dataframe consisting of IDVs with baseline value

    Method
    ------
    # If action is Rolling Max, then take max value in the last n weeks.
    # If action is Set Average, then baseline value is set to the value specified.
    # If action is As is, then it is taken as is.
    # Variables not specified in the baseline file are set to 0.

    Returns
    -------
    pd.DataFrame
    """

    # Get baseline value

    if baseline_value is None:
        baseline_value = {
            'wk_sold_median_base_price_byppg_log': ['Rolling Max', 26],
            'tpr_discount_byppg': ['Set Average', 0],
            'tpr_discount_byppg_lag1': ['Set Average', 0],
            'tpr_discount_byppg_lag2': ['Set Average', 0],
            'ACV_Feat_Only': ['Set Average', 0],
            'ACV_Disp_Only': ['Set Average', 0],
            'ACV_Feat_Disp': ['Set Average', 0],
            'ACV_Selling': ['As is', 0],
            'flag_qtr2': ['As is', 0],
            'flag_qtr3': ['As is', 0],
            'flag_qtr4': ['As is', 0],
            'category_trend': ['As is', 0],
            'monthno': ['As is', 0],
            }
        baseline_value = pd.DataFrame.from_dict(baseline_value,
                orient='index').reset_index()
        baseline_value = \
            baseline_value.rename(columns={'index': 'Parameters',
                                  0: 'Action', 1: 'Value'})

        baseline_value = baseline_value[baseline_value['Parameters'
                ].isin(model_coef[var_col])]
        comp_features = \
            model_coef.loc[~model_coef['model_coefficient_name'
                           ].isin(baseline_value['Parameters'
                           ].to_list() + [intercept_name]), [var_col]]

        if len(comp_features) > 0:
            comp_features = \
                comp_features.rename(columns={'model_coefficient_name': 'Parameters'
                    })
            comp_features['Action'] = comp_features.apply(lambda x: \
                    ('Set Average' if 'PromotedDiscount'
                    in x['Parameters'] else 'As is'), axis=1)
            comp_features['Action'] = comp_features.apply(lambda x: \
                    ('Rolling Max' if 'RegularPrice' in x['Parameters'
                    ] else x['Action']), axis=1)
            comp_features['Value'] = comp_features.apply(lambda x: \
                    (26 if 'Rolling Max' in x['Action'] else 0), axis=1)
            baseline_value = baseline_value.append(comp_features,
                    ignore_index=True)

    # Initialization....

    base_var = [intercept_name] \
        + baseline_value.loc[baseline_value['Action'
                             ].isin(['Rolling Max', 'As is']),
                             'Parameters'].to_list()
    baseline_df = model_df.loc[:,
                               model_df.columns.isin(model_coef[var_col])].copy()

    # Set baseline value

    for i in baseline_value['Parameters'].to_list():
        action = baseline_value.loc[baseline_value['Parameters'] == i,
                                    'Action'].to_list()[0]
        value = baseline_value.loc[baseline_value['Parameters'] == i,
                                   'Value'].to_list()[0]
        if action == 'Set Average':
            baseline_df[i] = value
        elif action == 'As is':
            baseline_df[i] = model_df[i]
        elif action == 'Rolling Max':
            tmp_df = model_df[['Date', i]].set_index('Date')
            tmp_df['Final_baseprice'] = tmp_df[i].rolling(value,
                    min_periods=1).max()

            # TODO: Check do price variants make sense for other parameters
            # tmp_df["Final_baseprice"] = tmp_df.apply(lambda x: x["median_baseprice"] if x["Final_baseprice"] == (-np.Inf) else x["Final_baseprice"], axis=1)
            # tmp_df["Final_baseprice"] = tmp_df.apply(lambda x: x["wk_sold_avg_price_byppg"] if np.isnan(x["Final_baseprice"]) else x["Final_baseprice"], axis=1)

            baseline_df[i] = tmp_df['Final_baseprice'
                                    ].reset_index(drop=True)
        else:
            pass

            # Do nothing

            # print ()

    # Parameters not specified in  Baseline condition

    other_params = [i for i in baseline_df.columns if i
                    not in baseline_value['Parameters'].to_list() and i
                    != intercept_name]
    baseline_df.loc[:, other_params] = 0

    return (base_var, baseline_df)


def get_var_contribution_wt_baseline_defined(
    df,
    model_coef,
    wk_sold_price,
    var_col='model_coefficient_name',
    coef_col='model_coefficient_value',
    intercept_name='(Intercept)',
    baseline_value=None,
    ):
    """Get variable contribution with baseline defined.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe consisting of IDVs
    model_coef : pd.DataFrame
        Dataframe consisting of model variables and their coefficient estimate
    wk_sold_price : pd.DataFrame
        Dataframe consisting of weekly sold price        
    var_col : str
        Column containing model variable
    coef_col : str
        Column containing model variables' coefficient estimate
    intercept_name : str
        Name of intercept variable
    baseline_value : pd.DataFrame
        Dataframe consisting of IDVs with baseline value

    Returns
    -------
    tuple of pd.DataFrame
    """

    # Predict Sales

    ppg_cols = ['PPG_Cat', 'PPG_MFG', 'PPG_Item_No', 'PPG_Description']
    model_df = df.drop(columns=ppg_cols).copy()
    model_df['Predicted_sales'] = np.exp(_predict(model_df, model_coef,
            var_col=var_col, coef_col=coef_col,
            intercept_name=intercept_name))

    # Predict Baseline Sales

    tmp_model_df = model_df.copy()
    (base_var, baseline_df) = set_baseline_value(
        tmp_model_df,
        model_coef,
        var_col=var_col,
        coef_col=coef_col,
        intercept_name=intercept_name,
        baseline_value=baseline_value,
        )
    baseline_df['Predicted_baseline_sales'] = \
        np.exp(_predict(baseline_df, model_coef, var_col=var_col,
               coef_col=coef_col, intercept_name=intercept_name))
    model_df['Predicted_baseline_sales'] = \
        baseline_df['Predicted_baseline_sales']
    model_df['incremental_predicted'] = model_df['Predicted_sales'] \
        - model_df['Predicted_baseline_sales']

    # Calculate raw contribution

    model_df[intercept_name] = 1
    baseline_df[intercept_name] = 1
    pred_xb = _predict(model_df, model_coef, var_col=var_col,
                       coef_col=coef_col, intercept_name=intercept_name)
    rc_sum = 0
    abs_sum = 0

    for i in model_coef[var_col]:
        rc = 0
        rc = np.exp(pred_xb) - np.exp(pred_xb - model_df[i]
                * model_coef.loc[model_coef[var_col] == i,
                coef_col].to_list()[0] + baseline_df[i]
                * model_coef.loc[model_coef[var_col] == i,
                coef_col].to_list()[0])
        model_df[i + '_' + 'rc'] = rc
        rc_sum = rc_sum + rc
        abs_sum = abs_sum + abs(rc)

    # Calculate actual contribution

    y_b_s = model_df['incremental_predicted'] - rc_sum
    unit_dist_df = model_df[['Date', 'Predicted_sales',
                            'Predicted_baseline_sales']].copy()
    for i in model_coef[var_col]:
        rc = model_df[i + '_' + 'rc']
        ac = rc + abs(rc) / abs_sum * y_b_s
        i_adj = (i + '_contribution_inc_base' if i in base_var else i
                 + '_contribution_impact')
        unit_dist_df[i_adj] = ac
    unit_dist_df = unit_dist_df.fillna(0)

    # Get Dollar Sales

    price_dist_df = unit_dist_df.copy()
    numeric_cols = price_dist_df.select_dtypes(include='number'
            ).columns.to_list()
    price_dist_df[numeric_cols] = \
        price_dist_df[numeric_cols].mul(wk_sold_price.values, axis=0)

    # Get variable contribution variants

    (dt_unit_dist_df, qtr_unit_dist_df, yr_unit_dist_df) = \
        get_var_contribution_variants(unit_dist_df,
            'model_coefficient_name', value_col_name='units')
    (dt_price_dist_df, qtr_price_dist_df, yr_price_dist_df) = \
        get_var_contribution_variants(price_dist_df,
            'model_coefficient_name', value_col_name='price')
    overall_dt_dist_df = dt_unit_dist_df.merge(dt_price_dist_df,
            how='left', on=['Date', 'model_coefficient_name'])
    overall_qtr_dist_df = qtr_unit_dist_df.merge(qtr_price_dist_df,
            how='left', on=['Quarter', 'model_coefficient_name'])
    overall_yr_dist_df = yr_unit_dist_df.merge(yr_price_dist_df,
            how='left', on=['Year', 'model_coefficient_name'])

    ppg_df = df[ppg_cols].drop_duplicates().values.tolist()
    overall_dt_dist_df[ppg_cols] = pd.DataFrame(ppg_df,
            index=overall_dt_dist_df.index)
    overall_qtr_dist_df[ppg_cols] = pd.DataFrame(ppg_df,
            index=overall_qtr_dist_df.index)
    overall_yr_dist_df[ppg_cols] = pd.DataFrame(ppg_df,
            index=overall_yr_dist_df.index)

    return (overall_dt_dist_df, overall_qtr_dist_df, overall_yr_dist_df)

def get_var_contribution_wo_baseline_defined(
    df,
    model_coef,
    wk_sold_price,
    all_df=None,
    var_col='model_coefficient_name',
    coef_col='model_coefficient_value',
    intercept_name='(Intercept)',
    base_var=None,
    ):
    """Get variable contribution without baseline defined.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe consisting of IDVs
    model_coef : pd.DataFrame
        Dataframe consisting of model variables and their coefficient estimate
    wk_sold_price : pd.DataFrame
        Dataframe consisting of weekly sold price
    all_df : pd.DataFrame
        Dataframe consisting of IDVs of all PPGs        
    var_col : str
        Column containing model variable
    coef_col : str
        Column containing model variables' coefficient estimate
    intercept_name : str
        Name of intercept variable
    base_var : list of str
        Base variables

    Returns
    -------
    tuple of pd.DataFrame
    """

    # Predict Sales

    ppg_cols = ['PPG_Cat', 'PPG_MFG', 'PPG_Item_No', 'PPG_Description']
    ppg = df['PPG_Item_No'].to_list()[0]
    model_df = df.drop(columns=ppg_cols).copy()
    model_df['Predicted_sales'] = np.exp(_predict(model_df, model_coef,
            var_col=var_col, coef_col=coef_col,
            intercept_name=intercept_name))

    # Get base and impact variables

    if base_var is None:
        tpr_features = [i for i in model_coef[var_col]
                        if 'PromotedDiscount' in i]
        rp_features = [i for i in model_coef[var_col] if 'RegularPrice'
                       in i]
        base_var = ['wk_sold_median_base_price_byppg_log'] \
            + rp_features + [
            'ACV_Selling',
            'category_trend',
            'flag_qtr2',
            'flag_qtr3',
            'flag_qtr4',
            'monthno',
            ]
        base_var = [i for i in base_var if i
                    in model_coef[var_col].to_list()]

    base_var = [intercept_name] + base_var
    impact_var = [i for i in model_coef[var_col] if i not in base_var]

    # Get base and impact variables

    model_df[intercept_name] = 1
    tmp_model_coef = model_coef[model_coef[var_col].isin(base_var)]
    tmp_model_df = model_df.copy()
    if all_df is not None:
        if 'wk_sold_median_base_price_byppg_log' \
            in tmp_model_df.columns:
            tmp_model_df = \
                tmp_model_df.merge(all_df.loc[all_df['PPG_Item_No']
                                   == ppg, ['Date', 'Final_baseprice'
                                   ]].drop_duplicates(), how='left',
                                   on='Date')
            tmp_model_df['Final_baseprice'] = \
                tmp_model_df['Final_baseprice'].astype(np.float64)
            tmp_model_df['wk_sold_median_base_price_byppg_log'] = \
                np.log(tmp_model_df['Final_baseprice'])
            tmp_model_df = tmp_model_df.drop(columns=['Final_baseprice'
                    ])
        if tmp_model_df.columns.str.contains('.*RegularPrice.*RegularPrice'
                , regex=True).any():
            rp_interaction_cols = [col for col in tmp_model_df.columns
                                   if re.search('.*RegularPrice.*RegularPrice'
                                   , col) is not None]
            for col in rp_interaction_cols:
                col_adj = re.sub(ppg, '', col)
                col_adj = re.sub('_RegularPrice_', '', col_adj)
                col_adj = re.sub('_RegularPrice', '', col_adj)
                tmp_model_df = \
                    tmp_model_df.merge(all_df.loc[all_df['PPG_Item_No']
                        == ppg, ['Date', 'Final_baseprice']], how='left'
                        , on='Date')
                temp = all_df.loc[all_df['PPG_Item_No'] == col_adj,
                                  ['Date',
                                  'wk_sold_median_base_price_byppg_log'
                                  ]]
                temp = \
                    temp.rename(columns={'wk_sold_median_base_price_byppg_log': 'wk_price_log'
                                })
                tmp_model_df = tmp_model_df.merge(temp[['Date',
                        'wk_price_log']], how='left', on='Date')
                tmp_model_df[col] = \
                    np.log(tmp_model_df['Final_baseprice']) \
                    * tmp_model_df['wk_price_log']
                tmp_model_df = \
                    tmp_model_df.drop(columns=['Final_baseprice',
                        'wk_price_log'])
    base_val = \
        tmp_model_df[tmp_model_coef[var_col].to_list()].values.dot(tmp_model_coef[coef_col].values)

    tmp_model_coef = model_coef[model_coef[var_col].isin(impact_var)]
    impact_val = \
        model_df[tmp_model_coef[var_col].to_list()].values.dot(tmp_model_coef[coef_col].values)

    model_df['baseline_contribution'] = np.exp(base_val)
    model_df['incremental_contribution'] = model_df['Predicted_sales'] \
        - model_df['baseline_contribution']

    # Calculate raw contribution for impact variables

    row_sum = 0
    abs_sum = 0
    for i in impact_var:
        tmp_impact_val = model_df[i].values \
            * model_coef.loc[model_coef[var_col] == i,
                             coef_col].to_list()[0]
        model_df[i + '_contribution_impact'] = np.exp(base_val
                + impact_val) - np.exp(base_val + impact_val
                - tmp_impact_val)
        row_sum = row_sum + model_df[i + '_contribution_impact']
        abs_sum = abs_sum + abs(model_df[i + '_contribution_impact'])

    y_b_s = model_df['incremental_contribution'] - row_sum
    impact_contribution = model_df[['Date', 'Predicted_sales']].copy()
    for i in impact_var:
        i_adj = i + '_contribution_impact'
        impact_contribution[i_adj] = model_df[i_adj] \
            + abs(model_df[i_adj]) / abs_sum * y_b_s

    # Calculate raw contribution for base variables

    base_rc = model_coef.loc[model_coef[var_col] == intercept_name,
                             coef_col].to_list()[0]
    impact_contribution[intercept_name + '_contribution_base'] = \
        np.exp(base_rc)
    for i in base_var[1:]:
        t = tmp_model_df[i] * model_coef.loc[model_coef[var_col] == i,
                coef_col].to_list()[0] + base_rc
        impact_contribution[i + '_contribution_base'] = np.exp(t) \
            - np.exp(base_rc)
        base_rc = t
    impact_contribution = impact_contribution.fillna(0)
    unit_dist_df = impact_contribution.copy()

    # Get Dollar Sales

    price_dist_df = unit_dist_df.copy()
    numeric_cols = price_dist_df.select_dtypes(include='number'
            ).columns.to_list()
    price_dist_df[numeric_cols] = \
        price_dist_df[numeric_cols].mul(wk_sold_price.values, axis=0)

    # Get variable contribution variants

    (dt_unit_dist_df, qtr_unit_dist_df, yr_unit_dist_df) = \
        get_var_contribution_variants(unit_dist_df,
            'model_coefficient_name', value_col_name='units')
    (dt_price_dist_df, qtr_price_dist_df, yr_price_dist_df) = \
        get_var_contribution_variants(price_dist_df,
            'model_coefficient_name', value_col_name='price')
    overall_dt_dist_df = dt_unit_dist_df.merge(dt_price_dist_df,
            how='left', on=['Date', 'model_coefficient_name'])
    overall_qtr_dist_df = qtr_unit_dist_df.merge(qtr_price_dist_df,
            how='left', on=['Quarter', 'model_coefficient_name'])
    overall_yr_dist_df = yr_unit_dist_df.merge(yr_price_dist_df,
            how='left', on=['Year', 'model_coefficient_name'])

    ppg_df = df[ppg_cols].drop_duplicates().values.tolist()
    overall_dt_dist_df[ppg_cols] = pd.DataFrame(ppg_df,
            index=overall_dt_dist_df.index)
    overall_qtr_dist_df[ppg_cols] = pd.DataFrame(ppg_df,
            index=overall_qtr_dist_df.index)
    overall_yr_dist_df[ppg_cols] = pd.DataFrame(ppg_df,
            index=overall_yr_dist_df.index)

    return (overall_dt_dist_df, overall_qtr_dist_df, overall_yr_dist_df)


def get_var_contribution_variants(dist_df, var_col_name,
                                  value_col_name):
    """Get variable contribution by different variants.

    Parameters
    ----------
    dist_df : pd.DataFrame
        Dataframe consisting of IDVs
    var_col_name : str
        Column name for melted IDV columns
    value_col_name : str
        Column name for IDV values

    Returns
    -------
    tuple of pd.DataFrame
    """

    pct_dist_df = dist_df.copy()
    numeric_cols = pct_dist_df.select_dtypes(include='number'
            ).columns.to_list()
    pct_dist_df[numeric_cols] = \
        pct_dist_df[numeric_cols].div(pct_dist_df['Predicted_sales'],
            axis=0)
    dist_df_1 = pd.merge(dist_df.melt(id_vars=['Date'],
                         var_name=var_col_name,
                         value_name=value_col_name),
                         pct_dist_df.melt(id_vars=['Date'],
                         var_name=var_col_name, value_name='pct_'
                         + value_col_name), how='left', on=['Date',
                         var_col_name])

    # Quarterly Aggegration

    qtr_dist_df = dist_df.copy()
    qtr_dist_df['Quarter'] = qtr_dist_df['Date'].dt.year.astype(str) \
        + '-' + 'Q' + qtr_dist_df['Date'].dt.quarter.astype(str)
    qtr_dist_df = qtr_dist_df.drop(columns=['Date'])
    qtr_dist_df = qtr_dist_df.groupby(by=['Quarter'],
            as_index=False).agg(np.sum)

    pct_qtr_dist_df = qtr_dist_df.copy()
    pct_qtr_dist_df[numeric_cols] = \
        pct_qtr_dist_df[numeric_cols].div(pct_qtr_dist_df['Predicted_sales'
            ], axis=0)
    dist_df_2 = pd.merge(qtr_dist_df.melt(id_vars=['Quarter'],
                         var_name=var_col_name,
                         value_name=value_col_name),
                         pct_qtr_dist_df.melt(id_vars=['Quarter'],
                         var_name=var_col_name, value_name='pct_'
                         + value_col_name), how='left', on=['Quarter',
                         var_col_name])

    # Yearly Aggegration

    yr_dist_df = dist_df.copy()
    yr_dist_df['Year'] = yr_dist_df['Date'].dt.year.astype(str)
    yr_dist_df = yr_dist_df.drop(columns=['Date'])
    yr_dist_df = yr_dist_df.groupby(by=['Year'],
                                    as_index=False).agg(np.sum)

    pct_yr_dist_df = yr_dist_df.copy()
    pct_yr_dist_df[numeric_cols] = \
        pct_yr_dist_df[numeric_cols].div(pct_yr_dist_df['Predicted_sales'
            ], axis=0)
    dist_df_3 = pd.merge(yr_dist_df.melt(id_vars=['Year'],
                         var_name=var_col_name,
                         value_name=value_col_name),
                         pct_yr_dist_df.melt(id_vars=['Year'],
                         var_name=var_col_name, value_name='pct_'
                         + value_col_name), how='left', on=['Year',
                         var_col_name])

    return (dist_df_1, dist_df_2, dist_df_3)

# Base variable change as required

def base_var_cont(
    model_df,
    model_df1,
    baseline_var,
    baseline_var_othr,
    model_coef,
    ):
    m = 1
    dt_dist = pd.DataFrame()
    quar_dist = pd.DataFrame()
    year_dist = pd.DataFrame()
    year_dist1 = pd.DataFrame()

    price = 'wk_sold_avg_price_byppg'
    price_val = model_df[[price]]
    model_df[['PPG_Cat', 'PPG_MFG', 'PPG_Item_No', 'PPG_Description'
             ]] = pd.DataFrame([['TEMP'] * 4], index=model_df.index)
    base_var = baseline_var['Variable'].to_list()
    (overall_dt_dist_df, overall_qtr_dist_df, overall_yr_dist_df) = \
        get_var_contribution_wo_baseline_defined(
        model_df,
        model_coef,
        price_val,
        all_df=None,
        var_col='Variable',
        coef_col='Value',
        intercept_name='Intercept',
        base_var=base_var,
        )
    overall_dt_dist_df['Iteration'] = m
    overall_qtr_dist_df['Iteration'] = m
    overall_yr_dist_df['Iteration'] = m

    overall_yr_dist_df1 = \
        overall_yr_dist_df[overall_yr_dist_df['model_coefficient_name']
                           != 'Predicted_sales']
    yearly_1 = overall_yr_dist_df1[['Year', 'units']].groupby(by=['Year'
            ],
            as_index=False).agg(np.sum).rename(columns={'units': 'Yearly_Units'
            })
    overall_yr_dist_df1 = pd.merge(overall_yr_dist_df1, yearly_1,
                                   on='Year', how='left')
    overall_yr_dist_df1['units%'] = overall_yr_dist_df1['units'] \
        / overall_yr_dist_df1['Yearly_Units'] * 100
    dt_dist = pd.concat([dt_dist, overall_dt_dist_df],
                        ignore_index=False)
    quar_dist = pd.concat([quar_dist, overall_qtr_dist_df],
                          ignore_index=False)
    year_dist = pd.concat([year_dist, overall_yr_dist_df],
                          ignore_index=False)
    year_dist1 = pd.concat([year_dist1, overall_yr_dist_df1],
                           ignore_index=False)

  # ########Converting to Required Format

    dt_dist = pd.pivot_table(dt_dist[['Date', 'model_coefficient_name',
                             'units', 'Iteration']], index=['Date',
                             'Iteration'],
                             columns='model_coefficient_name')
    dt_dist = pd.DataFrame(dt_dist).reset_index()
    dt_dist.columns = dt_dist.columns.droplevel(0)
    # print dt_dist.columns
    a = ['Date', 'Iteration'] + list(dt_dist.columns[2:])
    dt_dist.columns = a

    year_dist = pd.pivot_table(year_dist[['Year',
                               'model_coefficient_name', 'units',
                               'Iteration']],
                               index=['model_coefficient_name',
                               'Iteration'], columns='Year')
    year_dist = pd.DataFrame(year_dist).reset_index()
    year_dist.columns = year_dist.columns.droplevel(0)
    # print year_dist.columns
    a = ['model_coefficient_name', 'Iteration'] \
        + list(year_dist.columns[2:])
    year_dist.columns = a

    year_dist1 = pd.pivot_table(year_dist1[['Year',
                                'model_coefficient_name', 'units%',
                                'Iteration']],
                                index=['model_coefficient_name',
                                'Iteration'], columns='Year')
    year_dist1 = pd.DataFrame(year_dist1).reset_index()
    year_dist1.columns = year_dist1.columns.droplevel(0)
    # print year_dist1.columns
    a = ['model_coefficient_name', 'Iteration'] \
        + list(year_dist1.columns[2:])
    year_dist1.columns = a

    aa = dt_dist.columns
    # print aa

  # ##Competitor Discounts

    Comp = [i for i in aa if 'PromotedDiscount' in i]
    dt_dist['Comp'] = dt_dist[Comp].sum(axis=1)

  # ##Give tpr related Variables
    

    Incremental = ['tpr_discount_byppg_contribution_impact'] + [i
            for i in aa if 'Catalogue' in i] + [i for i in aa
            if 'Display' in i] + [i for i in aa if 'flag_N_pls_1' in i]
    # print ('Incremental :', Incremental)
    dt_dist['Incremental'] = dt_dist[Incremental].sum(axis=1)

  # ##Give the remaining Base columns

#   baseprice_cols = ['wk_sold_median_base_price_byppg_log']+[i for i in model_cols if "RegularPrice" in i]
#   holiday =[i for i in model_cols if "day" and "flag" in i ]
#   SI_cols = [i for i in model_cols if "SI" in i ]
#   trend_cols = [i for i in model_cols if "trend" in i ]
#   base_list = baseprice_cols+SI_cols+trend_cols+[i for i in model_cols if "ACV_Selling" in i ]+holiday+[i for i in model_cols if "death_rate" in i]

    base_others = baseline_var_othr['Variable'].to_list()
    base = ['Intercept_contribution_base'] + [i + '_contribution_base'
            for i in base_var] + [i + '_contribution_impact' for i in
                                  base_others]
    # print ('base :', base)
    dt_dist['Base'] = dt_dist[base].sum(axis=1)

    model_df = model_df1.copy()
    req = model_df[['Date', 'Iteration', 'Promo_wave']]
    req['Date'] = pd.to_datetime(req['Date'], format='%Y-%m-%d')
    dt_dist = pd.merge(dt_dist, req, how='left')

    dt_dist['Base'] = dt_dist['Base'] + dt_dist['Comp']

    dt_dist['Lift'] = dt_dist['Incremental'] / dt_dist['Base']
    return dt_dist
  
def get_te_dict(baseline_data, config):
    
      # Function for calculating TE values

    Financial_information = baseline_data[[
        'Promo',
        'List_Price',
        'COGS',
        'Promotion_Cost',
        'TE',
        'tpr_discount_byppg',
        ]].drop_duplicates().reset_index(drop=True)
    promo_info = baseline_data[['Coinvestment', 'tpr_discount_byppg'
                               ]].drop_duplicates().reset_index(drop=True)
    mech_info = baseline_data[['Mechanic', 'tpr_discount_byppg'
                              ]].drop_duplicates().reset_index(drop=True)

  # # Changes added(only MARS_TPRS)
  # if((config['config_constrain']['only_addon_tpr'])&(len(config['MARS_TPRS']>0))):

    Financial_information = \
        Financial_information[Financial_information['tpr_discount_byppg'
                              ] == 0]
    promo_info = promo_info[promo_info['tpr_discount_byppg'] == 0]
    mech_info = mech_info[mech_info['tpr_discount_byppg'] == 0]

    TE_dict = dict(zip(Financial_information.tpr_discount_byppg,
                   Financial_information.TE))
    ret_inv_dict = dict(zip(promo_info.tpr_discount_byppg,
                        promo_info.Coinvestment))
    ret_mech_dict = dict(zip(mech_info.tpr_discount_byppg,
                         mech_info.Mechanic))

  # Additional TPRs

    tprs = config['MARS_TPRS']
    ret_inv = config['Co_investment']
    ret_mech = config['Promo_Mech']  # # Change 1

    if len(tprs) > 0:
        for i in range(len(tprs)):
            List_Price = baseline_data['List_Price'].unique()[0]
            TE_OFF = baseline_data['TE Off Inv'].unique()[0]
            TE_ON = baseline_data['TE On Inv'].unique()[0]
            COGS = baseline_data['COGS'].unique()[0]
            tpr = tprs[i] / 100
            Promotion_Cost = List_Price * tpr * (1 - TE_ON)
            TE = List_Price * (tpr + TE_ON + TE_OFF - tpr * TE_OFF
                               - TE_OFF * TE_ON - TE_ON * tpr + TE_OFF
                               * TE_ON * tpr)
            tpr_1 = tpr * 100
            tpr_1 = tpr_1 + ret_inv[i]

      # appending the new tpr and TE value to TE_dict

            TE_dict[tpr_1] = TE
            ret_inv_dict[tpr_1] = ret_inv[i]
            ret_mech_dict[tpr_1] = ret_mech[i]  # # Change 1
            # print (TE, Promotion_Cost)
    return (TE_dict, ret_inv_dict, ret_mech_dict)


def get_required_base(
    baseline_data,
    Model_Coeff,
    TE_dict,
    ret_inv_dict,
    ret_mech_dict,
    config
    ):

  # Optimizer scenario creation
  # getting model variables

    model_cols = Model_Coeff['names'].to_list()
    model_cols.remove('Intercept')
    Base = baseline_data[['wk_base_price_perunit_byppg', 'Promo', 'TE',
                         'List_Price', 'COGS'] + model_cols]
    i = 0
    TPR_list = list(TE_dict.keys())
    TPR_list.sort()
    TPR_list

#   ret_inv = config["Co_investment"]
  # TPR_list=[j for j in TPR_list if j>0]
  # creating optimizer scenarios for multiple tprs

    for tpr in TPR_list:
        Required_base = Base.copy()
        Required_base['tpr_discount_byppg'] = tpr
        if 'tpr_discount_byppg_lag1' in model_cols:
            Required_base['tpr_discount_byppg_lag1'] = 0
        if 'tpr_discount_byppg_lag2' in model_cols:
            Required_base['tpr_discount_byppg_lag2'] = 0
        Required_base['Promo'] = \
            Required_base['wk_base_price_perunit_byppg'] \
            - Required_base['wk_base_price_perunit_byppg'] \
            * Required_base['tpr_discount_byppg'] / 100

    # mapping TE values from TE dict

        Required_base.loc[Required_base['tpr_discount_byppg'
                          ].isin(TE_dict.keys()), 'TE'] = \
            Required_base['tpr_discount_byppg'].map(TE_dict)
        if 'Catalogue_Dist' in model_cols:

      # catalogue value from the baseline data

            Catalogue_temp = baseline_data['Catalogue_Dist'].max()
            Required_base['Catalogue_Dist'] = \
                np.where(Required_base['tpr_discount_byppg'] == 0, 0,
                         Catalogue_temp)
        if 'Catalogue' in model_cols:

      # catalogue value from the baseline data

            Catalogue_temp = baseline_data['Catalogue'].max()
            Required_base['Catalogue'] = \
                np.where(Required_base['tpr_discount_byppg'] == 0, 0,
                         Catalogue_temp)
        if 'Display' in model_cols:

      # display value from the baseline data

            Catalogue_temp = baseline_data['Display'].max()
            Required_base['Display'] = \
                np.where(Required_base['tpr_discount_byppg'] == 0, 0,
                         Catalogue_temp)

    # Changes Added

        if 'flag_N_pls_1' in model_cols:  # # Change 1
            if ret_mech_dict[tpr] in ['N + 1', 'N+1', '2 + 1 free',
                    '1 + 1 free', '3 + 1 free']:
                Required_base['flag_N_pls_1'] = 1
            else:
                Required_base['flag_N_pls_1'] = 0

        if 'flag_motivation' in model_cols:
            if ret_mech_dict[tpr] in ['Motivation', 'motivation',
                    'Motivational']:
                Required_base['flag_motivation'] = 1
            else:
                Required_base['flag_motivation'] = 0

        Required_base['Units'] = predict_sales(Model_Coeff,
                Required_base)
        Required_base['Promo Price'] = \
            Required_base['wk_base_price_perunit_byppg'] \
            - Required_base['wk_base_price_perunit_byppg'] \
            * (Required_base['tpr_discount_byppg'] / 100)
        Required_base['Sales'] = Required_base['Units'] \
            * Required_base['Promo Price']

    # creating flag for promo price based on the promo price constraint

        Required_base['Promo_price_flag'] = \
            np.where(Required_base['Promo Price']
                     > config['constrain_params']['promo_price'], 0, 1)

    # calculating the financial metrics

        Required_base['GSV'] = Required_base['Units'] \
            * Required_base['List_Price']
        Required_base['Trade_Expense'] = Required_base['Units'] \
            * Required_base['TE']
        Required_base['NSV'] = Required_base['GSV'] \
            - Required_base['Trade_Expense']
        Required_base['MAC'] = Required_base['NSV'] \
            - Required_base['Units'] * Required_base['COGS']  # (list_price-cogs-TE)
        Required_base['RP'] = Required_base['Sales'] \
            - Required_base['NSV']  # (Promo price-List price+TE)
        cols = Required_base.columns[Required_base.columns.isin([
            'Units',
            'Sales',
            'GSV',
            'Trade_Expense',
            'NSV',
            'MAC',
            'RP',
            'Promo_price_flag',
            ])]
        Required_base['TPR'] = tpr
        Required_base['Iteration'] = i
        if i == 0:
            new_base = Required_base
        else:
            new_base = new_base.append(Required_base)

  #   Required_base.rename(columns = dict(zip(cols, 'TPR_'+str(i)+cols )), inplace=True)

        i = i + 1
    Required_base = new_base
    Required_base = Required_base.reset_index(drop=True)
    Required_base['WK_ID'] = Required_base.index

  # creating unique ids for optimization

    Required_base['WK_ID'] = 'WK_' + Required_base['WK_ID'].astype(str) \
        + '_' + Required_base['Iteration'].astype(str)
    return Required_base


def optimizer_fun(baseline_data, Required_base, config):
    
      # calculating baseline numbers from baseline data

    baseline_df = baseline_data[[
        'Baseline_Prediction',
        'Baseline_Sales',
        'Baseline_GSV',
        'Baseline_Trade_Expense',
        'Baseline_NSV',
        'Baseline_MAC',
        'Baseline_RP',
        ]].sum().astype(int)

  # getting the user input

    config_constrain = config['config_constrain']
    constrain_params = config['constrain_params']

  # getting the number of promotions (including zero)

    promo_loop = Required_base['Iteration'].nunique()

  # defining variable type

    WK_DV_vars = list(Required_base['WK_ID'])
    WK_DV_tprvars = list(Required_base['Iteration'])
    WK_vars = LpVariable.dicts('RP', WK_DV_vars, cat='Binary')

  # selecting the objective based on user input

    if config['Objective'] == 'Maximize':
        # print 'Maximize'
        prob = LpProblem('Simple_Workaround_problem', LpMaximize)
    else:
        prob = LpProblem('Simple_Workaround_problem', LpMinimize)

#   slack = pulp.LpVariable('slack', lowBound=0, cat='Continuous')
  # defining objective function

    obj_metric = config['Objective_metric']
    prob += lpSum([WK_vars[Required_base['WK_ID'][i]]
                  * Required_base[obj_metric][i] for i in range(0,
                  Required_base.shape[0])])
    if obj_metric == 'MAC':
        obj_basevalue = baseline_df['Baseline_MAC']
    elif obj_metric == 'RP':
        obj_basevalue = baseline_df['Baseline_RP']
    elif obj_metric == 'Trade_Expense':
        obj_basevalue = baseline_df['Baseline_Trade_Expense']
    elif obj_metric == 'Units':
        obj_basevalue = baseline_df['Baseline_Prediction']
    elif obj_metric == 'NSV':
        obj_basevalue = baseline_df['Baseline_NSV']
    elif obj_metric == 'GSV':
        obj_basevalue = baseline_df['Baseline_GSV']
    elif obj_metric == 'Sales':
        obj_basevalue = baseline_df['Baseline_Sales']

  # Subject to
  # MAC constraint

    if config_constrain['MAC']:
        prob += lpSum([WK_vars[Required_base['WK_ID'][i]]
                      * Required_base['MAC'][i] for i in range(0,
                      Required_base.shape[0])]) \
            >= int(baseline_df['Baseline_MAC'] * constrain_params['MAC'
                   ])

  # RP constraint

    if config_constrain['RP']:
        prob += lpSum([WK_vars[Required_base['WK_ID'][i]]
                      * Required_base['RP'][i] for i in range(0,
                      Required_base.shape[0])]) \
            >= int(baseline_df['Baseline_RP'] * constrain_params['RP'])

#     prob+= slack <= baseline_df['Baseline_RP']*0.1
#     prob+=lpSum([WK_vars[Required_base['WK_ID'][i]]*slack  for i in range(0,Required_base.shape[0])])<=int(baseline_df['Baseline_RP']*0.1)
#     prob+=lpSum([WK_vars[Required_base['WK_ID'][i]]*slack  for i in range(0,Required_base.shape[0])])>=0

    if config_constrain['Trade_Expense']:
        prob += lpSum([WK_vars[Required_base['WK_ID'][i]]
                      * Required_base['Trade_Expense'][i] for i in
                      range(0, Required_base.shape[0])]) \
            <= int(baseline_df['Baseline_Trade_Expense']
                   * constrain_params['Trade_Expense'])

    if config_constrain['Units']:
        prob += lpSum([WK_vars[Required_base['WK_ID'][i]]
                      * Required_base['Units'][i] for i in range(0,
                      Required_base.shape[0])]) \
            >= int(baseline_df['Baseline_Prediction']
                   * constrain_params['Units'])

    if config_constrain['NSV']:
        prob += lpSum([WK_vars[Required_base['WK_ID'][i]]
                      * Required_base['NSV'][i] for i in range(0,
                      Required_base.shape[0])]) \
            >= int(baseline_df['Baseline_NSV'] * constrain_params['NSV'
                   ])

    if config_constrain['GSV']:
        prob += lpSum([WK_vars[Required_base['WK_ID'][i]]
                      * Required_base['GSV'][i] for i in range(0,
                      Required_base.shape[0])]) \
            >= int(baseline_df['Baseline_GSV'] * constrain_params['GSV'
                   ])

    if config_constrain['Sales']:
        prob += lpSum([WK_vars[Required_base['WK_ID'][i]]
                      * Required_base['Sales'][i] for i in range(0,
                      Required_base.shape[0])]) \
            >= int(baseline_df['Baseline_Sales']
                   * constrain_params['Sales'])

    if config_constrain['MAC_Perc']:
        L1 = lpSum([WK_vars[Required_base['WK_ID'][i]]
                   * Required_base['MAC'][i] for i in range(0,
                   Required_base.shape[0])])
        L2 = lpSum([WK_vars[Required_base['WK_ID'][i]]
                   * Required_base['NSV'][i] for i in range(0,
                   Required_base.shape[0])])
        prob += L1 >= L2 * (baseline_df['Baseline_MAC']
                            / baseline_df['Baseline_NSV']) \
            * constrain_params['MAC_Perc']

    if config_constrain['RP_Perc']:
        L1 = lpSum([WK_vars[Required_base['WK_ID'][i]]
                   * Required_base['RP'][i] for i in range(0,
                   Required_base.shape[0])])
        L2 = lpSum([WK_vars[Required_base['WK_ID'][i]]
                   * Required_base['Sales'][i] for i in range(0,
                   Required_base.shape[0])])

    # RP perc = RP/Sales

        prob += L1 >= L2 * (baseline_df['Baseline_RP']
                            / baseline_df['Baseline_Sales']) \
            * constrain_params['RP_Perc']

  # constraint for promo price

    if config_constrain['promo_price']:
        prob += lpSum([WK_vars[Required_base['WK_ID'][i]]
                      * Required_base['Promo_price_flag'][i] for i in
                      range(0, Required_base.shape[0])]) <= 0

  # Set up constraints such that only one tpr is chose for a week

    for i in range(0, 52):
        prob += lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]]
                      for j in range(0, promo_loop)]) <= 1
        prob += lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]]
                      for j in range(0, promo_loop)]) >= 1

  # constraint for compulsory non-promo weeks

    if len(constrain_params['compul_no_promo_weeks']) > 0:
        no_promo_list = constrain_params['compul_no_promo_weeks']
        no_promo_list[:] = [i - 1 for i in no_promo_list]
        prob += lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]]
                      for j in range(1, promo_loop) for i in
                      no_promo_list]) <= 0

   # constraint for compulsory promo weeks

    if len(constrain_params['compul_promo_weeks']) > 0:
        promo_list = constrain_params['compul_promo_weeks'].copy()
        promo_list[:] = [i - 1 for i in promo_list]
        prob += lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]]
                      for j in range(1, promo_loop) for i in
                      promo_list]) >= len(promo_list)

  # Costraint for No of promotions

    if config_constrain['tot_promo_max']:
        prob += lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]]
                      for j in range(1, promo_loop) for i in range(0,
                      52)]) <= constrain_params['tot_promo_max']
    if config_constrain['tot_promo_min']:
        prob += lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]]
                      for j in range(1, promo_loop) for i in range(0,
                      52)]) >= constrain_params['tot_promo_min']

  # Constraint for unique tpr in a promo wave

    k = 1
    loop = True
    while loop:
        if promo_loop - k == 1:
            loop = False
        for i in range(0, 51):
            for j in range(1, promo_loop - k):
                prob += lpSum([WK_vars[Required_base['WK_ID'][i + j
                              * 52]] + WK_vars[Required_base['WK_ID'][i
                              + 1 + (j + k) * 52]]]) <= 1
                prob += lpSum([WK_vars[Required_base['WK_ID'][i + (j
                              + k) * 52]]
                              + WK_vars[Required_base['WK_ID'][i + 1
                              + j * 52]]]) <= 1
        k += 1

  # Sub constraint for minimum weeks

    R0 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
               range(1, promo_loop) for i in range(0, 1)])
    R1 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
               range(1, promo_loop) for i in range(1, 2)])
    R2 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
               range(1, promo_loop) for i in range(2, 3)])
    R3 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
               range(1, promo_loop) for i in range(3, 4)])
    R4 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
               range(1, promo_loop) for i in range(4, 5)])
    R5 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
               range(1, promo_loop) for i in range(5, 6)])
    R6 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
               range(1, promo_loop) for i in range(6, 7)])
    R7 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
               range(1, promo_loop) for i in range(7, 8)])
    R8 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
               range(1, promo_loop) for i in range(8, 9)])
    R9 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
               range(1, promo_loop) for i in range(9, 10)])
    R10 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
                range(1, promo_loop) for i in range(10, 11)])
    R11 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
                range(1, promo_loop) for i in range(11, 12)])

  # prob+= 11*R11-R10-R9-R8-R7-R6-R5-R4-R3-R2-R1-R0 >=0
  # prob+= 10*R10-R9-R8-R7-R6-R5-R4-R3-R2-R1-R0 >=0
  # prob+= 9*R9-R8-R7-R6-R5-R4-R3-R2-R1-R0 >=0
  # prob+= 8*R8-R7-R6-R5-R4-R3-R2-R1-R0 >=0
  # prob+= 7*R7-R6-R5-R4-R3-R2-R1-R0 >=0
  # prob+= 6*R6-R5-R4-R3-R2-R1-R0 >=0
  # prob+= 5*R5-R4-R3-R2-R1-R0 >=0
  # prob+= 4*R4-R3-R2-R1-R0 >=0
  # prob+= 3*R3-R2-R1-R0 >=0
  # prob+= 2*R2-R1-R0 >=0
  # prob+=R1-R0 >=0

  # Sub constraint for minimum weeks

    R51 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
                range(1, promo_loop) for i in range(50, 51)])
    R52 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
                range(1, promo_loop) for i in range(51, 52)])
    R50 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
                range(1, promo_loop) for i in range(49, 50)])
    R49 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
                range(1, promo_loop) for i in range(48, 49)])
    R48 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
                range(1, promo_loop) for i in range(47, 48)])
    R47 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
                range(1, promo_loop) for i in range(46, 47)])
    R46 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
                range(1, promo_loop) for i in range(45, 46)])
    R45 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
                range(1, promo_loop) for i in range(44, 45)])
    R44 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
                range(1, promo_loop) for i in range(43, 44)])
    R43 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
                range(1, promo_loop) for i in range(42, 43)])
    R42 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
                range(1, promo_loop) for i in range(41, 42)])
    R41 = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]] for j in
                range(1, promo_loop) for i in range(40, 41)])

  # prob+= 11*R41-R42-R43-R44-R45-R46-R47-R48-R49-R50-R51-R52 >=0
  # prob+= 10*R42-R43-R44-R45-R46-R47-R48-R49-R50-R51-R52 >=0
  # prob+= 9*R43-R44-R45-R46-R47-R48-R49-R50-R51-R52 >=0
  # prob+= 8*R44-R45-R46-R47-R48-R49-R50-R51-R52 >=0
  # prob+= 7*R45-R46-R47-R48-R49-R50-R51-R52 >=0
  # prob+= 6*R46-R47-R48-R49-R50-R51-R52 >=0
  # prob+= 5*R47-R48-R49-R50-R51-R52 >=0
  # prob+= 4*R48-R49-R50-R51-R52 >=0
  # prob+= 3*R49-R50-R51-R52 >=0
  # prob+=2*R50-R51-R52 >=0
  # prob+=R51-R52 >=0

    if config_constrain['min_consecutive_promo']:
        if constrain_params['min_consecutive_promo'] == 2:
            prob += R51 - R52 >= 0
            prob += R1 - R0 >= 0
        if constrain_params['min_consecutive_promo'] == 3:
            prob += R51 - R52 >= 0
            prob += R1 - R0 >= 0
            prob += 2 * R50 - R51 - R52 >= 0
            prob += 2 * R2 - R1 - R0 >= 0
        if constrain_params['min_consecutive_promo'] == 4:
            prob += R51 - R52 >= 0
            prob += R1 - R0 >= 0
            prob += 2 * R50 - R51 - R52 >= 0
            prob += 2 * R2 - R1 - R0 >= 0
            prob += 3 * R49 - R50 - R51 - R52 >= 0
            prob += 3 * R3 - R2 - R1 - R0 >= 0
        if constrain_params['min_consecutive_promo'] == 5:
            prob += R51 - R52 >= 0
            prob += R1 - R0 >= 0
            prob += 2 * R50 - R51 - R52 >= 0
            prob += 2 * R2 - R1 - R0 >= 0
            prob += 3 * R49 - R50 - R51 - R52 >= 0
            prob += 3 * R3 - R2 - R1 - R0 >= 0
            prob += 4 * R48 - R49 - R50 - R51 - R52 >= 0
            prob += 4 * R4 - R3 - R2 - R1 - R0 >= 0
        if constrain_params['min_consecutive_promo'] == 6:
            prob += R51 - R52 >= 0
            prob += R1 - R0 >= 0
            prob += 2 * R50 - R51 - R52 >= 0
            prob += 2 * R2 - R1 - R0 >= 0
            prob += 3 * R49 - R50 - R51 - R52 >= 0
            prob += 3 * R3 - R2 - R1 - R0 >= 0
            prob += 4 * R48 - R49 - R50 - R51 - R52 >= 0
            prob += 4 * R4 - R3 - R2 - R1 - R0 >= 0
            prob += 5 * R47 - R48 - R49 - R50 - R51 - R52 >= 0
            prob += 5 * R5 - R4 - R3 - R2 - R1 - R0 >= 0
        if constrain_params['min_consecutive_promo'] == 7:
            prob += R51 - R52 >= 0
            prob += R1 - R0 >= 0
            prob += 2 * R50 - R51 - R52 >= 0
            prob += 2 * R2 - R1 - R0 >= 0
            prob += 3 * R49 - R50 - R51 - R52 >= 0
            prob += 3 * R3 - R2 - R1 - R0 >= 0
            prob += 4 * R48 - R49 - R50 - R51 - R52 >= 0
            prob += 4 * R4 - R3 - R2 - R1 - R0 >= 0
            prob += 5 * R47 - R48 - R49 - R50 - R51 - R52 >= 0
            prob += 5 * R5 - R4 - R3 - R2 - R1 - R0 >= 0
            prob += 6 * R46 - R47 - R48 - R49 - R50 - R51 - R52 >= 0
            prob += 6 * R6 - R5 - R4 - R3 - R2 - R1 - R0 >= 0
        if constrain_params['min_consecutive_promo'] == 8:
            prob += R51 - R52 >= 0
            prob += R1 - R0 >= 0
            prob += 2 * R50 - R51 - R52 >= 0
            prob += 2 * R2 - R1 - R0 >= 0
            prob += 3 * R49 - R50 - R51 - R52 >= 0
            prob += 3 * R3 - R2 - R1 - R0 >= 0
            prob += 4 * R48 - R49 - R50 - R51 - R52 >= 0
            prob += 4 * R4 - R3 - R2 - R1 - R0 >= 0
            prob += 5 * R47 - R48 - R49 - R50 - R51 - R52 >= 0
            prob += 5 * R5 - R4 - R3 - R2 - R1 - R0 >= 0
            prob += 6 * R46 - R47 - R48 - R49 - R50 - R51 - R52 >= 0
            prob += 6 * R6 - R5 - R4 - R3 - R2 - R1 - R0 >= 0
            prob += 7 * R45 - R46 - R47 - R48 - R49 - R50 - R51 - R52 \
                >= 0
            prob += 7 * R7 - R6 - R5 - R4 - R3 - R2 - R1 - R0 >= 0
        if constrain_params['min_consecutive_promo'] == 9:
            prob += R51 - R52 >= 0
            prob += R1 - R0 >= 0
            prob += 2 * R50 - R51 - R52 >= 0
            prob += 2 * R2 - R1 - R0 >= 0
            prob += 3 * R49 - R50 - R51 - R52 >= 0
            prob += 3 * R3 - R2 - R1 - R0 >= 0
            prob += 4 * R48 - R49 - R50 - R51 - R52 >= 0
            prob += 4 * R4 - R3 - R2 - R1 - R0 >= 0
            prob += 5 * R47 - R48 - R49 - R50 - R51 - R52 >= 0
            prob += 5 * R5 - R4 - R3 - R2 - R1 - R0 >= 0
            prob += 6 * R46 - R47 - R48 - R49 - R50 - R51 - R52 >= 0
            prob += 6 * R6 - R5 - R4 - R3 - R2 - R1 - R0 >= 0
            prob += 7 * R45 - R46 - R47 - R48 - R49 - R50 - R51 - R52 \
                >= 0
            prob += 7 * R7 - R6 - R5 - R4 - R3 - R2 - R1 - R0 >= 0
            prob += 8 * R44 - R45 - R46 - R47 - R48 - R49 - R50 - R51 \
                - R52 >= 0
            prob += 8 * R8 - R7 - R6 - R5 - R4 - R3 - R2 - R1 - R0 >= 0
    # print (config_constrain['max_consecutive_promo'], '-',
    #        constrain_params['max_consecutive_promo'])

  # contraint for max promo length

    if config_constrain['max_consecutive_promo']:
        for k in range(0, 52 - constrain_params['max_consecutive_promo'
                       ]):
            for j in range(1, promo_loop):
                prob += lpSum([WK_vars[Required_base['WK_ID'][i + j
                              * 52]] for i in range(k, k
                              + constrain_params['max_consecutive_promo'
                              ] + 1)]) \
                    <= constrain_params['max_consecutive_promo']
        for k in range(0, 52 - constrain_params['max_consecutive_promo'
                       ]):
            prob += lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]]
                          for i in range(k, k
                          + constrain_params['max_consecutive_promo']
                          + 1) for j in range(1, promo_loop)]) \
                <= constrain_params['max_consecutive_promo']

  # Constrain for min promo wave length

    if config_constrain['min_consecutive_promo']:
        for k in range(0, 50):
            R1_sum = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]]
                           for i in range(k + 1, min(k
                           + constrain_params['min_consecutive_promo']
                           + 1, 52)) for j in range(1, promo_loop)])
            R2_sum = lpSum([WK_vars[Required_base['WK_ID'][k + j * 52]]
                           for j in range(1, promo_loop)])
            R3_sum = lpSum([WK_vars[Required_base['WK_ID'][k + 1 + j
                           * 52]] for j in range(1, promo_loop)])
            gap_weeks = len(range(k + 1, min(52, k
                            + constrain_params['min_consecutive_promo']
                            + 1)))
            prob += R1_sum + gap_weeks * R2_sum >= gap_weeks * R3_sum

        for k in range(0, 50):
            for j in range(1, promo_loop):
                R1_sum = lpSum([WK_vars[Required_base['WK_ID'][i + j
                               * 52]] for i in range(k + 1, min(k
                               + constrain_params['min_consecutive_promo'
                               ] + 1, 52))])
                R2_sum = lpSum([WK_vars[Required_base['WK_ID'][k + j
                               * 52]]])
                R3_sum = lpSum([WK_vars[Required_base['WK_ID'][k + 1
                               + j * 52]]])
                gap_weeks = len(range(k + 1, min(52, k
                                + constrain_params['min_consecutive_promo'
                                ] + 1)))
                prob += R1_sum + gap_weeks * R2_sum >= gap_weeks \
                    * R3_sum

  #  week gap constraint

    if config_constrain['promo_gap']:
        for k in range(0, 51):
            R1_sum = lpSum([WK_vars[Required_base['WK_ID'][i + j * 52]]
                           for j in range(0, 1) for i in range(k + 1,
                           min(k + constrain_params['promo_gap'] + 1,
                           52))])
            R2_sum = lpSum([WK_vars[Required_base['WK_ID'][k + j * 52]]
                           for j in range(0, 1)])
            R3_sum = lpSum([WK_vars[Required_base['WK_ID'][k + 1 + j
                           * 52]] for j in range(0, 1)])
            gap_weeks = len(range(k + 1, min(52, k
                            + constrain_params['promo_gap'] + 1)))
            prob += R1_sum + gap_weeks * R2_sum >= gap_weeks * R3_sum

    prob.solve()

    # prob.solve(PULP_CBC_CMD(msg=True, maxSeconds=1200000, threads=90,
    #            keepFiles=1, fracGap=None))
    # print LpStatus[prob.status]
    # print pulp.value(prob.objective)
    Weeks = []
    value = []

  # creating the optimal calendar from the prob output

    for variable in prob.variables():
        if variable.varValue == 1.0:
            Weeks.append(str(variable.name))
            value.append(variable.varValue)

  #     print(variable.name, variable.varValue)

    df = pd.DataFrame(list(zip(Weeks, value)), columns=['Weeks', 'val'])
    df['Iteration'] = df['Weeks'].apply(lambda x: x.split('_'
            )[3]).astype(int)
    df['Week_no'] = df['Weeks'].apply(lambda x: x.split('_'
            )[2]).astype(int)
    df['Week_no'] = df['Week_no'] - df['Iteration'] * 52 + 1
    df = df.sort_values('Week_no',
                        ascending=True).reset_index(drop=True)

  # df.head(51)

    tprs = Required_base[['Iteration', 'TPR'
                         ]].drop_duplicates().reset_index(drop=True)
    tprs
    df = pd.merge(df, tprs, on=['Iteration'], how='left')
    df = df.sort_values('Week_no',
                        ascending=True).reset_index(drop=True)

  # getting the solution status and optimum value

    df['Solution'] = LpStatus[prob.status]
    optmized_value = int(pulp.value(prob.objective))

  # returning if the change is insiginficant(<0.01%)

    if ((LpStatus[prob.status]=='Optimal') and (((obj_basevalue*1.0001)>=optmized_value) or (optmized_value<=obj_basevalue*0.999))): ## change added new
        # print 'Getting optimal solution but value is same as baseline'
        df['Solution'] = 'Infeasible'
    return df


def optimal_summary_fun(
    baseline_data,
    Model_Coeff,
    optimal_calendar,
    TE_dict,
    ret_inv_dict,
    ret_mech_dict,
    ):

  # function for creating optimal calendar data
    # import pdb
    # pdb.set_trace()
    model_cols = Model_Coeff['names'].to_list()
    model_cols.remove('Intercept')

  # selecting the required columns

    Base = baseline_data[[
        'Date',
        'wk_base_price_perunit_byppg',
        'Promo',
        'TE',
        'List_Price',
        'COGS',
        'TE On Inv',
        'GMAC',
        'TE Off Inv',
        ] + model_cols]
    new_data = Base.copy()

  # co investment from the user input

    new_data['tpr_discount_byppg'] = optimal_calendar['TPR']

  # changing the variable that related to promotion
  # lag variable creation

    if 'tpr_discount_byppg_lag1' in model_cols:
        new_data['tpr_discount_byppg_lag1'] = \
            new_data['tpr_discount_byppg'].shift(1).fillna(0)
    if 'tpr_discount_byppg_lag2' in model_cols:
        new_data['tpr_discount_byppg_lag2'] = \
            new_data['tpr_discount_byppg'].shift(2).fillna(0)

  # catlogue variableRequi

    if 'Catalogue_Dist' in model_cols:
        Catalogue_temp = baseline_data['Catalogue_Dist'].max()
        new_data['Catalogue_Dist'] = \
            np.where(new_data['tpr_discount_byppg'] == 0, 0,
                     Catalogue_temp)
    if 'Catalogue' in model_cols:
        Catalogue_temp = baseline_data['Catalogue'].max()
        new_data['Catalogue'] = np.where(new_data['tpr_discount_byppg']
                == 0, 0, Catalogue_temp)
    if 'Display' in model_cols:

    # display value from the baseline data

        Catalogue_temp = baseline_data['Display'].max()
        new_data['Display'] = np.where(new_data['tpr_discount_byppg']
                == 0, 0, Catalogue_temp)

  # mapping the tpr and Coinvestment

    new_data.loc[new_data['tpr_discount_byppg'
                 ].isin(ret_inv_dict.keys()), 'Coinvestment'] = \
        new_data['tpr_discount_byppg'].map(ret_inv_dict)
    new_data.loc[new_data['tpr_discount_byppg'
                 ].isin(ret_mech_dict.keys()), 'Mechanic'] = \
        new_data['tpr_discount_byppg'].map(ret_mech_dict)
    if 'flag_N_pls_1' in model_cols:  # # Change 1(Promo_Mech_List)
        new_data['flag_N_pls_1'] = np.where(new_data['Mechanic'
                ].isin(['N + 1', 'N+1', '2 + 1 free', '1 + 1 free',
                '3 + 1 free']), 1, 0)

    if 'flag_motivation' in model_cols:
        new_data['flag_motivation'] = np.where(new_data['Mechanic'
                ].isin(['Motivation', 'motivation', 'Motivational']),
                1, 0)

    new_data['Units'] = predict_sales(Model_Coeff, new_data)
    new_data['Promo Price'] = new_data['wk_base_price_perunit_byppg'] \
        - new_data['wk_base_price_perunit_byppg'] \
        * (new_data['tpr_discount_byppg'] / 100)
    new_data.loc[new_data['tpr_discount_byppg'].isin(TE_dict.keys()),
                 'TE'] = new_data['tpr_discount_byppg'].map(TE_dict)
    new_data['wk_sold_avg_price_byppg'] = new_data['Promo Price']
    new_data['Sales'] = new_data['Units'] * new_data['Promo Price']
    new_data['GSV'] = new_data['Units'] * new_data['List_Price']
    new_data['Trade_Expense'] = new_data['Units'] * new_data['TE']
    new_data['NSV'] = new_data['GSV'] - new_data['Trade_Expense']
    new_data['MAC'] = new_data['NSV'] - new_data['Units'] \
        * new_data['COGS']
    new_data['RP'] = new_data['Sales'] - new_data['NSV']
    # print new_data[[
    #     'Sales',
    #     'GSV',
    #     'Trade_Expense',
    #     'NSV',
    #     'MAC',
    #     'RP',
    #     'Units',
    #     ]].sum().apply(lambda x: '%.3f' % x)
    return new_data


def get_opt_base_comparison(
    baseline_data,
    optimal_data,
    Model_Coeff,
    config,
    ):

  # Function for creating a metric comparison between baseline and optimal calendar

    Segment = config['Segment']
    train_data = baseline_data.copy()
    model_coef = Model_Coeff.copy()
    ret_inv = config['Co_investment']

#   train_data['tpr_discount_byppg']=np.where(train_data['tpr_discount_byppg']==0,0,train_data['tpr_discount_byppg']+ret_inv)
  # promo wave

    train_data['Promo_wave'] = promo_wave_cal(train_data)
    train_data['wk_sold_avg_price_byppg'] = \
        np.exp(train_data['wk_sold_median_base_price_byppg_log']) * (1
            - train_data['tpr_discount_byppg'] / 100)
    model_df = train_data.copy()

  # Base Variable Method

    model_coef.rename(columns={'names': 'Variable',
                      'model_coefficients': 'Value'}, inplace=True)
    baseline_var = pd.DataFrame(columns=['Variable'])
    baseline_var_othr = pd.DataFrame(columns=['Variable'])
    col = model_coef['Variable'].to_list()

  # baseline variables

    baseprice_cols = ['wk_sold_median_base_price_byppg_log'] + [i
            for i in col if 'RegularPrice' in i]
    holiday = [i for i in col if 'day' and 'flag' in i]
    SI_cols = [i for i in col if 'SI' in i]
    trend_cols = [i for i in col if 'trend' in i]
    base_list = baseprice_cols + SI_cols + trend_cols + [i for i in col
            if 'ACV_Selling' in i]
    base_others = holiday + [i for i in col if 'death_rate' in i] + [i
            for i in col if 'tpr_discount_byppg_lag1' in i] + [i
            for i in col if 'tpr_discount_byppg_lag2' in i]

  # base_list = ['wk_sold_median_base_price_byppg_log']+[i for i in col if "RegularPrice" in i]+[i for i in col if "day" and "flag" in i ]+["SI","ACV_Selling"]
  # base_list.remove( 'flag_date_1')
  # base_list.remove( 'flag_date_2')

    # print base_list
    baseline_var['Variable'] = base_list
    baseline_var_othr['Variable'] = base_others

    model_df['Iteration'] = 1
    model_df1 = model_df.copy()
    model_coef['Iteration'] = 1

    base_scenario = base_var_cont(model_df, model_df1, baseline_var,
                                  baseline_var_othr, model_coef)
    base_scenario = base_scenario.merge(train_data[[
        'Date',
        'tpr_discount_byppg',
        'List_Price',
        'GMAC',
        'TE Off Inv',
        'TE On Inv',
        'Coinvestment',
        ]], how='left', on='Date')

  # ROI calcultion for baseline scenario

    base_scenario['TPR'] = base_scenario['tpr_discount_byppg'] \
        - base_scenario['Coinvestment']
    base_scenario['TPR'] = base_scenario['TPR'] / 100
    base_scenario['Uplift LSV'] = base_scenario['Incremental'] \
        * base_scenario['List_Price']
    base_scenario['Uplift GMAC, LSV'] = base_scenario['Uplift LSV'] \
        * base_scenario['GMAC']
    base_scenario['Total LSV'] = (base_scenario['Incremental']
                                  + base_scenario['Base']) \
        * base_scenario['List_Price']
    base_scenario['Mars Uplift On-Invoice'] = base_scenario['Uplift LSV'
            ] * base_scenario['TE On Inv']
    base_scenario['Mars Total On-Invoice'] = base_scenario['Total LSV'] \
        * base_scenario['TE On Inv']
    base_scenario['Mars Uplift NRV'] = base_scenario['Uplift LSV'] \
        - base_scenario['Mars Uplift On-Invoice']
    base_scenario['Mars Total NRV'] = base_scenario['Total LSV'] \
        - base_scenario['Mars Total On-Invoice']
    base_scenario['Uplift Promo Cost'] = base_scenario['Mars Uplift NRV'
            ] * base_scenario['TPR']
    base_scenario['TPR Budget ROI'] = base_scenario['Mars Total NRV'] \
        * base_scenario['TPR']
    base_scenario['Mars Uplift Net Invoice Price'] = \
        base_scenario['Mars Uplift NRV'] \
        - base_scenario['Uplift Promo Cost']
    base_scenario['Mars Total Net Invoice Price'] = \
        base_scenario['Mars Total NRV'] - base_scenario['TPR Budget ROI'
            ]
    base_scenario['Mars Uplift Off-Invoice'] = \
        base_scenario['Mars Uplift Net Invoice Price'] \
        * base_scenario['TE Off Inv']
    base_scenario['Mars Total Off-Invoice'] = \
        base_scenario['Mars Total Net Invoice Price'] \
        * base_scenario['TE Off Inv']
    base_scenario['Uplift  Trade Expense'] = \
        base_scenario['Mars Uplift Off-Invoice'] \
        + base_scenario['TPR Budget ROI'] \
        + base_scenario['Mars Uplift On-Invoice']
    base_scenario['Total  Trade Expense'] = \
        base_scenario['Mars Total On-Invoice'] \
        + base_scenario['TPR Budget ROI'] \
        + base_scenario['Mars Total Off-Invoice']
    base_scenario['Uplift NSV'] = base_scenario['Uplift LSV'] \
        - base_scenario['Uplift  Trade Expense']
    base_scenario['Total NSV'] = base_scenario['Total LSV'] \
        - base_scenario['Total  Trade Expense']
    base_scenario['Segment'] = Segment
    base_scenario['Uplift Royalty'] = np.where(base_scenario['Segment']
            == 'Choco', 0, 0.5 * base_scenario['Uplift NSV'])
    base_scenario['Total Uplift Cost'] = base_scenario['Uplift Royalty'
            ] + base_scenario['Uplift  Trade Expense']
    base_scenario['ROI'] = base_scenario['Uplift GMAC, LSV'] \
        / base_scenario['Total Uplift Cost']
    base_scenario['ROI'] = base_scenario['ROI'].fillna(0)

   # lift wave calculation for base scenario

    lift_wave_base = base_scenario.groupby(['Promo_wave']).agg({
        'Uplift GMAC, LSV': [('Uplift GMAC, LSV', sum)],
        'Total Uplift Cost': [('Total Uplift Cost', sum)],
        'Incremental': [('Incremental', sum)],
        'Base': [('Base', sum)],
        })
    lift_wave_base.columns = ['Uplift_GMAC_LSV', 'Uplift_cost',
                              'incremental', 'base']
    lift_wave_base = lift_wave_base.reset_index()
    lift_wave_base['Lift %'] = lift_wave_base['incremental'] \
        / lift_wave_base['base']
    lift_wave_base['ROI'] = lift_wave_base['Uplift_GMAC_LSV'] \
        / lift_wave_base['Uplift_cost']
    lift_wave_base
    train_data = optimal_data.copy()

#   train_data['tpr_discount_byppg']=np.where(train_data['tpr_discount_byppg']==0,0,train_data['tpr_discount_byppg']+ret_inv)

    train_data['Promo_wave'] = promo_wave_cal(train_data)
    train_data['wk_sold_avg_price_byppg'] = \
        np.exp(train_data['wk_sold_median_base_price_byppg_log']) * (1
            - train_data['tpr_discount_byppg'] / 100)
    model_df = train_data.copy()
    model_df['Iteration'] = 1
    model_df1 = model_df.copy()
    model_coef['Iteration'] = 1

    optimum_scenario = base_var_cont(model_df, model_df1, baseline_var,
            baseline_var_othr, model_coef)
    optimum_scenario = optimum_scenario.merge(train_data[[
        'Date',
        'tpr_discount_byppg',
        'SI',
        'List_Price',
        'GMAC',
        'TE Off Inv',
        'TE On Inv',
        'Coinvestment',
        ]], how='left', on='Date')

  # ROI calculation for optimal scenario

    optimum_scenario['TPR'] = optimum_scenario['tpr_discount_byppg'] \
        - optimum_scenario['Coinvestment']
    optimum_scenario['TPR'] = optimum_scenario['TPR'] / 100
    optimum_scenario['Uplift LSV'] = optimum_scenario['Incremental'] \
        * optimum_scenario['List_Price']
    optimum_scenario['Uplift GMAC, LSV'] = optimum_scenario['Uplift LSV'
            ] * optimum_scenario['GMAC']
    optimum_scenario['Total LSV'] = (optimum_scenario['Incremental']
            + optimum_scenario['Base']) * optimum_scenario['List_Price']
    optimum_scenario['Mars Uplift On-Invoice'] = \
        optimum_scenario['Uplift LSV'] * optimum_scenario['TE On Inv']
    optimum_scenario['Mars Total On-Invoice'] = \
        optimum_scenario['Total LSV'] * optimum_scenario['TE On Inv']
    optimum_scenario['Mars Uplift NRV'] = optimum_scenario['Uplift LSV'
            ] - optimum_scenario['Mars Uplift On-Invoice']
    optimum_scenario['Mars Total NRV'] = optimum_scenario['Total LSV'] \
        - optimum_scenario['Mars Total On-Invoice']
    optimum_scenario['Uplift Promo Cost'] = \
        optimum_scenario['Mars Uplift NRV'] * optimum_scenario['TPR']
    optimum_scenario['TPR Budget ROI'] = \
        optimum_scenario['Mars Total NRV'] * optimum_scenario['TPR']
    optimum_scenario['Mars Uplift Net Invoice Price'] = \
        optimum_scenario['Mars Uplift NRV'] \
        - optimum_scenario['Uplift Promo Cost']
    optimum_scenario['Mars Total Net Invoice Price'] = \
        optimum_scenario['Mars Total NRV'] \
        - optimum_scenario['TPR Budget ROI']
    optimum_scenario['Mars Uplift Off-Invoice'] = \
        optimum_scenario['Mars Uplift Net Invoice Price'] \
        * optimum_scenario['TE Off Inv']
    optimum_scenario['Mars Total Off-Invoice'] = \
        optimum_scenario['Mars Total Net Invoice Price'] \
        * optimum_scenario['TE Off Inv']
    optimum_scenario['Uplift  Trade Expense'] = \
        optimum_scenario['Mars Uplift Off-Invoice'] \
        + optimum_scenario['TPR Budget ROI'] \
        + optimum_scenario['Mars Uplift On-Invoice']
    optimum_scenario['Total  Trade Expense'] = \
        optimum_scenario['Mars Total On-Invoice'] \
        + optimum_scenario['TPR Budget ROI'] \
        + optimum_scenario['Mars Total Off-Invoice']
    optimum_scenario['Uplift NSV'] = optimum_scenario['Uplift LSV'] \
        - optimum_scenario['Uplift  Trade Expense']
    optimum_scenario['Total NSV'] = optimum_scenario['Total LSV'] \
        - optimum_scenario['Total  Trade Expense']
    optimum_scenario['Segment'] = Segment
    optimum_scenario['Uplift Royalty'] = \
        np.where(optimum_scenario['Segment'] == 'Choco', 0, 0.5
                 * optimum_scenario['Uplift NSV'])
    optimum_scenario['Total Uplift Cost'] = \
        optimum_scenario['Uplift Royalty'] \
        + optimum_scenario['Uplift  Trade Expense']
    optimum_scenario['ROI'] = optimum_scenario['Uplift GMAC, LSV'] \
        / optimum_scenario['Total Uplift Cost']
    optimum_scenario['ROI'] = optimum_scenario['ROI'].fillna(0)

  # lift wave optimal scenario

    lift_wave_opt = optimum_scenario.groupby(['Promo_wave']).agg({
        'Uplift GMAC, LSV': [('Uplift GMAC, LSV', sum)],
        'Total Uplift Cost': [('Total Uplift Cost', sum)],
        'Incremental': [('Incremental', sum)],
        'Base': [('Base', sum)],
        })
    lift_wave_opt.columns = ['Uplift_GMAC_LSV', 'Uplift_cost',
                             'incremental', 'base']
    lift_wave_opt = lift_wave_opt.reset_index()
    lift_wave_opt['Lift %'] = lift_wave_opt['incremental'] \
        / lift_wave_opt['base']
    lift_wave_opt['ROI'] = lift_wave_opt['Uplift_GMAC_LSV'] \
        / lift_wave_opt['Uplift_cost']
    lift_wave_opt

  # Calenders(optimum and Baseline)

    base_scenario['Units'] = base_scenario['Base'] \
        + base_scenario['Incremental']
    optimum_scenario['Units'] = optimum_scenario['Base'] \
        + optimum_scenario['Incremental']
    col_req = [
        'Date',
        'tpr_discount_byppg',
        'Units',
        'Base',
        'Incremental',
        'ROI',
        'Lift',
        'Uplift GMAC, LSV',
        'Total Uplift Cost',
        ]
    base_scenario = base_scenario[col_req]
    optimum_scenario = optimum_scenario[col_req + ['SI']]

    base_scenario.rename(columns={
        'tpr_discount_byppg': 'Baseline_Promo',
        'Units': 'Baseline_Units',
        'Base': 'Baseline_Base',
        'Incremental': 'Baseline_Incremental',
        'ROI': 'Baseline_ROI',
        'Uplift GMAC, LSV': 'Baseline_Uplift_GMAC_LSV',
        'Total Uplift Cost': 'Baseline_Total_Uplift_Cost',
        'Lift': 'Baseline_Lift',
        }, inplace=True)

    optimum_scenario.rename(columns={
        'tpr_discount_byppg': 'Optimum_Promo',
        'Units': 'Optimum_Units',
        'Base': 'Optimum_Base',
        'Incremental': 'Optimum_Incremental',
        'ROI': 'Optimum_ROI',
        'Uplift GMAC, LSV': 'Optimum_Uplift_GMAC_LSV',
        'Total Uplift Cost': 'Optimum_Total_Uplift_Cost',
        'Lift': 'Optimum_Lift',
        }, inplace=True)
    opt_base = optimum_scenario.merge(base_scenario, on='Date',
            how='left')

  # opt_base['Optimum_Promo']=np.round(opt_base['Optimum_Promo'],4)
  # opt_base['Baseline_Promo']=np.round(opt_base['Baseline_Promo'],4)

    return opt_base


def get_calendar_summary(baseline_data, optimal_data, opt_base):
    
  # Function for creating an overall summary

    prd = 0
    base_scenario = baseline_data.copy()
    Optimal_scenario = optimal_data.copy()

  # Optimal_scenario=pd.read_csv(path+"Model_Results/Training_data_"+str(prd)+"_Optimal.csv")

    base_scenario.rename(columns={
        'Baseline_Prediction': 'Units',
        'Baseline_Sales': 'Sales',
        'Baseline_GSV': 'GSV',
        'Baseline_Trade_Expense': 'Trade_Expense',
        'Baseline_NSV': 'NSV',
        'Baseline_MAC': 'MAC',
        'Baseline_RP': 'RP',
        }, inplace=True)
    base_scenario['AvgSellingPrice'] = \
        base_scenario['wk_sold_avg_price_byppg']
    Optimal_scenario['AvgSellingPrice'] = \
        Optimal_scenario['wk_sold_avg_price_byppg']
    base_scenario['Product'] = 'Product' + str(prd)
    Optimal_scenario['Product'] = 'Product' + str(prd)

    summary_base = base_scenario.groupby('Product').sum()[[
        'Sales',
        'Units',
        'Trade_Expense',
        'GSV',
        'NSV',
        'MAC',
        'RP',
        'AvgSellingPrice',
        ]]

  # summary_base['AvgSellingPrice']=summary_base['AvgSellingPrice']/52

    summary_base['AvgSellingPrice'] = summary_base['Sales'] \
        / summary_base['Units']
    summary_base['Avg_PromoSellingPrice'] = \
        base_scenario.loc[base_scenario['tpr_discount_byppg']
                          != 0]['Sales'].sum() \
        / base_scenario.loc[base_scenario['tpr_discount_byppg']
                            != 0]['Units'].sum()
    summary_base['ROI'] = opt_base['Baseline_Uplift_GMAC_LSV'].sum() \
        / opt_base['Baseline_Total_Uplift_Cost'].sum()

    summary_opt = Optimal_scenario.groupby('Product').sum()[[
        'Sales',
        'Units',
        'Trade_Expense',
        'GSV',
        'NSV',
        'MAC',
        'RP',
        'AvgSellingPrice',
        ]]

  # summary_opt['AvgSellingPrice']=summary_opt['AvgSellingPrice']/52

    summary_opt['AvgSellingPrice'] = summary_opt['Sales'] \
        / summary_opt['Units']
    summary_opt['Avg_PromoSellingPrice'] = \
        Optimal_scenario.loc[Optimal_scenario['tpr_discount_byppg']
                             != 0]['Sales'].sum() \
        / Optimal_scenario.loc[Optimal_scenario['tpr_discount_byppg']
                               != 0]['Units'].sum()
    summary_opt['ROI'] = opt_base['Optimum_Uplift_GMAC_LSV'].sum() \
        / opt_base['Optimum_Total_Uplift_Cost'].sum()

  # Metric Summary
  # summary_base=pd.read_csv((path+  f"Model_Results/Summary_Metric_{prd}.csv"))
  # summary_opt=pd.read_csv((path+  f"Model_Results/Summary_Metric_{prd}_Optimal.csv"))

    summary_opt[[
        'Sales',
        'Units',
        'Trade_Expense',
        'GSV',
        'NSV',
        'MAC',
        'RP',
        ]] = summary_opt[[
        'Sales',
        'Units',
        'Trade_Expense',
        'GSV',
        'NSV',
        'MAC',
        'RP',
        ]].astype(int)
    summary_opt['RP_Perc'] = summary_opt['RP'] / summary_opt['Sales']
    summary_opt['Mac_Perc'] = summary_opt['MAC'] / summary_opt['NSV']

  # summary_opt[["RP_Perc",'Mac_Perc']]=summary_opt[["RP_Perc",'Mac_Perc']].apply(lambda x: round(x,3))

    summary_base[[
        'Sales',
        'Units',
        'Trade_Expense',
        'GSV',
        'NSV',
        'MAC',
        'RP',
        ]] = summary_base[[
        'Sales',
        'Units',
        'Trade_Expense',
        'GSV',
        'NSV',
        'MAC',
        'RP',
        ]].astype(int)
    summary_base['RP_Perc'] = summary_base['RP'] / summary_base['Sales']
    summary_base['Mac_Perc'] = summary_base['MAC'] / summary_base['NSV']

  # summary_base[["RP_Perc",'Mac_Perc']]=summary_base[["RP_Perc",'Mac_Perc']].apply(lambda x: round(x,3))

    summary_opt = summary_opt.reset_index(drop=True)
    summary_opt = summary_opt.transpose().reset_index(drop=False)
    summary_opt.rename(columns={'index': 'Metric',
                       0: 'Recommended_Scenario'}, inplace=True)

    summary_base = summary_base.reset_index(drop=True)
    summary_base = summary_base.transpose().reset_index(drop=False)
    summary_base.rename(columns={'index': 'Metric', 0: 'Base_Scenario'
                        }, inplace=True)

    summary_metric = summary_base.merge(summary_opt, on='Metric',
            how='left')

  # summary_metric['Change']=np.round(summary_metric['Recommended_Scenario']-summary_metric['Base_Scenario'],4)
  # summary_metric['Delta']=np.round(summary_metric['Change']/summary_metric['Base_Scenario'],4)

    summary_metric['Change'] = summary_metric['Recommended_Scenario'] \
        - summary_metric['Base_Scenario']
    summary_metric['Delta'] = summary_metric['Change'] \
        / summary_metric['Base_Scenario']
    return summary_metric


def config_lag_function(Model_Coeff, config_dict):

  # function for ppgs with lag variables
  # if a lag variable is present, we will increase/decrease the ub and lb by 1%

    model_vars = Model_Coeff['names'].to_list()
    if 'tpr_discount_byppg_lag1' in model_vars \
        or 'tpr_discount_byppg_lag2' in model_vars:
        fin_vars = ['MAC_Perc', 'RP', 'MAC']
        for i in fin_vars:
            if config_dict['config_constrain'][i]:
                config_dict['constrain_params'][i] = \
                    config_dict['constrain_params'][i] + 0.01
    return config_dict
  
      # print("parsed_summarycompleted",parsed_summary)
      # print("completed",summary)
    
#!/usr/bin/python
# -*- coding: utf-8 -*-


def process(
    constraints=None,
    optimizer_save=None,
    promo_week=None,
    pricing_week=None,
    ):

    if constraints:
        account_name = constraints['account_name']
        product_group = constraints['product_group']
        segment = constraints['corporate_segment']
        (model_data_all, ROI_data, model_coeff, coeff_mapping) = \
            pr.get_list_from_db(constraints['account_name'],
                                constraints['product_group'],
                                optimizer_save=optimizer_save)

    if optimizer_save:
        optimizer_save = list(optimizer_save)
        account_name = optimizer_save[0].model_meta.account_name
        product_group = optimizer_save[0].model_meta.product_group
        segment = optimizer_save[0].model_meta.corporate_segment
        (model_data_all, ROI_data, model_coeff, coeff_mapping) = \
            pr.get_list_from_db(account_name, product_group,
                                optimizer_save=optimizer_save)
    if promo_week:
        promo_week = list(promo_week)
        account_name = promo_week[0].pricing_save.account_name
        product_group = promo_week[0].pricing_save.product_group
        segment = promo_week[0].pricing_save.corporate_segment
        (model_data_all, ROI_data, model_coeff, coeff_mapping) = \
            pr.get_list_from_db(account_name, product_group,
                                promo_week=promo_week)
    if pricing_week:
        pricing_week = list(pricing_week)
        account_name = pricing_week[0].pricing_save.account_name
        product_group = pricing_week[0].pricing_save.product_group
        segment = pricing_week[0].pricing_save.corporate_segment
        (model_data_all, ROI_data, model_coeff, coeff_mapping) = \
            pr.get_list_from_db(account_name, product_group,
                                pricing_week=pricing_week)

      # min_consecutive_promo,max_consecutive_promo,min_promo_length_gap,tot_promo_min,tot_promo_max,no_of_promo, no_of_waves= get_promo_wave_values(model_data_all['TPR_Discount'])
      # slct_retailer = account_name
      # slct_ppg =product_group

      # config_constrain : Actiavte/deactivate config constraint -True/False
      # For financial metrics, a value of the form 1.xx or 0.xx where we want the maximum metric value to be xx*100 % higher or lower than the baseline value. Similary, for the # LowerBound_value and LB percentage
      # compulsory no_promo weeks and promo weeks : empty list means no compulsory weeks
      # Co_investment : Investment from the retailers
      # MARS_TPRS : additional tprs to include in the optimization
      # Fin_Pref_Order : The order of relaxing financial metric when we get a infeasible solution

    config = {
        'Retailer': 'Pyaterochka',
        'PPG': 'Orbit OTC',
        'Segment': 'Gum',
        'MARS_TPRS': [20, 30],
        'Co_investment': [8, 8],
        'Promo_Mech': ['N+1', 'N+1'],
        'Objective_metric': 'RP',
        'Objective': 'Maximize',
        'Fin_Pref_Order': ['Trade_Expense', 'RP_Perc', 'MAC_Perc', 'RP'
                           , 'MAC'],
        'config_constrain': {
            'MAC': True,
            'RP': True,
            'Trade_Expense': True,
            'Units': False,
            'NSV': False,
            'GSV': False,
            'Sales': False,
            'MAC_Perc': True,
            'RP_Perc': True,
            'min_consecutive_promo': True,
            'max_consecutive_promo': True,
            'promo_gap': True,
            'tot_promo_min': True,
            'tot_promo_max': True,
            'promo_price': False,
            },
        'constrain_params': {
            'MAC': 1.05,
            'RP': 1.05,
            'Trade_Expense': 1,
            'Units': 1,
            'NSV': 1,
            'GSV': 1,
            'Sales': 1,
            'MAC_Perc': 1,
            'RP_Perc': 1,
            'min_consecutive_promo': 3,
            'max_consecutive_promo': 6,
            'promo_gap': 6,
            'tot_promo_min': 12,
            'tot_promo_max': 18,
            'compul_no_promo_weeks': [],
            'compul_promo_weeks': [],
            'promo_price': 0,
            },
        }
    print(config , "before update")

    if constraints:
        _update_params(config, constraints)
   
    print(config , "after update")
    # import pdb
    # pdb.set_trace()
   

      # retailer, ppg filter

    slct_retailer = config['Retailer']
    slct_ppg = config['PPG']

      # getting coefficient name mapping and values for selected retailer, ppg
    coeff_mapping['Coefficient_Holiday'] = list(map(lambda x: x.startswith('Holiday'),coeff_mapping['Coefficient_new']))
    holiday_list = []
    for index, row in coeff_mapping.iterrows():
      if row['Coefficient_Holiday'] == True:
        holiday_list.append(row['Coefficient'])

    coeff_mapping_temp = coeff_mapping
    col_dict = dict(zip(coeff_mapping_temp['Coefficient_new'],
                    coeff_mapping_temp['Coefficient']))
    col_dict_2 = dict(zip(coeff_mapping_temp['Coefficient'],
                      coeff_mapping_temp['Coefficient_new']))
    col_dict_2.pop('Intercept')
    col_dict.pop('Intercept')

      # print col_dict

      # getting idvs present for the retailer, ppg

    idvs = coeff_mapping_temp['Coefficient_new'].to_list()
    idvs.remove('Intercept')

      # getting model data for retailer, ppg and renaming columns

    Model_Data = model_data_all
    Model_Data = Model_Data[['Date'] + idvs]
    Model_Data.rename(columns=col_dict, inplace=True)

      # getting model coefficients values with original names and format

    Model_Coeff = coeff_mapping_temp[['Coefficient', 'Value', 'PPG',
            'Account Name', 'Coefficient_new']]
    Model_Coeff.rename(columns={'Value': 'model_coefficients',
                       'Coefficient': 'names'}, inplace=True)
    Model_Coeff
    flag_vars = Model_Coeff.loc[Model_Coeff['Coefficient_new'
                                ].str.contains('promo')
                                | Model_Coeff['Coefficient_new'
                                ].str.contains('death')]['names'
            ].to_list()
    Model_Coeff

      # print Model_Data.shape
      # promo_list_PPG = ROI_data[(ROI_data['Account Name'] == slct_retailer)
      #                           & (ROI_data['PPG']
      #                           == slct_ppg)].reset_index(drop=True)

    promo_list_PPG = ROI_data.reset_index(drop=True)

      # print promo_list_PPG.shape
    # import pdb
    # pdb.set_trace()

    Period_data = promo_list_PPG[[
        'Date',
        'Discount, NRV %',
        'TE Off Inv',
        'TE On Inv',
        'GMAC',
        'COGS',
        'List_Price',
        'Mechanic',
        'Coinvestment',
        'Flag_promotype_N_pls_1',
        ]]
    Model_Coeff_list_Keep = list(Model_Coeff['names'])
    Model_Coeff_list_Keep.remove('Intercept')
    Period_data['Promotion_Cost'] = Period_data['List_Price'] \
        * Period_data['Discount, NRV %'] * (1 - Period_data['TE On Inv'
            ])
    Period_data['TE'] = Period_data['List_Price'] \
        * (Period_data['Discount, NRV %'] + Period_data['TE On Inv']
           + Period_data['TE Off Inv'] - Period_data['Discount, NRV %']
           * Period_data['TE Off Inv'] - Period_data['TE Off Inv']
           * Period_data['TE On Inv'] - Period_data['TE On Inv']
           * Period_data['Discount, NRV %'] + Period_data['TE Off Inv']
           * Period_data['TE On Inv'] * Period_data['Discount, NRV %'])
    Period_data['Promo_Depth'] = Period_data['Discount, NRV %'] * 100
    Period_data['tpr_discount_byppg'] = Period_data['Promo_Depth'] \
        + Period_data['Coinvestment']
    Model_Data.rename(columns={'tpr_discount_byppg': 'tpr_discount_byppg_train'
                      }, inplace=True)
    Period_data['Date'] = pd.to_datetime(Period_data['Date'],
            format='%Y-%m-%d')
    Model_Data['Date'] = pd.to_datetime(Model_Data['Date'],
            format='%Y-%m-%d')
    Final_Pred_Data = pd.merge(Period_data, Model_Data, how='left',
                               on='Date')
    Final_Pred_Data['wk_base_price_perunit_byppg'] = \
        np.exp(Final_Pred_Data['wk_sold_median_base_price_byppg_log'])
    Final_Pred_Data['Promo'] = \
        np.where(Final_Pred_Data['tpr_discount_byppg'] == 0,
                 Final_Pred_Data['wk_base_price_perunit_byppg'],
                 Final_Pred_Data['wk_base_price_perunit_byppg'] * (1
                 - Final_Pred_Data['tpr_discount_byppg'] / 100))
    Final_Pred_Data['wk_sold_avg_price_byppg'] = Final_Pred_Data['Promo'
            ]

      # changing the promo related variable if there is any change

    if 'Catalogue_Dist' in Model_Coeff_list_Keep:
        Catalogue_temp = \
            Final_Pred_Data[Final_Pred_Data['tpr_discount_byppg']
                            > 0]['Catalogue_Dist'].max()
    if 'Catalogue' in Model_Coeff_list_Keep:
        Catalogue_temp = \
            Final_Pred_Data[Final_Pred_Data['tpr_discount_byppg']
                            > 0]['Catalogue'].max()
    if 'Display' in Model_Coeff_list_Keep:
        Display_temp = \
            Final_Pred_Data[Final_Pred_Data['tpr_discount_byppg']
                            > 0]['Display'].max()
    if 'Catalogue_Dist' in Model_Coeff_list_Keep:
        Final_Pred_Data['Catalogue_Dist'] = \
            np.where(Final_Pred_Data['tpr_discount_byppg'] == 0, 0,
                     Catalogue_temp)
    if 'Catalogue' in Model_Coeff_list_Keep:
        Final_Pred_Data['Catalogue'] = \
            np.where(Final_Pred_Data['tpr_discount_byppg'] == 0, 0,
                     Catalogue_temp)
    if 'Display' in Model_Coeff_list_Keep:
        Final_Pred_Data['Catalogue'] = \
            np.where(Final_Pred_Data['Display'] == 0, 0, Display_temp)
    if 'tpr_discount_byppg_lag1' in Model_Coeff_list_Keep:
        Final_Pred_Data['tpr_discount_byppg_lag1'] = \
            Final_Pred_Data['tpr_discount_byppg'].shift(1).fillna(0)
    if 'tpr_discount_byppg_lag2' in Model_Coeff_list_Keep:
        Final_Pred_Data['tpr_discount_byppg_lag2'] = \
            Final_Pred_Data['tpr_discount_byppg'].shift(2).fillna(0)

      # if 'flag_N_pls_1' in Model_Coeff_list_Keep:
      #   Final_Pred_Data['flag_N_pls_1']=np.where(Final_Pred_Data['Flag_promotype_N_pls_1']==1,1,0)

    if 'flag_N_pls_1' in Model_Coeff_list_Keep:
        Final_Pred_Data['flag_N_pls_1'] = \
            np.where(Final_Pred_Data['Mechanic'].isin(['N + 1', 'N+1',
                     '2 + 1 free', '1 + 1 free', '3 + 1 free']), 1, 0)
    if 'flag_motivation' in Model_Coeff_list_Keep:  # # change added
        Final_Pred_Data['flag_motivation'] = \
            np.where(Final_Pred_Data['Mechanic'].isin(['Motivation',
                     'motivation', 'Motivational']), 1, 0)

    Final_Pred_Data['Baseline_Prediction'] = predict_sales(Model_Coeff,
            Final_Pred_Data)
    Final_Pred_Data['Baseline_Sales'] = \
        Final_Pred_Data['Baseline_Prediction'] * Final_Pred_Data['Promo'
            ]
    Final_Pred_Data['Baseline_GSV'] = \
        Final_Pred_Data['Baseline_Prediction'] \
        * Final_Pred_Data['List_Price']
    Final_Pred_Data['Baseline_Trade_Expense'] = \
        Final_Pred_Data['Baseline_Prediction'] * Final_Pred_Data['TE']
    Final_Pred_Data['Baseline_NSV'] = Final_Pred_Data['Baseline_GSV'] \
        - Final_Pred_Data['Baseline_Trade_Expense']
    Final_Pred_Data['Baseline_MAC'] = Final_Pred_Data['Baseline_NSV'] \
        - Final_Pred_Data['Baseline_Prediction'] \
        * Final_Pred_Data['COGS']
    Final_Pred_Data['Baseline_RP'] = Final_Pred_Data['Baseline_Sales'] \
        - Final_Pred_Data['Baseline_NSV']

      # print Final_Pred_Data[[
      #     'Baseline_Prediction',
      #     'Baseline_Sales',
      #     'Baseline_GSV',
      #     'Baseline_Trade_Expense',
      #     'Baseline_NSV',
      #     'Baseline_MAC',
      #     'Baseline_RP',
      #     ]].sum().apply(lambda x: '%.3f' % x)

    baseline_df = Final_Pred_Data[[
        'Baseline_Prediction',
        'Baseline_Sales',
        'Baseline_GSV',
        'Baseline_Trade_Expense',
        'Baseline_NSV',
        'Baseline_MAC',
        'Baseline_RP',
        ]].sum().astype(int)
    baseline_df['Baseline_MAC']

    baseline_data = Final_Pred_Data.copy()

    opt_pop_up_flag = -1
    if len(config['MARS_TPRS']) == 0:
        pass
    else:

          # print 'No TPRs are given, please provide TPRs for optimisation'

        opt_pop_up_flag = 0
        opt_pop_up_config = {}

        # financial metric preference order

        fin_pref_order = config['Fin_Pref_Order']

        # getting TE for all the tprs

        (TE_dict, ret_inv_dict, ret_mech_dict) = \
            get_te_dict(baseline_data, config)

        # Optimizer scenario creation

        Required_base = get_required_base(
            baseline_data,
            Model_Coeff,
            TE_dict,
            ret_inv_dict,
            ret_mech_dict,
            config,
            )
        Optimal_calendar_fin = pd.DataFrame()
        infeasible_solution = True

        # creating a copy of config for ppgs with lag variables

        config_temp = copy.deepcopy(config)

        # increasing/decresing the limit to satisfy the actual constraints in ppgs with lag variables

        config_temp = config_lag_function(Model_Coeff, config_temp)

        # config and config temp will be same in all the ppgs without lag variables

          # print ('Init dict', config['constrain_params'])
          # print ('Init dict', config_temp['constrain_params'])

        Optimal_calendar = optimizer_fun(baseline_data, Required_base,
                config_temp)
        if Optimal_calendar.shape[0] == 52 \
            and Optimal_calendar['Solution'].unique() == 'Optimal':

          # Default Run Based On User's Config Step 101

            opt_pop_up_flag = 1
            opt_pop_up_config = copy.deepcopy(config)
            Optimal_calendar_fin = Optimal_calendar.copy()
            infeasible_solution = False

        # An iterative method to automate infeasible scenarios
        # step1 : decreasing the metrics with lower/upper limit higher/lower than baseline

        if infeasible_solution:

              # print 'Infeasible solution : decreasing the metrics limit greater than baseline'

            fin_metrics = [
                'Units',
                'NSV',
                'GSV',
                'Sales',
                'Trade_Expense',
                'RP_Perc',
                'MAC_Perc',
                'RP',
                'MAC',
                ]
            config_constrain = config['config_constrain']
            Act_fin_metrics = []

          # getting the financial metric with lower/upper limit higher/lower than baseline
          # we will be checking only two iterations in this step

            for i in fin_metrics:
                if i == 'Trade_Expense':
                    if config_constrain[i] and config['constrain_params'
                            ][i] < 1:
                        Act_fin_metrics.append(i)
                else:
                    if config_constrain[i] and config['constrain_params'
                            ][i] > 1:
                        Act_fin_metrics.append(i)
            iteration = True
            delta_dict = {}
            iter_no = 0
            while iteration:
                if len(Act_fin_metrics) == 0:
                    config_temp = copy.deepcopy(config)

            # increasing/decresing the limit to satisfy the actual constraints in ppgs with lag variables

                    config_temp = config_lag_function(Model_Coeff,
                            config_temp)
                    config = copy.deepcopy(config_temp)
                    iteration = False
                    continue
                iter_no += 1
                for rel in Act_fin_metrics:
                    if rel == 'Trade_Expense':
                        if config['constrain_params'][rel] < 1 \
                            and iter_no == 1:
                            delta_dict[rel] = (1
                                    - config['constrain_params'][rel]) \
                                / 2  # 0.005
                        if config['constrain_params'][rel] < 1:
                            config['constrain_params'][rel] = \
                                config['constrain_params'][rel] \
                                + delta_dict[rel]
                        else:
                            config['constrain_params'][rel] = \
                                config['constrain_params'][rel] + delta
                    else:
                        if config['constrain_params'][rel] > 1 \
                            and iter_no == 1:
                            delta_dict[rel] = (config['constrain_params'
                                    ][rel] - 1) / 2
                        if config['constrain_params'][rel] > 1:
                            config['constrain_params'][rel] = \
                                config['constrain_params'][rel] \
                                - delta_dict[rel]

                  # print config['constrain_params']

                config_temp = copy.deepcopy(config)

            # increasing/decresing the limit to satisfy the actual constraints in ppgs with lag variables

                config_temp = config_lag_function(Model_Coeff,
                        config_temp)
                Optimal_calendar = optimizer_fun(baseline_data,
                        Required_base, config_temp)
                if Optimal_calendar.shape[0] == 52 \
                    and Optimal_calendar['Solution'].unique() \
                    == 'Optimal':

              # Slight Relaxation in All True/Activatated Fin Metrics Run Step 201

                    opt_pop_up_flag = 2
                    opt_pop_up_config = copy.deepcopy(config)

                      # print config['constrain_params']

                    Optimal_calendar_fin = Optimal_calendar.copy()
                    iteration = False
                    infeasible_solution = False
                if iter_no == 2:
                    config = copy.deepcopy(config_temp)
                    iteration = False

        # if solution is infeasible after step 1 , we will relax each finacial metric one by one and check which metric is causing the infeasible solution

        if infeasible_solution:

              # print 'Infeasible solution - relaxing financial metrics'

          # selecting the secondory finacial metric based on the financial preferance order from the user input

            Sec_fin_metrics = fin_pref_order[:-3]

          # selecting the primary finacial metric based on the financial preferance order from the user input

            Prim_fin_metrics = fin_pref_order[-3:]

          # calendar metrics

            calendar_metrics = [
                'compul_no_promo_weeks',
                'compul_promo_weeks',
                'promo_gap',
                'max_consecutive_promo',
                'min_consecutive_promo',
                'tot_promo_min',
                'tot_promo_max',
                ]

            config_constrain = config['config_constrain']
            Act_Prim_fin_metrics = []

          # getting the active secondary and primary financial metrics(config with True)

            for i in Prim_fin_metrics:
                if config_constrain[i]:
                    Act_Prim_fin_metrics.append(i)
            Act_Sec_fin_metrics = []
            for i in Sec_fin_metrics:
                if config_constrain[i]:
                    Act_Sec_fin_metrics.append(i)

          # creating the combinations of primary and secondary financial metrics

            Sec_combination = sum([list(map(list,
                                  combinations(Act_Sec_fin_metrics,
                                  i))) for i in
                                  range(len(Act_Sec_fin_metrics) + 1)],
                                  [])
            Sec_combination = [i for i in Sec_combination if len(i) > 0]
            Prim_combination = sum([list(map(list,
                                   combinations(Act_Prim_fin_metrics,
                                   i))) for i in
                                   range(len(Act_Prim_fin_metrics)
                                   + 1)], [])
            Prim_combination = [i for i in Prim_combination if len(i)
                                > 0]
            iteration = True
            while iteration:
                relaxed_sec_metrics_opt = []
                relaxed_sec_metrics = []
                relaxed_prim_metrics_opt = []
                relaxed_prim_metrics = []

            # first we will start by relaxing secondary financial variables

                for i in range(len(Sec_combination)):

                      # print Sec_combination[i]

                    metrics = Sec_combination[i]
                    for metric in metrics:
                        config['config_constrain'][metric] = False

                      # print config['config_constrain']

                    Optimal_calendar = optimizer_fun(baseline_data,
                            Required_base, config)
                    if Optimal_calendar.shape[0] == 52 \
                        and Optimal_calendar['Solution'].unique() \
                        == 'Optimal':

                # Complete Relaxation in All Seconderoy Fin Metrics Run Step 301

                        opt_pop_up_flag = 3
                        opt_pop_up_config = copy.deepcopy(config)

                # break the loop if we get a feasible solution

                        relaxed_sec_metrics_opt = Sec_combination[i]
                        for metric in metrics:
                            config['config_constrain'][metric] = True

                          # print 'breaking while loop'

                        iteration = False
                        break
                    if i == len(Sec_combination) - 1:

                          # print i

                        relaxed_sec_metrics = Sec_combination[i]
                    else:

                          # print ('Relaxing :', relaxed_sec_metrics)

                        for metric in metrics:
                            config['config_constrain'][metric] = True

            # if we dont get any feasible solution by relaxing all the secondary metrics, we relax primary metrics one by one

                if iteration == False:
                    continue
                for i in range(len(Prim_combination)):

                      # print Prim_combination[i]

                    metrics = Prim_combination[i]
                    for metric in metrics:
                        config['config_constrain'][metric] = False

                      # print config['config_constrain']

                    Optimal_calendar = optimizer_fun(baseline_data,
                            Required_base, config)
                    if Optimal_calendar.shape[0] == 52 \
                        and Optimal_calendar['Solution'].unique() \
                        == 'Optimal':

                # Complete Relaxation in All Primary Fin Metrics Run Step 351

                        opt_pop_up_flag = 4
                        opt_pop_up_config = copy.deepcopy(config)

                # break the loop if we get a feasible solution

                        relaxed_prim_metrics_opt = Prim_combination[i]
                        for metric in metrics:
                            config['config_constrain'][metric] = True
                        iteration = False
                        break
                    if i == len(Prim_combination) - 1:

                          # print i

                        relaxed_prim_metrics = Prim_combination[i]
                        for metric in metrics:
                            config['config_constrain'][metric] = True
                        iteration = False
                        break
                    else:
                        for metric in metrics:
                            config['config_constrain'][metric] = True
            relaxed_metrics = []
            if len(relaxed_sec_metrics_opt) > 0:
                relaxed_metrics = relaxed_sec_metrics_opt
            elif len(relaxed_prim_metrics_opt) > 0:
                relaxed_metrics = relaxed_prim_metrics_opt
            relaxed_metrics

          # Till here everything is same in all methods

            delta = 0.01
            opt_soln = False
            metric_list = fin_pref_order.copy()
            metric_list.reverse()

          # decresing/increasing the lower/limit of metric to get a feasible solution
          # if there is more than one metric, first we relax the metric with highest preference and decativate all other metrics. Once we get the solution with first metric, we relax the second metric

            if len(relaxed_metrics) > 0:
                iter_metrics = [i for i in metric_list if i
                                in relaxed_metrics]
                n_metrics = len(iter_metrics)
                rel_no = 0
                for rel in iter_metrics:  # [MAC,RP,]#TOCHECK
                    rel_no += 1
                    iteration = True
                    if rel == len(iter_metrics):
                        break
                    othr_metrics = iter_metrics[rel_no:]  # TODO
                    for metric in othr_metrics:
                        config['config_constrain'][metric] = False

                      # print othr_metrics
                      # print config['config_constrain']

                    iter_no = 0
                    while iteration:
                        iter_no += 1
                        if rel == 'Trade_Expense':
                            config['constrain_params'][rel] = \
                                config['constrain_params'][rel] + delta
                        else:
                            config['constrain_params'][rel] = \
                                config['constrain_params'][rel] - delta

                          # print config['constrain_params']

                        Optimal_calendar = optimizer_fun(baseline_data,
                                Required_base, config)
                        if Optimal_calendar.shape[0] == 52 \
                            and Optimal_calendar['Solution'].unique() \
                            == 'Optimal':

                  # Slight Relaxation in Optimized Combined Fin Metrics(Primary/Secondry) Run Step 401

                            opt_pop_up_flag = 5
                            opt_pop_up_config = copy.deepcopy(config)
                            opt_soln = True
                            Optimal_calendar_fin = \
                                Optimal_calendar.copy()
                            for metric in othr_metrics:
                                config['config_constrain'][metric] = \
                                    True
                            iteration = False
                        if iter_no == 15:
                            iteration = False
                    for metric in othr_metrics:
                        config['config_constrain'][metric] = True
                if opt_soln:
                    dta = delta / 4
                    k = 0

              # we will be checking extra 3 ietration to check feasible solution between last limit and and the feasible limit

                    while k < 3:
                        for rel in relaxed_metrics:
                            if rel == 'Trade_Expense':
                                config['constrain_params'][rel] = \
                                    config['constrain_params'][rel] \
                                    - dta
                            else:
                                config['constrain_params'][rel] = \
                                    config['constrain_params'][rel] \
                                    + dta
                        Optimal_calendar = optimizer_fun(baseline_data,
                                Required_base, config)
                        if Optimal_calendar.shape[0] == 52 \
                            and Optimal_calendar['Solution'].unique() \
                            == 'Optimal':

                  # Slight Restricting on top of feasible limit(above section) Fin Metrics Run Step 501

                            opt_pop_up_flag = 6
                            opt_pop_up_config = copy.deepcopy(config)

                              # print config['constrain_params']

                            Optimal_calendar_fin = \
                                Optimal_calendar.copy()
                        k += 1
        if Optimal_calendar_fin.shape[0] == 0:
            opt_pop_up_flag = 0
            opt_pop_up_config = {}
            Optimal_calendar_fin['TPR'] = \
                baseline_data['tpr_discount_byppg']
            Financial_information1= baseline_data[['TE','tpr_discount_byppg']].drop_duplicates().reset_index(drop=True) ## change added new
            promo_info1 = baseline_data[['Coinvestment','tpr_discount_byppg']].drop_duplicates().reset_index(drop=True) ## change added new
            mech_info1 = baseline_data[['Mechanic','tpr_discount_byppg']].drop_duplicates().reset_index(drop=True) ## change added new
            
        
            TE_dict= dict(zip(Financial_information1.tpr_discount_byppg, Financial_information1.TE)) ## change added new
            ret_inv_dict = dict(zip(promo_info1.tpr_discount_byppg, promo_info1.Coinvestment)) ## change added new
            ret_mech_dict = dict(zip(mech_info1.tpr_discount_byppg, mech_info1.Mechanic)) ## change added new

        # getting data with optimal calendar

        Optimal_data = optimal_summary_fun(
            baseline_data,
            Model_Coeff,
            Optimal_calendar_fin,
            TE_dict,
            ret_inv_dict,
            ret_mech_dict,
            )

        # getting comparison between baseline and optimal calendar

        opt_base = get_opt_base_comparison(baseline_data, Optimal_data,
                Model_Coeff, config)

        # metric summary comparison between optimal and baseline calendar

        summary = get_calendar_summary(baseline_data, Optimal_data,
                opt_base)
        # import pdb
        # pdb.set_trace()

      # Optmised Pop Up Final Message

    opt_pop_up_flag_final = 0
    opt_pop_up_config_final = {}

      # TPR's not provided from User

    if opt_pop_up_flag == -1:
        opt_pop_up_flag_final = -1
    elif opt_pop_up_flag == 1:

      # #Optimum Soln based on user's constraints

        opt_pop_up_flag_final = 1
    elif opt_pop_up_flag in [2, 3, 4, 5, 6]:

      # Optimum Soln based on slight change in user's constraints; provide updated constraints

        opt_pop_up_flag_final = 2
        opt_pop_up_config_final = copy.deepcopy(opt_pop_up_config)
    else:

      # Infeasible Soln

        opt_pop_up_flag_final = 0
    print(opt_pop_up_flag , "opt_pop_up_flag")
    print(opt_pop_up_flag_final , "opt_pop_up_flag_final")
    print(opt_pop_up_config_final , "opt_pop_up_config_final")
    if len(holiday_list) > 0:
        for value in holiday_list:
          opt_base[value] = Optimal_data[value]
    opt_base['week'] = opt_base.sort_values("Date").index + 1
    opt_base['Coinvestment']  = Optimal_data.sort_values("Date")['Coinvestment']
    opt_base["Optimum_Promo"] = opt_base["Optimum_Promo"] - opt_base['Coinvestment']
    # import pdb
    # pdb.set_trace()
    opt_base['Mechanic']  = Optimal_data.sort_values("Date")['Mechanic']
    
    parsed_summary = json.loads(summary.to_json(orient="records"))
    parsed_base = json.loads(opt_base.to_json(orient="records"))
    # import pdb
    # pdb.set_trace()
    
    financial_metrics = mixin.calculate_finacial_metrics_for_optimizer(account_name,product_group,parsed_base,model_coeff,model_data_all,ROI_data)
  #   print ('completed', opt_base)
  #   print ('completed', summary)
    return {
    "holiday" : holiday_list,
    "summary" : parsed_summary,
    "optimal" : parsed_base,
    "financial_metrics": financial_metrics
  }


