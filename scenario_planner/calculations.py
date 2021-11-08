from typing import List
from django.db.models.fields import DateField
from django.db.models.query import QuerySet
import pandas as pd
# from core.models import db_model.ModelData, db_model.ModelMeta
from core import models as db_model
from utils import util as util
from utils import models as model
from functools import reduce
from itertools import groupby
from operator import itemgetter
import utils
# from utils import units_calculation as uc
# from .query import roi_values , data_values
from utils.constants import (
     DATA_VALUES as data_values , ROI_VALUES as roi_values , 
     COEFFICIENT_VALUES as coeff_values
     
)
import math
import decimal
import copy
# import json

def equation(predicted_sales , new_retail_price , old_retail_price,price_elasticity,
                                  tpr_coefficient , new_tpr,old_tpr):
    # import pdb
    # pdb.set_trace()
    
    price = math.pow(round((1+(round(new_retail_price,2) - round(old_retail_price,2))/round(old_retail_price,2)),2) , price_elasticity)
    discount = math.exp(tpr_coefficient * (new_tpr - old_tpr))
    res = round(decimal.Decimal(predicted_sales),2) * decimal.Decimal(price) * decimal.Decimal(discount)
    # import pdb
    # pdb.set_trace()
    # print(float(res))
    return res

def base_price_const_promo_change():
    pass

    

def get_related_value(promo_data : db_model.ModelData):
    # import pdb
    # pdb.set_trace()
    # print("get related value")
    # return promo_data.model_meta.coefficient.get()
    
    return promo_data.model_meta.prefetched_coeff[0]
def get_roi_by_week(promo_data : db_model.ModelData):
    return promo_data.model_meta.prefetched_roi[promo_data.week-1]
def promo_simulator_calculations(promo_data : db_model.ModelData):
    
    result = get_related_value(promo_data).intercept
    + (promo_data.median_base_price_log * get_related_value(promo_data).median_base_price_log)
    + (promo_data.tpr_discount * get_related_value(promo_data).tpr_discount)
    + (promo_data.tpr_discount_lag1 * get_related_value(promo_data).tpr_discount_lag1)
    + (promo_data.tpr_discount_lag2 * get_related_value(promo_data).tpr_discount_lag2)
    + (promo_data.catalogue * get_related_value(promo_data).catalogue)
    + (promo_data.display * get_related_value(promo_data).display)
    + (promo_data.acv * get_related_value(promo_data).acv)
    + (promo_data.si * get_related_value(promo_data).si)
    + (promo_data.si_month * get_related_value(promo_data).si_month)
    + (promo_data.si_quarter * get_related_value(promo_data).si_quarter)
    + (promo_data.c_1_crossretailer_discount * get_related_value(promo_data).c_1_crossretailer_discount)
    + (promo_data.c_1_crossretailer_log_price * get_related_value(promo_data).c_1_crossretailer_log_price)
    + (promo_data.c_1_intra_discount * get_related_value(promo_data).c_1_intra_discount)
    + (promo_data.c_2_intra_discount * get_related_value(promo_data).c_2_intra_discount)
    + (promo_data.c_3_intra_discount * get_related_value(promo_data).c_3_intra_discount)
    + (promo_data.c_4_intra_discount * get_related_value(promo_data).c_4_intra_discount)
    + (promo_data.c_5_intra_discount * get_related_value(promo_data).c_5_intra_discount)
    + (promo_data.c_1_intra_log_price * get_related_value(promo_data).c_1_intra_log_price)
    + (promo_data.c_2_intra_log_price * get_related_value(promo_data).c_2_intra_log_price)
    + (promo_data.c_3_intra_log_price * get_related_value(promo_data).c_3_intra_log_price)
    + (promo_data.c_4_intra_log_price * get_related_value(promo_data).c_4_intra_log_price)
    + (promo_data.c_5_intra_log_price * get_related_value(promo_data).c_5_intra_log_price)
    + (promo_data.category_trend * get_related_value(promo_data).category_trend)
    + (promo_data.trend_month * get_related_value(promo_data).trend_month)
    + (promo_data.trend_quarter * get_related_value(promo_data).trend_quarter)
    + (promo_data.trend_year * get_related_value(promo_data).trend_year)
    + (promo_data.month_no * get_related_value(promo_data).month_no)
    + (promo_data.flag_promotype_motivation * get_related_value(promo_data).flag_promotype_motivation)
    + (promo_data.flag_promotype_n_pls_1 * get_related_value(promo_data).flag_promotype_n_pls_1)
    + (promo_data.flag_promotype_traffic * get_related_value(promo_data).flag_promotype_traffic)
    + (promo_data.flag_nonpromo_1 * get_related_value(promo_data).flag_nonpromo_1)
    + (promo_data.flag_nonpromo_2 * get_related_value(promo_data).flag_nonpromo_2)
    + (promo_data.flag_nonpromo_3 * get_related_value(promo_data).flag_nonpromo_3)
    + (promo_data.flag_promo_1 * get_related_value(promo_data).flag_promo_1)
    + (promo_data.flag_promo_2 * get_related_value(promo_data).flag_promo_2)
    + (promo_data.flag_promo_3 * get_related_value(promo_data).flag_promo_3)
    + (promo_data.holiday_flag_1 * get_related_value(promo_data).holiday_flag_1)
    + (promo_data.holiday_flag_2 * get_related_value(promo_data).holiday_flag_2)
    + (promo_data.holiday_flag_3 * get_related_value(promo_data).holiday_flag_3)
    + (promo_data.holiday_flag_4 * get_related_value(promo_data).holiday_flag_4)
    + (promo_data.holiday_flag_5 * get_related_value(promo_data).holiday_flag_5)
    + (promo_data.holiday_flag_6 * get_related_value(promo_data).holiday_flag_6)
    + (promo_data.holiday_flag_7 * get_related_value(promo_data).holiday_flag_7)
    + (promo_data.holiday_flag_8 * get_related_value(promo_data).holiday_flag_8)
    + (promo_data.holiday_flag_9 * get_related_value(promo_data).holiday_flag_9)
    + (promo_data.holiday_flag_10 * get_related_value(promo_data).holiday_flag_10)
    
    # print(result , "result")
    # import pdb
    # pdb.set_trace()
    # model.UnitModel()
    # print(promo_data.promo_data.off_inv , "promo off")
    # print(promo_data.on_inv , "promo on")
    ob = model.UnitModel(base_units=  decimal.Decimal(math.exp(result)),
                    on_inv_percent=promo_data.on_inv * 100,
                    list_price=promo_data.list_price,
                    tpr_percent=decimal.Decimal(promo_data.tpr_discount),
                    off_inv_percent = promo_data.off_inv * 100,
                    gmac_percent_lsv = promo_data.gmac_percent_lsv * 100
                    )
    # print(ob , "OB")
    # import pdb
    # pdb.set_trace()
    # print(result , "actual result")
    # print(math.exp(result) , "expo result")
    
    return ob.__dict__

def promo_simulator_calculations_test(promo_data : db_model.ModelData):
   
    intercept = get_related_value(promo_data).intercept
    median = (promo_data.median_base_price_log * get_related_value(promo_data).median_base_price_log)
    tpr_discount =  (promo_data.tpr_discount * get_related_value(promo_data).tpr_discount)
    tpr_discount_lag1 = (promo_data.tpr_discount_lag1 * get_related_value(promo_data).tpr_discount_lag1)
    tpr_discount_lag2 = (promo_data.tpr_discount_lag2 * get_related_value(promo_data).tpr_discount_lag2)
    catalogue =  (promo_data.catalogue * get_related_value(promo_data).catalogue)
    display = (promo_data.display * get_related_value(promo_data).display)
    acv = (promo_data.acv * get_related_value(promo_data).acv)
    si = (promo_data.si * get_related_value(promo_data).si)
    si_month = (promo_data.si_month * get_related_value(promo_data).si_month)
    si_quarter = (promo_data.si_quarter * get_related_value(promo_data).si_quarter)
    c_1_crossretailer_discount =  (promo_data.c_1_crossretailer_discount * get_related_value(promo_data).c_1_crossretailer_discount)
    c_1_crossretailer_log_price = (promo_data.c_1_crossretailer_log_price * get_related_value(promo_data).c_1_crossretailer_log_price)
    c_1_intra_discount = (promo_data.c_1_intra_discount * get_related_value(promo_data).c_1_intra_discount)
    c_2_intra_discount = (promo_data.c_2_intra_discount * get_related_value(promo_data).c_2_intra_discount)
    c_3_intra_discount = (promo_data.c_3_intra_discount * get_related_value(promo_data).c_3_intra_discount)
    c_4_intra_discount = (promo_data.c_4_intra_discount * get_related_value(promo_data).c_4_intra_discount)
    c_5_intra_discount = (promo_data.c_5_intra_discount * get_related_value(promo_data).c_5_intra_discount)
    c_1_intra_log_price = (promo_data.c_1_intra_log_price * get_related_value(promo_data).c_1_intra_log_price)
    c_2_intra_log_price = (promo_data.c_2_intra_log_price * get_related_value(promo_data).c_2_intra_log_price)
    c_3_intra_log_price =  (promo_data.c_3_intra_log_price * get_related_value(promo_data).c_3_intra_log_price)
    c_4_intra_log_price = (promo_data.c_4_intra_log_price * get_related_value(promo_data).c_4_intra_log_price)
    c_5_intra_log_price = (promo_data.c_5_intra_log_price * get_related_value(promo_data).c_5_intra_log_price)
    category_trend = (promo_data.category_trend * get_related_value(promo_data).category_trend)
    trend_month = (promo_data.trend_month * get_related_value(promo_data).trend_month)
    trend_quarter = (promo_data.trend_quarter * get_related_value(promo_data).trend_quarter)
    trend_year = (promo_data.trend_year * get_related_value(promo_data).trend_year)
    month_no =  (promo_data.month_no * get_related_value(promo_data).month_no)
    flag_promotype_motivation =  (promo_data.flag_promotype_motivation * get_related_value(promo_data).flag_promotype_motivation)
    flag_promotype_n_pls_1 = (promo_data.flag_promotype_n_pls_1 * get_related_value(promo_data).flag_promotype_n_pls_1)
    flag_promotype_traffic = (promo_data.flag_promotype_traffic * get_related_value(promo_data).flag_promotype_traffic)
    flag_nonpromo_1 =(promo_data.flag_nonpromo_1 * get_related_value(promo_data).flag_nonpromo_1)
    flag_nonpromo_2 = (promo_data.flag_nonpromo_2 * get_related_value(promo_data).flag_nonpromo_2)
    flag_nonpromo_3 = (promo_data.flag_nonpromo_3 * get_related_value(promo_data).flag_nonpromo_3)
    flag_promo_1= (promo_data.flag_promo_1 * get_related_value(promo_data).flag_promo_1)
    flag_promo_2 = (promo_data.flag_promo_2 * get_related_value(promo_data).flag_promo_2)
    flag_promo_3 = (promo_data.flag_promo_3 * get_related_value(promo_data).flag_promo_3)
    holiday_flag_1 = (promo_data.holiday_flag_1 * get_related_value(promo_data).holiday_flag_1)
    holiday_flag_2 = (promo_data.holiday_flag_2 * get_related_value(promo_data).holiday_flag_2)
    holiday_flag_3= (promo_data.holiday_flag_3 * get_related_value(promo_data).holiday_flag_3)
    holiday_flag_4 = (promo_data.holiday_flag_4 * get_related_value(promo_data).holiday_flag_4)
    holiday_flag_5 = (promo_data.holiday_flag_5 * get_related_value(promo_data).holiday_flag_5)
    holiday_flag_6 = (promo_data.holiday_flag_6 * get_related_value(promo_data).holiday_flag_6)
    holiday_flag_7= (promo_data.holiday_flag_7 * get_related_value(promo_data).holiday_flag_7)
    holiday_flag_8 = (promo_data.holiday_flag_8 * get_related_value(promo_data).holiday_flag_8)
    holiday_flag_9= (promo_data.holiday_flag_9 * get_related_value(promo_data).holiday_flag_9)
    holiday_flag_10 = (promo_data.holiday_flag_10 * get_related_value(promo_data).holiday_flag_10)
    result = median + intercept + tpr_discount + tpr_discount_lag1 + tpr_discount_lag2 + catalogue +\
    display +acv + si+si_month+si_quarter + c_1_crossretailer_discount + c_1_crossretailer_log_price +\
        c_1_intra_discount+c_2_intra_discount + c_3_intra_discount+c_4_intra_discount+c_5_intra_discount+\
            c_1_intra_log_price + c_2_intra_log_price+c_3_intra_log_price+c_4_intra_log_price+c_5_intra_log_price+\
                category_trend + trend_month + trend_quarter + trend_year + month_no + flag_promotype_motivation +\
                    flag_promotype_n_pls_1 + flag_promotype_traffic + flag_nonpromo_1 + flag_nonpromo_2 + flag_nonpromo_3+\
                        flag_promo_1 + flag_promo_2 + flag_promo_3 + holiday_flag_1 + holiday_flag_2 + holiday_flag_3+\
                            holiday_flag_4 + holiday_flag_5 + holiday_flag_6 + holiday_flag_7 + holiday_flag_8+\
                                holiday_flag_9 + holiday_flag_10
                                

    roi_model = get_roi_by_week(promo_data)
    ob = model.UnitModel(
                    predicted_units=  decimal.Decimal(math.exp(result)),
                    on_inv_percent=decimal.Decimal(roi_model.on_inv * 100),
                    list_price=decimal.Decimal(roi_model.list_price),
                    tpr_percent=decimal.Decimal(promo_data.tpr_discount),
                    off_inv_percent = decimal.Decimal(roi_model.off_inv * 100),
                    gmac_percent_lsv = decimal.Decimal(roi_model.gmac * 100),
                    average_selling_price = decimal.Decimal(promo_data.wk_sold_avg_price_byppg),
                    product_group_weight_in_grams = decimal.Decimal(promo_data.weighted_weight_in_grams),
                    median_base_price_log = decimal.Decimal(promo_data.median_base_price_log),
                    incremental_unit = decimal.Decimal(promo_data.incremental_unit),
                    base_unit = decimal.Decimal(promo_data.base_unit)
                    )
   
    
    return ob.__dict__
  
def update_week_value(promo_data:db_model.ModelMeta , querydict):
  
    for i in querydict.keys():
        week_regex = util._regex(r'week-\d{1,2}',i)
        if week_regex:
            week = int(util._regex(r'\d{1,2}',week_regex.group()).group())
            if querydict['param_depth_all']:
                # print("setting depth all")
                promo_data.prefetched_data[week-1].tpr_discount = querydict['param_depth_all'] 
            else: 
                promo_data.prefetched_data[week-1].tpr_discount = querydict[i]['promo_depth'] 
                
def get_holiday_information(data):
    holidays = ['holiday_flag_1','holiday_flag_2','holiday_flag_3',
    'holiday_flag_4','holiday_flag_5','holiday_flag_6','holiday_flag_7','holiday_flag_8','holiday_flag_9', 
    'holiday_flag_10']
    for i in holidays:
        if(data[data_values.index(i)]):
            return i
    return None

def calculate_financial_mertrics(data_list ,roi_list,unit_info , flag,promo_elasticity = 0,pricing = None):
    '''
    To calculate financial metrics for each week as well as total
    '''

    weekly_units = []
    total_units = model.TotalUnit()
    
    
    for i in range(0,len(data_list)):
        # import pdb
        # pdb.set_trace()
      
        roi = roi_list[i]
        unit = unit_info[i]
        data = data_list[i]
        
        retail_price = decimal.Decimal(math.exp(data[data_values.index('median_base_price_log')]))
        list_price =roi[roi_values.index('list_price')]
        gmac_percent_lsv = roi[roi_values.index('gmac')] * 100
        cogs = list_price - abs(list_price * (gmac_percent_lsv/100))
        # if(pricing):
        #     if(isinstance(pricing , List)):
              
        #         retail_price = retail_price + (retail_price * (pricing[i].rsp_increase)/100)
        #         list_price = list_price + (list_price * ((pricing[i].lp_increase)/100))
        #         cogs = cogs + (cogs * ((pricing[i].cogs_increase)/100))
        #     else:
        #         retail_price = retail_price + (retail_price * (decimal.Decimal(pricing['rsp'])/100))
        #         list_price = list_price + (list_price * (decimal.Decimal(pricing['lpi'])/100))
        #         cogs = cogs + (cogs * (decimal.Decimal(pricing['cogs'])/100))
            # pass
            # {'lpi': 5, 'rsp': 5, 'cogs': 4, 'elasticity': -1.18}
            # import pdb
            # pdb.set_trace()
        # print(data[data_values.index('promo_depth')] , "promo depth value for iteration " ,i )
        try:

            ob = model.UnitModel(
                data[data_values.index('date')],
                week = int(data[data_values.index('week')]),
                year = int(data[data_values.index('year')]),
                quater = (data[data_values.index('quater')]),
                month = (data[data_values.index('month')]),
                period = (data[data_values.index('period')]),
                flag_promotype_motivation = int(data[data_values.index('flag_promotype_motivation')]),
                flag_promotype_n_pls_1 = int(data[data_values.index('flag_promotype_n_pls_1')]),
                flag_promotype_traffic = int(data[data_values.index('flag_promotype_traffic')]),
                si = data[data_values.index('si')],
                predicted_units=decimal.Decimal(unit['Predicted_sales']),
                on_inv_percent=roi[roi_values.index('on_inv')] * 100,
                list_price = list_price,
                promo_depth=decimal.Decimal(data[data_values.index('promo_depth')]),
                off_inv_percent = roi[roi_values.index('off_inv')] * 100, 
                gmac_percent_lsv =gmac_percent_lsv,

                # average_selling_price = data[data_values.index('wk_sold_avg_price_byppg')],
                product_group_weight_in_grams = data[data_values.index('weighted_weight_in_grams')], 
                median_base_price_log = retail_price,
                incremental_unit = decimal.Decimal(unit['Incremental']),
                base_unit = decimal.Decimal(unit['Base']),
                promo_elasticity=promo_elasticity,
                co_investment = decimal.Decimal(data[data_values.index('co_investment')]),
                cogs = cogs,
                is_vat_applied=data[data_values.index('model_meta__account_name')] != 'Lenta',
                royalty_increase = 0.0 if util._transform_corporate_segment(
                    data[data_values.index('model_meta__corporate_segment')]
                ) == 'Choco' else 0.5
                
            )
            # import pdb
            # pdb.set_trace()
            update_total(total_units , ob)
            ob_dict = ob.__dict__
            ob_dict['holiday'] = get_holiday_information(data)
            weekly_units.append(ob_dict)
        except Exception as e:
            raise e
            # import pdb
            # pdb.set_trace()
            # passa
    # import pdb
    # pdb.set_trace()
    aggregate_total(total_units)
    
    return {
        flag : {
            'total' :  total_units.__dict__,
            'weekly' : weekly_units
        }
       
    }


 
def calculate_financial_mertrics_equation( data_list ,roi_list,unit_info , flag,coeff_list , promo_elasticity = 0,
                                 base_list = None , base_split = None , form = None):
    '''
    To calculate financial metrics for each week as well as total
    '''

    weekly_units = []
    total_units = model.TotalUnit()
    print("-----------------------------" + flag + "--------------------------")
    
    for i in range(0,len(data_list)):
        # import pdb
        # pdb.set_trace()
        
      
        roi = roi_list[i]  # current roi data
        unit = unit_info[i] # curent unit , base incren=ment sllit
        data = data_list[i] # current data list
        is_vat_applied = data[data_values.index('model_meta__account_name')] != 'Lenta' # for lenta dont apply vat
        
        predicted_units = decimal.Decimal(unit['Predicted_sales'])
        promo_depth = data[data_values.index('promo_depth')]
        # data[data_values.index('tpr_discount')] - data[data_values.index('co_investment')]
        
        incremental_units = decimal.Decimal(unit['Incremental'])
        base_units = decimal.Decimal(unit['Base']) 
        retail_price = decimal.Decimal(math.exp(data[data_values.index('median_base_price_log')]))
        # retail_price = retail_price * decimal.Decimal(1 - (20/100)) if is_vat_applied else retail_price
        # 
        list_price =roi[roi_values.index('list_price')]
        # import pdb
        # pdb.set_trace()
        gmac_percent_lsv = roi[roi_values.index('gmac')] * 100
        cogs = list_price - abs(list_price * (gmac_percent_lsv/100))
        if(flag=="simulated"):
            # if(i == 10):
            #     import pdb
            #     pdb.set_trace()
            
            base_val = base_list[i]
            base_split_val = base_split[i] 
            if(form['follow_competition']):
                elasticity = form['inc_net_elasticity']
            else:
                elasticity = form['inc_elasticity']
            
        
            predicted_units = equation(base_split_val['Predicted_sales'],
                    decimal.Decimal(math.exp(data[data_values.index('median_base_price_log')])),
                    decimal.Decimal(math.exp(base_val[data_values.index('median_base_price_log')])),
                    elasticity,
                    coeff_list[coeff_values.index('tpr_discount')],
                    data[data_values.index('tpr_discount')],
                    base_val[data_values.index('tpr_discount')]
                 
                 )
            # import pdb
            # pdb.set_trace()
            base_percent = (base_split_val['Base']/base_split_val['Predicted_sales'])*100
            inc_percent = 100 - base_percent
            base_units = decimal.Decimal((base_percent / 100)) * predicted_units
            incremental_units = decimal.Decimal((inc_percent / 100)) * predicted_units
            promo_depth  = data[data_values.index('tpr_discount')] - data[data_values.index('co_investment')]
            # import pdb
            # pdb.set_trace()
        # if(pricing):
        #     if(isinstance(pricing , List)):
              
        #         retail_price = retail_price + (retail_price * (pricing[i].rsp_increase)/100)
        #         list_price = list_price + (list_price * ((pricing[i].lp_increase)/100))
        #         cogs = cogs + (cogs * ((pricing[i].cogs_increase)/100))
        #     else:
        #         retail_price = retail_price + (retail_price * (decimal.Decimal(pricing['rsp'])/100))
        #         list_price = list_price + (list_price * (decimal.Decimal(pricing['lpi'])/100))
        #         cogs = cogs + (cogs * (decimal.Decimal(pricing['cogs'])/100))
            # pass
            # {'lpi': 5, 'rsp': 5, 'cogs': 4, 'elasticity': -1.18}
            # import pdb
            # pdb.set_trace()
        # print(data[data_values.index('promo_depth')] , "promo depth value for iteration " ,i )
        try:

            ob = model.UnitModel(
                data[data_values.index('date')],
                week = int(data[data_values.index('week')]),
                year = int(data[data_values.index('year')]),
                quater = (data[data_values.index('quater')]),
                month = (data[data_values.index('month')]),
                period = (data[data_values.index('period')]),
                flag_promotype_motivation = int(data[data_values.index('flag_promotype_motivation')]),
                flag_promotype_n_pls_1 = int(data[data_values.index('flag_promotype_n_pls_1')]),
                flag_promotype_traffic = int(data[data_values.index('flag_promotype_traffic')]),
                si = data[data_values.index('si')],
                predicted_units= predicted_units,
                on_inv_percent=roi[roi_values.index('on_inv')] * 100,
                list_price = list_price,
                promo_depth=promo_depth,
                off_inv_percent = roi[roi_values.index('off_inv')] * 100, 
                gmac_percent_lsv =gmac_percent_lsv,

                # average_selling_price = data[data_values.index('wk_sold_avg_price_byppg')],
                product_group_weight_in_grams = data[data_values.index('weighted_weight_in_grams')], 
                median_base_price_log = retail_price,
                incremental_unit = incremental_units,
                base_unit = base_units,
                promo_elasticity=promo_elasticity,
                co_investment = decimal.Decimal(data[data_values.index('co_investment')]),
                cogs = cogs,
                is_vat_applied=is_vat_applied,
                royalty_increase = 0.0 if util._transform_corporate_segment(
                    data[data_values.index('model_meta__corporate_segment')]
                ) == 'Choco' else 0.5
                
            )
            # import pdb
            # pdb.set_trace()
            update_total(total_units , ob)
            ob_dict = ob.__dict__
            ob_dict['holiday'] = get_holiday_information(data)
            weekly_units.append(ob_dict)
        except Exception as e:
            raise e
            # import pdb
            # pdb.set_trace()
            # passa
    # import pdb
    # pdb.set_trace()
    aggregate_total(total_units)
    
    return {
        flag : {
            'total' :  total_units.__dict__,
            'weekly' : weekly_units
        }
       
    }

def calculate_financial_mertrics_from_pricing( data_list ,roi_list,unit_info , flag,pricing_week:List[db_model.PricingWeek]):
    '''
    To calculate financial metrics for each week as well as total
    '''
  
    weekly_units = []
    total_units = model.TotalUnit()
    
    for i in range(0,len(data_list)):
      
        roi = roi_list[i]
        unit = unit_info[i]
        data = data_list[i]
        
        ob = model.UnitModelPrice(
            data[data_values.index('date')],
            week = int(data[data_values.index('week')]),
            year = int(data[data_values.index('year')]),
            quater = (data[data_values.index('quater')]),
            month = (data[data_values.index('month')]),
            period = (data[data_values.index('period')]),
            flag_promotype_motivation = int(data[data_values.index('flag_promotype_motivation')]),
            flag_promotype_n_pls_1 = int(data[data_values.index('flag_promotype_n_pls_1')]),
            flag_promotype_traffic = int(data[data_values.index('flag_promotype_traffic')]),
            si = data[data_values.index('si')],
            predicted_units=decimal.Decimal(unit['Predicted_sales']),
            on_inv_percent=roi[roi_values.index('on_inv')] * 100,
            list_price = pricing_week[i].lp_increase,
            promo_depth=decimal.Decimal(data[data_values.index('promo_depth')]),
            off_inv_percent = roi[roi_values.index('off_inv')] * 100, 
            gmac_percent_lsv = roi[roi_values.index('gmac')] * 100,
            # average_selling_price = data[data_values.index('wk_sold_avg_price_byppg')],
            product_group_weight_in_grams = data[data_values.index('weighted_weight_in_grams')], 
            # median_base_price_log = math.log(pricing_week[i].rsp_increase),
            median_base_price_log = data[data_values.index('median_base_price_log')],
            incremental_unit = decimal.Decimal(unit['Incremental']),
            base_unit = decimal.Decimal(unit['Base']),
            promo_elasticity=0,
            co_investment = decimal.Decimal(data[data_values.index('co_investment')]),
            mars_cogs_per_unit = pricing_week[i].cogs_increase,
            is_vat_applied=data[data_values.index('model_meta__account_name')] != 'Lenta',
                royalty_increase = 0.0 if util._transform_corporate_segment(
                    data[data_values.index('model_meta__corporate_segment')]
                ) == 'Choco' else 0.5
            #  is_vat_applied=data_values.index('model_meta__account_name') != 'Lenta',
            #    royalty_increase = 0.0 if util._transform_corporate_segment(
            #         data_values.index('model_meta__corporate_segment')
            #     ) == 'Choco' else 0.5
            
        )
        update_pricing_total(total_units , ob)
        ob_dict = ob.__dict__
        ob_dict['holiday'] = get_holiday_information(data)
        weekly_units.append(ob_dict)
    # print(weekly_units)
    aggregate_total(total_units)
    
    return {
        flag : {
            'total' :  total_units.__dict__,
            'weekly' : weekly_units
        }
       
    }
    
def aggregate_total(total_unit:model.TotalUnit):
    total_unit.rp_percent = util._divide(total_unit.rp,total_unit.total_rsv_w_o_vat) * 100
    total_unit.te_percent_of_lsv = util._divide(total_unit.te,total_unit.lsv) * 100
    total_unit.mac_percent = util._divide( total_unit.mac,total_unit.nsv)* 100
    total_unit.te_per_unit = util._divide( total_unit.te,total_unit.volume)
    total_unit.lift = util._divide(total_unit.increment_units,total_unit.base_units)
    total_unit.roi = util._divide(total_unit.uplift_gmac_lsv,total_unit.total_uplift_cost)
    total_unit.asp = util._divide(total_unit.asp,52)
    # lift_ = (unit_model.incremental_unit / unit_model.base_unit)
    
    # pass



def update_pricing_total(total_unit:model.TotalUnit ,unit_model : model.PricingUnit ):
    total_unit.total_rsv_w_o_vat = total_unit.total_rsv_w_o_vat + unit_model.total_rsv_w_o_vat
    total_unit.cogs = total_unit.cogs + unit_model.total_cogs
    total_unit.units = total_unit.units + unit_model.predicted_units
    total_unit.te = total_unit.te + unit_model.trade_expense
    total_unit.lsv = total_unit.lsv + unit_model.total_lsv
    total_unit.nsv = total_unit.nsv + unit_model.total_nsv
    total_unit.mac = total_unit.mac + unit_model.mars_mac
    total_unit.rp = total_unit.rp + unit_model.retailer_margin
    total_unit.asp = total_unit.asp + unit_model.asp
    # util.average(total_unit.asp,unit_model.asp) 
    # import pdb
    # pdb.set_trace()
    total_unit.uplift_gmac_lsv = total_unit.uplift_gmac_lsv + unit_model.uplift_gmac_lsv
    total_unit.total_uplift_cost= total_unit.total_uplift_cost + unit_model.total_uplift_cost  
    # if (unit_model.promo_depth + unit_model.co_investment):
    #     total_unit.avg_promo_selling_price =  util.average(total_unit.avg_promo_selling_price,unit_model.promo_asp) 
    # total_unit.roi = total_unit.roi + unit_model.roi
    # if(unit_model.roi):
    #     # print(unit_model.roi , "roi")
    #     total_unit.roi = util.average(total_unit.roi,unit_model.roi)
        # print(total_unit.roi , "average")
    # total_unit.rp_percent = util.average(total_unit.rp_percent,unit_model.retailer_margin_percent_of_rsp)
     
    # total_unit.mac_percent = util.average( total_unit.mac_percent,unit_model.mars_mac_percent_of_nsv)
   
    total_unit.volume = total_unit.volume  + unit_model.total_weight_in_tons
    # total_unit.te_per_unit = total_unit.te_per_unit + unit_model.te_per_units
    # total_unit.te_percent_of_lsv = util.average(total_unit.te_percent_of_lsv ,unit_model.te_percent_of_lsv)
    total_unit.base_units = total_unit.base_units + unit_model.base_unit
    total_unit.increment_units =total_unit.increment_units + unit_model.incremental_unit


def update_total(total_unit:model.TotalUnit ,unit_model : model.UnitModel ):
    total_unit.total_rsv_w_o_vat = total_unit.total_rsv_w_o_vat + unit_model.total_rsv_w_o_vat
    total_unit.cogs = total_unit.cogs + unit_model.total_cogs
    total_unit.units = total_unit.units + unit_model.predicted_units
    total_unit.te = total_unit.te + unit_model.trade_expense
    total_unit.lsv = total_unit.lsv + unit_model.total_lsv
    total_unit.nsv = total_unit.nsv + unit_model.total_nsv
    total_unit.mac = total_unit.mac + unit_model.mars_mac
    total_unit.rp = total_unit.rp + unit_model.retailer_margin
    total_unit.asp = total_unit.asp + unit_model.asp
    # util.average(total_unit.asp,unit_model.asp) 
    # import pdb
    # pdb.set_trace()
    total_unit.uplift_gmac_lsv = total_unit.uplift_gmac_lsv + unit_model.uplift_gmac_lsv
    total_unit.total_uplift_cost= total_unit.total_uplift_cost + unit_model.total_uplift_cost  
    if (unit_model.promo_depth + unit_model.co_investment):
        total_unit.avg_promo_selling_price =  util.average(total_unit.avg_promo_selling_price,unit_model.promo_asp) 
    # total_unit.roi = total_unit.roi + unit_model.roi
    # if(unit_model.roi):
    #     # print(unit_model.roi , "roi")
    #     total_unit.roi = util.average(total_unit.roi,unit_model.roi)
        # print(total_unit.roi , "average")
    # total_unit.rp_percent = util.average(total_unit.rp_percent,unit_model.retailer_margin_percent_of_rsp)
     
    # total_unit.mac_percent = util.average( total_unit.mac_percent,unit_model.mars_mac_percent_of_nsv)
   
    total_unit.volume = total_unit.volume  + unit_model.total_weight_in_tons
    # total_unit.te_per_unit = total_unit.te_per_unit + unit_model.te_per_units
    # total_unit.te_percent_of_lsv = util.average(total_unit.te_percent_of_lsv ,unit_model.te_percent_of_lsv)
    total_unit.base_units = total_unit.base_units + unit_model.base_unit
    total_unit.increment_units =total_unit.increment_units + unit_model.incremental_unit
    # lift_ = (unit_model.incremental_unit / unit_model.base_unit)
    # if(lift_):
    #     total_unit.lift = util.average(total_unit.lift, lift_)
        # total_unit.lift + 
     
    # pass
        # unit_info[0]['Predicted_sales']
        # roi_list[0][7] # 7 -week 8->on_inv 9->off_inv 10->list_price 11->gmac
        # data_list[0] # 11->date , 12->week , 15-> tpr discount , 
        # import pdb
        # pdb.set_trace()
    
    #  ob = model.UnitModel(
    #                 predicted_units=  decimal.Decimal(math.exp(result)),
    #                 on_inv_percent=decimal.Decimal(roi_model.off_inv * 100),
    #                 list_price=decimal.Decimal(roi_model.list_price),
    #                 tpr_percent=decimal.Decimal(promo_data.tpr_discount),
    #                 off_inv_percent = decimal.Decimal(roi_model.on_inv * 100),
    #                 gmac_percent_lsv = decimal.Decimal(roi_model.gmac * 100),
    #                 average_selling_price = decimal.Decimal(promo_data.wk_sold_avg_price_byppg),
    #                 product_group_weight_in_grams = decimal.Decimal(promo_data.weighted_weight_in_grams),
    #                 median_base_price_log = decimal.Decimal(promo_data.median_base_price_log),
    #                 incremental_unit = decimal.Decimal(promo_data.incremental_unit),
    #                 base_unit = decimal.Decimal(promo_data.base_unit)
    #                 )
    # pass
  
def _get_promotion_flag(promo_from_req):
    print(promo_from_req , "promo from request")
    if promo_from_req == "TPR":
        return
    val ={"Motivation" : "flag_promotype_motivation", 
    "N+1": "flag_promotype_n_pls_1", 
        "Traffic": "flag_promotype_traffic", 
        }  
    return val.get(promo_from_req)

def update_from_request(data_list , querydict):
    # import pdb
    # pdb.set_trace()
    cloned_list = copy.deepcopy(data_list)
    cataloge_average = 0
    catalogue_index = []
    for i in querydict.keys():
        week_regex = util._regex(r'week-\d{1,2}',i)
        if week_regex:
            week = int(util._regex(r'\d{1,2}',week_regex.group()).group())
            index = week -1
            cat = cloned_list[index][data_values.index('catalogue')]
            if cat :
                cataloge_average = util.average(cataloge_average , cat)
                cloned_list[index][data_values.index('catalogue')] = 0
             
            if querydict[i]['promo_mechanics'] and querydict[i]['promo_mechanics']!="TPR":
                cloned_list[index][data_values.index(
                    _get_promotion_flag(querydict[i]['promo_mechanics']))] = 1
            cloned_list[index][data_values.index('co_investment')] = querydict[i]['co_investment']
            if querydict['param_depth_all']:
                cloned_list[index][data_values.index('promo_depth')] =querydict['param_depth_all']
                if index + 1 < len(cloned_list):
                    cloned_list[index+1][data_values.index('tpr_discount_lag1')] = querydict['param_depth_all']
                if index + 2 < len(cloned_list):
                    cloned_list[index+2][data_values.index('tpr_discount_lag2')] =querydict['param_depth_all']
                catalogue_index.append(index)
            else: 
                cloned_list[index][data_values.index('promo_depth')] = querydict[i]['promo_depth']
                if index + 1 < len(cloned_list):
                    cloned_list[index+1][data_values.index('tpr_discount_lag1')] = querydict[i]['promo_depth']
                if index + 2 < len(cloned_list):
                    cloned_list[index+2][data_values.index('tpr_discount_lag2')] =querydict[i]['promo_depth']
                if querydict[i]['promo_depth']:
                    catalogue_index.append(index)
    for value in catalogue_index:
        cloned_list[value][data_values.index('catalogue')] = cataloge_average
    return cloned_list

def update_for_optimizer(data_list , querydict):
    cloned_list = copy.deepcopy(data_list)
    cataloge_average = 0
    catalogue_index = []
    for i in range(0,len(querydict)):
        week = querydict[i]['week']
        index = week -1
        cat = cloned_list[index][data_values.index('catalogue')]
        if cat :
            cataloge_average = util.average(cataloge_average , cat)
            cloned_list[index][data_values.index('catalogue')] = 0
        if(querydict[i]['Mechanic'] in ['N + 1', 'N+1', '2 + 1 free',
                    '1 + 1 free', '3 + 1 free']):
            cloned_list[index][data_values.index('flag_promotype_n_pls_1')] = 1
        if(querydict[i]['Mechanic'] in ['Motivation', 'motivation',
                    'Motivational']):
            cloned_list[index][data_values.index('flag_promotype_motivation')] = 1
        if(not querydict[i]['Mechanic']):
            cloned_list[index][data_values.index('flag_promotype_n_pls_1')] = 0
            cloned_list[index][data_values.index('flag_promotype_motivation')] = 0
            
            
        cloned_list[index][data_values.index('co_investment')] = querydict[i]['Coinvestment']
        cloned_list[index][data_values.index('promo_depth')] = querydict[i]['Optimum_Promo']
        if index + 1 < len(cloned_list):
            cloned_list[index+1][data_values.index('tpr_discount_lag1')] = querydict[i]['Optimum_Promo']
        if index + 2 < len(cloned_list):
            cloned_list[index+2][data_values.index('tpr_discount_lag2')] =querydict[i]['Optimum_Promo']
        if querydict[i]['Optimum_Promo']:
                catalogue_index.append(index)
    for value in catalogue_index:
        cloned_list[value][data_values.index('catalogue')] = cataloge_average
        
    # import pdb
    # pdb.set_trace()
    return cloned_list
    # cloned_list = copy.deepcopy(data_list)
    # for i in range(0,len(querydict)):
    #     week = querydict[i]['week']
    #     index = week -1
    #     cloned_list[index]['Promo_Depth'] = querydict[i]['Optimum_Promo']
    #     cloned_list[index]['Coinvestment'] = querydict[i]['Coinvestment']
    # return cloned_list

def update_optimizer_from_pricing(data_list , pricing_week : QuerySet[db_model.PricingWeek] , roi_list):
    cloned_list = copy.deepcopy(data_list)
    cloned_roi = copy.deepcopy(roi_list)
    for week in pricing_week:
        index = week.week - 1
        cloned_list[index][data_values.index('median_base_price_log')] = math.log(week.rsp_increase)
        if roi_list:
            cloned_roi[index][roi_values.index('list_price')] = week.lp_increase
            if len(cloned_roi[index]) > len(roi_values):
                cloned_roi[index][-1] = week.cogs_increase
    return cloned_list , roi_list

def update_from_saved_data(data_list , promo_week : QuerySet[db_model.PromoWeek] , roi_list = None):
    cloned_list = copy.deepcopy(data_list)
    if roi_list : 
        cloned_roi = copy.deepcopy(roi_list)
    saved_pricing = promo_week[0].pricing_save.saved_pricing
    if saved_pricing:
        pricing_week = db_model.PricingWeek.objects.filter(pricing_save = saved_pricing)
        for week in pricing_week:
            index = week.week - 1
            cloned_list[index][data_values.index('median_base_price_log')] = math.log(week.rsp_increase)
            if roi_list:
                cloned_roi[index][roi_values.index('list_price')] = week.lp_increase
                if len(cloned_roi[index]) > len(roi_values):
                    cloned_roi[index][-1] = week.cogs_increase
               
    if promo_week:
        for week in promo_week:
            index = week.week - 1
            cloned_list[index][data_values.index('co_investment')] = week.co_investment
            if week.promo_mechanic:
                cloned_list[index][data_values.index(
                        _get_promotion_flag(week.promo_mechanic))] = 1
            cloned_list[index][data_values.index('promo_depth')] = week.promo_depth
            if index + 1 < len(promo_week):
                cloned_list[index+1][data_values.index('tpr_discount_lag1')] = week.promo_depth
            if index + 2 < len(promo_week):
                cloned_list[index+2][data_values.index('tpr_discount_lag2')] =week.promo_depth
    if roi_list:
        return cloned_list , roi_list
    return cloned_list


def update_from_optimizer(data_list , optimizer_week:List[db_model.OptimizerSave]):
    cloned_list = copy.deepcopy(data_list)
    cataloge_average = 0
    catalogue_index = []
    for week in optimizer_week:  
        index = week.week -1
        cat = cloned_list[index][data_values.index('catalogue')]
        if cat :
            cataloge_average = util.average(cataloge_average , cat)
            cloned_list[index][data_values.index('catalogue')] = 0
            
        if week.optimum_promo:
            catalogue_index.append(index)
        if(week.mechanic in ['N + 1', 'N+1', '2 + 1 free',
                    '1 + 1 free', '3 + 1 free']):
            cloned_list[index][data_values.index('flag_promotype_n_pls_1')] = 1
        if(week.mechanic in ['Motivation', 'motivation',
                    'Motivational']):
            cloned_list[index][data_values.index('flag_promotype_motivation')] = 1
        if(not week.mechanic):
            cloned_list[index][data_values.index('flag_promotype_n_pls_1')] = 0
            cloned_list[index][data_values.index('flag_promotype_motivation')] = 0
        cloned_list[index][data_values.index('co_investment')]= week.optimum_co_investment
        cloned_list[index][data_values.index('promo_depth')] = week.optimum_promo
        # if week.mechanic in (['N + 1', 'N+1', '2 + 1 free','1 + 1 free', '3 + 1 free']):
        #     cloned_list[index][data_values.index('flag_promotype_n_pls_1')] = 1
        # if week.mechanic in (['Motivation', 'motivation','Motivational']):
        #     cloned_list[index][data_values.index('flag_promotype_motivation')] = 1
        if index + 1 < len(optimizer_week):
            cloned_list[index+1][data_values.index('tpr_discount_lag1')] = week.optimum_promo
        if index + 2 < len(optimizer_week):
            cloned_list[index+2][data_values.index('tpr_discount_lag2')] = week.optimum_promo
    for value in catalogue_index:
        cloned_list[value][data_values.index('catalogue')] = cataloge_average
    return cloned_list

def update_price(data_list,filtered_roi,filtered_coeff,product):
    # util.is_date_greater_or_equal(price['date'] , form['list_price_date']
   
    cloned_list = copy.deepcopy(data_list)
    cloned_roi = copy.deepcopy(filtered_roi)
    cloned_coeff = copy.deepcopy(filtered_coeff)
    form = product[0]
    const_tpr = form['is_tpr_constant']
    cataloge_average = 0
    catalogue_index = []
    if(form['follow_competition']):
        cloned_coeff[0][coeff_values.index('median_base_price_log')] = form['inc_net_elasticity']
    else:
        cloned_coeff[0][coeff_values.index('median_base_price_log')] = form['inc_elasticity']
    for i in range(0,len(cloned_list)):
        current_rp = decimal.Decimal(math.exp(cloned_list[i][data_values.index('median_base_price_log')]))
        new_rp = current_rp
        tpr = cloned_list[i][data_values.index('tpr_discount')]
        new_tpr = tpr
        new_rp = current_rp * (1 + (decimal.Decimal(form['inc_rsp'])/100))
        if(tpr):
            if util.is_date_greater_or_equal(cloned_list[i][data_values.index('date')] , form['promo_date']):
                current_promo_price = current_rp * (1 -(tpr/100))
                new_promo_price = current_promo_price * (1 + (decimal.Decimal(form['inc_promo_price'])/100))
                if not const_tpr:
                    new_tpr= ((new_rp - new_promo_price)/new_rp) * 100
        cat = cloned_list[i][data_values.index('catalogue')]
        if cat:
            cataloge_average = util.average(cataloge_average , cat)
            cloned_list[i][data_values.index('catalogue')] = 0
        cloned_list[i][data_values.index('tpr_discount')] = new_tpr
        if i + 1 < len(cloned_list):
            cloned_list[i+1][data_values.index('tpr_discount_lag1')] = new_tpr
        if i + 2 < len(cloned_list):
            cloned_list[i+2][data_values.index('tpr_discount_lag2')] =new_tpr
        if new_tpr:
            catalogue_index.append(i)
        if util.is_date_greater_or_equal(cloned_list[i][data_values.index('date')] , form['rsp_date']):
            cloned_list[i][data_values.index('median_base_price_log')] = decimal.Decimal(math.log(new_rp))
        baselp = cloned_roi[i][roi_values.index('list_price')]
        if util.is_date_greater_or_equal(cloned_list[i][data_values.index('date')] , form['list_price_date']):
            cloned_roi[i][roi_values.index('list_price')] = baselp + baselp * (decimal.Decimal(form['inc_list_price'])/100)
    for value in catalogue_index:
        cloned_list[value][data_values.index('catalogue')] = cataloge_average   
    # import pdb
    # pdb.set_trace() 
    return cloned_list , cloned_roi , cloned_coeff


def update_tpr_from_pricing(tpr_list , pricing_week):
     
    cloned_list = copy.deepcopy(tpr_list)
    pricing_week_list = list(pricing_week)
    for i in range(0,len(cloned_list)):
        current_rp = pricing_week_list[i].base_retail_price
        new_rp = current_rp
        tpr = cloned_list[i]['tpr_discount']
        new_tpr = tpr
        
        if(tpr):
            new_rp = current_rp * (1 + (pricing_week_list[i].rsp_increase)/100)
            current_promo_price = current_rp * (1 -(tpr/100))
            new_promo_price = current_promo_price * (1 + (pricing_week_list[i].promo_increase/100))
            new_tpr= ((new_rp - new_promo_price)/new_rp) * 100
            new_tpr =round(decimal.Decimal(new_tpr),2)
            # print(new_tpr , "newtPR" , type(new_tpr))
        
        cloned_list[i]['promo_depth'] = decimal.Decimal(new_tpr) - cloned_list[i]['co_investment']
        cloned_list[i]['tpr_discount'] = new_tpr
     
    return cloned_list 


def update_from_pricing(data_list,filtered_roi,filtered_coeff, pricing_week,promo_week : QuerySet[db_model.PromoWeek] ):
   
    cloned_list = copy.deepcopy(data_list)
    cloned_roi = copy.deepcopy(filtered_roi)
    cloned_coeff = copy.deepcopy(filtered_coeff)
    pricing_save_obj = pricing_week[0].pricing_save
    form= {
        "follow_competition" : pricing_save_obj.follow_competition,
        "inc_net_elasticity" : pricing_save_obj.inc_net_elasticity,
        "inc_elasticity" :  pricing_save_obj.inc_elasticity
        
    }
    # pricing_week[0]
    cataloge_average = 0
    catalogue_index = []
    if(form['follow_competition']):
        cloned_coeff[0][coeff_values.index('median_base_price_log')] = form['inc_net_elasticity']
    else:
        cloned_coeff[0][coeff_values.index('median_base_price_log')] = form['inc_elasticity']
    for i in range(0,len(cloned_list)):
        current_rp = decimal.Decimal(math.exp(cloned_list[i][data_values.index('median_base_price_log')]))
        new_rp = current_rp
        tpr = cloned_list[i][data_values.index('tpr_discount')]
        new_tpr = tpr
        
        if(tpr):
            new_rp = current_rp * (1 + (pricing_week[i].rsp_increase)/100)
            current_promo_price = current_rp * (1 -(tpr/100))
            new_promo_price = current_promo_price * (1 + (pricing_week[i].promo_increase/100))
            new_tpr= ((new_rp - new_promo_price)/new_rp) * 100
            new_tpr =round(decimal.Decimal(new_tpr),2)
            # print(new_tpr , "newtPR" , type(new_tpr))
        cat = cloned_list[i][data_values.index('catalogue')]
        if cat:
            cataloge_average = util.average(cataloge_average , cat)
            cloned_list[i][data_values.index('catalogue')] = 0
        cloned_list[i][data_values.index('promo_depth')] = decimal.Decimal(new_tpr) - cloned_list[i][data_values.index('co_investment')]
        cloned_list[i][data_values.index('tpr_discount')] = new_tpr
        if i + 1 < len(cloned_list):
            cloned_list[i+1][data_values.index('tpr_discount_lag1')] = new_tpr
        if i + 2 < len(cloned_list):
            cloned_list[i+2][data_values.index('tpr_discount_lag2')] =new_tpr
        if new_tpr:
            catalogue_index.append(i)
        cloned_list[i][data_values.index('median_base_price_log')] = math.log(new_rp)
        baselp = cloned_roi[i][roi_values.index('list_price')]
        cloned_roi[i][roi_values.index('list_price')] = baselp + baselp * (pricing_week[i].lp_increase/100)
    for value in catalogue_index:
        cloned_list[value][data_values.index('catalogue')] = cataloge_average    
    return cloned_list , cloned_roi , cloned_coeff , form

    
    
    
    
    
    
    
    # if promo_week:
    
    #     promo_week = list(promo_week)
        
        
        
    #     cataloge_average = 0
    #     catalogue_index = []
    #     for i in range(0,len(promo_week)):
    #         week = promo_week[i].week
    #         index = week -1
    #         cat = cloned_list[index][data_values.index('catalogue')]
    #         if cat :
    #             cataloge_average = util.average(cataloge_average , cat)
    #             cloned_list[index][data_values.index('catalogue')] = 0
    #         if(promo_week[i].promo_mechanic in ['N + 1', 'N+1', '2 + 1 free',
    #                     '1 + 1 free', '3 + 1 free']):
    #             cloned_list[index][data_values.index('flag_promotype_n_pls_1')] = 1
    #         if(promo_week[i].promo_mechanic in ['Motivation', 'motivation',
    #                     'Motivational']):
    #             cloned_list[index][data_values.index('flag_promotype_motivation')] = 1
    #         if(not promo_week[i].promo_mechanic):
    #             cloned_list[index][data_values.index('flag_promotype_n_pls_1')] = 0
    #             cloned_list[index][data_values.index('flag_promotype_motivation')] = 0
                
                
    #         cloned_list[index][data_values.index('co_investment')] =promo_week[i].co_investment
    #         cloned_list[index][data_values.index('promo_depth')] = promo_week[i].promo_depth
    #         if index + 1 < len(cloned_list):
    #             cloned_list[index+1][data_values.index('tpr_discount_lag1')] = promo_week[i].promo_depth
    #         if index + 2 < len(cloned_list):
    #             cloned_list[index+2][data_values.index('tpr_discount_lag2')] =promo_week[i].promo_depth
    #         if promo_week[i].promo_depth:
    #                 catalogue_index.append(index)
    #     for value in catalogue_index:
    #         cloned_list[value][data_values.index('catalogue')] = cataloge_average
            
    
    # return cloned_list


# def update_from_pricing(data_list , promo_week : QuerySet[db_model.PromoWeek] , roi_list = None):
       
#     cloned_list = copy.deepcopy(data_list)
#     if promo_week:
    
#         promo_week = list(promo_week)
#         # import pdb
#         # pdb.set_trace()
        
        
#         cataloge_average = 0
#         catalogue_index = []
#         for i in range(0,len(promo_week)):
#             week = promo_week[i].week
#             index = week -1
#             cat = cloned_list[index][data_values.index('catalogue')]
#             if cat :
#                 cataloge_average = util.average(cataloge_average , cat)
#                 cloned_list[index][data_values.index('catalogue')] = 0
#             if(promo_week[i].promo_mechanic in ['N + 1', 'N+1', '2 + 1 free',
#                         '1 + 1 free', '3 + 1 free']):
#                 cloned_list[index][data_values.index('flag_promotype_n_pls_1')] = 1
#             if(promo_week[i].promo_mechanic in ['Motivation', 'motivation',
#                         'Motivational']):
#                 cloned_list[index][data_values.index('flag_promotype_motivation')] = 1
#             if(not promo_week[i].promo_mechanic):
#                 cloned_list[index][data_values.index('flag_promotype_n_pls_1')] = 0
#                 cloned_list[index][data_values.index('flag_promotype_motivation')] = 0
                
                
#             cloned_list[index][data_values.index('co_investment')] =promo_week[i].co_investment
#             cloned_list[index][data_values.index('promo_depth')] = promo_week[i].promo_depth
#             if index + 1 < len(cloned_list):
#                 cloned_list[index+1][data_values.index('tpr_discount_lag1')] = promo_week[i].promo_depth
#             if index + 2 < len(cloned_list):
#                 cloned_list[index+2][data_values.index('tpr_discount_lag2')] =promo_week[i].promo_depth
#             if promo_week[i].promo_depth:
#                     catalogue_index.append(index)
#         for value in catalogue_index:
#             cloned_list[value][data_values.index('catalogue')] = cataloge_average
#         # if roi_list:
#         #     cloned_roi = copy.deepcopy(roi_list)
#         # for week in pricing_week:
#         #     index = week.week - 1
#         #     cloned_list[index][data_values.index('median_base_price_log')] = math.log(week.rsp_increase)
            
    
#     return cloned_list

def update_from_pricing_promo(data_list , pricing_week : List[db_model.PricingWeek] , promo_week : List[db_model.PromoWeek]):
    cloned_list = copy.deepcopy(data_list)
    
    for week in pricing_week:
        # print(week , "Pricing update...")
        index = week.week - 1
        cloned_list[index][data_values.index('median_base_price_log')] = math.log(week.rsp_increase)
    
    if promo_week:
        for week in promo_week:
            # print(week , "Promo update...")
            index = week.week - 1
            
            if week.promo_mechanic:
                cloned_list[index][data_values.index(
                        _get_promotion_flag(week.promo_mechanic))] = 1
            cloned_list[index][data_values.index('co_investment')] = week.co_investment
            cloned_list[index][data_values.index('promo_depth')] = week.promo_depth
            if index + 1 < len(promo_week):
                cloned_list[index+1][data_values.index('tpr_discount_lag1')] = week.promo_depth
            if index + 2 < len(promo_week):
                cloned_list[index+2][data_values.index('tpr_discount_lag2')] = week.promo_depth
    return cloned_list



def calculate_financial_mertrics_from_pricing_promo(data_list ,roi_list,unit_info , flag,pricing_week:List[db_model.PricingWeek],promo_elasticity = 0):
    weekly_units = []
    total_units = model.TotalUnit()
    
    for i in range(0,len(data_list)):
      
        roi = roi_list[i]
        unit = unit_info[i]
        data = data_list[i]
        
        ob = model.UnitModelPrice(
            data[data_values.index('date')],
            week = int(data[data_values.index('week')]),
            predicted_units=decimal.Decimal(unit['Predicted_sales']),
            on_inv_percent=roi[roi_values.index('on_inv')] * 100,
            list_price = pricing_week[i].lp_increase,
            promo_depth=decimal.Decimal(data[data_values.index('promo_depth')]),
            off_inv_percent = roi[roi_values.index('off_inv')] * 100, 
            gmac_percent_lsv = roi[roi_values.index('gmac')] * 100,
            # average_selling_price = data[data_values.index('wk_sold_avg_price_byppg')],
            product_group_weight_in_grams = data[data_values.index('weighted_weight_in_grams')], 
            # median_base_price_log = math.log(pricing_week[i].rsp_increase),
            median_base_price_log = data[data_values.index('median_base_price_log')],
            incremental_unit = decimal.Decimal(unit['Incremental']),
            base_unit = decimal.Decimal(unit['Base']),
            promo_elasticity=promo_elasticity,
            co_investment = decimal.Decimal(data[data_values.index('co_investment')]),
            mars_cogs_per_unit = pricing_week[i].cogs_increase
            
        )
        update_total(total_units , ob)
        ob_dict = ob.__dict__
        ob_dict['holiday'] = get_holiday_information(data)
        weekly_units.append(ob_dict)
    
    return {
        flag : {
            'total' :  total_units.__dict__,
            'weekly' : weekly_units
        }
       
    }



       
       
def calculate_financial_mertrics_for_pricing_request(price_list , flag):
    '''
    To calculate financial metrics for each week as well as total
    '''

   
    
    retailers = []
    for key, price_list in groupby(price_list,
                          key = itemgetter('account_name' , 'product_group')):
        # import pdb
        # pdb.set_trace()
        retailer = {}
        retailer['account_name'] = key[0]
        retailer['product'] = key[1]
        weekly_units = []
        total_units = model.TotalUnit()
        price_list = list(price_list)
        # print(price_list , "price list list")

    
    
        for price in price_list:
            #equation base_price_change_promo_const
            
            # if(flag == 'simulated'):
            #     import pdb
            #     pdb.set_trace()
            # base_price_change_promo_const()
            try:

                ob = model.PricingUnit(
                    price['week'],
                    price['year'],
                    price['date'],
                    price['quarter'],
                    price['base_units'],
                    price['base_split'],
                    price['incremental_split'],
                    price['list_price'],
                    price['retail_median_base_price_w_o_vat'],
                    price['cogs'],
                    price['on_inv'],
                    price['off_inv'],
                    price['tpr_discount'],
                    price['gmac'],
                    price['product_weight_in_grams'],
                    
                    royalty_increase = 0.0 if util._transform_corporate_segment(
                    price['corporate_segment']
                    ) == 'Choco' else 0.5,  
                    is_vat_applied= price['account_name'] != 'Lenta',
                    
                    
                    
                )
                # import pdb
                # pdb.set_trace()
                update_pricing_total(total_units , ob)
                ob_dict = ob.__dict__
                weekly_units.append(ob_dict)
            except Exception as e:
                raise e
                # import pdb
                # pdb.set_trace()
                # passa
        # import pdb
        # pdb.set_trace()
        aggregate_total(total_units)
        retailer['total'] =total_units.__dict__
        retailer['weekly'] = weekly_units
        retailers.append(retailer)
        
        #      'total' :  
        #     'weekly' : 
            
        # }
    
    return {
        flag : retailers
       
    }

def update_increased_pricing(pricing_list , form_data):
    cloned_list =  copy.deepcopy(pricing_list)
    print("-------------------------------------------")
    print(form_data , "update increased pricing formdata")
    print("-------------------------------------------")
    
    #   this.new_base_units =
    #   this.competition == 'Follows'
    #     ? this.base_units *
    #       (1 +
    #         this.net_elasticity *
    #           ((this.suggested_retailer_median_base_price_w_o_vat -
    #             this.retailer_median_base_price_w_o_vat) /
    #             this.retailer_median_base_price_w_o_vat))
    #     : this.base_units *
    #       (1 +
    #         this.base_price_elasticity *
    #           ((this.suggested_retailer_median_base_price_w_o_vat -
    #             this.retailer_median_base_price_w_o_vat) /
    #             this.retailer_median_base_price_w_o_vat));

    local_memory = {
    }
    for price in cloned_list:
        
        if((price['account_name']+ price['product_group']) in local_memory):
            form = local_memory[(price['account_name']+ price['product_group'])]

        else:

            form = list(filter(lambda forms: (price['account_name'] == forms['account_name'] and price['product_group'] == forms['product_group']), form_data))[0]
            local_memory[form['retailer']] = form
        # import pdb
        # pdb.set_trace()
        if('list_price_date' in form and util.is_date_greater_or_equal(price['date'] , form['list_price_date'])):
            if(form["inc_list_price"]):
                price["list_price_change"] = True
            price['list_price'] = float(float(form["list_price"]) + (float(form["list_price"]) * float(form['inc_list_price']))/100)
        # print()
        if('rsp_date' in form and util.is_date_greater_or_equal(price['date'] , form['rsp_date'])):
            if(form["inc_rsp"]):
                price['retail_price_change'] = True
            price['retail_median_base_price_w_o_vat'] = float(float(form['rsp']) + (float(form['rsp']) * float(form['inc_rsp']))/100)    
        if('cogs_date' in  form and util.is_date_greater_or_equal(price['date'] , form['cogs_date'])):
            if(form['inc_cogs']):
                price['cogs_change'] = True
            price['cogs'] = float(float(form['cogs']) + (float(form['cogs']) * float(form['inc_cogs']))/100)    
       
       
         
        price['base_price_elasticity'] = float(form['inc_elasticity'])
        price['net_elasticity'] = float(form['inc_net_elasticity'])
        
    # import pdb
    # pdb.set_trace()
    return cloned_list

def update_increased_pricing_loading(price_data , pricing_list , pricing_week):
    cloned_list = copy.deepcopy(price_data)
    # updated_list = []
    
    pricing_week_list = list(pricing_week)
    # pricing_week_list[0].pricing_save.account_name
    # pricing_week_list[0].pricing_save.product_group
    # pricing_week_list[0].week
    
    for i in range(0,len(pricing_week_list)):
        acc  = pricing_week_list[i].pricing_save.account_name
        prod = pricing_week_list[i].pricing_save.product_group
        week  = pricing_week_list[i].week
        lpi = pricing_week_list[i].lp_increase
        rsp = pricing_week_list[i].rsp_increase
        cogs = pricing_week_list[i].cogs_increase
        # import pdb
        # pdb.set_trace()
        update = next((item for item in cloned_list 
                       if item["account_name"] == acc
                       and item['product_group'] == prod
                       and item["week"] == week), None)
        if(update):
            update['retail_median_base_price_w_o_vat'] = float(rsp)
            update['list_price'] = float(lpi)
            update['cogs'] = float(cogs)
            # updated_list.append(update)
    return cloned_list


def _update_log(log_val , inc):
    base = math.exp(log_val)
    inc = base + base * (inc/100) 
    return math.log(inc)
    

    
    # import pdb
    # pdb.set_trace()
    #   this.new_base_units =
    #   this.competition == 'Follows'
    #     ? this.base_units *
    #       (1 +
    #         this.net_elasticity *
    #           ((this.suggested_retailer_median_base_price_w_o_vat -
    #             this.retailer_median_base_price_w_o_vat) /
    #             this.retailer_median_base_price_w_o_vat))
    #     : this.base_units *
    #       (1 +
    #         this.base_price_elasticity *
    #           ((this.suggested_retailer_median_base_price_w_o_vat -
    #             this.retailer_median_base_price_w_o_vat) /
    #             this.retailer_median_base_price_w_o_vat));

    # local_memory = {
    # }
    # for price in pricing_list:
    #     if((price['account_name']+ price['product_group']) in local_memory):
    #         form = local_memory[(price['account_name']+ price['product_group'])]

    #     else:

    #         form = list(filter(lambda forms: (price['account_name'] == forms['account_name'] and price['product_group'] == forms['product_group']), form_data))[0]
    #         local_memory[form['retailer']] = form
    #     price['list_price'] = float(form['inc_list_price'])
    #     price['retail_median_base_price_w_o_vat'] = float(form['inc_rsp']) 
    #     price['base_price_elasticity'] = float(form['inc_elasticity'])
    #     price['net_elasticity'] = float(form['inc_net_elasticity'])
    #     price['cogs'] = float(form['inc_cogs'])
    # return pricing_list

        # price[]
        # pass
        
        # if(price['account_name'] == )