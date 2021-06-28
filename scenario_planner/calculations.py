from typing import List
from django.db.models.fields import DateField
from django.db.models.query import QuerySet
# from core.models import db_model.ModelData, db_model.ModelMeta
from core import models as db_model
from utils import util as util
from utils import models as model
from .query import roi_values , data_values
import math
import decimal
import copy

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
 
def calculate_financial_mertrics( data_list ,roi_list,unit_info , flag,promo_elasticity = 0):
    '''
    To calculate financial metrics for each week as well as total
    '''
  
    weekly_units = []
    total_units = model.TotalUnit()
    
    for i in range(0,len(data_list)):
      
        roi = roi_list[i]
        unit = unit_info[i]
        data = data_list[i]
        
        ob = model.UnitModel(
            data[data_values.index('date')],
            week = int(data[data_values.index('week')]),
            predicted_units=decimal.Decimal(unit['Predicted_sales']),
            on_inv_percent=roi[roi_values.index('on_inv')] * 100,
            list_price = roi[roi_values.index('list_price')],
            promo_depth=decimal.Decimal(data[data_values.index('promo_depth')]),
            off_inv_percent = roi[roi_values.index('off_inv')] * 100, 
            gmac_percent_lsv = roi[roi_values.index('gmac')] * 100,
            # average_selling_price = data[data_values.index('wk_sold_avg_price_byppg')],
            product_group_weight_in_grams = data[data_values.index('weighted_weight_in_grams')], 
            median_base_price_log = data[data_values.index('median_base_price_log')],
            incremental_unit = decimal.Decimal(unit['Incremental']),
            base_unit = decimal.Decimal(unit['Base']),
            promo_elasticity=promo_elasticity,
            co_investment = decimal.Decimal(data[data_values.index('co_investment')])
            
        )
        update_total(total_units , ob)
        weekly_units.append(ob.__dict__)
    
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
            predicted_units=decimal.Decimal(unit['Predicted_sales']),
            on_inv_percent=roi[roi_values.index('on_inv')] * 100,
            list_price = pricing_week[i].lp_increase,
            promo_depth=decimal.Decimal(data[data_values.index('promo_depth')]),
            off_inv_percent = roi[roi_values.index('off_inv')] * 100, 
            gmac_percent_lsv = roi[roi_values.index('gmac')] * 100,
            # average_selling_price = data[data_values.index('wk_sold_avg_price_byppg')],
            product_group_weight_in_grams = data[data_values.index('weighted_weight_in_grams')], 
            median_base_price_log = math.log(pricing_week[i].rsp_increase),
            incremental_unit = decimal.Decimal(unit['Incremental']),
            base_unit = decimal.Decimal(unit['Base']),
            promo_elasticity=0,
            co_investment = decimal.Decimal(data[data_values.index('co_investment')]),
            mars_cogs_per_unit = pricing_week[i].cogs_increase
            
        )
        update_total(total_units , ob)
        weekly_units.append(ob.__dict__)
    
    return {
        flag : {
            'total' :  total_units.__dict__,
            'weekly' : weekly_units
        }
       
    }


def update_total(total_unit:model.TotalUnit ,unit_model : model.UnitModel ):
    total_unit.total_rsv_w_o_vat = total_unit.total_rsv_w_o_vat + unit_model.total_rsv_w_o_vat
    total_unit.units = total_unit.units + unit_model.predicted_units
    total_unit.te = total_unit.te + unit_model.trade_expense
    total_unit.lsv = total_unit.lsv + unit_model.total_lsv
    total_unit.nsv = total_unit.nsv + unit_model.total_nsv
    total_unit.mac = total_unit.mac + unit_model.mars_mac
    total_unit.rp = total_unit.rp + unit_model.retailer_margin
    total_unit.asp = total_unit.asp + unit_model.asp 
    total_unit.avg_promo_selling_price =  util.average(total_unit.avg_promo_selling_price,unit_model.promo_asp) 
    total_unit.roi = total_unit.roi + unit_model.roi
    total_unit.rp_percent = util.average(total_unit.rp_percent,unit_model.retailer_margin_percent_of_rsp)
    total_unit.mac_percent = util.average( total_unit.mac_percent,unit_model.mars_mac_percent_of_nsv)
    total_unit.volume = util.average(total_unit.volume ,unit_model.total_weight_in_tons)
    total_unit.te_per_unit = total_unit.te_per_unit + unit_model.te_per_units
    total_unit.te_percent_of_lsv = util.average(total_unit.te_percent_of_lsv ,unit_model.te_percent_of_lsv)
    total_unit.base_units = total_unit.base_units + unit_model.base_unit
    total_unit.increment_units =total_unit.increment_units + unit_model.incremental_unit
    total_unit.lift = total_unit.lift + (unit_model.incremental_unit / unit_model.base_unit)
     
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
    val ={"Flag_promotype_Motivation" : "flag_promotype_motivation", 
    "Flag_promotype_N_pls_1": "flag_promotype_n_pls_1", 
        "Flag_promotype_traffic": "flag_promotype_traffic", 
        }  
    return val[promo_from_req]

def update_from_request(data_list , querydict):
    cloned_list = copy.deepcopy(data_list)
    for i in querydict.keys():
        week_regex = util._regex(r'week-\d{1,2}',i)
        if week_regex:
            week = int(util._regex(r'\d{1,2}',week_regex.group()).group())
            # import pdb
            # pdb.set_trace()
            if querydict[i]['promo_mechanics']:
                cloned_list[week-1][data_values.index(
                    _get_promotion_flag(querydict[i]['promo_mechanics']))] = 1
            # import pdb
            # pdb.set_trace()
            # cloned_list[week-1].append({'co_investment' :querydict[i]['co_investment'] })
            cloned_list[week-1][data_values.index('co_investment')] = querydict[i]['co_investment']
            # cloned_list[week-1][data_values.index('promo_depth')] = querydict[i]['promo_depth']
            if querydict['param_depth_all']:
                # print("setting depth all")
                cloned_list[week-1][data_values.index('promo_depth')] = querydict[i]['promo_depth']
                # promo_data.prefetched_data[week-1].tpr_discount = querydict['param_depth_all'] 
            else: 
                cloned_list[week-1][data_values.index('promo_depth')] = querydict[i]['promo_depth']
    return cloned_list

def update_from_saved_data(data_list , promo_week : QuerySet[db_model.PromoWeek]):
    cloned_list = copy.deepcopy(data_list)
    # import pdb
    # pdb.set_trace()
    if promo_week:
        for week in promo_week:
            index = week.week - 1
            cloned_list[index][data_values.index('co_investment')] = week.co_investment
            if week.promo_mechanic:
                cloned_list[index][data_values.index(
                        _get_promotion_flag(week.promo_mechanic))] = 1
            cloned_list[index][data_values.index('promo_depth')] = week.promo_depth
    return cloned_list


def update_from_pricing(data_list):
    cloned_list = copy.deepcopy(data_list)
    return cloned_list