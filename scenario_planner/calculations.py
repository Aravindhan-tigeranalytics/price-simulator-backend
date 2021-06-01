from core.models import ModelData, ModelMeta
from utils import util as util
from utils import models as model
import math
import decimal

def get_related_value(promo_data : ModelData):
    # import pdb
    # pdb.set_trace()
    # print("get related value")
    # return promo_data.model_meta.coefficient.get()
    
    return promo_data.model_meta.prefetched_coeff[0]
def get_roi_by_week(promo_data : ModelData):
    return promo_data.model_meta.prefetched_roi[promo_data.week-1]
def promo_simulator_calculations(promo_data : ModelData):
    
    # if promo_data.week == 1:
    #     print(promo_data.tpr_discount , "tpr discount value")
    
    # print(promo_data.week , "week")
    # print(promo_data.tpr_discount , "promo_data.tpr_discount")
    # print(get_related_value(promo_data).tpr_discount , "get_related_value(promo_data).tpr_discount")
    # print(promo_data.tpr_discount * get_related_value(promo_data).tpr_discount , "result")
    # import pdb
    # pdb.set_trace()
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
    
    print(result , "result")
    # import pdb
    # pdb.set_trace()
    # model.UnitModel()
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

def promo_simulator_calculations_test(promo_data : ModelData):
    # import pdb
    # pdb.set_trace()
    # result = 1.4
    # if promo_data.week == 1:
    #     print(promo_data.tpr_discount , "tpr discount value")
    
    # print(promo_data.week , "week")
    # print(promo_data.tpr_discount , "promo_data.tpr_discount")
    # print(get_related_value(promo_data).tpr_discount , "get_related_value(promo_data).tpr_discount")
    # print(promo_data.tpr_discount * get_related_value(promo_data).tpr_discount , "result")
    # import pdb
    # pdb.set_trace()
    # result =
    # import pdb
    # pdb.set_trace()
# if promo_data.week ==1 :
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
    # print(promo_data.trend_month , "model trend month")
    # print(get_related_value(promo_data).trend_month , "coff trend month")
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
                                

    # print(intercept , "intercept")
    # print(median , "median base price log")
    # print(acv , "acv")
    # print(si , "SI")
    # print(c_2_intra_discount , "c2 intra discount")
    # print(c_3_intra_discount , "c3 intra discount")
    # print(trend_month , "trend month")
    # print(promo_data.week , ":week no")
    # print(result , ":result")
    # print(math.exp(result) , ":math.exp(result) result")
    # import pdb
    # pdb.set_trace()
    # model.UnitModel()
    roi_model = get_roi_by_week(promo_data)
    ob = model.UnitModel(predicted_units=  decimal.Decimal(math.exp(result)),
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
    # print(ob , "OB")
    # import pdb
    # pdb.set_trace()
    # print(result , "actual result")
    # print(math.exp(result) , "expo result")
    
    return ob.__dict__
    # import pdb
    # pdb.set_trace()
    # return 
    
def update_week_value(promo_data:ModelMeta , querydict):
    # import pdb
    # pdb.set_trace()
    for i in querydict.keys():
        week_regex = util._regex(r'week-\d{1,2}',i)
        if week_regex:
            week = int(util._regex(r'\d{1,2}',week_regex.group()).group())
            # import pdb
            # pdb.set_trace()
            # promo_data.data.get(week = week).
            # print(promo_data.get(week = week) , "promo data week ")
            if querydict['param_depth_all']:
                print("setting depth all")
                promo_data.prefetched_data[week-1].tpr_discount = querydict['param_depth_all'] 
            else: 
                promo_data.prefetched_data[week-1].tpr_discount = querydict[i]['promo_depth'] 
            # promo_data.data.get(week = week).tpr_discount = 2
    # return promo_data
    # import pdb
    # pdb.set_trace()
    # promo_data.get(week)
    # pass
    