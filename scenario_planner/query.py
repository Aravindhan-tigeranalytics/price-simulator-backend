from typing import List
from core import models as model
from utils import constants as const
from django.db.models import Q
import math
import decimal
# from django.db.models import Model 
# coeff_values = [ 'model_meta__id','model_meta__account_name', 
#                     'model_meta__corporate_segment', 'model_meta__product_group', 'model_meta__brand_filter',
#                     'model_meta__brand_format_filter', 'model_meta__strategic_cell_filter','wmape', 'rsq',
#                     'intercept', 'median_base_price_log', 'tpr_discount', 'tpr_discount_lag1',
#                     'tpr_discount_lag2', 'catalogue', 'display', 'acv', 'si', 
#                     'si_month', 'si_quarter', 'c_1_crossretailer_discount', 'c_1_crossretailer_log_price', 'c_1_intra_discount', 
#                     'c_2_intra_discount', 'c_3_intra_discount', 'c_4_intra_discount', 'c_5_intra_discount',
#                     'c_1_intra_log_price', 'c_2_intra_log_price', 'c_3_intra_log_price', 'c_4_intra_log_price', 'c_5_intra_log_price', 'category_trend', 'trend_month', 'trend_quarter', 'trend_year', 'month_no', 'flag_promotype_motivation', 'flag_promotype_n_pls_1', 'flag_promotype_traffic', 'flag_nonpromo_1', 'flag_nonpromo_2', 'flag_nonpromo_3', 'flag_promo_1', 'flag_promo_2', 'flag_promo_3', 'holiday_flag_1', 'holiday_flag_2', 'holiday_flag_3', 'holiday_flag_4', 'holiday_flag_5', 'holiday_flag_6', 'holiday_flag_7', 'holiday_flag_8', 'holiday_flag_9', 'holiday_flag_10' 
#                     ]
# data_values = ['model_meta__id','model_meta__account_name', 
#                 'model_meta__corporate_segment', 'model_meta__product_group', 'model_meta__brand_filter',
#                 'model_meta__brand_format_filter', 'model_meta__strategic_cell_filter',
#                 'year','quater','month','period','date','week',
#                 'intercept', 'median_base_price_log', 'tpr_discount','promo_depth','co_investment', 'tpr_discount_lag1',
#                 'tpr_discount_lag2', 'catalogue', 'display', 'acv', 'si', 
#                 'si_month', 'si_quarter', 'c_1_crossretailer_discount', 'c_1_crossretailer_log_price', 'c_1_intra_discount', 
#                 'c_2_intra_discount', 'c_3_intra_discount', 'c_4_intra_discount', 'c_5_intra_discount',
#                 'c_1_intra_log_price', 'c_2_intra_log_price', 'c_3_intra_log_price', 'c_4_intra_log_price', 'c_5_intra_log_price', 'category_trend', 'trend_month', 'trend_quarter', 'trend_year', 'month_no', 'flag_promotype_motivation', 'flag_promotype_n_pls_1', 'flag_promotype_traffic', 'flag_nonpromo_1', 'flag_nonpromo_2', 'flag_nonpromo_3', 'flag_promo_1', 'flag_promo_2', 'flag_promo_3', 'holiday_flag_1', 'holiday_flag_2', 'holiday_flag_3', 'holiday_flag_4', 'holiday_flag_5', 'holiday_flag_6', 'holiday_flag_7', 'holiday_flag_8', 'holiday_flag_9', 'holiday_flag_10', 
#                 'wk_sold_avg_price_byppg',
#                 'average_weight_in_grams','weighted_weight_in_grams']

# roi_values = [
#     'model_meta__id','model_meta__account_name', 
#                 'model_meta__corporate_segment', 'model_meta__product_group', 'model_meta__brand_filter',
#                 'model_meta__brand_format_filter', 'model_meta__strategic_cell_filter','week','on_inv','off_inv',
#                 'list_price','gmac'
# ]

def _get_calendar_values(retailer) : 
    if(retailer.lower() == "x5"):
        return "x5_flag"
    elif(retailer.lower() == "tander" or retailer.lower() == 'magnit'):
        return "magnit_flag"
    else:
        return "x5_magnit_flag"
    
def get_coefficient(retailer , ppg):
    return model.ModelCoefficient.objects.select_related('model_meta').filter(
                    model_meta__account_name__iexact = retailer,
                    model_meta__product_group__iexact = ppg
                ).values_list(*const.COEFFICIENT_VALUES)
    
def get_model_data(retailer , ppg):
    return model.ModelData.objects.select_related('model_meta').filter(
                    model_meta__account_name__iexact = retailer,
                    model_meta__product_group__iexact = ppg
                ).values_list(*const.DATA_VALUES).order_by('week')
def get_roi(retailer , ppg):
    return model.ModelROI.objects.select_related('model_meta').filter(
                    model_meta__account_name__iexact = retailer,
                    model_meta__product_group__iexact = ppg
                ).values_list(*const.ROI_VALUES).order_by('week') 
    
def get_coeff_map(retailer,ppg):
    return model.CoeffMap.objects.select_related('model_meta').filter(
        model_meta__account_name__iexact = retailer,
                    model_meta__product_group__iexact = ppg
    ).values('coefficient_old' , 'coefficient_new') 
    
def get_holiday_calendar(retailer):
    holiday_cal =  model.HolidayCalendar.objects.filter(year = 2022).values(
       'week' ,  _get_calendar_values(retailer)
        
    )
    return [{i['week']:i[ _get_calendar_values(retailer)]} for i in holiday_cal ]

def get_list_value_from_query(coeff_model:model.ModelCoefficient,
                              data_model:model.ModelData,
                              roi_model:model.ModelROI,
                              retailer,ppg):
    '''
    returns list form of ORM query
    '''
    # import pdb
    # pdb.set_trace()
    coefficient = get_coefficient(retailer , ppg)
    data = get_model_data(retailer , ppg)
    roi = get_roi(retailer , ppg)
    coeff_map = get_coeff_map(retailer ,ppg)
    # import pdb
    # pdb.set_trace()
    coeff_list = [list(i) for i in coefficient]
    # data_list = [list(i) for i in data]
    data_list = [_check_if_vat_applied(list(i)) for i in data]
    roi_list = [list(i) for i in roi] 
    holiday_calendar_list = get_holiday_calendar(retailer)
    return coeff_list , data_list, roi_list , coeff_map,holiday_calendar_list


def get_list_value_from_query_all(request_id):
    '''
    returns list form of ORM query
    '''
    # import pdb
    # pdb.set_trace()
    coefficient = model.ModelCoefficient.objects.select_related('model_meta').values_list(
        *const.COEFFICIENT_VALUES).filter(model_meta__in = request_id)
    data = model.ModelData.objects.select_related('model_meta').values_list(
        *const.DATA_VALUES).order_by('week').filter(model_meta__in = request_id)
    roi = model.ModelROI.objects.select_related('model_meta').values_list(*const.ROI_VALUES).order_by('week').filter(model_meta__in = request_id)
    coeff_list = [list(i) for i in coefficient]
    data_list = [_check_if_vat_applied(list(i)) for i in data]
    roi_list = [list(i) for i in roi] 
    return coeff_list , data_list, roi_list

def _check_if_vat_applied(model_data):
    # import pdb
    # pdb.set_trace()
    # for lenta dont apply vat
    if(model_data[const.DATA_VALUES.index('model_meta__account_name')] != 'Lenta'):
        rsp = model_data[const.DATA_VALUES.index('median_base_price_log')]
        rsp = math.exp(rsp)
        rsp = decimal.Decimal(rsp) * decimal.Decimal(1 - (20/100))
        model_data[const.DATA_VALUES.index('median_base_price_log')] = decimal.Decimal(math.log(rsp))
        
    return model_data

def get_list_value_from_query_by_name(retailer:List):
    '''
    returns list form of ORM query
    '''
    # import pdb
    # pdb.set_trace()
    # from django.db.models import Q
    query = Q()
    for i in retailer:
        query.add(Q(Q(model_meta__account_name=i['account_name']) & Q(model_meta__product_group = i['product_group'])),Q.OR)

    # print(query , "generate query")
    
    # import pdb
    # pdb.set_trace()
    
    # query = Q(first_name='mark')
    # query.add(Q(email='mark@test.com'), Q.OR)
    # query.add(Q(last_name='doe'), Q.AND)

    # queryset = User.objects.filter(query)
    coefficient = model.ModelCoefficient.objects.select_related('model_meta').values_list(
        *const.COEFFICIENT_VALUES).filter(
            query
            )
    data = model.ModelData.objects.select_related('model_meta').values_list(
        *const.DATA_VALUES).order_by('week').filter(query)
    # import pdb
    # pdb.set_trace()
    roi = model.ModelROI.objects.select_related('model_meta').values_list(*const.ROI_VALUES).order_by('week').filter(
       query)
    coeff_list = [list(i) for i in coefficient]
    data_list = [_check_if_vat_applied(list(i)) for i in data]
    roi_list = [list(i) for i in roi] 
    return coeff_list , data_list, roi_list