from core import models as model

# from django.db.models import Model 
coeff_values = [ 'model_meta__id','model_meta__account_name', 
                    'model_meta__corporate_segment', 'model_meta__product_group', 'model_meta__brand_filter',
                    'model_meta__brand_format_filter', 'model_meta__strategic_cell_filter','wmape', 'rsq',
                    'intercept', 'median_base_price_log', 'tpr_discount', 'tpr_discount_lag1',
                    'tpr_discount_lag2', 'catalogue', 'display', 'acv', 'si', 
                    'si_month', 'si_quarter', 'c_1_crossretailer_discount', 'c_1_crossretailer_log_price', 'c_1_intra_discount', 
                    'c_2_intra_discount', 'c_3_intra_discount', 'c_4_intra_discount', 'c_5_intra_discount',
                    'c_1_intra_log_price', 'c_2_intra_log_price', 'c_3_intra_log_price', 'c_4_intra_log_price', 'c_5_intra_log_price', 'category_trend', 'trend_month', 'trend_quarter', 'trend_year', 'month_no', 'flag_promotype_motivation', 'flag_promotype_n_pls_1', 'flag_promotype_traffic', 'flag_nonpromo_1', 'flag_nonpromo_2', 'flag_nonpromo_3', 'flag_promo_1', 'flag_promo_2', 'flag_promo_3', 'holiday_flag_1', 'holiday_flag_2', 'holiday_flag_3', 'holiday_flag_4', 'holiday_flag_5', 'holiday_flag_6', 'holiday_flag_7', 'holiday_flag_8', 'holiday_flag_9', 'holiday_flag_10' 
                    ]
data_values = ['model_meta__id','model_meta__account_name', 
                'model_meta__corporate_segment', 'model_meta__product_group', 'model_meta__brand_filter',
                'model_meta__brand_format_filter', 'model_meta__strategic_cell_filter',
                'year','quater','month','period','date','week',
                'intercept', 'median_base_price_log', 'tpr_discount', 'tpr_discount_lag1',
                'tpr_discount_lag2', 'catalogue', 'display', 'acv', 'si', 
                'si_month', 'si_quarter', 'c_1_crossretailer_discount', 'c_1_crossretailer_log_price', 'c_1_intra_discount', 
                'c_2_intra_discount', 'c_3_intra_discount', 'c_4_intra_discount', 'c_5_intra_discount',
                'c_1_intra_log_price', 'c_2_intra_log_price', 'c_3_intra_log_price', 'c_4_intra_log_price', 'c_5_intra_log_price', 'category_trend', 'trend_month', 'trend_quarter', 'trend_year', 'month_no', 'flag_promotype_motivation', 'flag_promotype_n_pls_1', 'flag_promotype_traffic', 'flag_nonpromo_1', 'flag_nonpromo_2', 'flag_nonpromo_3', 'flag_promo_1', 'flag_promo_2', 'flag_promo_3', 'holiday_flag_1', 'holiday_flag_2', 'holiday_flag_3', 'holiday_flag_4', 'holiday_flag_5', 'holiday_flag_6', 'holiday_flag_7', 'holiday_flag_8', 'holiday_flag_9', 'holiday_flag_10', 
                'wk_sold_avg_price_byppg',
                'average_weight_in_grams','weighted_weight_in_grams']

roi_values = [
    'model_meta__id','model_meta__account_name', 
                'model_meta__corporate_segment', 'model_meta__product_group', 'model_meta__brand_filter',
                'model_meta__brand_format_filter', 'model_meta__strategic_cell_filter','week','on_inv','off_inv',
                'list_price','gmac'
]



def get_list_value_from_query(coeff_model:model.ModelCoefficient,
                              data_model:model.ModelData,
                              roi_model:model.ModelROI,
                              retailer,ppg):
    coefficient = coeff_model.objects.select_related('model_meta').filter(
                    model_meta__account_name = retailer,
                    model_meta__product_group = ppg
                ).values_list(*coeff_values)
    data = data_model.objects.select_related('model_meta').filter(
                    model_meta__account_name = retailer,
                    model_meta__product_group = ppg
                ).values_list(*data_values).order_by('week')
    roi = roi_model.objects.select_related('model_meta').filter(
                    model_meta__account_name = retailer,
                    model_meta__product_group = ppg
                ).values_list(*roi_values).order_by('week')
    coeff_list = [list(i) for i in coefficient]
    data_list = [list(i) for i in data]
    roi_list = [list(i) for i in roi] 
    return coeff_list , data_list, roi_list