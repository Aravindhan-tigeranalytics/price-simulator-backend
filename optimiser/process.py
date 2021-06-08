# Example usage:
import json
import pandas as pd
from core import models as model


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


# Index(['Nielsen SKU Name', 'Year', 'PPG Name', 'Retailer', 'DATE', 'Level',
#        'Activity name', 'Mechanic', 'Discount, NRV %', 'Weeknum', 'TE Off Inv',
#        'TE On Inv', 'COGS', 'GMAC', 'PPG', 'List Price Nov 2019 Price per PC',
#        'List Price July 2020 Price per PC', 'List_Price'],
#       dtype='object')- ROI
# Index(['model_coefficients', 'names', 'PPG_Item'], dtype='object') - Coeff
# Index(['Unnamed: 0', 'Promo_flg_date_1', 'wk_sold_median_base_price_byppg_log',
#        'Catalogue_Dist', 'ITEM_2010_Magnit_RegularPrice',
#        'ITEM_2028_Magnit_PromotedDiscount',
#        'ITEM_2029_Magnit_PromotedDiscount', 'SI', 'flag_old_mans_day',
#        'tpr_discount_byppg'], - Data

def get_list_value_from_query(retailer,ppg):
    coeff_model = model.ModelCoefficient
    data_model = model.ModelData
    roi_model = model.ModelROI
    retailer = 'Tander'
                              
    coefficient = coeff_model.objects.select_related('model_meta').filter(
                    model_meta__account_name = retailer,
                    model_meta__product_group = ppg
                ).values_list(*coeff_values)
    # print(coefficient,"coefficient")
    data = data_model.objects.select_related('model_meta').filter(
                    model_meta__account_name = retailer,
                    model_meta__product_group = ppg
                ).values_list(*data_values).order_by('week')
    # print(data,"data")
    roi = roi_model.objects.select_related('model_meta').filter(
                    model_meta__account_name = retailer,
                    model_meta__product_group = ppg
                ).values_list(*roi_values).order_by('week')
    # print(roi,"roi")
    coeff_list = [list(i) for i in coefficient]
    data_list = [list(i) for i in data]
    roi_list = [list(i) for i in roi]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    coeff_list = pd.DataFrame(coeff_list, columns = coeff_values)
    coeff_list.rename(columns={'intercept':'Intercept','flag_promo_1':'Promo_flg_date_1','median_base_price_log': 'wk_sold_median_base_price_byppg_log', 'catalogue':'Catalogue_Dist', 'c_1_intra_log_price':'ITEM_2010_Magnit_RegularPrice',
    'c_1_crossretailer_discount':'ITEM_2028_Magnit_PromotedDiscount', 'c_2_intra_discount':'ITEM_2029_Magnit_PromotedDiscount',
    'si':'SI','flag_promotype_traffic':'flag_old_mans_day', 'tpr_discount': 'tpr_discount_byppg'},inplace=True)
    
    data_list = pd.DataFrame(data_list, columns = data_values)
    data_list.rename(columns={'flag_promo_1': 'Promo_flg_date_1', 'wk_sold_avg_price_byppg': 'wk_sold_median_base_price_byppg_log',
       'catalogue': 'Catalogue_Dist','si': 'SI','tpr_discount': 'tpr_discount_byppg'},inplace=True)

    roi_list = pd.DataFrame(roi_list, columns = roi_values)
    roi_list.rename(columns={'model_meta__brand_filter':'Nielsen SKU Name', 'model_meta__account_name': 'PPG Name',
        'model_meta__corporate_segment': 'Retailer',  'week' : 'Weeknum', 'off_inv': 'TE Off Inv','on_inv': 'TE On Inv',  'gmac': 'GMAC', 'model_meta__product_group': 'PPG','list_price': 'List_Price'},inplace=True)

    return coeff_list , data_list, roi_list

# roi_data = [{'Nielsen SKU Name': 'KORKUNOV ASSORT A DARK&MLK.CHOC.COAT ASR. 0.11KG K A', 'Year' : 2021, 'PPG Name': 'A.Korkunov 110g',
#  'Retailer': 'Magnit', 'DATE': '14-11-2021', 'Level': 'NA',
#        'Activity name': '21_NEW YEAR MULTIBRAND', 'Mechanic' : 'Single price off % discount', 'Discount, NRV %': 0.3, 'Weeknum' : 46,
#         'TE Off Inv': 0.05,'TE On Inv': 0.2436, 'COGS': 83.8571428571428, 'GMAC': 0.502124663912944, 'PPG': 'A.Korkunov 110g',
#          'List Price Nov 2019 Price per PC': 168.43,
#        'List Price July 2020 Price per PC': 168.43, 'List_Price': 168.43}]

# model_coeff = [
#     {'model_coefficients': 11.534, 'names': 'Intercept', 'PPG_Item': 'ITEM_201_Magnit'},
#     {'model_coefficients': 1.472, 'names': 'Promo_flg_date_1', 'PPG_Item': 'ITEM_201_Magnit'},
#     {'model_coefficients': -1.083, 'names': 'wk_sold_median_base_price_byppg_log', 'PPG_Item': 'ITEM_201_Magnit'},
#     {'model_coefficients': 0.049, 'names': 'Catalogue_Dist', 'PPG_Item': 'ITEM_201_Magnit'},
#     {'model_coefficients': 0.75, 'names': 'ITEM_2010_Magnit_RegularPrice', 'PPG_Item': 'ITEM_201_Magnit'},
#     {'model_coefficients': -0.001, 'names': 'ITEM_2028_Magnit_PromotedDiscount', 'PPG_Item': 'ITEM_201_Magnit'},
#     {'model_coefficients': -0.001, 'names': 'ITEM_2029_Magnit_PromotedDiscount', 'PPG_Item': 'ITEM_201_Magnit'},
#     {'model_coefficients': 0.272, 'names': 'SI', 'PPG_Item': 'ITEM_201_Magnit'},
#     {'model_coefficients': 0.185, 'names': 'flag_old_mans_day', 'PPG_Item': 'ITEM_201_Magnit'},
#     {'model_coefficients': 0.034, 'names': 'tpr_discount_byppg', 'PPG_Item': 'ITEM_201_Magnit'},
#     ]

# model_data = [{'Unnamed: 0': '31-12-2017', 'Promo_flg_date_1': 0, 'wk_sold_median_base_price_byppg_log': 5.39891376270662,
#        'Catalogue_Dist': 0, 'ITEM_2010_Magnit_RegularPrice': 5.17412215311023,
#        'ITEM_2028_Magnit_PromotedDiscount': 0,
#        'ITEM_2029_Magnit_PromotedDiscount': 0, 'SI': 1.90397303539481, 'flag_old_mans_day': 0,'tpr_discount_byppg': 0}]


# dataframe = pd.DataFrame.from_dict(roi_data, orient='columns')
# coe_dataframe = pd.DataFrame.from_dict(model_coeff, orient='columns')
# model_dataframe = pd.DataFrame.from_dict(model_data, orient='columns')
# # dataframe.rename(columns={'name':'user_name'},inplace=True)

# print(dataframe)
# print(coe_dataframe)
# print(model_dataframe)