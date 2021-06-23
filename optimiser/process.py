
from django.db.models import F , Value
from django.db.models.functions import Concat
from utils import constants as const
import json
import pandas as pd
from core import models as model
from scenario_planner import query as cols

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

def get_list_from_db_bk(retailer,ppg):
    '''
    convert quey data to list to match the data frame format
    
    data_dt> data frame list for model data
    roi_dt -> data frame list for roi data
    coeff_dt -> data frame list for coefficient data
    '''
    
  
    # coefficient new used in db
    # coefficient old used in dataframe
    coeff_list = []
    data_frame_map = {
        
    }
    coeff_map_values = [
        'model_meta__id','model_meta__account_name', 
                    'model_meta__corporate_segment', 'model_meta__product_group','coefficient_new',
                    'value','coefficient_old'
        
    ]
    data_values = [
                    'date',
                    ]
    coeff_map = model.CoeffMap.objects.select_related('model_meta').filter(
        model_meta__account_name = retailer,model_meta__product_group = ppg
        ).values_list(
           *coeff_map_values
            ).annotate( 
                       PPG_Item=Concat(
                         Value('ITEM_'),
                         F('model_meta__account_name'),Value('_'), F('model_meta__product_group')))
    
    roi = model.ModelROI.objects.select_related('model_meta').filter(
        model_meta__account_name = retailer,model_meta__product_group = ppg
        ).values_list(
            'neilson_sku_name','year','model_meta__product_group','model_meta__account_name','date',
            'activity_name','mechanic','discount_nrv','week','off_inv','on_inv','gmac',
            'model_meta__product_group','list_price'
            ).annotate(
                cogs=F('list_price') - (F('list_price') * F('gmac'))
                )
   
    for i in coeff_map:
        coeff_list.append(i[-3:])
        data_values.append(const.CALCULATION_METRIC[i[-4]])
        # import pdb
        # pdb.set_trace()
        data_frame_map[i[-4]] = i[-2]
    data = model.ModelData.optimizer_objects.select_related('model_meta').filter(
        model_meta__account_name = retailer,
        model_meta__product_group = ppg
    ).values_list(*data_values)
    data_col = ['Unnamed: 0']
    data_col = data_col +  [data_frame_map[const.get_key_from_value(const.CALCULATION_METRIC , i)] for i in data_values[1:]]
    coeff_list = [list(i[-3:]) for i in coeff_map]
    data_list = [list(i) for i in data]
    roi_list = [list(i) for i in roi]
    coeff_col = ['model_coefficients','names','PPG_Item']
    roi_col = [
'Nielsen SKU Name','Year','PPG Name','Retailer','DATE','Activity name','Mechanic','Discount, NRV %','Weeknum','TE On Inv','TE Off Inv',
'GMAC','PPG' , 'List_Price' , 'COGS'

]
    coeff_dt = pd.DataFrame(coeff_list, columns = coeff_col)
    roi_dt = pd.DataFrame(roi_list, columns = roi_col)
    data_dt = pd.DataFrame(data_list, columns = data_col)
    coeff_dt['model_coefficients'] = coeff_dt['model_coefficients'].astype(float) # convert object to float for calculation
    dec_col = ['Discount, NRV %', 'TE On Inv',
    'TE Off Inv', 'GMAC', 'List_Price', 'COGS']
    roi_dt['DATE'] = pd.to_datetime(roi_dt['DATE']) # convert to dataframe date format
    for i in dec_col:
        roi_dt[i]= roi_dt[i].astype(float)
    # import pdb
    # pdb.set_trace()
    data_dt['Unnamed: 0']= pd.to_datetime(data_dt['Unnamed: 0'])
    # import pdb
    # pdb.set_trace()
    for i in data_col[1:]:
        data_dt[i] = data_dt[i].astype(float)
   
    return data_dt , roi_dt , coeff_dt






def get_list_from_db(retailer,ppg):
    coeff_map_values = [
        'model_meta__account_name','model_meta__corporate_segment', 'model_meta__product_group','model_meta__brand_filter',
        'model_meta__brand_format_filter', 'model_meta__strategic_cell_filter','coefficient_new','value','coefficient_old'
    ]

    roi_values = [
        'model_meta__account_name','model_meta__corporate_segment', 'model_meta__product_group','model_meta__brand_filter',
        'model_meta__brand_format_filter', 'model_meta__strategic_cell_filter','neilson_sku_name','date','year','week','activity_name',
        'mechanic','discount_nrv','off_inv','on_inv','gmac','list_price'
    ]

    model_coeff_values = ['model_meta__account_name', 
                    'model_meta__corporate_segment', 'model_meta__product_group', 'model_meta__brand_filter',
                    'model_meta__brand_format_filter', 'model_meta__strategic_cell_filter','wmape', 'rsq',
                    'intercept', 'median_base_price_log', 'tpr_discount', 'tpr_discount_lag1',
                    'tpr_discount_lag2', 'catalogue', 'display', 'acv', 'si', 
                    'si_month', 'si_quarter', 'c_1_crossretailer_discount', 'c_1_crossretailer_log_price', 'c_1_intra_discount', 
                    'c_2_intra_discount', 'c_3_intra_discount', 'c_4_intra_discount', 'c_5_intra_discount',
                    'c_1_intra_log_price', 'c_2_intra_log_price', 'c_3_intra_log_price', 'c_4_intra_log_price', 'c_5_intra_log_price', 'category_trend', 'trend_month', 'trend_quarter', 'trend_year', 'month_no', 'flag_promotype_motivation', 'flag_promotype_n_pls_1', 'flag_promotype_traffic', 'flag_nonpromo_1', 'flag_nonpromo_2', 'flag_nonpromo_3', 'flag_promo_1', 'flag_promo_2', 'flag_promo_3', 'holiday_flag_1', 'holiday_flag_2', 'holiday_flag_3', 'holiday_flag_4', 'holiday_flag_5', 'holiday_flag_6', 'holiday_flag_7', 'holiday_flag_8', 'holiday_flag_9', 'holiday_flag_10' 
                    ]

    model_data_values = ['model_meta__account_name', 
                'model_meta__corporate_segment', 'model_meta__product_group', 'model_meta__brand_filter',
                'model_meta__brand_format_filter', 'model_meta__strategic_cell_filter',
                'year','quater','month','period','date','week',
                'intercept', 'median_base_price_log', 'tpr_discount','promo_depth','co_investment', 'tpr_discount_lag1',
                'tpr_discount_lag2', 'catalogue', 'display', 'acv', 'si', 
                'si_month', 'si_quarter', 'c_1_crossretailer_discount', 'c_1_crossretailer_log_price', 'c_1_intra_discount', 
                'c_2_intra_discount', 'c_3_intra_discount', 'c_4_intra_discount', 'c_5_intra_discount',
                'c_1_intra_log_price', 'c_2_intra_log_price', 'c_3_intra_log_price', 'c_4_intra_log_price', 'c_5_intra_log_price', 'category_trend', 'trend_month', 'trend_quarter', 'trend_year', 'month_no', 'flag_promotype_motivation', 'flag_promotype_n_pls_1', 'flag_promotype_traffic', 'flag_nonpromo_1', 'flag_nonpromo_2', 'flag_nonpromo_3', 'flag_promo_1', 'flag_promo_2', 'flag_promo_3', 'holiday_flag_1', 'holiday_flag_2', 'holiday_flag_3', 'holiday_flag_4', 'holiday_flag_5', 'holiday_flag_6', 'holiday_flag_7', 'holiday_flag_8', 'holiday_flag_9', 'holiday_flag_10', 
                'wk_sold_avg_price_byppg',
                'average_weight_in_grams','weighted_weight_in_grams','optimiser_flag']

    model_coefficient_cols = ['Account Name', 'Corporate Segment', 'PPG', 'Brand Filter',
       'Brand Format Filter', 'Strategic Cell Filter',
       'Coefficient_new', 'Value', 'Coefficient']

    coeff_map = model.CoeffMap.objects.select_related('model_meta').filter(
        model_meta__account_name = retailer,model_meta__product_group = ppg
        ).values_list(
           *coeff_map_values
            )
    
    model_coeff = model.ModelCoefficient.objects.select_related('model_meta').filter(
        model_meta__account_name = retailer,model_meta__product_group = ppg
        ).values_list(
           *model_coeff_values
            )
    
    model_data = model.ModelData.objects.select_related('model_meta').filter(
        model_meta__account_name = retailer,model_meta__product_group = ppg
        ).values_list(
           *model_data_values
            )

    roi = model.ModelROI.objects.select_related('model_meta').filter(
    model_meta__account_name = retailer,model_meta__product_group = ppg
    ).values_list(
        *roi_values
        ).annotate(
            cogs=F('list_price') - (F('list_price') * F('gmac'))
            )
    # pd.set_option('display.max_columns', None)  # or 1000
    # pd.set_option('display.max_rows', 100)  # or 1000

    coeff_dt = pd.DataFrame(list(model_coeff), columns= ['Account Name', 'Corporate Segment', 'PPG', 'Brand Filter',
       'Brand Format Filter', 'Strategic Cell Filter', 'WMAPE', 'Rsq',
       'Intercept', 'Median_Base_Price_log', 'TPR_Discount',
       'TPR_Discount_lag1', 'TPR_Discount_lag2', 'Catalogue', 'Display', 'ACV',
       'SI', 'SI_month', 'SI_quarter', 'C_1_crossretailer_discount',
       'C_1_crossretailer_log_price', 'C_1_intra_discount',
       'C_2_intra_discount', 'C_3_intra_discount', 'C_4_intra_discount',
       'C_5_intra_discount', 'C_1_intra_log_price', 'C_2_intra_log_price',
       'C_3_intra_log_price', 'C_4_intra_log_price', 'C_5_intra_log_price',
       'Category trend', 'Trend_month', 'Trend_quarter', 'Trend_year',
       'month_no', 'Flag_promotype_Motivation', 'Flag_promotype_N_pls_1',
       'Flag_promotype_traffic', 'Flag_nonpromo_1', 'Flag_nonpromo_2',
       'Flag_nonpromo_3', 'Flag_promo_1', 'Flag_promo_2', 'Flag_promo_3',
       'Holiday_Flag1', 'Holiday_Flag2', 'Holiday_Flag3', 'Holiday_Flag4',
       'Holiday_Flag5', 'Holiday_Flag6', 'Holiday_Flag7', 'Holiday_Flag8',
       'Holiday_Flag9', 'Holiday_Flag10'])

    data_dt = pd.DataFrame(list(model_data), columns= ['Account Name', 'Corporate Segment', 'PPG', 'Brand Filter',
       'Brand Format Filter', 'Strategic Cell Filter', 'Year', 'Quarter',      
       'Month', 'Period', 'Date', 'Week', 'Intercept', 'Median_Base_Price_log',
       'TPR_Discount','Promo_Depth', 'Coinvestment', 'TPR_Discount_lag1', 'TPR_Discount_lag2', 'Catalogue',  
       'Display', 'ACV', 'SI', 'SI_month', 'SI_quarter',
       'C_1_crossretailer_discount', 'C_1_crossretailer_log_price',
       'C_1_intra_discount', 'C_2_intra_discount', 'C_3_intra_discount',       
       'C_4_intra_discount', 'C_5_intra_discount', 'C_1_intra_log_price',      
       'C_2_intra_log_price', 'C_3_intra_log_price', 'C_4_intra_log_price',    
       'C_5_intra_log_price', 'Category trend', 'Trend_month', 'Trend_quarter',
       'Trend_year', 'month_no', 'Flag_promotype_Motivation',
       'Flag_promotype_N_pls_1', 'Flag_promotype_traffic', 'Flag_nonpromo_1',  
       'Flag_nonpromo_2', 'Flag_nonpromo_3', 'Flag_promo_1', 'Flag_promo_2',   
       'Flag_promo_3', 'Holiday_Flag1', 'Holiday_Flag2', 'Holiday_Flag3',      
       'Holiday_Flag4', 'Holiday_Flag5', 'Holiday_Flag6', 'Holiday_Flag7',     
       'Holiday_Flag8', 'Holiday_Flag9', 'Holiday_Flag10',
       'wk_sold_avg_price_byppg', 'Average Weight in grams',
       'Weighted Weight in grams','Optimiser_flag'])

    data_dt['Date']= pd.to_datetime(data_dt['Date'])
    data_dt['Coinvestment'] = data_dt['Coinvestment'].astype(float)
    data_dt['Flag_promotype_N_pls_1'] = data_dt['Flag_promotype_N_pls_1'].astype(float)

    roi_dt = pd.DataFrame.from_records(roi, columns = ['Account Name', 'Corporate Segment', 'PPG', 'Brand Filter',
       'Brand Format Filter', 'Strategic Cell Filter', 'Nielsen SKU Name',
       'Date', 'Year', 'Week', 'Activity name',
       'Mechanic', 'Discount, NRV %', 'TE Off Inv', 'TE On Inv',
       'GMAC', 'List_Price','COGS'])

    roi_dt['Coinvestment'] = data_dt['Coinvestment']
    roi_dt['Flag_promotype_N_pls_1'] = data_dt['Flag_promotype_N_pls_1']

    coeff_map_dt = pd.DataFrame(list(coeff_map), columns = model_coefficient_cols)

    dec_col = ['Discount, NRV %', 'TE On Inv',
    'TE Off Inv', 'GMAC', 'List_Price', 'COGS','Coinvestment','Flag_promotype_N_pls_1']
    roi_dt['Date'] = pd.to_datetime(roi_dt['Date']) # convert to dataframe date format
    for i in dec_col:
        roi_dt[i]= roi_dt[i].astype(float)

    to_integer_cols = ['Trend_month']
    for i in to_integer_cols:
        coeff_dt[i]= coeff_dt[i].astype(int)
        data_dt[i]= data_dt[i].astype(int)
    
    to_float_cols = ['TPR_Discount','Median_Base_Price_log','SI']
    for i in to_float_cols:
        coeff_dt[i]= coeff_dt[i].astype(float)
        data_dt[i]= data_dt[i].astype(float)

    # Model Data datatype coversion
    model_data_to_float = ['wk_sold_avg_price_byppg','TPR_Discount_lag1','TPR_Discount_lag2','Catalogue','ACV','SI_month','SI_quarter',
    'C_1_crossretailer_discount','C_1_crossretailer_log_price','C_1_intra_discount','C_2_intra_discount','C_3_intra_discount','C_4_intra_discount',
    'C_1_intra_log_price','C_2_intra_log_price','C_3_intra_log_price','C_4_intra_log_price','C_5_intra_log_price','Category trend','Average Weight in grams',
    'Weighted Weight in grams']
    for i in model_data_to_float: 
        data_dt[i] = data_dt[i].astype(float)

    model_data_to_int = ['Intercept','Promo_Depth','Coinvestment','Optimiser_flag','Display','C_5_intra_discount','Trend_month',
    'Trend_quarter','Trend_year','month_no','Flag_promotype_Motivation','Flag_promotype_N_pls_1','Flag_promotype_traffic',
    'Flag_nonpromo_1','Flag_nonpromo_2','Flag_nonpromo_3','Flag_promo_1','Flag_promo_2','Flag_promo_3','Holiday_Flag1','Holiday_Flag2',
    'Holiday_Flag3','Holiday_Flag4','Holiday_Flag5','Holiday_Flag6','Holiday_Flag7','Holiday_Flag8','Holiday_Flag9','Holiday_Flag10']
    for i in model_data_to_int: 
        data_dt[i] = data_dt[i].astype(int)

    # Model Coeff datatype coversion
    model_coeff_to_float = ['WMAPE','Rsq','Intercept','TPR_Discount_lag1','TPR_Discount_lag2','Catalogue','Display','ACV','SI_month',
    'SI_quarter','C_1_crossretailer_discount','C_1_crossretailer_log_price','C_1_intra_discount','C_2_intra_discount','C_3_intra_discount',
    'C_4_intra_discount','C_1_intra_log_price','C_2_intra_log_price','C_3_intra_log_price','C_4_intra_log_price','C_5_intra_log_price','Category trend',
    'Trend_month','Trend_quarter','Trend_year','month_no','Flag_promotype_Motivation','Flag_promotype_N_pls_1','Flag_promotype_traffic','Flag_nonpromo_1',
    'Flag_nonpromo_2','Flag_promo_1', 'Flag_promo_2', 'Flag_promo_3', 'Holiday_Flag1','Holiday_Flag2','Holiday_Flag3','Holiday_Flag4','Holiday_Flag5',
    'Holiday_Flag6','Holiday_Flag7','Holiday_Flag8']
    for i in model_coeff_to_float: 
        coeff_dt[i] = coeff_dt[i].astype(float)

    model_coeff_to_int = ['C_5_intra_discount','Flag_nonpromo_3','Holiday_Flag9','Holiday_Flag10']
    for i in model_coeff_to_int: 
        coeff_dt[i] = coeff_dt[i].astype(int)

    
    # Coeff Mapping datatype coversion
    coeff_mapping_to_float = ['Value']
    for i in coeff_mapping_to_float: 
        coeff_map_dt[i] = coeff_map_dt[i].astype(float)


    # ROI data datatype coversion
    # roi_dt_to_int = ['Year','Week','Coinvestment','Flag_promotype_N_pls_1']
    # for i in roi_dt_to_int: 
    #     roi_dt[i] = roi_dt[i].astype(int)

    return data_dt,roi_dt,coeff_dt,coeff_map_dt
    
