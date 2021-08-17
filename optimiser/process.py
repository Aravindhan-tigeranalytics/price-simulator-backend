
from django.db import models
from django.db.models import F , Value
from django.db.models.functions import Concat
from utils import constants as const
import json
import pandas as pd
from core import models as model
from scenario_planner import query as cols
from scenario_planner import calculations as cal

coeff_values = const.COEFFICIENT_VALUES
data_values = const.DATA_VALUES
coeff_map_values = const.COEFFICIENT_MAP_VALUES
roi_values = const.ROI_VALUES


def get_list_from_db(retailer,ppg , optimizer_save = None , promo_week = None , pricing_week = None):
    
  
    model_coefficient_cols = ['Account Name', 'Corporate Segment', 'PPG', 'Brand Filter',
       'Brand Format Filter', 'Strategic Cell Filter',
       'Coefficient_new', 'Value', 'Coefficient']

    coeff_map = model.CoeffMap.objects.select_related('model_meta').filter(
        model_meta__account_name__iexact = retailer,model_meta__product_group__iexact = ppg
        ).values_list(
           *coeff_map_values
            )
    
    model_coeff = model.ModelCoefficient.objects.select_related('model_meta').filter(
        model_meta__account_name__iexact = retailer,model_meta__product_group__iexact = ppg
        ).values_list(
           *coeff_values
            )
    
    model_data = model.ModelData.objects.select_related('model_meta').filter(
        model_meta__account_name__iexact = retailer,model_meta__product_group__iexact = ppg
        ).values_list(
           *data_values
            )

    roi = model.ModelROI.objects.select_related('model_meta').filter(
    model_meta__account_name__iexact = retailer,model_meta__product_group__iexact = ppg
    ).values_list(
        *roi_values
        ).annotate(
            cogs=F('list_price') - (F('list_price') * F('gmac'))
            )

    model_data_list = [list(i) for i in model_data]
    roi_data_list = [list(i) for i in roi]
  
    if optimizer_save:
        model_data_list = cal.update_from_optimizer(model_data_list , optimizer_save)
    
    if promo_week:
        model_data_list,roi_data_list = cal.update_from_saved_data(model_data_list , promo_week,roi_list=roi_data_list)
    
    if pricing_week:
         model_data_list,roi_data_list = cal.update_optimizer_from_pricing(model_data_list , pricing_week,roi_list=roi_data_list)
        
         
    
    
    # cal.update_from_pricing(model_data_list , roi_data_list)
        
        
    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', 100)  # or 1000

    
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

    data_dt = pd.DataFrame(model_data_list, columns= ['Account Name', 'Corporate Segment', 'PPG', 'Brand Filter',
       'Brand Format Filter', 'Strategic Cell Filter', 'Year', 'Quarter',      
       'Month', 'Period', 'Date', 'Week', 'Intercept', 'Median_Base_Price_log',
       'TPR_Discount','TPR_Discount_lag1', 'TPR_Discount_lag2', 'Catalogue',  
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
       'Weighted Weight in grams','death_rate','Promo_Depth', 'Coinvestment','Optimiser_flag'])

    data_dt['Date']= pd.to_datetime(data_dt['Date'])
    data_dt['Coinvestment'] = data_dt['Coinvestment'].astype(float)
    data_dt['Flag_promotype_N_pls_1'] = data_dt['Flag_promotype_N_pls_1'].astype(float)

    roi_dt = pd.DataFrame.from_records(roi_data_list, columns = ['Account Name', 'Corporate Segment', 'PPG', 'Brand Filter',
       'Brand Format Filter', 'Strategic Cell Filter', 'Nielsen SKU Name',
       'Date', 'Year', 'Week', 'Activity name',
       'Mechanic', 'Discount, NRV %', 'TE Off Inv', 'TE On Inv',
       'GMAC', 'List_Price','COGS'])

    roi_dt['Coinvestment'] = data_dt['Coinvestment']
    roi_dt['Flag_promotype_N_pls_1'] = data_dt['Flag_promotype_N_pls_1']

    coeff_map_dt = pd.DataFrame(list(coeff_map), columns = model_coefficient_cols)
    
    # convert to dataframe datatypes

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
    'Weighted Weight in grams','death_rate']
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

    roi_dt['Promo_Depth'] = data_dt['Promo_Depth']
    roi_dt['Promo_Depth'] = roi_dt['Promo_Depth'].astype(float)
    roi_dt['Month'] = data_dt['Month']
    roi_dt['Quarter'] = data_dt['Quarter']
    roi_dt['Period'] = data_dt['Period']

    return data_dt,roi_dt,coeff_dt,coeff_map_dt



def update_db(opt_base,retailer,ppg):   
    meta = model.ModelMeta.objects.get(
        account_name = retailer,
        product_group = ppg
    )
    bulk_obj = []
    for ind in opt_base.index:
        bulk_obj.append(model.OptimizerSave(
            model_meta =  meta,
            date = opt_base['Date'][ind],
            optimum_promo = opt_base['Optimum_Promo'][ind],
            optimum_units = opt_base['Optimum_Units'][ind],
            optimum_base = opt_base['Optimum_Base'][ind],
            optimum_incremental = opt_base['Optimum_Incremental'][ind],
            base_promo = opt_base['Baseline_Promo'][ind],
            base_units = opt_base['Baseline_Units'][ind],
            base_base = opt_base['Baseline_Base'][ind],
            base_incremental = opt_base['Baseline_Incremental'][ind],
            
            ))
    model.OptimizerSave.objects.bulk_create(bulk_obj)     
    











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






