PATH = 'D:/projects/MARSBACKEND/price-simulator-backend/data/'
ROI_FILE_NAME = "ROI_DATA_Internal_Promo_2021.csv"
MODEL_COEFF_FILE_NAME =  "Model_Coeff_Lenta_Orbit OTC.csv"
MODEL_DATA_FILE_NAME ="Model_Data_Lenta_Orbit OTC.csv" 
RETAILER ="Lenta"
PRODUCT_GROUP ="Orbit OTC"
SKU ="ORBIT(MARS) YAGODNIY MIKS BERRY MIX F 10P 0.0136KG WP"

DATA_HEADER = ['Account Name' , 'Corporate Segment' , 'PPG','Brand Filter' ,
               'Brand Format Filter','Strategic Cell Filter','Year','Quarter','Month','Period',
    'Date','Week',
    'Intercept','Median_Base_Price_log' , 'TPR_Discount','TPR_Discount_lag1','TPR_Discount_lag2',
    'Catalogue','Display','ACV','SI','SI_month','SI_quarter','C_1_crossretailer_discount',
    'C_1_crossretailer_log_price','C_1_intra_discount','C_2_intra_discount','C_3_intra_discount',
    'C_4_intra_discount','C_5_intra_discount','C_1_intra_log_price','C_2_intra_log_price'
    ,'C_3_intra_log_price','C_4_intra_log_price','C_5_intra_log_price','Category trend','Trend_month',
    'Trend_quarter','Trend_year','month_no','Flag_promotype_Motivation','Flag_promotype_N_pls_1',
    'Flag_promotype_traffic','Flag_nonpromo_1','Flag_nonpromo_2','Flag_nonpromo_3','Flag_promo_1',
    'Flag_promo_2','Flag_promo_3','Holiday_Flag1','Holiday_Flag2','Holiday_Flag3','Holiday_Flag4',
    'Holiday_Flag5','Holiday_Flag6','Holiday_Flag7','Holiday_Flag8','Holiday_Flag9','Holiday_Flag10',
    'wk_sold_avg_price_byppg','Average Weight in grams','Weighted Weight in grams']

COEFF_HEADER = ['Account Name' , 'Corporate Segment' , 'PPG' ,
                'Brand Filter' ,
               'Brand Format Filter','Strategic Cell Filter',
                'WMAPE','Rsq',
    'Intercept','Median_Base_Price_log' , 'TPR_Discount','TPR_Discount_lag1','TPR_Discount_lag2',
    'Catalogue','Display','ACV','SI','SI_month','SI_quarter','C_1_crossretailer_discount',
    'C_1_crossretailer_log_price','C_1_intra_discount','C_2_intra_discount','C_3_intra_discount',
    'C_4_intra_discount','C_5_intra_discount','C_1_intra_log_price','C_2_intra_log_price'
    ,'C_3_intra_log_price','C_4_intra_log_price','C_5_intra_log_price','Category trend','Trend_month',
    'Trend_quarter','Trend_year','month_no','Flag_promotype_Motivation','Flag_promotype_N_pls_1',
    'Flag_promotype_traffic','Flag_nonpromo_1','Flag_nonpromo_2','Flag_nonpromo_3','Flag_promo_1',
    'Flag_promo_2','Flag_promo_3','Holiday_Flag1','Holiday_Flag2','Holiday_Flag3','Holiday_Flag4',
    'Holiday_Flag5','Holiday_Flag6','Holiday_Flag7','Holiday_Flag8','Holiday_Flag9','Holiday_Flag10']

ROI_HEADER = [
    'Account Name','Corporate Segment','PPG','Brand Filter','Brand Format Filter','Strategic Cell Filter',
    'Year','Week','TE Off Inv','TE On Inv','GMAC','List_Price'
]

PROMO_MODEL_META_MAP = {
    'Account Name':'account_name',
    'Corporate Segment':'corporate_segment',
    'PPG':'product_group',
    'Brand Filter' :'brand_filter',
    'Brand Format Filter' : 'brand_format_filter',
    'Strategic Cell Filter':'strategic_cell_filter'
}
CALCULATION_METRIC = {
    'Intercept' : 'intercept',
    'Median_Base_Price_log':'median_base_price_log' ,
    'TPR_Discount':'tpr_discount',
    'TPR_Discount_lag1':'tpr_discount_lag1',
    'TPR_Discount_lag2' : 'tpr_discount_lag2',
    'Catalogue':'catalogue',
    'Display':'display',
    'ACV':'acv',
    'SI':'si',
    'SI_month':'si_month',
    'SI_quarter':'si_quarter',
    'C_1_crossretailer_discount':'c_1_crossretailer_discount',
    'C_1_crossretailer_log_price' : 'c_1_crossretailer_log_price',
    'C_1_intra_discount':'c_1_intra_discount',
    'C_2_intra_discount':'c_2_intra_discount',
    'C_3_intra_discount':'c_3_intra_discount',
    'C_4_intra_discount':'c_4_intra_discount',
    'C_5_intra_discount':'c_5_intra_discount',
    'C_1_intra_log_price':'c_1_intra_log_price',
    'C_2_intra_log_price':'c_2_intra_log_price',
    'C_3_intra_log_price':'c_3_intra_log_price',
    'C_4_intra_log_price':'c_4_intra_log_price',
    'C_5_intra_log_price':'c_5_intra_log_price',
    'Category trend':'category_trend',
    'Trend_month':'trend_month',
    'Trend_quarter':'trend_quarter',
    'Trend_year':'trend_year',
    'month_no':'month_no',
    'Flag_promotype_Motivation':'flag_promotype_motivation',
    'Flag_promotype_N_pls_1':'flag_promotype_n_pls_1',
    'Flag_promotype_traffic':'flag_promotype_traffic',
    'Flag_nonpromo_1':'flag_nonpromo_1',
    'Flag_nonpromo_2':'flag_nonpromo_2',
    'Flag_nonpromo_3':'flag_nonpromo_3',
    'Flag_promo_1':'flag_promo_1',
    'Flag_promo_2':'flag_promo_2',
    'Flag_promo_3':'flag_promo_3',
    'Holiday_Flag1':'holiday_flag_1',
    'Holiday_Flag2':'holiday_flag_2',
    'Holiday_Flag3':'holiday_flag_3',
    'Holiday_Flag4':'holiday_flag_4',
    'Holiday_Flag5':'holiday_flag_5',
    'Holiday_Flag6':'holiday_flag_6',
    'Holiday_Flag7':'holiday_flag_7',
    'Holiday_Flag8':'holiday_flag_8',
    'Holiday_Flag9':'holiday_flag_9',
    'Holiday_Flag10':'holiday_flag_10',
 
}
COEFF_MAP = {
    'WMAPE':'wmape',
    'Rsq':'rsq',
}
DATA_MAP = {
    'Year':'year',
    'Quarter':'quater',
    'Month':'month',
    'Period':'period',
    'Date':'date',
    'Week':'week',
    'wk_sold_avg_price_byppg' : 'wk_sold_avg_price_byppg',
     'Average Weight in grams':'average_weight_in_grams',
    'Weighted Weight in grams':'weighted_weight_in_grams'
}
ROI_MAP = {
    'Year':'year',
    'Week':'week',
    'TE Off Inv':'off_inv',
    'TE On Inv':'on_inv',
    'GMAC' : 'gmac',
    'List_Price' : 'list_price',
}
PROMO_MODEL_COEFF_MAP = {**CALCULATION_METRIC , **COEFF_MAP}
PROMO_MODEL_DATA_MAP = {**CALCULATION_METRIC , **DATA_MAP}
