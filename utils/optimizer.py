import numpy as np
import pandas as pd
from pulp import *
import time
import datetime
import math
import itertools
# import openpyxl
from itertools import chain
from utils import constants as CONST

def predict_sales(coeffs,data):
    predict = 0
    for i in coeffs['names']:
        if(i=="Intercept"):
            predict = predict + coeffs[coeffs['names']==i]["model_coefficients"].values
        else:
            predict = predict+ data[i]* coeffs[coeffs['names']==i]["model_coefficients"].values
    data['pred_vol'] = predict
    data['Predicted_Volume'] = np.exp(data['pred_vol'])
    return(data['Predicted_Volume'])

def update_params(config_constrain , constrain_params,constraints):
      # {'config_mac': True, 'config_rp': True, 'config_trade_expense': False, 'config_units': False,
      #  'config_mac_perc': False, 'config_min_length': True, 'config_max_length': True, 
      #  'config_promo_gap': True, 'param_mac': 1.0, 'param_rp': 1.0, 'param_trade_expense': 1.0, 
      #  'param_units': 1.0, 'param_mac_perc': 1.0, 'param_min_length': 2.0, 'param_max_length': 5.0, 
      #  'param_promo_gap': 4.0} constaints
      # print(constraints , "constaints ")
      config_constrain['MAC'] = constraints['config_mac']
      config_constrain['RP'] = constraints['config_rp']
      config_constrain['Trade_Expense'] = constraints['config_trade_expense']
      config_constrain['Units'] = constraints['config_units']
      config_constrain['MAC_Perc'] = constraints['config_mac_perc']
      config_constrain['min_length'] = constraints['config_min_length']
      config_constrain['max_length'] = constraints['config_max_length']
      config_constrain['promo_gap'] = constraints['config_promo_gap']
      
      constrain_params['MAC'] = constraints['param_mac']
      constrain_params['RP'] = constraints['param_rp']
      constrain_params['Trade_Expense'] = constraints['param_trade_expense']
      constrain_params['Units'] = constraints['param_units']
      constrain_params['MAC_Perc'] = constraints['param_mac_perc']
      constrain_params['min_length'] = constraints['param_min_length']
      constrain_params['max_length'] = constraints['param_max_length']
      constrain_params['promo_gap'] = constraints['param_promo_gap']
      # pass
      

def process(constraints = None):
      min_promotion = 16
      max_promotion = 23
      objective_function = 'MAC'
      MINIMIZE_PARAMS = ['Trade_Expense']
      
      config_constrain = {'MAC':True,'RP':True,'Trade_Expense':False,'Units':False,'MAC_Perc':False,'min_length':True,'max_length':True,
                        'promo_gap':True}
      constrain_params ={'MAC':1,'RP':1,'Trade_Expense':1,'Units':1,'MAC_Perc':1,'min_length':2,'max_length':5,
                        'promo_gap':4}
      # print(constrain_params , "constraint params before")
      if constraints:
            # print(constraints , "constainst from api")
            update_params(config_constrain , constrain_params , constraints)
            objective_function = constraints['objective_function']
            min_promotion = constraints['min_promotion']
            max_promotion = constraints['max_promotion']
            
        
      #changeinput
      # print(constrain_params , "constraint params aafter ")
    
    
    #   path = 'C:/Users/aravindhan.mathi/Videos/project/data/'
      path = CONST.PATH

      ROI_data = pd.read_csv(path + CONST.ROI_FILE_NAME)
      Model_Coeff=pd.read_csv(path + CONST.MODEL_COEFF_FILE_NAME )
      Model_Data=pd.read_csv(path + CONST.MODEL_DATA_FILE_NAME)
    #   print(Model_Data , "model data")
      Ret_name=CONST.RETAILER
      PPG_name= CONST.PRODUCT_GROUP
      Any_SKU_Name =CONST.SKU
      promo_list_PPG = ROI_data[(ROI_data['Retailer'] == Ret_name) & (ROI_data['PPG Name'] == PPG_name)].reset_index(drop=True)
      promo_list_PPG = ROI_data[(ROI_data['Retailer'] == Ret_name) & (ROI_data['PPG Name'] == PPG_name) & (ROI_data['Discount, NRV %'] == 0.25) ].reset_index(drop=True)
    #   print(promo_list_PPG.shape , "ssha[e")

      # print(promo_list_PPG , "promo list ppg")

      promo_list_PPG = ROI_data[(ROI_data['Retailer'] == Ret_name) & (ROI_data['PPG Name'] == PPG_name) & (ROI_data['Discount, NRV %'] == 0.25) ].reset_index(drop=True)
      # print(promo_list_PPG.shape)
      # promo_list_PPG['Discount, NRV %'] = np.where(promo_list_PPG['Discount, NRV %']==0.1,0.25,promo_list_PPG['Discount, NRV %'])
      # promo_list_PPG['Discount, NRV %']=0.33
      # Period_data["Promo"] = np.where(Period_data['Discount, NRV %'] == 0.25, 0.33,Period_data['Discount, NRV %']))
      Period_map = pd.DataFrame(pd.date_range("2021", freq="W", periods=52), columns=['Date'])
      Period_map['WK Num']=Period_map.index+1
      promo_list_SKU = promo_list_PPG[promo_list_PPG['Nielsen SKU Name'] == Any_SKU_Name].reset_index(drop=True)
      promo_list_SKU = promo_list_SKU.rename(columns={'Weeknum':'WK Num'})
      promo_list_PPG=promo_list_SKU[['Discount, NRV %','TE Off Inv','TE On Inv','COGS','List_Price','WK Num']]
      Period_data = pd.merge(Period_map[["Date", "WK Num"]],promo_list_PPG,how = "left",on = "WK Num")
      Period_data["Discount, NRV %"] = np.where(pd.isna(Period_data["Discount, NRV %"]),0,Period_data["Discount, NRV %"])

      Period_data["List_Price"] = np.where(pd.isna(Period_data["List_Price"]),
                                            Period_data.dropna(subset = ["List_Price"])['List_Price'].unique()
                                            ,Period_data["List_Price"])
        
      Period_data["TE Off Inv"] = np.where(pd.isna(Period_data["TE Off Inv"]),
                                            Period_data.dropna(subset = ["TE Off Inv"])['TE Off Inv'].unique()
                                            ,Period_data["TE Off Inv"])
        
      Period_data["TE On Inv"] = np.where(pd.isna(Period_data["TE On Inv"]),
                                            Period_data.dropna(subset = ["TE On Inv"])['TE On Inv'].unique()
                                            ,Period_data["TE On Inv"])
        
      Period_data["COGS"] = np.where(pd.isna(Period_data["COGS"]),
                                            Period_data.dropna(subset = ["COGS"])['COGS'].unique()
                                            ,Period_data["COGS"])
        
      Period_data["Promotion_Cost"] = Period_data['List_Price'] * Period_data['Discount, NRV %'] * (1 - Period_data['TE On Inv'])
      Period_data["TE"] = Period_data["List_Price"] * (Period_data["Discount, NRV %"] + Period_data["TE On Inv"] + Period_data["TE Off Inv"] - 
                                                              Period_data["Discount, NRV %"] * Period_data["TE Off Inv"] - Period_data["TE Off Inv"] * Period_data["TE On Inv"] - 
                                                              Period_data["TE On Inv"] * Period_data["Discount, NRV %"] + Period_data["TE Off Inv"] * Period_data["TE On Inv"] * Period_data["Discount, NRV %"])
      baseprice_pre_July = math.exp(max(Model_Data[((pd.DatetimeIndex(Model_Data['Unnamed: 0']).year == 2020) & (pd.DatetimeIndex(Model_Data['Unnamed: 0']).month >=1 ))]["wk_sold_median_base_price_byppg_log"]))
      baseprice_post_July= 1.07016*baseprice_pre_July
      Period_data["wk_base_price_perunit_byppg"] = baseprice_post_July
      Period_data["Promo"] = np.where(Period_data['Discount, NRV %'] == 0, Period_data['wk_base_price_perunit_byppg'],
                                        Period_data['wk_base_price_perunit_byppg'] * (1-Period_data['Discount, NRV %']))
      Period_data["wk_sold_avg_price_byppg"] = np.where(pd.isna(Period_data['Promo']), Period_data['wk_base_price_perunit_byppg'],
                                                          Period_data['Promo'])
      Period_data  = Period_data[['Date', 'wk_sold_avg_price_byppg', 'wk_base_price_perunit_byppg', 'Promo', "TE", "List_Price", "Promotion_Cost","COGS"]]
      Period_data['tpr_discount_byppg']=0
      Period_data['median_baseprice']=baseprice_post_July
      var='median_baseprice'
      Period_data['tpr_discount_byppg']=((Period_data['median_baseprice']-Period_data['wk_sold_avg_price_byppg'])/Period_data['median_baseprice'])*100
      # Filter the model data for last 52 weeks
      index_2019=Model_Data.shape[0]-52
      Model_Data['Year_Flag']=Model_Data['Unnamed: 0'].apply(lambda x: x.split('-')[0])
      index_2020=Model_Data[Model_Data['Year_Flag']=='2020'].index.tolist()[0]
      pred_data1=Model_Data[index_2020:Model_Data.shape[0]]
      pred_data2=Model_Data[index_2019:index_2020]
      pred_data = pred_data1.append(pred_data2).reset_index(drop = "True")
      Model_Coeff_list_Keep=list(Model_Coeff['names'])
      Model_Coeff_list_Keep.remove(Model_Coeff_list_Keep[0])
      pred_data=pred_data[Model_Coeff_list_Keep]
      pred_data['Date']=pd.date_range("2021", freq="W", periods=52)
      pred_data['flag_promo_under1']=0
      pred_data['flag_promo_under2']=0
      pred_data['trend_year']=3
      pred_data['flag_christmas_day_new']=np.where(pred_data["Date"] =='2021-05-30',1,0)

      #####Regular Price
      pred_data['ITEM5006_Lenta_RegularPrice'] = np.log(1.07016) + max(pred_data['ITEM5006_Lenta_RegularPrice'])

      Catalogue_temp=pred_data[pred_data['tpr_discount_byppg']>0]['Catalogue_Dist'].mean()
      # Catalogue_temp = 0.803
      # print(Catalogue_temp)
      pred_data=pred_data.drop(['wk_sold_median_base_price_byppg_log',],axis=1)
      pred_data=pred_data.rename(columns={'tpr_discount_byppg':'tpr_discount_byppg_train'})
      Final_Pred_Data=pd.merge(Period_data,pred_data,how="left",on="Date")
      # If the change is very less for eg in train 23% and ROI 24% Please proceed to the next step but if the difference is non tpr week is tpr week check the ROI file for dates of promotion redo and come to this stage
      Final_Pred_Data['QC']=Final_Pred_Data['tpr_discount_byppg'].astype(int)-Final_Pred_Data['tpr_discount_byppg_train'].astype(int)
      Final_Pred_Data['QC'].sum()
      # print(Final_Pred_Data[Final_Pred_Data['QC']>0])
      # Final_Pred_Data['tpr_discount_byppg']==33
      # Final_Pred_Data['tpr_discount_byppg']=np.where(Final_Pred_Data["QC"] ==33,0,Final_Pred_Data['tpr_discount_byppg'])
      # Final_Pred_Data=Final_Pred_Data.drop(['QC'],axis=1)
      # Satish ROI data 8/8/2021 to 29/8/2021 has Promotion but actual data doesnt have  the other differences are Okay this run goes with the assumption that based on the ROI data we have promotions have happened in August and output is based on the same Please confirm on this and let me know 
      temp=Final_Pred_Data[Final_Pred_Data['tpr_discount_byppg']>0]['Catalogue_Dist'].mean()
      Final_Pred_Data['Catalogue_Dist']=np.where(Final_Pred_Data['tpr_discount_byppg']==0,0,Catalogue_temp)
      Final_Pred_Data['tpr_discount_byppg_train']=Final_Pred_Data['tpr_discount_byppg']
      Final_Pred_Data['wk_sold_median_base_price_byppg_log']=np.log(Final_Pred_Data['median_baseprice'])
      Financial_information=Final_Pred_Data[['Promo','List_Price','COGS','median_baseprice','TE','tpr_discount_byppg','Promotion_Cost']].drop_duplicates().reset_index(drop=True)
      TE_dict= dict(zip(Financial_information.tpr_discount_byppg, Financial_information.TE))
      prd=0
      # Here for OTC and XXL alone we calculate the predicted units for 33% all other calculation for 25%
      FPD_units=Final_Pred_Data.copy()
      training_data_optimal = Final_Pred_Data.copy()
      TPR_list=Final_Pred_Data['tpr_discount_byppg'].unique()
      TPR_list.sort()
    #   print(TPR_list , "tpr list")

      FPD_units['tpr_discount_byppg']=np.where(FPD_units['tpr_discount_byppg']>=20,33,FPD_units['tpr_discount_byppg'])

      Final_Pred_Data['Baseline_Prediction'] = predict_sales(Model_Coeff,FPD_units)
      # print(Final_Pred_Data[Final_Pred_Data['tpr_discount_byppg']>0]['Baseline_Prediction'])
      Final_Pred_Data['Baseline_Sales']=Final_Pred_Data['Baseline_Prediction'] *Final_Pred_Data['Promo']
      Final_Pred_Data["Baseline_GSV"] = Final_Pred_Data['Baseline_Prediction'] * Final_Pred_Data['List_Price']
      Final_Pred_Data["Baseline_Trade_Expense"] = Final_Pred_Data['Baseline_Prediction'] * Final_Pred_Data['TE']
      Final_Pred_Data["Baseline_NSV"] = Final_Pred_Data['Baseline_GSV'] - Final_Pred_Data["Baseline_Trade_Expense"]
      Final_Pred_Data["Baseline_MAC"] = Final_Pred_Data["Baseline_NSV"]-Final_Pred_Data['Baseline_Prediction'] * Final_Pred_Data['COGS']
      Final_Pred_Data["Baseline_RP"] = Final_Pred_Data['Baseline_Sales']-Final_Pred_Data["Baseline_NSV"]
      #Export Baseline Training Data
      prd=0
      Final_Pred_Data.to_csv(path+"Model_Results/Training_data_LPBase_newbase.csv",index=False)
      Final_Pred_Data.to_csv(path+"Model_Results/Training_data_0.csv",index=False)
      Financial_information.to_csv(path+"Output/Promos_mapping_"+str(prd)+".csv")

      # print(TE_dict)
      Final_Pred_Data['Baseline_Sales']=Final_Pred_Data['Baseline_Prediction'] *Final_Pred_Data['Promo']
      Final_Pred_Data["Baseline_GSV"] = Final_Pred_Data['Baseline_Prediction'] * Final_Pred_Data['List_Price']
      Final_Pred_Data["Baseline_Trade_Expense"] = Final_Pred_Data['Baseline_Prediction'] * Final_Pred_Data['TE']
      Final_Pred_Data["Baseline_NSV"] = Final_Pred_Data['Baseline_GSV'] - Final_Pred_Data["Baseline_Trade_Expense"]
      Final_Pred_Data["Baseline_MAC"] = Final_Pred_Data["Baseline_NSV"]-Final_Pred_Data['Baseline_Prediction'] * Final_Pred_Data['COGS']
      Final_Pred_Data["Baseline_RP"] = Final_Pred_Data['Baseline_Sales']-Final_Pred_Data["Baseline_NSV"]
      Final_Pred_Data[['Baseline_Prediction','Baseline_Sales',"Baseline_GSV","Baseline_Trade_Expense","Baseline_NSV","Baseline_MAC","Baseline_RP"]].sum().apply(lambda x: '%.3f' % x)

      baseline_df =Final_Pred_Data[['Baseline_Prediction','Baseline_Sales',"Baseline_GSV","Baseline_Trade_Expense","Baseline_NSV","Baseline_MAC","Baseline_RP"]].sum().astype(int)
      baseline_df['Baseline_MAC']

    #   print(Final_Pred_Data , "Final pre data")
      TPR_list=list(TE_dict.keys())
      TPR_list=[i for i in TPR_list if i>0]
      # TPR_list.remove(24.999999999999993)
      # TPR_list.remove(9.999999999999993)
      # print(TPR_list)

      prd=0
      OTC_train_Data=pd.read_csv(path+"Model_Results/Training_data_LPBase_newbase.csv")
      Base=OTC_train_Data[['wk_base_price_perunit_byppg','Promo', 'TE', 'List_Price','COGS','tpr_discount_byppg','flag_promo_under1',
            'flag_promo_under2', 'flag_children_protection_day', 'trend_year','Catalogue_Dist', 'ACV_Selling', 'SI',
            'ITEM5006_Lenta_PromotedDiscount', 'ITEM5006_Lenta_RegularPrice',
            'flag_christmas_day_new','wk_sold_median_base_price_byppg_log']]

      # all 33%
      Catalogue_temp=0.803
      i=0
      TPR_list=list(TE_dict.keys())
      # TPR_list.remove(24.999999999999993)
      # TPR_list.remove(9.999999999999993)
      TPR_list.sort()
      # print(TPR_list)
      # TPR_list=[j for j in TPR_list if j>0]
      for tpr in TPR_list:
        Required_base=Base.copy() 
        Required_base['tpr_discount_byppg']=tpr
        Required_base['Promo']	=Required_base['wk_base_price_perunit_byppg']-(Required_base['wk_base_price_perunit_byppg']*Required_base['tpr_discount_byppg']/100)
        Required_base.loc[Required_base['tpr_discount_byppg'].isin(TE_dict.keys()), 'TE'] = Required_base['tpr_discount_byppg'].map(TE_dict)
        Required_base['Catalogue_Dist']=np.where(Required_base['tpr_discount_byppg']==0,0,Catalogue_temp)
        if tpr>10:
          # print(tpr)
          Required_temp=Required_base.copy()
          Required_temp['tpr_discount_byppg']=33
          # print(Required_temp['tpr_discount_byppg'].unique())
          # print("entering loop")
          Required_base['Units']=predict_sales(Model_Coeff,Required_temp)  
        else:
          # print("no")
          Required_base['Units']=predict_sales(Model_Coeff,Required_base)  
      #   Required_base['Units']=predict_sales(Model_Coeff,Required_base)
        Required_base['Promo Price']=Required_base['wk_base_price_perunit_byppg']-(Required_base['wk_base_price_perunit_byppg']*(Required_base['tpr_discount_byppg']/100))  
        Required_base['Sales']=Required_base['Units'] *Required_base['Promo Price']
        Required_base["GSV"] = Required_base['Units'] * Required_base['List_Price']
        Required_base["Trade_Expense"] = Required_base['Units'] * Required_base['TE']
        Required_base["NSV"] = Required_base['GSV'] - Required_base["Trade_Expense"]
        Required_base["MAC"] = Required_base["NSV"]-Required_base['Units'] * Required_base['COGS']
        Required_base["RP"] = Required_base['Sales']-Required_base["NSV"]
        cols = Required_base.columns[Required_base.columns.isin(['Units','Sales',"GSV","Trade_Expense","NSV","MAC","RP"])]
        Required_base["TPR"]=tpr
        Required_base["Iteration"]=i
        if i==0:
          new_base=Required_base
        else:
          new_base=new_base.append(Required_base)

        i=i+1
      new_base.to_csv(path+"Model_Results/New_base.csv")

      Required_base=new_base
      Required_base=Required_base.reset_index(drop=True)
      Required_base['WK_ID']=Required_base.index
      Required_base['WK_ID'] = 'WK_' + Required_base['WK_ID'].astype(str)+'_'+Required_base['Iteration'].astype(str)
      WK_DV_vars = list(Required_base['WK_ID'])
      WK_DV_tprvars = list(Required_base['Iteration'])

      Required_base['WK_ID']=Required_base.index
      Required_base['WK_ID'] = 'WK_' + Required_base['WK_ID'].astype(str)+'_'+Required_base['Iteration'].astype(str)
      WK_DV_vars = list(Required_base['WK_ID'])
      WK_DV_tprvars = list(Required_base['Iteration'])

      Required_base.head()
      promo_loop=Required_base['Iteration'].nunique()
      prob = LpProblem("Simple_Workaround_problem",LpMinimize if(objective_function in MINIMIZE_PARAMS) else LpMaximize)
      WK_vars = LpVariable.dicts("RP",WK_DV_vars,cat='Binary')

      prob+=lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base[objective_function][i]  for i in range(0,Required_base.shape[0])])
      # prob+=lpSum([WK_vars[i]  for i in WK_DV_vars ] )
      # Subject to 
      if(config_constrain['MAC']):
        prob+=lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base['MAC'][i]  for i in range(0,Required_base.shape[0])])>=int(baseline_df['Baseline_MAC']*constrain_params['MAC'])

      if(config_constrain['RP']):
        prob+=lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base['RP'][i]  for i in range(0,Required_base.shape[0])])>=int(baseline_df['Baseline_RP']*constrain_params['RP'])

      if(config_constrain['Trade_Expense']):
        prob+=lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base['Trade_Expense'][i]  for i in range(0,Required_base.shape[0])])<=int(baseline_df['Baseline_Trade_Expense']*constrain_params['Trade_Expense'])

      if(config_constrain['Units']):
        prob+=lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base['Units'][i]  for i in range(0,Required_base.shape[0])])>=int(baseline_df['Baseline_Prediction']*constrain_params['Units'])
        
      if(config_constrain['MAC_Perc']):
        L1 =lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base['MAC'][i]  for i in range(0,Required_base.shape[0])])
        L2=lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base['NSV'][i]  for i in range(0,Required_base.shape[0])])
        prob+= L1 >= L2*(baseline_df['Baseline_MAC']/baseline_df['Baseline_NSV'])*constrain_params['MAC_Perc']

      # Set up constraints such that only one tpr is chose for a week
      for i in range(0,52):
        prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]  for j in range(0,promo_loop)])<=1
        prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for j in range(0,promo_loop)])>=1  

      # Costraint for No of promotions 
      prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]  for j in range(1,promo_loop) for i in range(0,52)])<=max_promotion
      prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for j in range(1,promo_loop) for i in range(0,52)])>=min_promotion
      # prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for j in range(1,2) for i in range(0,52)])<=10

      # Constraint for unique tpr in a promo wave
      for i in range(0,51):
        prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]+WK_vars[Required_base['WK_ID'][i+1+(j+1)*52]] for j in range(1,promo_loop-1) ])<=1
        prob+=lpSum([WK_vars[Required_base['WK_ID'][i+(j+1)*52]]+WK_vars[Required_base['WK_ID'][i+1+j*52]] for j in range(1,promo_loop-1) ])<=1

      # Sub constraint for minimum weeks
      R0=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(0,1)])
      R1=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(1,2)])
      R2=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(2,3)])
      R3=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(3,4)])
      R4=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(4,5)])
      R5=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(5,6)])
      R6=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(6,7)])
      R7=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(7,8)])
      R8=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(8,9)])
      R9=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(9,10)])
      R10=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(10,11)])
      R11=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(11,12)])

      # Sub constraint for minimum weeks
      R51=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(50,51)])
      R52=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(51,52)])
      R50=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(49,50)])
      R49=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(48,49)])
      R48=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(47,48)])
      R47=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(46,47)])
      R46=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(45,46)])
      R45=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(44,45)])
      R44=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(43,44)])
      R43=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(42,43)])
      R42=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(41,42)])
      R41=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]for j in range(1,promo_loop) for i in range(40,41)])


      if(config_constrain['min_length']):
        if constrain_params['min_length']==2:
          prob+=R51-R52 >=0
          prob+=R1-R0 >=0
        if constrain_params['min_length']==3:
          prob+=R51-R52 >=0
          prob+=R1-R0 >=0
          prob+=2*R50-R51-R52 >=0
          prob+= 2*R2-R1-R0 >=0
        if constrain_params['min_length']==4:
          prob+=R51-R52 >=0
          prob+=R1-R0 >=0
          prob+=2*R50-R51-R52 >=0
          prob+= 2*R2-R1-R0 >=0
          prob+= 3*R49-R50-R51-R52 >=0
          prob+= 3*R3-R2-R1-R0 >=0
        if constrain_params['min_length']==5:
          prob+=R51-R52 >=0
          prob+=R1-R0 >=0
          prob+=2*R50-R51-R52 >=0
          prob+= 2*R2-R1-R0 >=0
          prob+= 3*R49-R50-R51-R52 >=0
          prob+= 3*R3-R2-R1-R0 >=0
          prob+= 4*R48-R49-R50-R51-R52 >=0
          prob+= 4*R4-R3-R2-R1-R0 >=0
      # print(config_constrain['max_length'],"-",constrain_params['max_length'])
      # contraint for max promo length
      if(config_constrain['max_length']):
        for k in range(0,52-constrain_params['max_length']):
          for j in range(1,promo_loop):
            prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for i in range(k,k+constrain_params['max_length']+1)])<=constrain_params['max_length']
        for k in range(0,52-constrain_params['max_length']):  
            prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for i in range(k,k+constrain_params['max_length']+1) for j in range(1,promo_loop)])<=constrain_params['max_length']

      # Constrain for min promo wave length
      if(config_constrain['min_length']):
        for k in range(0,50):
          R1_sum = lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for i in range(k+1,min(k+constrain_params['min_length']+1,52)) for j in range(1,promo_loop)])
          R2_sum = lpSum([WK_vars[Required_base['WK_ID'][k+j*52]] for j in range(1,promo_loop)])
          R3_sum = lpSum([WK_vars[Required_base['WK_ID'][k+1+j*52]] for j in range(1,promo_loop)])
          gap_weeks = len(range(k+1, min(52, k+constrain_params['min_length']+1)))
          prob+= R1_sum + gap_weeks * R2_sum >= gap_weeks * R3_sum

        for k in range(0,50):
          for j in range(1,promo_loop):
            R1_sum = lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for i in range(k+1,min(k+constrain_params['min_length']+1,52))])
            R2_sum = lpSum([WK_vars[Required_base['WK_ID'][k+j*52]] ])
            R3_sum = lpSum([WK_vars[Required_base['WK_ID'][k+1+j*52]]])
            gap_weeks = len(range(k+1, min(52, k+constrain_params['min_length']+1)))
            prob+= R1_sum + gap_weeks * R2_sum >= gap_weeks * R3_sum
      # 4 week gap
      if(config_constrain['promo_gap']):
        for k in range(0,51):
          R1_sum = lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for j in range(0,1) for i in range(k+1,min(k+constrain_params['promo_gap']+1,52))])
          R2_sum= lpSum([WK_vars[Required_base['WK_ID'][k+j*52]] for j in range(0,1)])
          R3_sum = lpSum([WK_vars[Required_base['WK_ID'][k+1+j*52]] for j in range(0,1)])
          gap_weeks = len(range(k+1, min(52, k+constrain_params['promo_gap']+1)))
          prob+= R1_sum + gap_weeks * R2_sum >= gap_weeks * R3_sum
      prob.solve(PULP_CBC_CMD(msg=False))

      # print(prob , "prob result")

      # print(LpStatus[prob.status])
      # print(pulp.value(prob.objective))
      Weeks =[]
      value = []
      for variable in prob.variables():
        if variable.varValue==1.0:
          Weeks.append(str(variable.name))
          value.append(variable.varValue)
      #     print(variable.name, variable.varValue)
      df= pd.DataFrame(list(zip(Weeks, value)), 
                    columns =['Weeks', 'val']) 
      df['Iteration']=df["Weeks"].apply(lambda x: x.split('_')[3]).astype(int)
      df['Week_no']=df["Weeks"].apply(lambda x: x.split('_')[2]).astype(int)
      df['Week_no']=(df['Week_no']-(df['Iteration']*52))+1
      df = df.sort_values('Week_no',ascending = True).reset_index(drop=True)
      # df.head(51)
      tprs = new_base[['Iteration','TPR']].drop_duplicates().reset_index(drop=True)
      tprs
      df = pd.merge(df,tprs,on=['Iteration'],how='left')
      df = df.sort_values('Week_no',ascending = True).reset_index(drop=True)
      # print(df , "DF")

      new_data=training_data_optimal.copy()
      new_data['tpr_discount_byppg']=df['TPR']
      new_data['Catalogue_Dist']=np.where(new_data['tpr_discount_byppg']==0,0,Catalogue_temp)
      new_data_temp=new_data.copy()
      new_data_temp['tpr_discount_byppg']=np.where(new_data_temp['tpr_discount_byppg']>10,33,new_data_temp['tpr_discount_byppg'])
      new_data['Units']=predict_sales(Model_Coeff,new_data_temp)  
      # new_data['Units']=predict_sales(Model_Coeff,new_data)     
      new_data.loc[new_data['tpr_discount_byppg'].isin(TE_dict.keys()), 'TE'] = new_data['tpr_discount_byppg'].map(TE_dict)
      new_data['Promo Price']=new_data['median_baseprice']-(new_data['median_baseprice']*(new_data['tpr_discount_byppg']/100)) 
      new_data['wk_sold_avg_price_byppg']=new_data['Promo Price']
      new_data['Sales']=new_data['Units'] *new_data['Promo Price']
      new_data["GSV"] = new_data['Units'] * new_data['List_Price']
      new_data["Trade_Expense"] = new_data['Units'] * new_data['TE']
      new_data["NSV"] = new_data['GSV'] - new_data["Trade_Expense"]
      new_data["MAC"] = new_data["NSV"]-new_data['Units'] * new_data['COGS']
      new_data["RP"] = new_data['Sales']-new_data["NSV"]
      new_data.to_csv(path+"Model_Results/Training_data_0_1.csv")
      new_data[['Sales',"GSV","Trade_Expense","NSV","MAC","RP","Units"]].sum().apply(lambda x: '%.3f' % x)

    #   print(new_data , "new data value")
    #   res = new_data.to_json(orient="records")
    # #   print(res , "res")
    #   parsed = json.loads(res)
    # #   print(parsed , "parsed")
    #   fin = json.dumps(parsed, indent=4) 
    
      parsed = json.loads(new_data.to_json(orient="records"))
    #   print(parsed , "parsed")
      return parsed
