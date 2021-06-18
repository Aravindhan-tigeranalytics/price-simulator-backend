from django.db.models import base
import numpy as np
import pandas as pd
import datetime
from decimal import Decimal
import numpy as np
import pandas as pd
import re


def get_incremental_base_var_cont(var_name,li):
    return [var_name]+[i for i in li if "Catalogue" in i]

def predict_sales(coeffs,data):
    # import pdb
    # pdb.set_trace()
    predict = 0
    for i in coeffs['Variable']:
        if(i=="Intercept"):
            predict = predict + coeffs[coeffs['Variable']==i]["Value"].values
        else:
            data[i] =  data[i].astype(float)
            predict = predict+ data[i]* coeffs[coeffs['Variable']==i]["Value"].values
    data['pred_vol'] = predict
    data['Predicted_Volume'] = np.exp(data['pred_vol'])
    return(data['Predicted_Volume'])

def promo_wave_cal(tpr_data):
  tpr_data['Promo_wave']=0
  c=1
  i=0
  while(i<=tpr_data.shape[0]-1):
      if(tpr_data.loc[i,'TPR_Discount']>0):#####Also tpr ??since in validation consdered tpr
          tpr_data.loc[i,'Promo_wave']=c
          j=i+1
          if(j==tpr_data.shape[0]):
                  break
          while((j<=tpr_data.shape[0]-1) & (tpr_data.loc[j,'TPR_Discount']>0)):
              tpr_data.loc[j,'Promo_wave']=c
              i = j+1
              j = i
              if(j==tpr_data.shape[0]):
                  break
          c=c+1
      i=i+1
  return tpr_data['Promo_wave']    

def _predict(pred_df, model_coef, var_col="model_coefficient_name", coef_col="model_coefficient_value", intercept_name="(Intercept)"):
    """Predict Sales.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Dataframe consisting of IDVs
    model_coef : pd.DataFrame
        Dataframe consisting of model coefficient and its value
    var_col : str
        Column containing model variable
    coef_col : str
        Column containing model variables and their coefficient estimate
    intercept_name : str
        Name of intercept variable

    Returns
    -------
    ndarray
    """
    idv_cols = [col for col in model_coef[var_col] if col != intercept_name]
    idv_coef = model_coef.loc[model_coef[var_col].isin(idv_cols), coef_col]
    idv_df = pred_df.loc[:, idv_cols]

    intercept = model_coef.loc[~model_coef[var_col].isin(idv_cols), coef_col].to_list()[0]
    prediction = idv_df.values.dot(idv_coef.values) + intercept
    
    return prediction

def get_var_contribution_wo_baseline_defined(df, model_coef, wk_sold_price,  all_df=None, var_col="model_coefficient_name", coef_col="model_coefficient_value", intercept_name="(Intercept)", base_var=None):
    """Get variable contribution without baseline defined.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe consisting of IDVs
    model_coef : pd.DataFrame
        Dataframe consisting of model variables and their coefficient estimate
    wk_sold_price : pd.DataFrame
        Dataframe consisting of weekly sold price
    all_df : pd.DataFrame
        Dataframe consisting of IDVs of all PPGs        
    var_col : str
        Column containing model variable
    coef_col : str
        Column containing model variables' coefficient estimate
    intercept_name : str
        Name of intercept variable
    base_var : list of str
        Base variables

    Returns
    -------
    tuple of pd.DataFrame
    """
    # Predict Sales
    ppg_cols = ["PPG_Cat", "PPG_MFG", "PPG_Item_No", "PPG_Description"]
    ppg = df["PPG_Item_No"].to_list()[0]
    model_df = df.drop(columns=ppg_cols).copy()
    model_df["Predicted_sales"] = np.exp(_predict(model_df, model_coef, var_col=var_col, coef_col=coef_col, intercept_name=intercept_name))

    # Get base and impact variables
    if base_var is None:
        tpr_features = [i for i in model_coef[var_col] if "intra_discount" in i]
        rp_features = [i for i in model_coef[var_col] if "intra_log_price" in i]
        base_var = ["Median_Base_Price_log"] + rp_features + ["ACV", "Category trend", "flag_qtr2", "flag_qtr3", "flag_qtr4", "month_no"]
        base_var = [i for i in base_var if i in model_coef[var_col].to_list()]
        
    base_var = [intercept_name] + base_var
    impact_var = [i for i in model_coef[var_col] if i not in base_var]

    # Get base and impact variables
    model_df[intercept_name] = 1
    tmp_model_coef = model_coef[model_coef[var_col].isin(base_var)]
    tmp_model_df = model_df.copy()
    if all_df is not None:
        if "Median_Base_Price_log" in tmp_model_df.columns:
            tmp_model_df = tmp_model_df.merge(all_df.loc[all_df["PPG_Item_No"] == ppg, ["Date", "Final_baseprice"]].drop_duplicates(), how="left", on="Date")
            tmp_model_df["Final_baseprice"] = tmp_model_df["Final_baseprice"].astype(np.float64)
            tmp_model_df["Median_Base_Price_log"] = np.log(tmp_model_df["Final_baseprice"])
            tmp_model_df = tmp_model_df.drop(columns=["Final_baseprice"])
        if tmp_model_df.columns.str.contains(".*RegularPrice.*RegularPrice", regex=True).any():
            rp_interaction_cols = [col for col in tmp_model_df.columns if re.search(".*RegularPrice.*RegularPrice", col) is not None]
            for col in rp_interaction_cols:
                col_adj = re.sub(ppg, "", col)
                col_adj = re.sub("_RegularPrice_", "", col_adj)
                col_adj = re.sub("_RegularPrice", "", col_adj)
                tmp_model_df = tmp_model_df.merge(all_df.loc[all_df["PPG_Item_No"] == ppg, ["Date", "Final_baseprice"]], how="left", on="Date")
                temp = all_df.loc[all_df["PPG_Item_No"] == col_adj, ["Date", "Median_Base_Price_log"]]
                temp = temp.rename(columns={"Median_Base_Price_log": "wk_price_log"})
                tmp_model_df = tmp_model_df.merge(temp[["Date", "wk_price_log"]], how="left", on="Date")
                tmp_model_df[col] = np.log(tmp_model_df["Final_baseprice"]) * tmp_model_df["wk_price_log"]
                tmp_model_df = tmp_model_df.drop(columns=["Final_baseprice", "wk_price_log"])
    base_val = tmp_model_df[tmp_model_coef[var_col].to_list()].values.dot(tmp_model_coef[coef_col].values)
    
    tmp_model_coef = model_coef[model_coef[var_col].isin(impact_var)]
    impact_val = model_df[tmp_model_coef[var_col].to_list()].values.dot(tmp_model_coef[coef_col].values)
    
    model_df["baseline_contribution"] = np.exp(base_val)
    model_df["incremental_contribution"] = model_df["Predicted_sales"] - model_df["baseline_contribution"]
    
    # Calculate raw contribution for impact variables
    row_sum =0
    abs_sum = 0
    for i in impact_var:
        tmp_impact_val = model_df[i].values * model_coef.loc[model_coef[var_col]==i, coef_col].to_list()[0]
        model_df[i + "_contribution_impact"] = np.exp(base_val + impact_val) - np.exp(base_val + impact_val - tmp_impact_val)
        row_sum = row_sum + model_df[i + "_contribution_impact"]
        abs_sum = abs_sum + abs(model_df[i + "_contribution_impact"])

    y_b_s = model_df["incremental_contribution"] - row_sum
    impact_contribution = model_df[["Date", "Predicted_sales"]].copy()
    for i in  impact_var:
        i_adj = i + "_contribution_impact"
        impact_contribution[i_adj] = model_df[i_adj] + (abs(model_df[i_adj])/abs_sum) * y_b_s

    # Calculate raw contribution for base variables
    base_rc = model_coef.loc[model_coef[var_col]==intercept_name, coef_col].to_list()[0]
    impact_contribution[intercept_name + "_contribution_base"] = np.exp(base_rc)
    for i in base_var[1:]:
        t = tmp_model_df[i] * model_coef.loc[model_coef[var_col]==i, coef_col].to_list()[0] + base_rc
        impact_contribution[i + "_contribution_base"] = np.exp(t) - np.exp(base_rc)
        base_rc = t
    impact_contribution = impact_contribution.fillna(0)
    unit_dist_df = impact_contribution.copy()
    
    # Get Dollar Sales
    price_dist_df = unit_dist_df.copy()
    numeric_cols = price_dist_df.select_dtypes(include="number").columns.to_list()
    price_dist_df[numeric_cols] = price_dist_df[numeric_cols].mul(wk_sold_price.values, axis=0)
    
    # Get variable contribution variants
    dt_unit_dist_df, qtr_unit_dist_df, yr_unit_dist_df = get_var_contribution_variants(unit_dist_df, "model_coefficient_name", value_col_name="units")
    dt_price_dist_df, qtr_price_dist_df, yr_price_dist_df = get_var_contribution_variants(price_dist_df, "model_coefficient_name", value_col_name="price")
    overall_dt_dist_df = dt_unit_dist_df.merge(dt_price_dist_df, how="left", on=["Date", "model_coefficient_name"])
    overall_qtr_dist_df = qtr_unit_dist_df.merge(qtr_price_dist_df, how="left", on=["Quarter", "model_coefficient_name"])
    overall_yr_dist_df = yr_unit_dist_df.merge(yr_price_dist_df, how="left", on=["Year", "model_coefficient_name"])
    
    ppg_df = df[ppg_cols].drop_duplicates().values.tolist()
    overall_dt_dist_df[ppg_cols] = pd.DataFrame(ppg_df, index=overall_dt_dist_df.index)
    overall_qtr_dist_df[ppg_cols] = pd.DataFrame(ppg_df, index=overall_qtr_dist_df.index)
    overall_yr_dist_df[ppg_cols] = pd.DataFrame(ppg_df, index=overall_yr_dist_df.index)

    return overall_dt_dist_df, overall_qtr_dist_df, overall_yr_dist_df

def get_var_contribution_variants(dist_df, var_col_name, value_col_name):
    """Get variable contribution by different variants.

    Parameters
    ----------
    dist_df : pd.DataFrame
        Dataframe consisting of IDVs
    var_col_name : str
        Column name for melted IDV columns
    value_col_name : str
        Column name for IDV values

    Returns
    -------
    tuple of pd.DataFrame
    """
    pct_dist_df = dist_df.copy()
    numeric_cols = pct_dist_df.select_dtypes(include="number").columns.to_list()
    pct_dist_df[numeric_cols] = pct_dist_df[numeric_cols].div(pct_dist_df["Predicted_sales"], axis=0)
    dist_df_1 = pd.merge(dist_df.melt(id_vars=["Date"], var_name=var_col_name, value_name=value_col_name),
                         pct_dist_df.melt(id_vars=["Date"], var_name=var_col_name, value_name="pct_" + value_col_name),
                         how="left", on=["Date", var_col_name])
    
    # Quarterly Aggegration
    qtr_dist_df = dist_df.copy()
    qtr_dist_df["Quarter"] = qtr_dist_df["Date"].dt.year.astype(str) + "-" + "Q" + qtr_dist_df["Date"].dt.quarter.astype(str)
    qtr_dist_df = qtr_dist_df.drop(columns=["Date"])
    qtr_dist_df = qtr_dist_df.groupby(by=["Quarter"], as_index=False).agg(np.sum)

    pct_qtr_dist_df = qtr_dist_df.copy()
    pct_qtr_dist_df[numeric_cols] = pct_qtr_dist_df[numeric_cols].div(pct_qtr_dist_df["Predicted_sales"], axis=0)
    dist_df_2 = pd.merge(qtr_dist_df.melt(id_vars=["Quarter"], var_name=var_col_name, value_name=value_col_name),
                         pct_qtr_dist_df.melt(id_vars=["Quarter"], var_name=var_col_name, value_name="pct_" + value_col_name),
                         how="left", on=["Quarter", var_col_name])
    
    # Yearly Aggegration 
    yr_dist_df = dist_df.copy()
    yr_dist_df["Year"] = yr_dist_df["Date"].dt.year.astype(str)
    yr_dist_df = yr_dist_df.drop(columns=["Date"])
    yr_dist_df = yr_dist_df.groupby(by=["Year"], as_index=False).agg(np.sum)
    
    pct_yr_dist_df = yr_dist_df.copy()
    pct_yr_dist_df[numeric_cols] = pct_yr_dist_df[numeric_cols].div(pct_yr_dist_df["Predicted_sales"], axis=0)
    dist_df_3 = pd.merge(yr_dist_df.melt(id_vars=["Year"], var_name=var_col_name, value_name=value_col_name),
                         pct_yr_dist_df.melt(id_vars=["Year"], var_name=var_col_name, value_name="pct_" + value_col_name),
                         how="left", on=["Year", var_col_name])
    
    return dist_df_1, dist_df_2, dist_df_3

def base_var_cont(model_df,model_df1,baseline_var,baseline_var_othr,model_coef):
  m=1
  dt_dist = pd.DataFrame()
  quar_dist = pd.DataFrame()
  year_dist = pd.DataFrame()
  year_dist1 = pd.DataFrame()

  price = "wk_sold_avg_price_byppg"
  price_val = model_df[[price]]
  model_df[["PPG_Cat", "PPG_MFG", "PPG_Item_No", "PPG_Description"]] = pd.DataFrame([["TEMP"]*4], index=model_df.index)
  base_var = baseline_var["Variable"].to_list()
  overall_dt_dist_df, overall_qtr_dist_df, overall_yr_dist_df = get_var_contribution_wo_baseline_defined(model_df, model_coef, price_val, all_df=None, var_col="Variable", coef_col="Value", intercept_name="Intercept", base_var=base_var)
  overall_dt_dist_df['Iteration'] = m
  overall_qtr_dist_df['Iteration'] = m
  overall_yr_dist_df['Iteration'] = m

  overall_yr_dist_df1 = overall_yr_dist_df[overall_yr_dist_df['model_coefficient_name']!= 'Predicted_sales']
  yearly_1 = overall_yr_dist_df1[['Year','units']].groupby(by=["Year"], as_index=False).agg(np.sum).rename(columns = {'units':"Yearly_Units"})
  overall_yr_dist_df1 = pd.merge(overall_yr_dist_df1,yearly_1,on = "Year",how = "left")
  overall_yr_dist_df1['units%'] = overall_yr_dist_df1['units']/overall_yr_dist_df1['Yearly_Units']*100
  dt_dist  = pd.concat([dt_dist,overall_dt_dist_df],ignore_index = False)
  quar_dist  = pd.concat([quar_dist,overall_qtr_dist_df],ignore_index = False)
  year_dist  = pd.concat([year_dist,overall_yr_dist_df],ignore_index = False)
  year_dist1  = pd.concat([year_dist1,overall_yr_dist_df1],ignore_index = False)



  #########Converting to Required Format
  dt_dist = pd.pivot_table(dt_dist[['Date','model_coefficient_name','units','Iteration']],index = ["Date","Iteration"],columns = "model_coefficient_name")
  dt_dist = pd.DataFrame(dt_dist).reset_index()
  dt_dist.columns = dt_dist.columns.droplevel(0) 
  print(dt_dist.columns)
  a = ["Date","Iteration"]+list(dt_dist.columns[2:])
  dt_dist.columns = a

  year_dist = pd.pivot_table(year_dist[['Year','model_coefficient_name','units','Iteration']],index = ["model_coefficient_name","Iteration"],columns = "Year")
  year_dist = pd.DataFrame(year_dist).reset_index()
  year_dist.columns = year_dist.columns.droplevel(0) 
  print(year_dist.columns)
  a = ["model_coefficient_name","Iteration"]+list(year_dist.columns[2:])
  year_dist.columns = a

  year_dist1 = pd.pivot_table(year_dist1[['Year','model_coefficient_name','units%','Iteration']],index = ["model_coefficient_name","Iteration"],columns = "Year")
  year_dist1 = pd.DataFrame(year_dist1).reset_index()
  year_dist1.columns = year_dist1.columns.droplevel(0) 
  print(year_dist1.columns)
  a = ["model_coefficient_name","Iteration"]+list(year_dist1.columns[2:])
  year_dist1.columns = a

  aa = dt_dist.columns
  print(aa)

  ###Competitor Discounts
  Comp = [i for i in aa if "_intra_discount" in i]
  dt_dist['Comp'] = dt_dist[Comp].sum(axis = 1)
  ###Give tpr related Variables
  Incremental = ['TPR_Discount_contribution_impact']+[i for i in aa if "Catalogue" in i]
  print('Incremental :',Incremental)
  dt_dist['Incremental'] = dt_dist[Incremental].sum(axis = 1)
  ###Give the remaining Base columns
  
  base_others = baseline_var_othr["Variable"].to_list()
  base = [ 'Intercept_contribution_base']+[i+'_contribution_base' for i in base_var]+[i+'_contribution_impact' for i in base_others]
  print("base :",base)
  dt_dist['Base'] = dt_dist[base].sum(axis = 1)

  model_df = model_df1.copy()
#   req = model_df[['Date','Iteration','Promo_wave']]
  req = model_df[['Date','Iteration']]
  req['Date'] = pd.to_datetime(req['Date'], format='%Y-%m-%d')
  dt_dist = pd.merge(dt_dist,req,how = "left")
  
  dt_dist['Base']=(dt_dist['Base']+dt_dist['Comp'])


#   dt_dist['Lift'] = dt_dist['Incremental']/(dt_dist['Base'])
  return dt_dist


def main(data_frame,coeff_frame):
    base_data = data_frame
    
    train_coef = coeff_frame
    base_data['Date'] = pd.to_datetime(base_data['Date'])

    
    coeff = train_coef.melt(var_name = "Variable",value_name='Value')
    coeff = coeff.iloc[9:].reset_index(drop=True) # taking only the numeric rows  8 -> 9
    coeff['Value'] = coeff['Value'].astype(float)
    ### train_data is the original or the simulated model data
    train_data = base_data.copy() 
    model_coef = coeff.copy()

    # train_data['Promo_wave']=promo_wave_cal(train_data)
    train_data['Units']=predict_sales(model_coef,train_data)

    train_data['wk_sold_avg_price_byppg']=np.exp(train_data['Median_Base_Price_log'])*(1-train_data['TPR_Discount']/100)
    model_df=train_data.copy()


    #Base Variable Method
    baseline_var = pd.DataFrame(columns =["Variable"])
    baseline_var_othr = pd.DataFrame(columns =["Variable"])
    col = model_coef["Variable"].to_list()

    # baseline variables
    baseprice_cols = ['Median_Base_Price_log']+[i for i in col if "_intra_log_price" in i]
    holiday =[i for i in col if "Holiday" in i ]
    SI_cols = ["SI","SI_month","SI_quarter" ]
    trend_cols = ["Trend_month","Trend_quarter","Trend_year"]
    base_list = baseprice_cols+SI_cols+trend_cols+["ACV"]
    base_others = holiday+[i for i in col if "death_rate" in i]+[i for i in col if 'TPR_Discount_lag' in i]
    print(base_list)
    baseline_var["Variable"]=base_list
    baseline_var_othr["Variable"]=base_others

    model_df['Iteration']=1
    model_df1=model_df.copy()
    model_coef['Iteration']=1

    base_scenario=base_var_cont(model_df,model_df1,baseline_var,baseline_var_othr,model_coef) # base and increment function
    return base_scenario


def main_file(file1,file2, retailer , ppg, segment):
    
    
    model_data = pd.read_excel(file1,sheet_name='MODEL_DATA')
    model_coeff = pd.read_excel(file1,sheet_name='MODEL_COEFFICIENT')
     

    ret = retailer
    ppg = ppg
    tag_path  = ""
    Segment=segment 
    prd=0
    base_data = model_data.loc[(model_data['Account Name']==ret) & (model_data['PPG']==ppg)]

    train_coef = model_coeff.loc[(model_coeff['Account Name']==ret) & (model_coeff['PPG']==ppg)]
    
    coeff = train_coef.melt(var_name = "Variable",value_name='Value')
    coeff = coeff.iloc[8:].reset_index(drop=True) # taking only the numeric rows  8 -> 9
    coeff['Value'] = coeff['Value'].astype(float)
    ### train_data is the original or the simulated model data
    train_data = base_data.copy() 
    model_coef = coeff.copy()

    # train_data['Promo_wave']=promo_wave_cal(train_data)
    train_data['Units']=predict_sales(model_coef,train_data)

    train_data['wk_sold_avg_price_byppg']=np.exp(train_data['Median_Base_Price_log'])*(1-train_data['TPR_Discount']/100)
    model_df=train_data.copy()


    #Base Variable Method
    baseline_var = pd.DataFrame(columns =["Variable"])
    baseline_var_othr = pd.DataFrame(columns =["Variable"])
    col = model_coef["Variable"].to_list()

    # baseline variables
    baseprice_cols = ['Median_Base_Price_log']+[i for i in col if "_intra_log_price" in i]
    holiday =[i for i in col if "Holiday" in i ]
    SI_cols = ["SI","SI_month","SI_quarter" ]
    trend_cols = ["Trend_month","Trend_quarter","Trend_year"]
    base_list = baseprice_cols+SI_cols+trend_cols+["ACV"]
    base_others = holiday+[i for i in col if "death_rate" in i]+[i for i in col if 'TPR_Discount_lag' in i]
    print(base_list)
    baseline_var["Variable"]=base_list
    baseline_var_othr["Variable"]=base_others

    model_df['Iteration']=1
    model_df1=model_df.copy()
    model_coef['Iteration']=1

    base_scenario=base_var_cont(model_df,model_df1,baseline_var,baseline_var_othr,model_coef) # base and increment function
    print(base_scenario , "base scenario")
    return base_scenario[['Date','Incremental' , 'Base','Lift']]



coeff_columns = ['meta_id','Account Name', 'Corporate Segment', 'PPG', 'Brand Filter',
       'Brand Format Filter', 'Strategic Cell Filter', 'WMAPE', 'Rsq','Intercept', 'Median_Base_Price_log', 'TPR_Discount', 'TPR_Discount_lag1', 'TPR_Discount_lag2', 'Catalogue', 'Display', 'ACV', 'SI', 'SI_month', 'SI_quarter', 'C_1_crossretailer_discount', 'C_1_crossretailer_log_price', 'C_1_intra_discount', 'C_2_intra_discount', 'C_3_intra_discount', 'C_4_intra_discount', 'C_5_intra_discount', 'C_1_intra_log_price', 'C_2_intra_log_price', 'C_3_intra_log_price', 'C_4_intra_log_price', 'C_5_intra_log_price', 'Category trend', 'Trend_month', 'Trend_quarter', 'Trend_year', 'month_no', 'Flag_promotype_Motivation', 'Flag_promotype_N_pls_1', 'Flag_promotype_traffic', 'Flag_nonpromo_1', 'Flag_nonpromo_2', 'Flag_nonpromo_3', 'Flag_promo_1', 'Flag_promo_2', 'Flag_promo_3', 'Holiday_Flag1', 'Holiday_Flag2', 'Holiday_Flag3', 'Holiday_Flag4', 'Holiday_Flag5', 'Holiday_Flag6', 'Holiday_Flag7', 'Holiday_Flag8', 'Holiday_Flag9', 'Holiday_Flag10']
# coeff_columns
data_columns =  ['meta_id','Account Name', 'Corporate Segment', 'PPG', 'Brand Filter',
       'Brand Format Filter', 'Strategic Cell Filter','Year', 'Quarter',
       'Month', 'Period', 'Date', 'Week','Intercept', 'Median_Base_Price_log', 'TPR_Discount','promo_depth','co investment', 'TPR_Discount_lag1', 'TPR_Discount_lag2', 'Catalogue', 'Display', 'ACV', 'SI', 'SI_month', 'SI_quarter', 'C_1_crossretailer_discount', 'C_1_crossretailer_log_price', 'C_1_intra_discount', 'C_2_intra_discount', 'C_3_intra_discount', 'C_4_intra_discount', 'C_5_intra_discount', 'C_1_intra_log_price', 'C_2_intra_log_price', 'C_3_intra_log_price', 'C_4_intra_log_price', 'C_5_intra_log_price', 'Category trend', 'Trend_month', 'Trend_quarter', 'Trend_year', 'month_no', 'Flag_promotype_Motivation', 'Flag_promotype_N_pls_1', 'Flag_promotype_traffic', 'Flag_nonpromo_1', 'Flag_nonpromo_2', 'Flag_nonpromo_3', 'Flag_promo_1', 'Flag_promo_2', 'Flag_promo_3', 'Holiday_Flag1', 'Holiday_Flag2', 'Holiday_Flag3', 'Holiday_Flag4', 'Holiday_Flag5', 'Holiday_Flag6', 'Holiday_Flag7', 'Holiday_Flag8', 'Holiday_Flag9',
        'Holiday_Flag10', 'wk_sold_avg_price_byppg','Average Weight in grams','Weighted Weight in grams']


# coeff_dt = pd.DataFrame(coeff, columns = coeff_columns)
# coeff_dt
# data_dt = pd.DataFrame(data, columns = data_columns)
# data_dt

# main(data_dt,coeff_dt,retailer,ppg,segment)
def list_to_frame_bkp(coeff,data,flag=False):
    coeff_dt = pd.DataFrame(coeff, columns = coeff_columns)
    
    if flag:
        co_dt = pd.DataFrame([[i.pop()['co_investment']] for i in data] , columns = ['co_inv'])
        data_dt = pd.DataFrame(data, columns = data_columns)
        data_dt['TPR_Discount'] = data_dt['TPR_Discount'] + co_dt['co_inv']
        
    else:
        data_dt = pd.DataFrame(data, columns = data_columns)
        
    retuned_dt = main( data_dt,coeff_dt )
    if flag:
        retuned_dt['co_inv'] = co_dt['co_inv']
    print(retuned_dt['Incremental'][0] , "Incremental")
    print(retuned_dt['Base'][0] , "Base")
    print(retuned_dt['Predicted_sales'][0] , "Predicted_sales")
    return retuned_dt

def list_to_frame(coeff,data):
    coeff_dt = pd.DataFrame(coeff, columns = coeff_columns)
   

    data_dt = pd.DataFrame(data, columns = data_columns)
    data_dt['TPR_Discount'] = data_dt['promo_depth'] + data_dt['co investment']
        
   
    return main( data_dt,coeff_dt )


