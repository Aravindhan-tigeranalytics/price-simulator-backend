import os
import numpy as np
import pandas as pd
import re
import time
import datetime
import math
import itertools
# import openpyxl
from itertools import chain
from joblib import Parallel, delayed
import statistics 
from pulp import *
import pandas as pd
from itertools import combinations
from utils import constants as CONST
from optimiser import process

class my_dictionary(dict): 
  
    # __init__ function 
    def __init__(self): 
        self = dict() 
          
    # Function to add key:value 
    def add(self, key, value): 
        self[key] = value 


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


##### Promo wave/slot calculation#######  
def promo_wave_cal(tpr_data):
  tpr_data['Promo_wave']=0
  c=1
  i=0
  while(i<=tpr_data.shape[0]-1):
      if(tpr_data.loc[i,'tpr_discount_byppg']>0):#####Also tpr ??since in validation consdered tpr
          tpr_data.loc[i,'Promo_wave']=c
          j=i+1
          if(j==tpr_data.shape[0]):
                  break
          while((j<=tpr_data.shape[0]-1) & (tpr_data.loc[j,'tpr_discount_byppg']>0)):
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


def set_baseline_value(model_df, model_coef, var_col="model_coefficient_name", coef_col="model_coefficient_value", intercept_name="(Intercept)", baseline_value=None):
    """Set baseline value.

    Parameters
    ----------
    model_df : pd.DataFrame
        Dataframe consisting of IDVs
    model_coef : pd.DataFrame
        Dataframe consisting of model variables and their coefficient estimate
    var_col : str
        Column containing model variable
    coef_col : str
        Column containing model variables' coefficient estimate
    intercept_name : str
        Name of intercept variable
    baseline_value : pd.DataFrame
        Dataframe consisting of IDVs with baseline value

    Method
    ------
    # If action is Rolling Max, then take max value in the last n weeks.
    # If action is Set Average, then baseline value is set to the value specified.
    # If action is As is, then it is taken as is.
    # Variables not specified in the baseline file are set to 0.

    Returns
    -------
    pd.DataFrame
    """
    # Get baseline value
    if baseline_value is None:
        baseline_value = {"wk_sold_median_base_price_byppg_log": ["Rolling Max", 26],
                          "tpr_discount_byppg": ["Set Average", 0],
                          "tpr_discount_byppg_lag1": ["Set Average", 0],
                          "tpr_discount_byppg_lag2": ["Set Average", 0],                  
                          "ACV_Feat_Only": ["Set Average", 0],
                          "ACV_Disp_Only": ["Set Average", 0],
                          "ACV_Feat_Disp": ["Set Average", 0],
                          "ACV_Selling": ["As is", 0],
                          "flag_qtr2": ["As is", 0],
                          "flag_qtr3": ["As is", 0],
                          "flag_qtr4": ["As is", 0],
                          "category_trend": ["As is", 0],
                          "monthno": ["As is", 0]}
        baseline_value = pd.DataFrame.from_dict(baseline_value, orient="index").reset_index()
        baseline_value = baseline_value.rename(columns={"index": "Parameters", 0: "Action", 1: "Value"})
        
        baseline_value = baseline_value[baseline_value["Parameters"].isin(model_coef[var_col])]
        comp_features = model_coef.loc[~model_coef["model_coefficient_name"].isin(baseline_value["Parameters"].to_list() + [intercept_name]), [var_col]]
        
        if len(comp_features)>0:
            comp_features = comp_features.rename(columns={"model_coefficient_name": "Parameters"})
            comp_features["Action"] = comp_features.apply(lambda x: "Set Average" if "PromotedDiscount" in x["Parameters"] else "As is", axis=1)
            comp_features["Action"] = comp_features.apply(lambda x: "Rolling Max" if "RegularPrice" in x["Parameters"] else x["Action"], axis=1)
            comp_features["Value"] = comp_features.apply(lambda x: 26 if "Rolling Max" in x["Action"] else 0, axis=1)
            baseline_value = baseline_value.append(comp_features, ignore_index=True)
    
    # Initialization	
    base_var = [intercept_name] + baseline_value.loc[baseline_value["Action"].isin(["Rolling Max", "As is"]), "Parameters"].to_list()
    baseline_df = model_df.loc[:, model_df.columns.isin(model_coef[var_col])].copy()
    
    # Set baseline value
    for i in baseline_value["Parameters"].to_list():
        action = baseline_value.loc[baseline_value["Parameters"]==i, "Action"].to_list()[0]
        value = baseline_value.loc[baseline_value["Parameters"]==i, "Value"].to_list()[0]
        if action == "Set Average":
            baseline_df[i] = value
        elif action == "As is":
            baseline_df[i] = model_df[i]
        elif action == "Rolling Max":
            tmp_df = model_df[["Date", i]].set_index("Date")
            tmp_df["Final_baseprice"] = tmp_df[i].rolling(value, min_periods=1).max()
            
            # TODO: Check do price variants make sense for other parameters
            # tmp_df["Final_baseprice"] = tmp_df.apply(lambda x: x["median_baseprice"] if x["Final_baseprice"] == (-np.Inf) else x["Final_baseprice"], axis=1)
            # tmp_df["Final_baseprice"] = tmp_df.apply(lambda x: x["wk_sold_avg_price_byppg"] if np.isnan(x["Final_baseprice"]) else x["Final_baseprice"], axis=1)
            baseline_df[i] = tmp_df["Final_baseprice"].reset_index(drop=True)
        else:
            # Do nothing
            print()
            
    # Parameters not specified in  Baseline condition
    other_params = [i for i in baseline_df.columns if (i not in baseline_value["Parameters"].to_list()) and (i !=intercept_name)]
    baseline_df.loc[:, other_params] = 0

    return base_var, baseline_df


def get_var_contribution_wt_baseline_defined(df, model_coef, wk_sold_price, var_col="model_coefficient_name", coef_col="model_coefficient_value", intercept_name="(Intercept)", baseline_value=None):
    """Get variable contribution with baseline defined.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe consisting of IDVs
    model_coef : pd.DataFrame
        Dataframe consisting of model variables and their coefficient estimate
    wk_sold_price : pd.DataFrame
        Dataframe consisting of weekly sold price        
    var_col : str
        Column containing model variable
    coef_col : str
        Column containing model variables' coefficient estimate
    intercept_name : str
        Name of intercept variable
    baseline_value : pd.DataFrame
        Dataframe consisting of IDVs with baseline value

    Returns
    -------
    tuple of pd.DataFrame
    """
    # Predict Sales
    ppg_cols = ["PPG_Cat", "PPG_MFG", "PPG_Item_No", "PPG_Description"]
    model_df = df.drop(columns=ppg_cols).copy()
    model_df["Predicted_sales"] = np.exp(_predict(model_df, model_coef, var_col=var_col, coef_col=coef_col, intercept_name=intercept_name))
    
    # Predict Baseline Sales
    tmp_model_df = model_df.copy()    
    base_var, baseline_df = set_baseline_value(tmp_model_df, model_coef, var_col=var_col, coef_col=coef_col, intercept_name=intercept_name, baseline_value=baseline_value)
    baseline_df["Predicted_baseline_sales"] = np.exp(_predict(baseline_df, model_coef, var_col=var_col, coef_col=coef_col, intercept_name=intercept_name))
    model_df["Predicted_baseline_sales"] = baseline_df["Predicted_baseline_sales"]
    model_df["incremental_predicted"] = model_df["Predicted_sales"] - model_df["Predicted_baseline_sales"]
    
    # Calculate raw contribution
    model_df[intercept_name] = 1
    baseline_df[intercept_name] = 1
    pred_xb = _predict(model_df, model_coef, var_col=var_col, coef_col=coef_col, intercept_name=intercept_name)
    rc_sum = 0
    abs_sum = 0
        
    for i in model_coef[var_col]:
        rc = 0
        rc = np.exp(pred_xb) - np.exp(pred_xb - 
                                      (model_df[i]*model_coef.loc[model_coef[var_col]==i, coef_col].to_list()[0]) +
                                      (baseline_df[i]*model_coef.loc[model_coef[var_col]==i, coef_col].to_list()[0]))
        model_df[i + "_" + "rc"] = rc
        rc_sum = rc_sum + rc
        abs_sum = abs_sum + abs(rc)

    # Calculate actual contribution
    y_b_s = model_df["incremental_predicted"] - rc_sum
    unit_dist_df = model_df[["Date", "Predicted_sales", "Predicted_baseline_sales"]].copy()
    for i in model_coef[var_col]:
        rc = model_df[i + "_" + "rc"]
        ac = rc + (abs(rc)/abs_sum) * y_b_s
        i_adj = i + "_contribution_inc_base" if i in base_var else i + "_contribution_impact"
        unit_dist_df[i_adj] = ac
    unit_dist_df = unit_dist_df.fillna(0)
    
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
        tpr_features = [i for i in model_coef[var_col] if "PromotedDiscount" in i]
        rp_features = [i for i in model_coef[var_col] if "RegularPrice" in i]
        base_var = ["wk_sold_median_base_price_byppg_log"] + rp_features + ["ACV_Selling", "category_trend", "flag_qtr2", "flag_qtr3", "flag_qtr4", "monthno"]
        base_var = [i for i in base_var if i in model_coef[var_col].to_list()]
        
    base_var = [intercept_name] + base_var
    impact_var = [i for i in model_coef[var_col] if i not in base_var]

    # Get base and impact variables
    model_df[intercept_name] = 1
    tmp_model_coef = model_coef[model_coef[var_col].isin(base_var)]
    tmp_model_df = model_df.copy()
    if all_df is not None:
        if "wk_sold_median_base_price_byppg_log" in tmp_model_df.columns:
            tmp_model_df = tmp_model_df.merge(all_df.loc[all_df["PPG_Item_No"] == ppg, ["Date", "Final_baseprice"]].drop_duplicates(), how="left", on="Date")
            tmp_model_df["Final_baseprice"] = tmp_model_df["Final_baseprice"].astype(np.float64)
            tmp_model_df["wk_sold_median_base_price_byppg_log"] = np.log(tmp_model_df["Final_baseprice"])
            tmp_model_df = tmp_model_df.drop(columns=["Final_baseprice"])
        if tmp_model_df.columns.str.contains(".*RegularPrice.*RegularPrice", regex=True).any():
            rp_interaction_cols = [col for col in tmp_model_df.columns if re.search(".*RegularPrice.*RegularPrice", col) is not None]
            for col in rp_interaction_cols:
                col_adj = re.sub(ppg, "", col)
                col_adj = re.sub("_RegularPrice_", "", col_adj)
                col_adj = re.sub("_RegularPrice", "", col_adj)
                tmp_model_df = tmp_model_df.merge(all_df.loc[all_df["PPG_Item_No"] == ppg, ["Date", "Final_baseprice"]], how="left", on="Date")
                temp = all_df.loc[all_df["PPG_Item_No"] == col_adj, ["Date", "wk_sold_median_base_price_byppg_log"]]
                temp = temp.rename(columns={"wk_sold_median_base_price_byppg_log": "wk_price_log"})
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
#Base variable change as required
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
  Comp = [i for i in aa if "PromotedDiscount" in i]
  dt_dist['Comp'] = dt_dist[Comp].sum(axis = 1)
  ###Give tpr related Variables
  Incremental = ['tpr_discount_byppg_contribution_impact']+[i for i in aa if "Catalogue" in i]
  print('Incremental :',Incremental)
  dt_dist['Incremental'] = dt_dist[Incremental].sum(axis = 1)
  ###Give the remaining Base columns
  
  #   baseprice_cols = ['wk_sold_median_base_price_byppg_log']+[i for i in model_cols if "RegularPrice" in i]
  #   holiday =[i for i in model_cols if "day" and "flag" in i ]
  #   SI_cols = [i for i in model_cols if "SI" in i ]
  #   trend_cols = [i for i in model_cols if "trend" in i ]
  #   base_list = baseprice_cols+SI_cols+trend_cols+[i for i in model_cols if "ACV_Selling" in i ]+holiday+[i for i in model_cols if "death_rate" in i]
  base_others = baseline_var_othr["Variable"].to_list()
  base = [ 'Intercept_contribution_base']+[i+'_contribution_base' for i in base_var]+[i+'_contribution_impact' for i in base_others]
  print("base :",base)
  dt_dist['Base'] = dt_dist[base].sum(axis = 1)

  model_df = model_df1.copy()
  req = model_df[['Date','Iteration','Promo_wave']]
  req['Date'] = pd.to_datetime(req['Date'], format='%Y-%m-%d')
  dt_dist = pd.merge(dt_dist,req,how = "left")
  
  dt_dist['Base']=(dt_dist['Base']+dt_dist['Comp'])


  dt_dist['Lift'] = dt_dist['Incremental']/(dt_dist['Base'])
  return dt_dist



def process(constraints = None):
  # path = CONST.PATH
  BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  path = os.path.join(BASE_DIR + "/data/")

  # ROI_data = pd.read_csv(path + CONST.OPT_ROI_FILE_NAME)
  # Model_Coeff = pd.read_csv(path + CONST.OPT_MODEL_COEFF_FILE_NAME)
  # Model_Data = pd.read_csv(path + CONST.OPT_MODEL_DATA_FILE_NAME)
  
  # print(ROI_data.columns)
  # print(Model_Coeff.columns)
  # print(Model_Data.columns)

  Model_Coeff,Model_Data,ROI_data = process.get_list_value_from_query(CONST.OPT_RETAILER_NAME,CONST.OPT_PRODUCT_GROUP_NAME)

  # print(ROI_data)
  # print(Model_Coeff)
  # print(Model_Data)

  # exit()
  Ret_name = CONST.OPT_RETAILER_NAME
  # PPG_name = CONST.OPT_PRODUCT_GROUP_NAME
  PPG_name = 'Tander'
  Any_SKU_Name = CONST.OPT_SKU_NAME
  promo_list_PPG = ROI_data[(ROI_data['Retailer'] == Ret_name) & (ROI_data['PPG Name'] == PPG_name)].reset_index(drop=True)

  print(promo_list_PPG.shape)
  # Removing mismatched promo
  promo_list_PPG = promo_list_PPG.loc[~(promo_list_PPG['Weeknum'].isin([1,2])) ].reset_index(drop=True)
  # promo_list_PPG['Discount, NRV %']=np.where(promo_list_PPG['Discount, NRV %']!=0,0.15,0)
  print(promo_list_PPG.shape)
  # Create Period Mapping File 
  Period_map = pd.DataFrame(pd.date_range("2022", freq="W", periods=52), columns=['Date'])
  Period_map['WK Num']=Period_map.index+1
  promo_list_SKU = promo_list_PPG[promo_list_PPG['Nielsen SKU Name'] == Any_SKU_Name].reset_index(drop=True)
  promo_list_SKU = promo_list_SKU.rename(columns={'Weeknum':'WK Num'})
  promo_list_PPG=promo_list_SKU[['Discount, NRV %','TE Off Inv','TE On Inv','GMAC','COGS','List_Price','WK Num']]
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
  Period_data["GMAC"] = np.where(pd.isna(Period_data["GMAC"]),
                                        Period_data.dropna(subset = ["GMAC"])['GMAC'].unique()
                                        ,Period_data["GMAC"])
    
  Period_data["Promotion_Cost"] = Period_data['List_Price'] * Period_data['Discount, NRV %'] * (1 - Period_data['TE On Inv'])
  Period_data["TE"] = Period_data["List_Price"] * (Period_data["Discount, NRV %"] + Period_data["TE On Inv"] + Period_data["TE Off Inv"] - 
                                                          Period_data["Discount, NRV %"] * Period_data["TE Off Inv"] - Period_data["TE Off Inv"] * Period_data["TE On Inv"] - 
                                                          Period_data["TE On Inv"] * Period_data["Discount, NRV %"] + Period_data["TE Off Inv"] * Period_data["TE On Inv"] * Period_data["Discount, NRV %"])


  baseprice_post_July = math.exp(max(Model_Data[((pd.DatetimeIndex(Model_Data['Unnamed: 0']).year == 2020) & (pd.DatetimeIndex(Model_Data['Unnamed: 0'],dayfirst=True).month >=7 ))]["wk_sold_median_base_price_byppg_log"]))
  # baseprice_post_July = 1.097257*baseprice_pre_July
  # print(baseprice_pre_July)
  print(baseprice_post_July)
  Period_data["wk_base_price_perunit_byppg"] = baseprice_post_July
  Period_data["Promo"] = np.where(Period_data['Discount, NRV %'] == 0, Period_data['wk_base_price_perunit_byppg'],
                                    Period_data['wk_base_price_perunit_byppg'] * (1-Period_data['Discount, NRV %']))
  Period_data["wk_sold_avg_price_byppg"] = np.where(pd.isna(Period_data['Promo']), Period_data['wk_base_price_perunit_byppg'],
                                                      Period_data['Promo'])
  Period_data  = Period_data[['Date', 'wk_sold_avg_price_byppg', 'wk_base_price_perunit_byppg', 'Promo', "TE", "List_Price", "Promotion_Cost","COGS",'GMAC',"TE Off Inv","TE On Inv"]]
  Period_data['tpr_discount_byppg']=0
  Period_data['median_baseprice']=baseprice_post_July
  var='median_baseprice'
  Period_data['tpr_discount_byppg']=((Period_data['median_baseprice']-Period_data['wk_sold_avg_price_byppg'])/Period_data['median_baseprice'])*100

  # Filter the model data for last 52 weeks
  index_2019=Model_Data.shape[0]-52
  Model_Data['Year_Flag']=Model_Data['Unnamed: 0'].apply(lambda x: x.split('-')[0])  ## for otc make it 2
  index_2020=Model_Data[Model_Data['Year_Flag']=='2020'].index.tolist()[0]
  pred_data1=Model_Data[index_2020:Model_Data.shape[0]]
  pred_data2=Model_Data[index_2019:index_2020]
  pred_data = pred_data1.append(pred_data2).reset_index(drop = "True")
  Model_Coeff_list_Keep=list(Model_Coeff['names'])
  Model_Coeff_list_Keep.remove(Model_Coeff_list_Keep[0])
  pred_data=pred_data[Model_Coeff_list_Keep]
  pred_data['Date']=pd.date_range("2022", freq="W", periods=52)
  pred_data['flag_old_mans_day']=np.where(pred_data["Date"].isin(['2022-10-09']),1,0)
  pred_data['Promo_flg_date_1'] = 0
  # pred_data['flag_builders_day']=np.where(pred_data["Date"] == '2022-08-14',1,0)
  a=[i for i in pred_data.columns if re.search("RegularPrice",i)]
  print(a)
  for comp in a:
    print(max(pred_data[comp]))
    pred_data[comp] = max(pred_data[comp])
  #   pred_data[comp] = np.log(1.097257) + max(pred_data[comp])
    print(max(pred_data[comp]))
  # Add competitor price hike regular price
  Catalogue_temp=pred_data[pred_data['tpr_discount_byppg']>0]['Catalogue_Dist'].mean()
  pred_data=pred_data.drop(['wk_sold_median_base_price_byppg_log',],axis=1)
  pred_data=pred_data.rename(columns={'tpr_discount_byppg':'tpr_discount_byppg_train'})
  Final_Pred_Data=pd.merge(Period_data,pred_data,how="left",on="Date")

  # If the change is very less for eg in train 23% and ROI 24% Please proceed to the next step but if the difference is non tpr week is tpr week check the ROI file for dates of promotion redo and come to this stage
  Final_Pred_Data['QC']=Final_Pred_Data['tpr_discount_byppg'].astype(int)-Final_Pred_Data['tpr_discount_byppg_train'].astype(int)
  print(Final_Pred_Data['QC'].sum())
  Final_Pred_Data[Final_Pred_Data['QC']>0]


  Final_Pred_Data=Final_Pred_Data.drop(['QC'],axis=1)
  temp=Final_Pred_Data[Final_Pred_Data['tpr_discount_byppg']>0]['Catalogue_Dist'].mean()
  Final_Pred_Data['Catalogue_Dist']=np.where(Final_Pred_Data['tpr_discount_byppg']==0,0,Catalogue_temp)
  Final_Pred_Data['tpr_discount_byppg_train']=Final_Pred_Data['tpr_discount_byppg']
  Final_Pred_Data['wk_sold_median_base_price_byppg_log']=np.log(Final_Pred_Data['median_baseprice'])
  # Final_Pred_Data['tpr_discount_byppg_lag1'] = Final_Pred_Data['tpr_discount_byppg'].shift(1).fillna(0)
  # acv = Final_Pred_Data['ACV_Selling'].to_list()
  # acv = acv[38:52]
  # acv_med = statistics.median(acv) 
  # Final_Pred_Data['ACV_Selling']=acv_med
  training_data_optimal =Final_Pred_Data.copy()
  Final_Pred_Data['Baseline_Prediction']=predict_sales(Model_Coeff,Final_Pred_Data)

  Financial_information=Final_Pred_Data[['Promo','List_Price','COGS','median_baseprice','Promotion_Cost','TE','tpr_discount_byppg']].drop_duplicates().reset_index(drop=True)
  TE_dict= dict(zip(Financial_information.tpr_discount_byppg, Financial_information.TE))

  prd=0
  Final_Pred_Data.to_csv(path+"Training_data_LPBase"+str(prd)+".csv",index=False)
  #export Promos_mapping_0#Promotion_Cost
  Financial_information['RRP']=Financial_information['median_baseprice']
  os.makedirs(path+"/Output", exist_ok=True)
  Financial_information.to_csv(path+"Output/Promos_mapping_"+str(prd)+".csv")

  ### Baseline Vars Calculation
  Final_Pred_Data['Baseline_Prediction']=predict_sales(Model_Coeff,Final_Pred_Data)
  Final_Pred_Data['Baseline_Sales']=Final_Pred_Data['Baseline_Prediction'] *Final_Pred_Data['Promo']
  Final_Pred_Data["Baseline_GSV"] = Final_Pred_Data['Baseline_Prediction'] * Final_Pred_Data['List_Price']
  Final_Pred_Data["Baseline_Trade_Expense"] = Final_Pred_Data['Baseline_Prediction'] * Final_Pred_Data['TE']
  Final_Pred_Data["Baseline_NSV"] = Final_Pred_Data['Baseline_GSV'] - Final_Pred_Data["Baseline_Trade_Expense"]
  Final_Pred_Data["Baseline_MAC"] = Final_Pred_Data["Baseline_NSV"]-Final_Pred_Data['Baseline_Prediction'] * Final_Pred_Data['COGS']
  Final_Pred_Data["Baseline_RP"] = Final_Pred_Data['Baseline_Sales']-Final_Pred_Data["Baseline_NSV"]

  prd=0
  Final_Pred_Data.to_csv(path+"Training_data_"+str(prd)+".csv",index=False)
  Final_Pred_Data[['Baseline_Prediction','Baseline_Sales',"Baseline_GSV","Baseline_Trade_Expense","Baseline_NSV","Baseline_MAC","Baseline_RP"]].sum().apply(lambda x: '%.3f' % x)


  baseline_df =Final_Pred_Data[['Baseline_Prediction','Baseline_Sales',"Baseline_GSV","Baseline_Trade_Expense","Baseline_NSV","Baseline_MAC","Baseline_RP"]].sum().astype(int)
  baseline_df['Baseline_MAC']

  config = {"Reatiler":"Magnit","PPG":'A.Korkunov_192g',"MARS_TPRS":[],"Co_investment":0,
         "Objective_metric":"MAC","Objective":"Maximize",
        "config_constrain":{'MAC':True,'RP':True,'Trade_Expense':False,'Units':False,"NSV":False,"GSV":False,"Sales":False
                            ,'MAC_Perc':True,"RP_Perc":False,'min_consecutive_promo':True,'max_consecutive_promo':True,
                   'promo_gap':True},
         "constrain_params": {'MAC':1,'RP':1,'Trade_Expense':1,'Units':1,'NSV':1,'GSV':1,'Sales':1,'MAC_Perc':1,'RP_Perc':1,
                              'min_consecutive_promo':6,'max_consecutive_promo':6,
                   'promo_gap':2,'tot_promo_min':10,'tot_promo_max':26,'compul_no_promo_weeks':[],'compul_promo_weeks' :[]}}

  baseline_data = Final_Pred_Data.copy()

  config = {"Reatiler":"Magnit","PPG":'A.Korkunov_192g','Segment':"Choco","MARS_TPRS":[],"Co_investment":0,
          "Objective_metric":"MAC","Objective":"Maximize",
          "config_constrain":{'MAC':True,'RP':True,'Trade_Expense':True,'Units':False,"NSV":False,"GSV":False,"Sales":False
                              ,'MAC_Perc':True,"RP_Perc":True,'min_consecutive_promo':True,'max_consecutive_promo':True,
                    'promo_gap':True},
          "constrain_params": {'MAC':1,'RP':1,'Trade_Expense':1,'Units':1,'NSV':1,'GSV':1,'Sales':1,'MAC_Perc':1,'RP_Perc':1,
                                'min_consecutive_promo':6,'max_consecutive_promo':6,
                    'promo_gap':2,'tot_promo_min':10,'tot_promo_max':26,'compul_no_promo_weeks':[],'compul_promo_weeks' :[]}}
  
  TE_dict = get_te_dict(baseline_data,config,Financial_information)
  Required_base = get_required_base(baseline_data,Model_Coeff,TE_dict,config)
  Required_base.tail()
  Optimal_calendar = optimizer_fun(baseline_data,Required_base,config)
  if((Optimal_calendar.shape[0]!=52) or (Optimal_calendar['Solution'].unique()!='Optimal')):
    print("Infeasible solution - relaxing financial metrics")
    Sec_fin_metrics = ['Units','NSV','GSV','Sales','Trade_Expense',"RP_Perc"]
    Prim_fin_metrics = ['MAC_Perc','RP','MAC']
    calendar_metrics = ['compul_no_promo_weeks','compul_promo_weeks','promo_gap','max_consecutive_promo','min_consecutive_promo','tot_promo_min',
                      'tot_promo_max']

    config_constrain = config['config_constrain']
    Act_Prim_fin_metrics = []
    for i in Prim_fin_metrics:
      if config_constrain[i]:
        Act_Prim_fin_metrics.append(i)
    Act_Sec_fin_metrics =[]
    for i in Sec_fin_metrics:
      if config_constrain[i]:
        Act_Sec_fin_metrics.append(i)
    Sec_combination = sum([list(map(list, combinations(Act_Sec_fin_metrics, i))) for i in range(len(Act_Sec_fin_metrics) + 1)], [])
    Sec_combination = [i for i in Sec_combination if len(i)>0]
    Prim_combination = sum([list(map(list, combinations(Act_Prim_fin_metrics, i))) for i in range(len(Act_Prim_fin_metrics) + 1)], [])
    Prim_combination = [i for i in Prim_combination if len(i)>0]
    iteration =True
    while iteration:
      relaxed_sec_metrics_opt=[]
      relaxed_sec_metrics = []
      relaxed_prim_metrics_opt=[]
      relaxed_prim_metrics = []
      for i in range(len(Sec_combination)):
        print(Sec_combination[i])
        metrics = Sec_combination[i]
        for metric in metrics:
          config['config_constrain'][metric]=False
        print(config['config_constrain'])
        Optimal_calendar = optimizer_fun(baseline_data,Required_base,config)
        if ((Optimal_calendar.shape[0]==52) and (Optimal_calendar['Solution'].unique()=='Optimal')):
          relaxed_sec_metrics_opt = Sec_combination[i]
          for metric in metrics:
            config['config_constrain'][metric]=True
          print("breaking while loop")
          iteration = False
          break
        if i==(len(Sec_combination)-1):
          print(i)
          relaxed_sec_metrics = Sec_combination[i]
          print("Relaxing :",relaxed_sec_metrics)
        else:
          for metric in metrics:
            config['config_constrain'][metric]=True
      for i in range(len(Prim_combination)):
        print(Prim_combination[i])
        metrics = Prim_combination[i]
        for metric in metrics:
          config['config_constrain'][metric]=False
        print(config['config_constrain'])
        Optimal_calendar = optimizer_fun(baseline_data,Required_base,config)
        if ((Optimal_calendar.shape[0]==52) and (Optimal_calendar['Solution'].unique()=='Optimal')):
          relaxed_prim_metrics_opt = Prim_combination[i]
          for metric in metrics:
            config['config_constrain'][metric]=True
          iteration = False
          break
        if i==(len(Sec_combination)-1):
          print(i)
          relaxed_prim_metrics = Prim_combination[i]
          for metric in metrics:
            config['config_constrain'][metric]=True
          iteration = False
          break
        else:
          for metric in metrics:
            config['config_constrain'][metric]=True
    relaxed_metrics = []
    if len(relaxed_sec_metrics_opt)>0:
      relaxed_metrics = relaxed_sec_metrics_opt
    elif len(relaxed_prim_metrics_opt)>0:
      relaxed_metrics = relaxed_prim_metrics_opt
    relaxed_metrics
    delta = 0.01
    iter_no =0
    opt_soln = False
    if len(relaxed_metrics)>0:
      iteration = True
      while iteration:
        iter_no+=1
        for rel in relaxed_metrics:
          if(rel=='Trade_Expense'):
            config['constrain_params'][rel]=config['constrain_params'][rel]+delta
          else:
            config['constrain_params'][rel]=config['constrain_params'][rel]-delta
        Optimal_calendar_fin = optimizer_fun(baseline_data,Required_base,config)
        if ((Optimal_calendar_fin.shape[0]==52) and (Optimal_calendar_fin['Solution'].unique()=='Optimal')):
          print(config['constrain_params'])
          iteration = False
          opt_soln = True
        if(iter_no==10):
          iteration = False
      if opt_soln:
        dta=delta/4
        k=0
        while k<3:
          for rel in relaxed_metrics:
            if(rel=='Trade_Expense'):
              config['constrain_params'][rel]=config['constrain_params'][rel]-dta
            else:
              config['constrain_params'][rel]=config['constrain_params'][rel]+dta
          Optimal_calendar = optimizer_fun(baseline_data,Required_base,config)
          if ((Optimal_calendar.shape[0]==52) and (Optimal_calendar['Solution'].unique()=='Optimal')):
            print(config['constrain_params'])
            Optimal_calendar_fin = Optimal_calendar.copy()
          k+=1
  Optimal_data = optimal_summary_fun(baseline_data,Model_Coeff,Optimal_calendar_fin,TE_dict,config)
  opt_base = get_opt_base_comparison(baseline_data,Optimal_data,Model_Coeff,config)
  summary = get_calendar_summary(baseline_data,Optimal_data,opt_base)
  parsed = json.loads(summary.to_json(orient="records"))
  return parsed



def get_te_dict(baseline_data,config,Financial_information):
  Fin_info= baseline_data[['Promo','List_Price','COGS','median_baseprice','Promotion_Cost','TE','tpr_discount_byppg']].drop_duplicates().reset_index(drop=True)
  TE_dict= dict(zip(Financial_information.tpr_discount_byppg, Financial_information.TE))
  tprs = config["MARS_TPRS"]
  if len(tprs)>0:
    for tpr in tprs:
      List_Price = baseline_data["List_Price"].unique()[0] 
      TE_OFF=baseline_data["TE Off Inv"].unique()[0]  
      TE_ON= baseline_data["TE On Inv"].unique()[0]  
      COGS = baseline_data["COGS"].unique()[0]
      tpr = tpr/100
      Promotion_Cost = List_Price * tpr * (1 -TE_ON)
      TE = List_Price * (tpr + TE_ON + TE_OFF - tpr * TE_OFF - TE_OFF * TE_ON -TE_ON * tpr + TE_OFF * TE_ON * tpr)
      tpr_1=tpr*100
      TE_dict[tpr_1]=TE
      print(TE,Promotion_Cost)
  return TE_dict


def get_required_base(baseline_data,Model_Coeff,TE_dict,config):
  model_cols = Model_Coeff['names'].to_list()
  model_cols.remove('Intercept')
  Base=baseline_data[['wk_base_price_perunit_byppg','Promo', 'TE', 'List_Price','COGS']+model_cols]
  i=0
  TPR_list=list(TE_dict.keys())
  TPR_list.sort()
  TPR_list
  ret_inv = config["Co_investment"]
  # TPR_list=[j for j in TPR_list if j>0]
  for tpr in TPR_list:
    Required_base=Base.copy()
    Required_base['tpr_discount_byppg']=tpr
    if 'tpr_discount_byppg_lag1' in model_cols:
      Required_base['tpr_discount_byppg_lag1'] =0
    Required_base['Promo']	=Required_base['wk_base_price_perunit_byppg']-(Required_base['wk_base_price_perunit_byppg']*Required_base['tpr_discount_byppg']/100)
    Required_base.loc[Required_base['tpr_discount_byppg'].isin(TE_dict.keys()), 'TE'] = Required_base['tpr_discount_byppg'].map(TE_dict)
    if 'Catalogue_Dist' in  model_cols:
      Catalogue_temp = baseline_data['Catalogue_Dist'].max()
      Required_base['Catalogue_Dist']=np.where(Required_base['tpr_discount_byppg']==0,0,Catalogue_temp)
  #     Required_base['Catalogue_Dist']=np.where(Required_base['tpr_discount_byppg']==0,0,Catalogue_temp)
    if (tpr!=0 and ret_inv!=0):
      print(tpr)
      Required_temp=Required_base.copy()
      Required_temp['tpr_discount_byppg']=tpr+ret_inv
      print(Required_temp['tpr_discount_byppg'].unique())
      print("entering loop")
      Required_base['Units']=predict_sales(Model_Coeff,Required_temp)
      Required_base['Promo Price']=Required_temp['wk_base_price_perunit_byppg']-(Required_temp['wk_base_price_perunit_byppg']*(Required_temp['tpr_discount_byppg']/100))
    else:
      Required_base['Units']=predict_sales(Model_Coeff,Required_base)
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
  #   Required_base.rename(columns = dict(zip(cols, 'TPR_'+str(i)+cols )), inplace=True)
    i=i+1
  Required_base=new_base
  Required_base=Required_base.reset_index(drop=True)
  Required_base['WK_ID']=Required_base.index
  Required_base['WK_ID'] = 'WK_' + Required_base['WK_ID'].astype(str)+'_'+Required_base['Iteration'].astype(str)
  return(Required_base)


def optimizer_fun(baseline_data,Required_base,config):
  baseline_df =baseline_data[['Baseline_Prediction','Baseline_Sales',"Baseline_GSV","Baseline_Trade_Expense","Baseline_NSV","Baseline_MAC","Baseline_RP"]].sum().astype(int)
  config_constrain = config['config_constrain']
  constrain_params = config['constrain_params']
  promo_loop=Required_base['Iteration'].nunique()
  WK_DV_vars = list(Required_base['WK_ID'])
  WK_DV_tprvars = list(Required_base['Iteration'])
  if config['Objective']=='Maximize':
    print("Maximize")
    prob = LpProblem("Simple_Workaround_problem",LpMaximize)
  else:
    prob = LpProblem("Simple_Workaround_problem",LpMinimize)
  WK_vars = LpVariable.dicts("RP",WK_DV_vars,cat='Binary')
  obj_metric = config['Objective_metric']
  prob+=lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base[obj_metric][i]  for i in range(0,Required_base.shape[0])])
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
    
  if(config_constrain['NSV']):
    prob+=lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base['NSV'][i]  for i in range(0,Required_base.shape[0])])>=int(baseline_df['Baseline_NSV']*constrain_params['NSV'])
    
  if(config_constrain['GSV']):
    prob+=lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base['GSV'][i]  for i in range(0,Required_base.shape[0])])>=int(baseline_df['Baseline_GSV']*constrain_params['GSV'])
    
  if(config_constrain['Sales']):
    prob+=lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base['Sales'][i]  for i in range(0,Required_base.shape[0])])>=int(baseline_df['Baseline_Sales']*constrain_params['Sales'])
    
  if(config_constrain['MAC_Perc']):
    L1 =lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base['MAC'][i]  for i in range(0,Required_base.shape[0])])
    L2=lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base['NSV'][i]  for i in range(0,Required_base.shape[0])])
    prob+= L1 >= L2*(baseline_df['Baseline_MAC']/baseline_df['Baseline_NSV'])*constrain_params['MAC_Perc']
  
  if(config_constrain['RP_Perc']):
    L1 =lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base['RP'][i]  for i in range(0,Required_base.shape[0])])
    L2=lpSum([WK_vars[Required_base['WK_ID'][i]]*Required_base['Sales'][i]  for i in range(0,Required_base.shape[0])])
    prob+= L1 >= L2*(baseline_df['Baseline_RP']/baseline_df['Baseline_Sales'])*constrain_params['RP_Perc']

  # Set up constraints such that only one tpr is chose for a week
  for i in range(0,52):
    prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]  for j in range(0,promo_loop)])<=1
    prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for j in range(0,promo_loop)])>=1  
  if len(constrain_params['compul_no_promo_weeks'])>0:
    prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]  for j in range(1,promo_loop) for i in constrain_params['compul_no_promo_weeks']])<=0
  if len(constrain_params['compul_promo_weeks'])>0:
    prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]  for j in range(1,promo_loop) for i in constrain_params['compul_promo_weeks']])>=len(constrain_params['compul_promo_weeks'])
  # Costraint for No of promotions 
  prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]]  for j in range(1,promo_loop) for i in range(0,52)])<=constrain_params['tot_promo_max']
  prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for j in range(1,promo_loop) for i in range(0,52)])>=constrain_params['tot_promo_min']

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

  # prob+= 11*R11-R10-R9-R8-R7-R6-R5-R4-R3-R2-R1-R0 >=0
  # prob+= 10*R10-R9-R8-R7-R6-R5-R4-R3-R2-R1-R0 >=0
  # prob+= 9*R9-R8-R7-R6-R5-R4-R3-R2-R1-R0 >=0
  # prob+= 8*R8-R7-R6-R5-R4-R3-R2-R1-R0 >=0
  # prob+= 7*R7-R6-R5-R4-R3-R2-R1-R0 >=0
  # prob+= 6*R6-R5-R4-R3-R2-R1-R0 >=0
  # prob+= 5*R5-R4-R3-R2-R1-R0 >=0
  # prob+= 4*R4-R3-R2-R1-R0 >=0
  # prob+= 3*R3-R2-R1-R0 >=0
  # prob+= 2*R2-R1-R0 >=0
  # prob+=R1-R0 >=0

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

  # prob+= 11*R41-R42-R43-R44-R45-R46-R47-R48-R49-R50-R51-R52 >=0
  # prob+= 10*R42-R43-R44-R45-R46-R47-R48-R49-R50-R51-R52 >=0
  # prob+= 9*R43-R44-R45-R46-R47-R48-R49-R50-R51-R52 >=0
  # prob+= 8*R44-R45-R46-R47-R48-R49-R50-R51-R52 >=0
  # prob+= 7*R45-R46-R47-R48-R49-R50-R51-R52 >=0
  # prob+= 6*R46-R47-R48-R49-R50-R51-R52 >=0
  # prob+= 5*R47-R48-R49-R50-R51-R52 >=0
  # prob+= 4*R48-R49-R50-R51-R52 >=0
  # prob+= 3*R49-R50-R51-R52 >=0
  # prob+=2*R50-R51-R52 >=0
  # prob+=R51-R52 >=0
  if(config_constrain['min_consecutive_promo']):
    if constrain_params['min_consecutive_promo']==2:
      prob+=R51-R52 >=0
      prob+=R1-R0 >=0
    if constrain_params['min_consecutive_promo']==3:
      prob+=R51-R52 >=0
      prob+=R1-R0 >=0
      prob+=2*R50-R51-R52 >=0
      prob+= 2*R2-R1-R0 >=0
    if constrain_params['min_consecutive_promo']==4:
      prob+=R51-R52 >=0
      prob+=R1-R0 >=0
      prob+=2*R50-R51-R52 >=0
      prob+= 2*R2-R1-R0 >=0
      prob+= 3*R49-R50-R51-R52 >=0
      prob+= 3*R3-R2-R1-R0 >=0
    if constrain_params['min_consecutive_promo']==5:
      prob+=R51-R52 >=0
      prob+=R1-R0 >=0
      prob+=2*R50-R51-R52 >=0
      prob+= 2*R2-R1-R0 >=0
      prob+= 3*R49-R50-R51-R52 >=0
      prob+= 3*R3-R2-R1-R0 >=0
      prob+= 4*R48-R49-R50-R51-R52 >=0
      prob+= 4*R4-R3-R2-R1-R0 >=0
    if constrain_params['min_consecutive_promo']==6:
      prob+=R51-R52 >=0
      prob+=R1-R0 >=0
      prob+=2*R50-R51-R52 >=0
      prob+= 2*R2-R1-R0 >=0
      prob+= 3*R49-R50-R51-R52 >=0
      prob+= 3*R3-R2-R1-R0 >=0
      prob+= 4*R48-R49-R50-R51-R52 >=0
      prob+= 4*R4-R3-R2-R1-R0 >=0
      prob+= 5*R47-R48-R49-R50-R51-R52 >=0
      prob+= 5*R5-R4-R3-R2-R1-R0 >=0
    if constrain_params['min_consecutive_promo']==7:
      prob+=R51-R52 >=0
      prob+=R1-R0 >=0
      prob+=2*R50-R51-R52 >=0
      prob+= 2*R2-R1-R0 >=0
      prob+= 3*R49-R50-R51-R52 >=0
      prob+= 3*R3-R2-R1-R0 >=0
      prob+= 4*R48-R49-R50-R51-R52 >=0
      prob+= 4*R4-R3-R2-R1-R0 >=0
      prob+= 5*R47-R48-R49-R50-R51-R52 >=0
      prob+= 5*R5-R4-R3-R2-R1-R0 >=0
      prob+= 6*R46-R47-R48-R49-R50-R51-R52 >=0
      prob+= 6*R6-R5-R4-R3-R2-R1-R0 >=0
    if constrain_params['min_consecutive_promo']==8:
      prob+=R51-R52 >=0
      prob+=R1-R0 >=0
      prob+=2*R50-R51-R52 >=0
      prob+= 2*R2-R1-R0 >=0
      prob+= 3*R49-R50-R51-R52 >=0
      prob+= 3*R3-R2-R1-R0 >=0
      prob+= 4*R48-R49-R50-R51-R52 >=0
      prob+= 4*R4-R3-R2-R1-R0 >=0
      prob+= 5*R47-R48-R49-R50-R51-R52 >=0
      prob+= 5*R5-R4-R3-R2-R1-R0 >=0
      prob+= 6*R46-R47-R48-R49-R50-R51-R52 >=0
      prob+= 6*R6-R5-R4-R3-R2-R1-R0 >=0
      prob+= 7*R45-R46-R47-R48-R49-R50-R51-R52 >=0
      prob+= 7*R7-R6-R5-R4-R3-R2-R1-R0 >=0
    if constrain_params['min_consecutive_promo']==9:
      prob+=R51-R52 >=0
      prob+=R1-R0 >=0
      prob+=2*R50-R51-R52 >=0
      prob+= 2*R2-R1-R0 >=0
      prob+= 3*R49-R50-R51-R52 >=0
      prob+= 3*R3-R2-R1-R0 >=0
      prob+= 4*R48-R49-R50-R51-R52 >=0
      prob+= 4*R4-R3-R2-R1-R0 >=0
      prob+= 5*R47-R48-R49-R50-R51-R52 >=0
      prob+= 5*R5-R4-R3-R2-R1-R0 >=0
      prob+= 6*R46-R47-R48-R49-R50-R51-R52 >=0
      prob+= 6*R6-R5-R4-R3-R2-R1-R0 >=0
      prob+= 7*R45-R46-R47-R48-R49-R50-R51-R52 >=0
      prob+= 7*R7-R6-R5-R4-R3-R2-R1-R0 >=0
      prob+= 8*R44-R45-R46-R47-R48-R49-R50-R51-R52 >=0
      prob+= 8*R8-R7-R6-R5-R4-R3-R2-R1-R0 >=0
  print(config_constrain['max_consecutive_promo'],"-",constrain_params['max_consecutive_promo'])
  # contraint for max promo length
  if(config_constrain['max_consecutive_promo']):
    for k in range(0,52-constrain_params['max_consecutive_promo']):
      for j in range(1,promo_loop):
        prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for i in range(k,k+constrain_params['max_consecutive_promo']+1)])<=constrain_params['max_consecutive_promo']
    for k in range(0,52-constrain_params['max_consecutive_promo']):  
        prob+=lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for i in range(k,k+constrain_params['max_consecutive_promo']+1) for j in range(1,promo_loop)])<=constrain_params['max_consecutive_promo']

  # Constrain for min promo wave length
  if(config_constrain['min_consecutive_promo']):
    for k in range(0,50):
      R1_sum = lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for i in range(k+1,min(k+constrain_params['min_consecutive_promo']+1,52)) for j in range(1,promo_loop)])
      R2_sum = lpSum([WK_vars[Required_base['WK_ID'][k+j*52]] for j in range(1,promo_loop)])
      R3_sum = lpSum([WK_vars[Required_base['WK_ID'][k+1+j*52]] for j in range(1,promo_loop)])
      gap_weeks = len(range(k+1, min(52, k+constrain_params['min_consecutive_promo']+1)))
      prob+= R1_sum + gap_weeks * R2_sum >= gap_weeks * R3_sum

    for k in range(0,50):
      for j in range(1,promo_loop):
        R1_sum = lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for i in range(k+1,min(k+constrain_params['min_consecutive_promo']+1,52))])
        R2_sum = lpSum([WK_vars[Required_base['WK_ID'][k+j*52]] ])
        R3_sum = lpSum([WK_vars[Required_base['WK_ID'][k+1+j*52]]])
        gap_weeks = len(range(k+1, min(52, k+constrain_params['min_consecutive_promo']+1)))
        prob+= R1_sum + gap_weeks * R2_sum >= gap_weeks * R3_sum
  # 4 week gap
  if(config_constrain['promo_gap']):
    for k in range(0,51):
      R1_sum = lpSum([WK_vars[Required_base['WK_ID'][i+j*52]] for j in range(0,1) for i in range(k+1,min(k+constrain_params['promo_gap']+1,52))])
      R2_sum= lpSum([WK_vars[Required_base['WK_ID'][k+j*52]] for j in range(0,1)])
      R3_sum = lpSum([WK_vars[Required_base['WK_ID'][k+1+j*52]] for j in range(0,1)])
      gap_weeks = len(range(k+1, min(52, k+constrain_params['promo_gap']+1)))
      prob+= R1_sum + gap_weeks * R2_sum >= gap_weeks * R3_sum
  prob.solve()
  print(LpStatus[prob.status])
  print(pulp.value(prob.objective))
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
  tprs = Required_base[['Iteration','TPR']].drop_duplicates().reset_index(drop=True)
  tprs
  df = pd.merge(df,tprs,on=['Iteration'],how='left')
  df = df.sort_values('Week_no',ascending = True).reset_index(drop=True)
  df['Solution']=LpStatus[prob.status]
  return(df)
  # prob.solve(PULP_CBC_CMD(msg=True, maxSeconds=1200000, threads=90, keepFiles=1, fracGap=None))



def optimal_summary_fun(baseline_data,Model_Coeff,optimal_calendar,TE_dict,config):
  model_cols = Model_Coeff['names'].to_list()
  model_cols.remove('Intercept')
  Base=baseline_data[["Date",'wk_base_price_perunit_byppg','Promo', 'TE', 'List_Price','COGS','TE On Inv', 'GMAC', 'TE Off Inv']+model_cols]
  new_data=Base.copy()
  ret_inv = config['Co_investment']
  new_data['tpr_discount_byppg']=optimal_calendar['TPR']
  if 'tpr_discount_byppg_lag1' in model_cols:
    new_data['tpr_discount_byppg_lag1'] = new_data['tpr_discount_byppg'].shift(1).fillna(0)
  if 'Catalogue_Dist' in model_cols:
    Catalogue_temp = baseline_data['Catalogue_Dist'].max()
    new_data['Catalogue_Dist']=np.where(new_data['tpr_discount_byppg']==0,0,Catalogue_temp)
  if (ret_inv!=0):
    new_data_temp=new_data.copy()
    new_data_temp['tpr_discount_byppg']=np.where(new_data_temp['tpr_discount_byppg']==0,0,new_data_temp['tpr_discount_byppg']+ret_inv)
    print(new_data_temp['tpr_discount_byppg'].unique())
    new_data['Units']=predict_sales(Model_Coeff,new_data_temp)
    new_data['Promo Price']=new_data_temp['wk_base_price_perunit_byppg']-(new_data_temp['wk_base_price_perunit_byppg']*(new_data_temp['tpr_discount_byppg']/100))
  else:
    new_data['Units']=predict_sales(Model_Coeff,new_data)
    new_data['Promo Price']=new_data['wk_base_price_perunit_byppg']-(new_data['wk_base_price_perunit_byppg']*(new_data['tpr_discount_byppg']/100))
  #   new_data['Units']=predict_sales(Model_Coeff,new_data)     
  new_data.loc[new_data['tpr_discount_byppg'].isin(TE_dict.keys()), 'TE'] = new_data['tpr_discount_byppg'].map(TE_dict)
  #   new_data['Promo Price']=new_data['wk_base_price_perunit_byppg']-(new_data['wk_base_price_perunit_byppg']*(new_data['tpr_discount_byppg']/100)) 
  new_data['wk_sold_avg_price_byppg']=new_data['Promo Price']
  new_data['Sales']=new_data['Units'] *new_data['Promo Price']
  new_data["GSV"] = new_data['Units'] * new_data['List_Price']
  new_data["Trade_Expense"] = new_data['Units'] * new_data['TE']
  new_data["NSV"] = new_data['GSV'] - new_data["Trade_Expense"]
  new_data["MAC"] = new_data["NSV"]-new_data['Units'] * new_data['COGS']
  new_data["RP"] = new_data['Sales']-new_data["NSV"]
  print(new_data[['Sales',"GSV","Trade_Expense","NSV","MAC","RP","Units"]].sum().apply(lambda x: '%.3f' % x))
  return(new_data)


def get_opt_base_comparison(baseline_data,optimal_data,Model_Coeff,config):
  Segment = config['Segment']
  train_data=baseline_data.copy()
  model_coef = Model_Coeff.copy()
  ret_inv = config['Co_investment']
  train_data['tpr_discount_byppg']=np.where(train_data['tpr_discount_byppg']==0,0,train_data['tpr_discount_byppg']+ret_inv)
  train_data['Promo_wave']=promo_wave_cal(train_data)
  train_data['wk_sold_avg_price_byppg']=np.exp(train_data['wk_sold_median_base_price_byppg_log'])*(1-train_data['tpr_discount_byppg']/100)
  model_df=train_data.copy()
  #Base Variable Method
  model_coef.rename(columns={"names":"Variable","model_coefficients":"Value"},inplace=True)
  baseline_var = pd.DataFrame(columns =["Variable"])
  baseline_var_othr = pd.DataFrame(columns =["Variable"])
  col = model_coef["Variable"].to_list()

  # baseline variables
  baseprice_cols = ['wk_sold_median_base_price_byppg_log']+[i for i in col if "RegularPrice" in i]
  holiday =[i for i in col if "day" and "flag" in i ]
  SI_cols = [i for i in col if "SI" in i ]
  trend_cols = [i for i in col if "trend" in i ]
  base_list = baseprice_cols+SI_cols+trend_cols+[i for i in col if "ACV_Selling" in i ]
  base_others = holiday+[i for i in col if "death_rate" in i]+[i for i in col if 'tpr_discount_byppg_lag1' in i]
  # base_list = ['wk_sold_median_base_price_byppg_log']+[i for i in col if "RegularPrice" in i]+[i for i in col if "day" and "flag" in i ]+["SI","ACV_Selling"]
  # base_list.remove( 'flag_date_1')
  # base_list.remove( 'flag_date_2')
  print(base_list)
  baseline_var["Variable"]=base_list
  baseline_var_othr["Variable"]=base_others

  model_df['Iteration']=1
  model_df1=model_df.copy()
  model_coef['Iteration']=1

  base_scenario=base_var_cont(model_df,model_df1,baseline_var,baseline_var_othr,model_coef)
  base_scenario=base_scenario.merge(train_data[['Date','tpr_discount_byppg','List_Price','GMAC',
                                               'TE Off Inv', 'TE On Inv']],how='left',on='Date')
  base_scenario['tpr_discount_byppg']=np.where(base_scenario['tpr_discount_byppg']==0,0,base_scenario['tpr_discount_byppg']-ret_inv)
  base_scenario['TPR']=base_scenario['tpr_discount_byppg']/100
  base_scenario['Uplift LSV'] = base_scenario['Incremental']*base_scenario['List_Price']
  base_scenario["Uplift GMAC, LSV"] = base_scenario["Uplift LSV"]*base_scenario["GMAC"]
  base_scenario['Total LSV'] = (base_scenario['Incremental']+base_scenario['Base'])*base_scenario['List_Price']
  base_scenario['Mars Uplift On-Invoice'] = base_scenario['Uplift LSV']*base_scenario['TE On Inv']
  base_scenario['Mars Total On-Invoice'] = base_scenario['Total LSV']*base_scenario['TE On Inv']
  base_scenario['Mars Uplift NRV'] = base_scenario['Uplift LSV']-base_scenario['Mars Uplift On-Invoice']
  base_scenario['Mars Total NRV'] = base_scenario['Total LSV']-base_scenario['Mars Total On-Invoice']
  base_scenario['Uplift Promo Cost'] = base_scenario['Mars Uplift NRV']*base_scenario['TPR']
  base_scenario['TPR Budget ROI'] = base_scenario['Mars Total NRV']*base_scenario['TPR']
  base_scenario['Mars Uplift Net Invoice Price'] = base_scenario['Mars Uplift NRV']-base_scenario['Uplift Promo Cost']
  base_scenario['Mars Total Net Invoice Price'] = base_scenario['Mars Total NRV']-base_scenario['TPR Budget ROI']
  base_scenario['Mars Uplift Off-Invoice'] = base_scenario['Mars Uplift Net Invoice Price']*base_scenario['TE Off Inv']
  base_scenario['Mars Total Off-Invoice'] = base_scenario['Mars Total Net Invoice Price']*base_scenario['TE Off Inv']
  base_scenario['Uplift  Trade Expense'] = base_scenario['Mars Uplift Off-Invoice']+base_scenario['TPR Budget ROI']+base_scenario['Mars Uplift On-Invoice']
  base_scenario['Total  Trade Expense'] = base_scenario['Mars Total On-Invoice']+base_scenario['TPR Budget ROI']+base_scenario['Mars Total Off-Invoice']
  base_scenario['Uplift NSV'] = base_scenario['Uplift LSV']-base_scenario['Uplift  Trade Expense']
  base_scenario['Total NSV'] = base_scenario['Total LSV']-base_scenario['Total  Trade Expense']
  base_scenario['Segment']=Segment
  base_scenario['Uplift Royalty'] =np.where(base_scenario['Segment']=='Choco',0,0.5*base_scenario['Uplift NSV'])
  base_scenario['Total Uplift Cost'] = base_scenario['Uplift Royalty'] + base_scenario['Uplift  Trade Expense']
  base_scenario['ROI'] = base_scenario["Uplift GMAC, LSV"]/base_scenario['Total Uplift Cost']
  base_scenario['ROI'] = base_scenario['ROI'].fillna(0)
  lift_wave_base = base_scenario.groupby(['Promo_wave']).agg({'Uplift GMAC, LSV': [("Uplift GMAC, LSV",sum)],
                                                                          'Total Uplift Cost' : [('Total Uplift Cost',sum)],
                                                                          'Incremental' : [('Incremental',sum)],
                                                                          'Base': [('Base',sum)]})
  lift_wave_base.columns = ['Uplift_GMAC_LSV', 'Uplift_cost', 'incremental', 'base']
  lift_wave_base = lift_wave_base.reset_index()
  lift_wave_base['Lift %'] = lift_wave_base['incremental']/lift_wave_base['base']
  lift_wave_base['ROI'] = lift_wave_base['Uplift_GMAC_LSV']/lift_wave_base['Uplift_cost']
  lift_wave_base
  train_data=optimal_data.copy()
  train_data['tpr_discount_byppg']=np.where(train_data['tpr_discount_byppg']==0,0,train_data['tpr_discount_byppg']+ret_inv)
  train_data['Promo_wave']=promo_wave_cal(train_data) 
  train_data['wk_sold_avg_price_byppg']=np.exp(train_data['wk_sold_median_base_price_byppg_log'])*(1-train_data['tpr_discount_byppg']/100)
  model_df=train_data.copy()
  model_df['Iteration']=1
  model_df1=model_df.copy()
  model_coef['Iteration']=1

  optimum_scenario=base_var_cont(model_df,model_df1,baseline_var,baseline_var_othr,model_coef)
  optimum_scenario=optimum_scenario.merge(train_data[['Date','tpr_discount_byppg','SI','List_Price','GMAC',
                                               'TE Off Inv', 'TE On Inv']],how='left',on='Date')
  optimum_scenario['tpr_discount_byppg']=np.where(optimum_scenario['tpr_discount_byppg']==0,0,optimum_scenario['tpr_discount_byppg']-ret_inv)
  optimum_scenario['TPR']=optimum_scenario['tpr_discount_byppg']/100
  optimum_scenario['Uplift LSV'] = optimum_scenario['Incremental']*optimum_scenario['List_Price']
  optimum_scenario["Uplift GMAC, LSV"] = optimum_scenario["Uplift LSV"]*optimum_scenario["GMAC"]
  optimum_scenario['Total LSV'] = (optimum_scenario['Incremental']+optimum_scenario['Base'])*optimum_scenario['List_Price']
  optimum_scenario['Mars Uplift On-Invoice'] = optimum_scenario['Uplift LSV']*optimum_scenario['TE On Inv']
  optimum_scenario['Mars Total On-Invoice'] = optimum_scenario['Total LSV']*optimum_scenario['TE On Inv']
  optimum_scenario['Mars Uplift NRV'] = optimum_scenario['Uplift LSV']-optimum_scenario['Mars Uplift On-Invoice']
  optimum_scenario['Mars Total NRV'] = optimum_scenario['Total LSV']-optimum_scenario['Mars Total On-Invoice']
  optimum_scenario['Uplift Promo Cost'] = optimum_scenario['Mars Uplift NRV']*optimum_scenario['TPR']
  optimum_scenario['TPR Budget ROI'] = optimum_scenario['Mars Total NRV']*optimum_scenario['TPR']
  optimum_scenario['Mars Uplift Net Invoice Price'] = optimum_scenario['Mars Uplift NRV']-optimum_scenario['Uplift Promo Cost']
  optimum_scenario['Mars Total Net Invoice Price'] = optimum_scenario['Mars Total NRV']-optimum_scenario['TPR Budget ROI']
  optimum_scenario['Mars Uplift Off-Invoice'] = optimum_scenario['Mars Uplift Net Invoice Price']*optimum_scenario['TE Off Inv']
  optimum_scenario['Mars Total Off-Invoice'] = optimum_scenario['Mars Total Net Invoice Price']*optimum_scenario['TE Off Inv']
  optimum_scenario['Uplift  Trade Expense'] = optimum_scenario['Mars Uplift Off-Invoice']+optimum_scenario['TPR Budget ROI']+optimum_scenario['Mars Uplift On-Invoice']
  optimum_scenario['Total  Trade Expense'] = optimum_scenario['Mars Total On-Invoice']+optimum_scenario['TPR Budget ROI']+optimum_scenario['Mars Total Off-Invoice']
  optimum_scenario['Uplift NSV'] = optimum_scenario['Uplift LSV']-optimum_scenario['Uplift  Trade Expense']
  optimum_scenario['Total NSV'] = optimum_scenario['Total LSV']-optimum_scenario['Total  Trade Expense']
  optimum_scenario['Segment']=Segment
  optimum_scenario['Uplift Royalty'] =np.where(optimum_scenario['Segment']=='Choco',0,0.5*optimum_scenario['Uplift NSV'])
  optimum_scenario['Total Uplift Cost'] = optimum_scenario['Uplift Royalty'] + optimum_scenario['Uplift  Trade Expense']
  optimum_scenario['ROI'] = optimum_scenario["Uplift GMAC, LSV"]/optimum_scenario['Total Uplift Cost']
  optimum_scenario['ROI'] = optimum_scenario['ROI'].fillna(0)
  lift_wave_opt = optimum_scenario.groupby(['Promo_wave']).agg({'Uplift GMAC, LSV': [("Uplift GMAC, LSV",sum)],
                                                                          'Total Uplift Cost' : [('Total Uplift Cost',sum)],
                                                                          'Incremental' : [('Incremental',sum)],
                                                                          'Base': [('Base',sum)]})
  lift_wave_opt.columns = ['Uplift_GMAC_LSV', 'Uplift_cost', 'incremental', 'base']
  lift_wave_opt = lift_wave_opt.reset_index()
  lift_wave_opt['Lift %'] = lift_wave_opt['incremental']/lift_wave_opt['base']
  lift_wave_opt['ROI'] = lift_wave_opt['Uplift_GMAC_LSV']/lift_wave_opt['Uplift_cost']
  lift_wave_opt

  #Calenders(optimum and Baseline)
  base_scenario['Units']=base_scenario['Base']+base_scenario['Incremental']
  optimum_scenario['Units']=optimum_scenario['Base']+optimum_scenario['Incremental']
  col_req=['Date','tpr_discount_byppg','Units','Base','Incremental','ROI','Lift',"Uplift GMAC, LSV",'Total Uplift Cost']
  base_scenario=base_scenario[col_req]
  optimum_scenario=optimum_scenario[col_req+['SI']]

  base_scenario.rename(columns={'tpr_discount_byppg':'Baseline_Promo','Units':'Baseline_Units','Base':'Baseline_Base',
                                 'Incremental':'Baseline_Incremental','ROI':'Baseline_ROI',"Uplift GMAC, LSV":"Baseline_Uplift_GMAC_LSV",
                                'Total Uplift Cost':'Baseline_Total_Uplift_Cost',"Lift":'Baseline_Lift'
                                },inplace=True)

  optimum_scenario.rename(columns={'tpr_discount_byppg':'Optimum_Promo','Units':'Optimum_Units','Base':'Optimum_Base',
                                 'Incremental':'Optimum_Incremental','ROI':'Optimum_ROI',"Uplift GMAC, LSV":"Optimum_Uplift_GMAC_LSV",
                                'Total Uplift Cost':'Optimum_Total_Uplift_Cost',"Lift":'Optimum_Lift'
                                },inplace=True)
  opt_base = optimum_scenario.merge(base_scenario,on='Date',how='left')
  # opt_base['Optimum_Promo']=np.round(opt_base['Optimum_Promo'],4)
  # opt_base['Baseline_Promo']=np.round(opt_base['Baseline_Promo'],4)
  return opt_base


def get_calendar_summary(baseline_data,optimal_data,opt_base):
  base_scenario=baseline_data.copy()
  Optimal_scenario=optimal_data.copy()
  # Optimal_scenario=pd.read_csv(path+"Training_data_"+str(prd)+"_Optimal.csv")
  base_scenario.rename(columns={'Baseline_Prediction':'Units','Baseline_Sales':'Sales',
                                'Baseline_GSV':'GSV','Baseline_Trade_Expense':'Trade_Expense',
                                'Baseline_NSV':'NSV','Baseline_MAC':'MAC',
                                'Baseline_RP':'RP'},inplace=True)
  base_scenario['AvgSellingPrice']=base_scenario['wk_sold_avg_price_byppg']
  Optimal_scenario['AvgSellingPrice']=Optimal_scenario['wk_sold_avg_price_byppg']
  prd=0
  base_scenario['Product']= "Product"+str(prd)
  Optimal_scenario['Product']= "Product"+str(prd)

  summary_base=base_scenario.groupby('Product').sum()[["Sales","Units","Trade_Expense","GSV","NSV","MAC","RP","AvgSellingPrice"]]
  # summary_base['AvgSellingPrice']=summary_base['AvgSellingPrice']/52
  summary_base['AvgSellingPrice'] = summary_base["Sales"]/summary_base["Units"]
  summary_base['Avg_PromoSellingPrice'] = base_scenario.loc[base_scenario['tpr_discount_byppg']!=0]['Sales'].sum()/base_scenario.loc[base_scenario['tpr_discount_byppg']!=0]['Units'].sum()
  summary_base['ROI'] = opt_base['Baseline_Uplift_GMAC_LSV'].sum()/opt_base['Baseline_Total_Uplift_Cost'].sum()

  summary_opt=Optimal_scenario.groupby('Product').sum()[["Sales","Units","Trade_Expense","GSV","NSV","MAC","RP","AvgSellingPrice"]]
  # summary_opt['AvgSellingPrice']=summary_opt['AvgSellingPrice']/52
  summary_opt['AvgSellingPrice'] = summary_opt["Sales"]/summary_opt["Units"]
  summary_opt['Avg_PromoSellingPrice'] = Optimal_scenario.loc[Optimal_scenario['tpr_discount_byppg']!=0]['Sales'].sum()/Optimal_scenario.loc[Optimal_scenario['tpr_discount_byppg']!=0]['Units'].sum()
  summary_opt['ROI'] = opt_base['Optimum_Uplift_GMAC_LSV'].sum()/opt_base['Optimum_Total_Uplift_Cost'].sum()

  #Metric Summary
  #summary_base=pd.read_csv((path+  f"Summary_Metric_{prd}.csv"))
  #summary_opt=pd.read_csv((path+  f"Summary_Metric_{prd}_Optimal.csv"))
  summary_opt[['Sales', 'Units', "Trade_Expense",'GSV', 'NSV', 'MAC', 'RP']]=summary_opt[['Sales', 'Units', "Trade_Expense",'GSV', 'NSV', 'MAC', 'RP']].astype(int)
  summary_opt["RP_Perc"]=summary_opt['RP']/summary_opt['Sales']
  summary_opt['Mac_Perc'] = summary_opt['MAC']/summary_opt['NSV']

  # summary_opt[["RP_Perc",'Mac_Perc']]=summary_opt[["RP_Perc",'Mac_Perc']].apply(lambda x: round(x,3))
  summary_base[['Sales', 'Units', "Trade_Expense",'GSV', 'NSV', 'MAC', 'RP']]=summary_base[['Sales', 'Units', "Trade_Expense",'GSV', 'NSV', 'MAC', 'RP']].astype(int)
  summary_base["RP_Perc"]=summary_base['RP']/summary_base['Sales']
  summary_base['Mac_Perc'] = summary_base['MAC']/summary_base['NSV']
  # summary_base[["RP_Perc",'Mac_Perc']]=summary_base[["RP_Perc",'Mac_Perc']].apply(lambda x: round(x,3))


  summary_opt=summary_opt.reset_index(drop=True)
  summary_opt=summary_opt.transpose().reset_index(drop=False)
  summary_opt.rename(columns={'index':'Metric',0:'Recommended_Scenario'},inplace=True)

  summary_base=summary_base.reset_index(drop=True)
  summary_base=summary_base.transpose().reset_index(drop=False)
  summary_base.rename(columns={'index':'Metric',0:'Base_Scenario'},inplace=True)

  summary_metric=summary_base.merge(summary_opt,on='Metric',how='left')
  # summary_metric['Change']=np.round(summary_metric['Recommended_Scenario']-summary_metric['Base_Scenario'],4)
  # summary_metric['Delta']=np.round(summary_metric['Change']/summary_metric['Base_Scenario'],4)
  summary_metric['Change']=summary_metric['Recommended_Scenario']-summary_metric['Base_Scenario']
  summary_metric['Delta']=summary_metric['Change']/summary_metric['Base_Scenario']
  return summary_metric



