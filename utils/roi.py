import os
import numpy as np
import pandas as pd
import time
# import pyomo.environ as pyo
import datetime
import itertools
# import libify
# import os
# import numpy as np
# import pandas as pd
import re 

def predict_sales(coeffs,data):
    predict = 0
    for i in coeffs['Variable']:
        if(i=="Intercept"):
            predict = predict + coeffs[coeffs['Variable']==i]["Value"].values
        else:
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
        baseline_value = {"Median_Base_Price_log": ["Rolling Max", 26],
                          "TPR_Discount": ["Set Average", 0],
                          "TPR_Discount_lag1": ["Set Average", 0],
                          "TPR_Discount_lag2": ["Set Average", 0],                  
                          "ACV_Feat_Only": ["Set Average", 0],
                          "ACV_Disp_Only": ["Set Average", 0],
                          "ACV_Feat_Disp": ["Set Average", 0],
                          "ACV": ["As is", 0],
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
        base_var = ["Median_Base_Price_log"] + rp_features + ["ACV", "category_trend", "flag_qtr2", "flag_qtr3", "flag_qtr4", "monthno"]
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
#   print('Incremental :',Incremental)
  dt_dist['Incremental'] = dt_dist[Incremental].sum(axis = 1)
  ###Give the remaining Base columns
  
#   baseprice_cols = ['Median_Base_Price_log']+[i for i in model_cols if "RegularPrice" in i]
#   holiday =[i for i in model_cols if "day" and "flag" in i ]
#   SI_cols = [i for i in model_cols if "SI" in i ]
#   trend_cols = [i for i in model_cols if "trend" in i ]
#   base_list = baseprice_cols+SI_cols+trend_cols+[i for i in model_cols if "ACV" in i ]+holiday+[i for i in model_cols if "death_rate" in i]
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


# path = '/dbfs/mnt/RAW/FILES/SRM_ANALYTICS_RUSSIA_MW/Model_Results/Promo_Tool_Simulator/'


def main(file1,file2, retailer , ppg, segment):
    
    model_data = pd.read_excel(file1,sheet_name='MODEL_DATA')
    model_coeff = pd.read_excel(file1,sheet_name='MODEL_COEFFICIENT')
    ROI_data = pd.read_csv(file2)
    ROI_data['Date'] = pd.to_datetime(ROI_data['Date'])

    ret = retailer
    ppg = ppg
    PPG_names = [ppg]

    tag_path  = ""
    prd=0 
    Segment=segment # Gum
    # import pdb
    # pdb.set_trace()
    base_data = model_data.loc[(model_data['Account Name']==ret) & (model_data['PPG']==ppg)]
    train_coef = model_coeff.loc[(model_coeff['Account Name']==ret) & (model_coeff['PPG']==ppg)]
    train_ROI = ROI_data.loc[(ROI_data['Account Name']==ret) & (ROI_data['PPG']==ppg)]

    base_data = base_data.merge(train_ROI[['Date','List_Price','GMAC', 'COGS',
                                                'TE Off Inv', 'TE On Inv']],how='left',on='Date')

    coeff = train_coef.melt(var_name = "Variable",value_name='Value')
    coeff = coeff.iloc[8:].reset_index(drop=True)
    coeff['Value'] = coeff['Value'].astype(float)
    print(coeff , "coefficient")
    coeff.info()
    base_data.info()
    prd=0
    # train_data=pd.read_csv(path+"Model_Results/Training_data_0.csv")
    # model_coef = pd.read_csv(path + "Model_Results/"+model_coeff_file[prd])

    ### train_data is the original or the simulated model data
    train_data = base_data.copy() 
    model_coef = coeff.copy()


    # doing financial calculations on base data
    train_data["Promotion_Cost"] = train_data['List_Price'] * train_data['TPR_Discount']/100 * (1 - train_data['TE On Inv'])
    train_data["TE"] = train_data["List_Price"] * (train_data["TPR_Discount"]/100 + train_data["TE On Inv"] + train_data["TE Off Inv"] - 
                                                            train_data["TPR_Discount"]/100 * train_data["TE Off Inv"] - train_data["TE Off Inv"] * train_data["TE On Inv"] - 
                                                            train_data["TE On Inv"] * train_data["TPR_Discount"]/100 + train_data["TE Off Inv"] * train_data["TE On Inv"] * train_data["TPR_Discount"]/100)

    print(np.exp(train_data['Median_Base_Price_log']).max())
    train_data["Promo"] = np.where(train_data['TPR_Discount'] == 0, np.exp(train_data['Median_Base_Price_log']),
                                    np.exp(train_data['Median_Base_Price_log']) * (1-train_data['TPR_Discount'])) 



    train_data['Promo_wave']=promo_wave_cal(train_data)
    # sdate='01-02-2022'
    # train_data['Date']=pd.date_range(sdate,periods=train_data.shape[0],freq='w') 
    train_data['Units']=predict_sales(model_coef,train_data)
    # train_data['Baseline_Prediction']=predict_sales(coeff,train_data)
    train_data['Sales']=train_data['Units'] *train_data['Promo']
    train_data["GSV"] = train_data['Units'] * train_data['List_Price']
    train_data["Trade_Expense"] = train_data['Units'] * train_data['TE']
    train_data["NSV"] = train_data['GSV'] - train_data["Trade_Expense"]
    train_data["MAC"] = train_data["NSV"]-train_data['Units'] * train_data['COGS']
    train_data["RP"] = train_data['Sales']-train_data["NSV"]

    train_data['wk_sold_avg_price_byppg']=np.exp(train_data['Median_Base_Price_log'])*(1-train_data['TPR_Discount']/100)
    model_df=train_data.copy()


    #Base Variable Method
    # model_coef.rename(columns={"names":"Variable","model_coefficients":"Value"},inplace=True)
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
    # base_list = ['Median_Base_Price_log']+[i for i in col if "RegularPrice" in i]+[i for i in col if "day" and "flag" in i ]+["SI","ACV"]
    # base_list.remove( 'flag_date_1')
    # base_list.remove( 'flag_date_2')
    print(base_list)
    baseline_var["Variable"]=base_list
    baseline_var_othr["Variable"]=base_others

    model_df['Iteration']=1
    model_df1=model_df.copy()
    model_coef['Iteration']=1

    base_scenario=base_var_cont(model_df,model_df1,baseline_var,baseline_var_othr,model_coef) # base and increment function

    base_scenario=base_scenario.merge(train_data[['Date','TPR_Discount','List_Price','GMAC',
                                                'TE Off Inv', 'TE On Inv']],how='left',on='Date')
    base_scenario['TPR']=base_scenario['TPR_Discount']/100
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


    # base_scenario
    # print(base_scenario , "base scenario value")
    return base_scenario
    # base_scenario.to_csv(path+"Model_Results/base_scenario.csv",index=False)
    # lift_wave_base = base_scenario.groupby(['Promo_wave']).agg({'Uplift GMAC, LSV': [("Uplift GMAC, LSV",sum)],
    #                                                                         'Total Uplift Cost' : [('Total Uplift Cost',sum)],
    #                                                                         'Incremental' : [('Incremental',sum)],
    #                                                                         'Base': [('Base',sum)]})
    # lift_wave_base.columns = ['Uplift_GMAC_LSV', 'Uplift_cost', 'incremental', 'base']
    # lift_wave_base = lift_wave_base.reset_index()
    # lift_wave_base['Lift %'] = lift_wave_base['incremental']/lift_wave_base['base']
    # lift_wave_base['ROI'] = lift_wave_base['Uplift_GMAC_LSV']/lift_wave_base['Uplift_cost']
    # lift_wave_base
    # print(lift_wave_base , "lift_wave_base value")
    # lift_wave_base.to_csv(path+"Model_Results/lift_wave_base.csv",index=False)


# path = 'C:/Users/aravindhan.mathi/Videos/project/data_roi/'
# file1 = path + 'Data/Promo_Simulator_Backend_All_tabs_All_Sprints_v7.xlsx'
# file2 = path + 'Data/ROI_Data_All_retailers_with_extra_columns_v3.csv'
# retailer = 'Lenta'
# ppg = 'Big Bars'
# segment = 'Gum'
# main(file1,file2,retailer,ppg,segment)