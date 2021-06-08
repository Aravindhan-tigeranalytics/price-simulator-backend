from django.db.models import base
import numpy as np
import pandas as pd
import datetime
from decimal import Decimal
import numpy as np
import pandas as pd
import re

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



# path = 'C:/Users/aravindhan.mathi/Videos/project/data_roi/'
# file1 = path + 'Data/Promo_Simulator_Backend_All_tabs_All_Sprints_v7.xlsx'
# file2 = path + 'Data/ROI_Data_All_retailers_with_extra_columns_v3.csv'
# retailer = 'Tander'
# ppg = 'A.Korkunov 192g'
# segment = 'BOXES'






# coeff=[[340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', Decimal('0.330000000000000'), Decimal('0.750000000000000'), Decimal('11.534000000000001'), Decimal('-1.083000000000000'), Decimal('0.034000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0.049000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0.272000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('-0.001000000000000'), Decimal('-0.001000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0.750000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('1.472000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0.185000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15')]]

# data = [[340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 4, 11, 'P12', datetime.date(2022, 11, 20), 47, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('36.428570917674477'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0.960000000000000'), Decimal('0E-15'), Decimal('76.700000000000003'), Decimal('1.084767852508583'), Decimal('0.876219937995377'), Decimal('1.502105643771855'), Decimal('0E-15'), Decimal('0E-15'), Decimal('15.415216310293390'), Decimal('5.194513154935898'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.588230625027972'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('59.000000000000000'), Decimal('20.000000000000000'), Decimal('5.000000000000000'), Decimal('59.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('226.983160706236788'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 4, 10, 'P11', datetime.date(2022, 10, 23), 43, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('22.857143197740829'), Decimal('40.000000596046448'), Decimal('40.000000596046448'), Decimal('0.970000000000000'), Decimal('0E-15'), Decimal('56.399999999999999'), Decimal('0.577094947835410'), Decimal('0.796726070054433'), Decimal('1.502105643771855'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.586699536340228'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('58.000000000000000'), Decimal('20.000000000000000'), Decimal('5.000000000000000'), Decimal('58.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('221.886938034985008'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 4, 10, 'P11', datetime.date(2022, 10, 16), 42, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('40.000000596046448'), Decimal('40.000000596046448'), Decimal('40.000000596046448'), Decimal('0.970000000000000'), Decimal('0E-15'), Decimal('59.799999999999997'), Decimal('0.602364870248981'), Decimal('0.796726070054433'), Decimal('1.502105643771855'), Decimal('0E-15'), Decimal('0E-15'), Decimal('10.412533048431650'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.576607193767206'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('58.000000000000000'), Decimal('20.000000000000000'), Decimal('5.000000000000000'), Decimal('58.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('214.079712364512403'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 3, 7, 'P07', datetime.date(2022, 7, 3), 27, Decimal('1.000000000000000'), Decimal('5.476593211756645'), 
# Decimal('20.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('53.200000000000003'), Decimal('0.454048384412045'), Decimal('0.447428882959249'), Decimal('0.508945206264278'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('8.147374701670639'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.562918470310414'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('55.000000000000000'), Decimal('19.000000000000000'), Decimal('5.000000000000000'), Decimal('55.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('249.852244505567910'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 2, 5, 'P05', datetime.date(2022, 5, 8), 19, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('19.999999659402029'), Decimal('0E-15'), Decimal('19.714286071913580'), Decimal('0.940000000000000'), Decimal('0E-15'), Decimal('72.700000000000003'), Decimal('0.505223977817316'), Decimal('0.549403384998270'), Decimal('0.508945206264278'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('15.833022909294099'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('53.000000000000000'), Decimal('18.000000000000000'), Decimal('5.000000000000000'), Decimal('53.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('216.437346081100912'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 1, 2, 'P02', datetime.date(2022, 2, 13), 7, 
# Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('26.857142789023261'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0.940000000000000'), Decimal('0E-15'), Decimal('79.500000000000000'), Decimal('1.185824920289000'), Decimal('1.226117080399179'), Decimal('1.307502887024804'), Decimal('0E-15'), Decimal('0E-15'), Decimal('11.966433837795879'), Decimal('31.627197072390441'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('50.000000000000000'), Decimal('17.000000000000000'), Decimal('5.000000000000000'), Decimal('50.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('208.406619675089701'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 1, 1, 'P01', datetime.date(2022, 1, 2), 1, Decimal('1.000000000000000'), Decimal('5.584779377133753'), Decimal('49.000000953674324'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0.060000000000000'), Decimal('0E-15'), Decimal('94.200000000000003'), Decimal('1.903973035394813'), Decimal('0.928719646479435'), Decimal('1.307502887024804'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570014571210206'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('49.000000000000000'), Decimal('17.000000000000000'), Decimal('5.000000000000000'), Decimal('49.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('223.261522386607510'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 1, 1, 'P01', datetime.date(2022, 1, 9), 2, Decimal('1.000000000000000'), Decimal('5.584779377133753'), Decimal('28.000000544956752'), Decimal('49.000000953674324'), Decimal('0E-15'), Decimal('0.060000000000000'), Decimal('0E-15'), Decimal('67.500000000000000'), Decimal('0.824729402816766'), Decimal('0.928719646479435'), Decimal('1.307502887024804'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('32.178807889744220'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570014571210206'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('49.000000000000000'), Decimal('17.000000000000000'), Decimal('5.000000000000000'), Decimal('49.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('249.004536289839194'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 1, 1, 'P01', datetime.date(2022, 1, 16), 3, Decimal('1.000000000000000'), Decimal('5.584779377133753'), Decimal('0E-15'), Decimal('28.000000544956752'), Decimal('49.000000953674324'), Decimal('0E-15'), Decimal('0E-15'), Decimal('55.700000000000003'), Decimal('0.650152836812828'), Decimal('0.928719646479435'), Decimal('1.307502887024804'), Decimal('0E-15'), Decimal('0E-15'), Decimal('9.352684966580703'), Decimal('30.432802679809878'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570014571210206'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('49.000000000000000'), Decimal('17.000000000000000'), Decimal('5.000000000000000'), Decimal('49.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('326.337207898246390'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 1, 1, 'P01', datetime.date(2022, 1, 23), 4, Decimal('1.000000000000000'), Decimal('5.588430488253191'), Decimal('0E-15'), Decimal('0E-15'), Decimal('28.000000544956752'), Decimal('0E-15'), Decimal('0E-15'), Decimal('51.200000000000003'), Decimal('0.640640804095574'), Decimal('0.928719646479435'), Decimal('1.307502887024804'), Decimal('0E-15'), Decimal('0E-15'), Decimal('12.739352093452680'), Decimal('19.654766196312700'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570014571210206'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('49.000000000000000'), Decimal('17.000000000000000'), Decimal('5.000000000000000'), Decimal('49.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('345.143842541066192'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 1, 2, 'P02', datetime.date(2022, 1, 30), 5, Decimal('1.000000000000000'), Decimal('5.588430488253191'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('53.700000000000003'), Decimal('0.522562988876450'), Decimal('0.928719646479435'), Decimal('1.307502887024804'), Decimal('0E-15'), Decimal('0E-15'), Decimal('15.758664746365300'), Decimal('16.537442096658321'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570014571210206'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('50.000000000000000'), Decimal('17.000000000000000'), Decimal('5.000000000000000'), Decimal('50.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), 
# Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('323.748971807010378'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 1, 2, 'P02', datetime.date(2022, 2, 6), 6, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('67.099999999999994'), Decimal('0.657093863558997'), Decimal('1.226117080399179'), Decimal('1.307502887024804'), Decimal('0E-15'), Decimal('0E-15'), Decimal('15.896586680624370'), Decimal('13.858275826552680'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('50.000000000000000'), Decimal('17.000000000000000'), Decimal('5.000000000000000'), Decimal('50.000000000000000'), Decimal('0E-15'), 
# Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('342.347074676845409'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 1, 2, 'P02', datetime.date(2022, 2, 20), 8, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('46.999999880790710'), Decimal('26.857142789023261'), Decimal('0E-15'), Decimal('0.950000000000000'), Decimal('0E-15'), Decimal('76.000000000000000'), Decimal('1.079453085940387'), Decimal('1.226117080399179'), Decimal('1.307502887024804'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('21.109759106271561'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('50.000000000000000'), Decimal('17.000000000000000'), Decimal('5.000000000000000'), Decimal('50.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('204.291871152669898'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 1, 3, 'P03', datetime.date(2022, 2, 27), 9, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('46.999999880790710'), Decimal('46.999999880790710'), Decimal('26.857142789023261'), Decimal('0.950000000000000'), Decimal('0E-15'), Decimal('89.799999999999997'), Decimal('1.982096451808331'), Decimal('1.226117080399179'), Decimal('1.307502887024804'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('7.206774572784147'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('51.000000000000000'), Decimal('17.000000000000000'), Decimal('5.000000000000000'), Decimal('51.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('201.516008201326088'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 1, 3, 'P03', datetime.date(2022, 3, 6), 10, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('46.999999880790710'), Decimal('46.999999880790710'), Decimal('46.999999880790710'), Decimal('0.930000000000000'), Decimal('0E-15'), Decimal('96.299999999999997'), Decimal('6.616209927395540'), Decimal('2.141922575375890'), Decimal('1.307502887024804'), Decimal('0E-15'), Decimal('0E-15'), Decimal('16.587611647665380'), Decimal('7.786105162978830'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('51.000000000000000'), Decimal('17.000000000000000'), Decimal('5.000000000000000'), Decimal('51.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('201.580825061080787'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 1, 3, 'P03', datetime.date(2022, 3, 13), 11, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('20.142857091767450'), Decimal('46.999999880790710'), Decimal('46.999999880790710'), Decimal('0.830000000000000'), Decimal('0E-15'), Decimal('82.099999999999994'), Decimal('0.806649516948754'), Decimal('2.141922575375890'), Decimal('1.307502887024804'), Decimal('0E-15'), Decimal('0E-15'), Decimal('11.694037341899209'), Decimal('19.776675298374911'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('51.000000000000000'), Decimal('17.000000000000000'), Decimal('5.000000000000000'), Decimal('51.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('224.189695378854992'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 1, 3, 'P03', datetime.date(2022, 3, 20), 12, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('0E-15'), Decimal('20.142857091767450'), Decimal('46.999999880790710'), Decimal('0E-15'), Decimal('0E-15'), Decimal('65.299999999999997'), Decimal('0.582584689740343'), Decimal('2.141922575375890'), Decimal('1.307502887024804'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('6.245538900785153'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('51.000000000000000'), Decimal('17.000000000000000'), Decimal('5.000000000000000'), Decimal('51.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('317.572672415087197'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 1, 3, 'P04', datetime.date(2022, 3, 27), 13, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('0E-15'), Decimal('0E-15'), Decimal('20.142857091767450'), Decimal('0E-15'), Decimal('0E-15'), Decimal('63.399999999999999'), Decimal('0.562246167418922'), Decimal('2.141922575375890'), Decimal('1.307502887024804'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.972335031627640'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('51.000000000000000'), Decimal('17.000000000000000'), Decimal('5.000000000000000'), Decimal('51.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('292.744517572594589'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 2, 4, 'P04', datetime.date(2022, 4, 3), 14, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('36.500000000000000'), Decimal('0.441753675250957'), Decimal('0.453911638279096'), Decimal('1.307502887024804'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('52.000000000000000'), Decimal('18.000000000000000'), Decimal('5.000000000000000'), Decimal('52.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), 
# Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('272.651541553753987'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 2, 4, 'P04', datetime.date(2022, 4, 10), 15, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('14.285714285714290'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('47.899999999999999'), Decimal('0.438412064258238'), Decimal('0.453911638279096'), Decimal('0.508945206264278'), Decimal('0E-15'), Decimal('0E-15'), Decimal('18.303247235108088'), Decimal('13.185119964780990'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('52.000000000000000'), Decimal('18.000000000000000'), Decimal('5.000000000000000'), Decimal('52.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('228.847952195315713'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], 
# [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 2, 4, 'P04', datetime.date(2022, 4, 
# 17), 16, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('37.000000476837158'), Decimal('14.285714285714290'), Decimal('0E-15'), Decimal('0.920000000000000'), Decimal('0E-15'), Decimal('55.899999999999999'), Decimal('0.461891335517736'), Decimal('0.453911638279096'), Decimal('0.508945206264278'), Decimal('0E-15'), Decimal('0E-15'), Decimal('21.020668732812432'), Decimal('8.760593220338986'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('52.000000000000000'), Decimal('18.000000000000000'), Decimal('5.000000000000000'), Decimal('52.000000000000000'), Decimal('0E-15'), 
# Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('212.531602852188996'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 2, 4, 'P05', datetime.date(2022, 4, 24), 17, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('19.714286071913580'), Decimal('37.000000476837158'), Decimal('14.285714285714290'), Decimal('0.920000000000000'), Decimal('0E-15'), Decimal('44.899999999999999'), Decimal('0.493200451091097'), Decimal('0.453911638279096'), Decimal('0.508945206264278'), Decimal('0E-15'), Decimal('0E-15'), Decimal('18.716732840938551'), Decimal('6.978813559322039'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('52.000000000000000'), Decimal('18.000000000000000'), Decimal('5.000000000000000'), Decimal('52.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('234.816323270455086'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 2, 5, 'P05', datetime.date(2022, 5, 1), 18, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('0E-15'), Decimal('19.714286071913580'), Decimal('37.000000476837158'), Decimal('0E-15'), Decimal('0E-15'), Decimal('66.000000000000000'), Decimal('0.452424280237041'), Decimal('0.453911638279096'), Decimal('0.508945206264278'), Decimal('0E-15'), Decimal('0E-15'), Decimal('17.672674614880339'), Decimal('7.297083903568802'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('53.000000000000000'), Decimal('18.000000000000000'), Decimal('5.000000000000000'), Decimal('53.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), 
# Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('231.426080585198406'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 2, 5, 'P05', datetime.date(2022, 5, 15), 20, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('14.999999744551520'), Decimal('19.999999659402029'), Decimal('0E-15'), Decimal('0.940000000000000'), Decimal('0E-15'), 
# Decimal('71.599999999999994'), Decimal('0.519601048278830'), Decimal('0.549403384998270'), Decimal('0.508945206264278'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('21.751705921197448'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('53.000000000000000'), Decimal('18.000000000000000'), Decimal('5.000000000000000'), Decimal('53.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('203.699880405013602'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 2, 5, 'P06', datetime.date(2022, 5, 22), 21, Decimal('1.000000000000000'), Decimal('5.680441239073231'), Decimal('19.999999659402029'), Decimal('14.999999744551520'), Decimal('19.999999659402029'), Decimal('0.870000000000000'), Decimal('0E-15'), Decimal('74.000000000000000'), Decimal('0.633663537945396'), Decimal('0.549403384998270'), Decimal('0.508945206264278'), Decimal('0E-15'), Decimal('0E-15'), Decimal('20.404882312542931'), Decimal('15.122300736275140'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('53.000000000000000'), Decimal('18.000000000000000'), Decimal('5.000000000000000'), Decimal('53.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('211.912722294337414'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 2, 6, 'P06', datetime.date(2022, 5, 29), 22, Decimal('1.000000000000000'), Decimal('5.706908366270659'), Decimal('14.999999744551520'), Decimal('19.999999659402029'), Decimal('14.999999744551520'), Decimal('0.870000000000000'), Decimal('0E-15'), Decimal('70.299999999999997'), Decimal('0.584662525471397'), Decimal('0.549403384998270'), Decimal('0.508945206264278'), Decimal('0E-15'), Decimal('0E-15'), Decimal('20.810901237947501'), Decimal('7.697574334898272'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('54.000000000000000'), Decimal('18.000000000000000'), Decimal('5.000000000000000'), Decimal('54.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), 
# Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('235.913896235343600'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 2, 6, 'P06', datetime.date(2022, 6, 5), 23, Decimal('1.000000000000000'), Decimal('5.706908366270659'), Decimal('0E-15'), Decimal('14.999999744551520'), Decimal('19.999999659402029'), Decimal('0E-15'), Decimal('0E-15'), Decimal('55.500000000000000'), Decimal('0.564761663157614'), Decimal('0.510594679495203'), Decimal('0.508945206264278'), Decimal('0E-15'), Decimal('0E-15'), Decimal('18.059750590627711'), Decimal('12.407338769458850'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.570019701919257'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('54.000000000000000'), Decimal('18.000000000000000'), Decimal('5.000000000000000'), Decimal('54.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('229.109917621662902'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 2, 6, 
# 'P06', datetime.date(2022, 6, 12), 24, Decimal('1.000000000000000'), Decimal('5.517471114330244'), Decimal('0E-15'), Decimal('0E-15'), Decimal('14.999999744551520'), Decimal('0E-15'), Decimal('0E-15'), Decimal('49.500000000000000'), Decimal('0.495153810430618'), Decimal('0.510594679495203'), Decimal('0.508945206264278'), Decimal('0E-15'), Decimal('0E-15'), Decimal('14.956813463819699'), Decimal('12.128325508607210'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.562918470310414'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('54.000000000000000'), Decimal('18.000000000000000'), Decimal('5.000000000000000'), Decimal('54.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('231.624794548623214'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 2, 6, 'P07', datetime.date(2022, 6, 19), 25, Decimal('1.000000000000000'), Decimal('5.517471114330244'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('53.399999999999999'), Decimal('0.490813999967702'), Decimal('0.510594679495203'), Decimal('0.508945206264278'), Decimal('0E-15'), Decimal('0E-15'), Decimal('14.232868648489550'), Decimal('11.752738654147100'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.562918470310414'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('54.000000000000000'), Decimal('18.000000000000000'), Decimal('5.000000000000000'), Decimal('54.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('239.030990707288709'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 2, 6, 'P07', datetime.date(2022, 6, 26), 26, Decimal('1.000000000000000'), Decimal('5.476593211756645'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('53.500000000000000'), Decimal('0.491649244424877'), Decimal('0.510594679495203'), Decimal('0.508945206264278'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('9.318603613774412'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.562918470310414'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('54.000000000000000'), Decimal('18.000000000000000'), Decimal('5.000000000000000'), Decimal('54.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('240.151323986063289'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 3, 7, 'P07', datetime.date(2022, 7, 10), 28, Decimal('1.000000000000000'), Decimal('5.476593211756645'), Decimal('8.571428571428573'), Decimal('20.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('54.700000000000003'), Decimal('0.448283875876944'), Decimal('0.447428882959249'), Decimal('0.668389955500459'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.542501360733876'), Decimal('0E-15'), Decimal('0E-15'), 
# Decimal('0E-15'), Decimal('5.562918470310414'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('55.000000000000000'), Decimal('19.000000000000000'), Decimal('5.000000000000000'), Decimal('55.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('235.245238735065612'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 3, 7, 'P08', datetime.date(2022, 7, 17), 29, Decimal('1.000000000000000'), Decimal('5.476593211756645'), Decimal('0E-15'), Decimal('8.571428571428573'), Decimal('20.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('65.700000000000003'), Decimal('0.431820670665140'), Decimal('0.447428882959249'), Decimal('0.668389955500459'), Decimal('0E-15'), Decimal('0E-15'), Decimal('9.026769471190121'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.562918470310414'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('55.000000000000000'), Decimal('19.000000000000000'), Decimal('5.000000000000000'), Decimal('55.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('231.386517648020089'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 3, 7, 'P08', datetime.date(2022, 7, 24), 30, Decimal('1.000000000000000'), Decimal('5.476593211756645'), Decimal('0E-15'), Decimal('0E-15'), Decimal('8.571428571428573'), Decimal('0E-15'), Decimal('0E-15'), Decimal('68.000000000000000'), Decimal('0.439572927052304'), Decimal('0.447428882959249'), Decimal('0.668389955500459'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('8.903253152484902'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.562918470310414'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('55.000000000000000'), Decimal('19.000000000000000'), Decimal('5.000000000000000'), Decimal('55.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('230.759366222581008'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 3, 8, 'P08', datetime.date(2022, 7, 31), 31, Decimal('1.000000000000000'), Decimal('5.476593211756645'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('63.700000000000003'), Decimal('0.452760524545528'), Decimal('0.447428882959249'), Decimal('0.668389955500459'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('14.717559647092351'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.559250366415653'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('56.000000000000000'), Decimal('19.000000000000000'), Decimal('5.000000000000000'), Decimal('56.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('231.544680723294505'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 3, 8, 'P08', datetime.date(2022, 8, 7), 32, Decimal('1.000000000000000'), Decimal('5.476593211756645'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('54.799999999999997'), Decimal('0.466039336659315'), Decimal('0.578446665880063'), Decimal('0.668389955500459'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('20.284896834647810'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.541920144000137'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('56.000000000000000'), Decimal('19.000000000000000'), Decimal('5.000000000000000'), Decimal('56.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('231.368750170264093'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 3, 8, 'P09', datetime.date(2022, 8, 14), 33, Decimal('1.000000000000000'), Decimal('5.476593211756645'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('53.100000000000001'), Decimal('0.505049507129710'), Decimal('0.578446665880063'), Decimal('0.668389955500459'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('21.211232383172629'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.538560396537948'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('56.000000000000000'), Decimal('19.000000000000000'), Decimal('5.000000000000000'), Decimal('56.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('234.721611554900392'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 3, 8, 'P09', datetime.date(2022, 8, 21), 34, Decimal('1.000000000000000'), 
# Decimal('5.476593211756645'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('56.600000000000001'), Decimal('0.546356790931130'), Decimal('0.578446665880063'), Decimal('0.668389955500459'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('18.533063657850011'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.538560396537948'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('56.000000000000000'), Decimal('19.000000000000000'), Decimal('5.000000000000000'), Decimal('56.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('243.599863591337908'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 3, 8, 'P09', datetime.date(2022, 8, 28), 35, Decimal('1.000000000000000'), Decimal('5.476593211756645'), Decimal('23.428571224212650'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0.950000000000000'), Decimal('0E-15'), Decimal('81.900000000000006'), Decimal('0.850671655351316'), Decimal('0.578446665880063'), Decimal('0.668389955500459'), Decimal('0E-15'), Decimal('0E-15'), Decimal('13.443135923154360'), Decimal('14.707651149316950'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.538560396537948'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('56.000000000000000'), Decimal('19.000000000000000'), Decimal('5.000000000000000'), Decimal('56.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('220.864937699583407'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 3, 9, 'P09', datetime.date(2022, 9, 4), 36, Decimal('1.000000000000000'), Decimal('5.476593211756645'), Decimal('17.571428418159481'), Decimal('23.428571224212650'), Decimal('0E-15'), Decimal('0.950000000000000'), Decimal('0E-15'), Decimal('83.900000000000006'), Decimal('0.771020459880263'), Decimal('0.797555774503776'), Decimal('0.668389955500459'), Decimal('0E-15'), Decimal('0E-15'), Decimal('16.484218315850509'), Decimal('11.104260791661460'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.541920144000137'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('57.000000000000000'), Decimal('19.000000000000000'), Decimal('5.000000000000000'), Decimal('57.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('226.250367224033397'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 
# 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 3, 9, 'P10', datetime.date(2022, 9, 11), 37, Decimal('1.000000000000000'), Decimal('5.476593211756645'), Decimal('0E-15'), Decimal('17.571428418159481'), Decimal('23.428571224212650'), Decimal('0E-15'), Decimal('0E-15'), Decimal('60.600000000000001'), Decimal('0.571607249319284'), Decimal('0.797555774503776'), Decimal('0.668389955500459'), Decimal('0E-15'), Decimal('0E-15'), Decimal('12.762372739406951'), Decimal('10.654471504971600'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.539805696847915'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('57.000000000000000'), Decimal('19.000000000000000'), Decimal('5.000000000000000'), Decimal('57.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('242.435366875946301'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 3, 9, 'P10', datetime.date(2022, 9, 18), 38, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('0E-15'), Decimal('0E-15'), Decimal('17.571428418159481'), Decimal('0E-15'), Decimal('0E-15'), Decimal('64.500000000000000'), Decimal('0.574709127710757'), Decimal('0.797555774503776'), Decimal('0.668389955500459'), Decimal('0E-15'), Decimal('0E-15'), Decimal('11.729239184485330'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.539805696847915'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('57.000000000000000'), Decimal('19.000000000000000'), Decimal('5.000000000000000'), Decimal('57.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('233.255040904966108'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 3, 9, 'P10', datetime.date(2022, 9, 25), 39, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('31.428571684019911'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0.970000000000000'), Decimal('0E-15'), Decimal('69.799999999999997'), Decimal('1.086315039043100'), Decimal('0.797555774503776'), Decimal('0.668389955500459'), Decimal('0E-15'), Decimal('0E-15'), Decimal('8.331520387866743'), Decimal('11.538534928540081'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.554827563867398'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('57.000000000000000'), Decimal('19.000000000000000'), Decimal('5.000000000000000'), Decimal('57.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('221.568619131883992'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 4, 10, 'P10', datetime.date(2022, 10, 2), 40, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('40.000000596046448'), Decimal('31.428571684019911'), Decimal('0E-15'), Decimal('0.970000000000000'), Decimal('0E-15'), Decimal('88.500000000000000'), Decimal('1.615180180251434'), Decimal('0.797555774503776'), Decimal('0.668389955500459'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('15.169635391592980'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.554827563867398'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('58.000000000000000'), Decimal('20.000000000000000'), Decimal('5.000000000000000'), Decimal('58.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('220.067575451869914'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], 
# [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 4, 10, 'P11', datetime.date(2022, 10, 9), 41, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('40.000000596046448'), Decimal('40.000000596046448'), Decimal('31.428571684019911'), Decimal('0.970000000000000'), Decimal('0E-15'), Decimal('74.000000000000000'), Decimal('0.748556885488473'), Decimal('0.796726070054433'), Decimal('1.502105643771855'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.576526137207336'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('58.000000000000000'), Decimal('20.000000000000000'), Decimal('5.000000000000000'), Decimal('58.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('221.062832091180610'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 4, 11, 'P11', datetime.date(2022, 10, 30), 44, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('0E-15'), Decimal('22.857143197740829'), Decimal('40.000000596046448'), Decimal('0E-15'), Decimal('0E-15'), Decimal('50.100000000000001'), Decimal('0.627281256659349'), Decimal('0.796726070054433'), Decimal('1.502105643771855'), Decimal('0E-15'), Decimal('0E-15'), Decimal('10.786507541853149'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.588230625027972'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('59.000000000000000'), Decimal('20.000000000000000'), Decimal('5.000000000000000'), Decimal('59.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('192.048502696835186'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 4, 11, 'P12', datetime.date(2022, 11, 6), 45, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('0E-15'), Decimal('0E-15'), Decimal('22.857143197740829'), Decimal('0E-15'), Decimal('0E-15'), Decimal('62.200000000000003'), Decimal('0.632767357375529'), Decimal('0.876219937995377'), Decimal('1.502105643771855'), Decimal('0E-15'), Decimal('0E-15'), Decimal('13.177523620089509'), Decimal('9.114402075949180'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.588230625027972'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('59.000000000000000'), Decimal('20.000000000000000'), Decimal('5.000000000000000'), Decimal('59.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('192.347139608105095'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 4, 11, 'P12', datetime.date(2022, 11, 13), 46, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('62.200000000000003'), Decimal('0.746834740529218'), Decimal('0.876219937995377'), Decimal('1.502105643771855'), Decimal('0E-15'), Decimal('0E-15'), Decimal('8.777425096548431'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.588230625027972'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('59.000000000000000'), Decimal('20.000000000000000'), Decimal('5.000000000000000'), Decimal('59.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('273.381103124990091'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 4, 11, 'P12', datetime.date(2022, 11, 27), 48, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('44.999998807907097'), Decimal('36.428570917674477'), Decimal('0E-15'), Decimal('0.960000000000000'), Decimal('0E-15'), Decimal('88.900000000000006'), Decimal('1.040509801568178'), Decimal('0.876219937995377'), Decimal('1.502105643771855'), 
# Decimal('0E-15'), Decimal('0E-15'), Decimal('21.348072327860489'), Decimal('15.764363326139010'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.588230625027972'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('59.000000000000000'), Decimal('20.000000000000000'), Decimal('5.000000000000000'), Decimal('59.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('217.691375862549393'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 4, 12, 'P13', datetime.date(2022, 12, 4), 49, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('44.999998807907097'), Decimal('44.999998807907097'), Decimal('36.428570917674477'), Decimal('0.960000000000000'), Decimal('0E-15'), Decimal('86.900000000000006'), Decimal('1.287751531723441'), Decimal('2.876721206378267'), Decimal('1.502105643771855'), Decimal('0E-15'), Decimal('0E-15'), Decimal('20.430648194705661'), Decimal('15.857030855906499'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.588230625027972'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('60.000000000000000'), Decimal('20.000000000000000'), Decimal('5.000000000000000'), Decimal('60.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('216.786536263712009'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 
# 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 4, 12, 'P13', datetime.date(2022, 12, 11), 50, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('44.999998807907097'), Decimal('44.999998807907097'), Decimal('44.999998807907097'), Decimal('0.960000000000000'), Decimal('0E-15'), Decimal('88.599999999999994'), Decimal('1.654395438320307'), Decimal('2.876721206378267'), Decimal('1.502105643771855'), Decimal('0E-15'), Decimal('0E-15'), Decimal('21.264786413315981'), Decimal('15.999179592613361'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.588230625027972'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('60.000000000000000'), Decimal('20.000000000000000'), Decimal('5.000000000000000'), Decimal('60.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('216.581990521326986'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 4, 12, 'P13', datetime.date(2022, 12, 18), 51, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('38.571427890232627'), Decimal('44.999998807907097'), Decimal('44.999998807907097'), Decimal('0.960000000000000'), Decimal('0E-15'), Decimal('93.700000000000003'), Decimal('2.927874703845202'), Decimal('2.876721206378267'), Decimal('1.502105643771855'), Decimal('0E-15'), Decimal('0E-15'), Decimal('12.573929187110449'), Decimal('5.377576907362758'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.588230625027972'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('60.000000000000000'), Decimal('20.000000000000000'), Decimal('5.000000000000000'), Decimal('60.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('213.265306122448891'), Decimal('192.000000000000000'), Decimal('192.000000000000000')], [340, 'Tander', 'BOXES', 'A.Korkunov 192g', 'KORKUNOV', 'KORKUNOV@Boxes', 'Win in premium and gifting', 2022, 4, 12, 'P13', datetime.date(2022, 12, 25), 52, Decimal('1.000000000000000'), Decimal('5.481269241258506'), Decimal('30.000000000000000'), Decimal('38.571427890232627'), Decimal('44.999998807907097'), Decimal('0E-15'), Decimal('0E-15'), Decimal('95.700000000000003'), Decimal('5.006893863810569'), Decimal('2.876721206378267'), Decimal('1.502105643771855'), Decimal('0E-15'), 
# Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('5.588230625027972'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('60.000000000000000'), Decimal('20.000000000000000'), Decimal('5.000000000000000'), Decimal('60.000000000000000'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('0E-15'), Decimal('215.037768404908007'), Decimal('192.000000000000000'), Decimal('192.000000000000000')]] 
coeff_columns = ['meta_id','Account Name', 'Corporate Segment', 'PPG', 'Brand Filter',
       'Brand Format Filter', 'Strategic Cell Filter', 'WMAPE', 'Rsq','Intercept', 'Median_Base_Price_log', 'TPR_Discount', 'TPR_Discount_lag1', 'TPR_Discount_lag2', 'Catalogue', 'Display', 'ACV', 'SI', 'SI_month', 'SI_quarter', 'C_1_crossretailer_discount', 'C_1_crossretailer_log_price', 'C_1_intra_discount', 'C_2_intra_discount', 'C_3_intra_discount', 'C_4_intra_discount', 'C_5_intra_discount', 'C_1_intra_log_price', 'C_2_intra_log_price', 'C_3_intra_log_price', 'C_4_intra_log_price', 'C_5_intra_log_price', 'Category trend', 'Trend_month', 'Trend_quarter', 'Trend_year', 'month_no', 'Flag_promotype_Motivation', 'Flag_promotype_N_pls_1', 'Flag_promotype_traffic', 'Flag_nonpromo_1', 'Flag_nonpromo_2', 'Flag_nonpromo_3', 'Flag_promo_1', 'Flag_promo_2', 'Flag_promo_3', 'Holiday_Flag1', 'Holiday_Flag2', 'Holiday_Flag3', 'Holiday_Flag4', 'Holiday_Flag5', 'Holiday_Flag6', 'Holiday_Flag7', 'Holiday_Flag8', 'Holiday_Flag9', 'Holiday_Flag10']
# coeff_columns
data_columns =  ['meta_id','Account Name', 'Corporate Segment', 'PPG', 'Brand Filter',
       'Brand Format Filter', 'Strategic Cell Filter','Year', 'Quarter',
       'Month', 'Period', 'Date', 'Week','Intercept', 'Median_Base_Price_log', 'TPR_Discount', 'TPR_Discount_lag1', 'TPR_Discount_lag2', 'Catalogue', 'Display', 'ACV', 'SI', 'SI_month', 'SI_quarter', 'C_1_crossretailer_discount', 'C_1_crossretailer_log_price', 'C_1_intra_discount', 'C_2_intra_discount', 'C_3_intra_discount', 'C_4_intra_discount', 'C_5_intra_discount', 'C_1_intra_log_price', 'C_2_intra_log_price', 'C_3_intra_log_price', 'C_4_intra_log_price', 'C_5_intra_log_price', 'Category trend', 'Trend_month', 'Trend_quarter', 'Trend_year', 'month_no', 'Flag_promotype_Motivation', 'Flag_promotype_N_pls_1', 'Flag_promotype_traffic', 'Flag_nonpromo_1', 'Flag_nonpromo_2', 'Flag_nonpromo_3', 'Flag_promo_1', 'Flag_promo_2', 'Flag_promo_3', 'Holiday_Flag1', 'Holiday_Flag2', 'Holiday_Flag3', 'Holiday_Flag4', 'Holiday_Flag5', 'Holiday_Flag6', 'Holiday_Flag7', 'Holiday_Flag8', 'Holiday_Flag9',
        'Holiday_Flag10', 'wk_sold_avg_price_byppg','Average Weight in grams','Weighted Weight in grams']


# coeff_dt = pd.DataFrame(coeff, columns = coeff_columns)
# coeff_dt
# data_dt = pd.DataFrame(data, columns = data_columns)
# data_dt

# main(data_dt,coeff_dt,retailer,ppg,segment)

def list_to_frame(coeff,data,flag=False):
    coeff_dt = pd.DataFrame(coeff, columns = coeff_columns)
    
    if flag:
        co_dt = pd.DataFrame([[i.pop()['co_investment']] for i in data] , columns = ['co_inv'])
        data_dt = pd.DataFrame(data, columns = data_columns)
        data_dt['TPR_Discount'] = data_dt['TPR_Discount'] + co_dt['co_inv']
        
        # import pdb
        # pdb.set_trace()
    else:
        data_dt = pd.DataFrame(data, columns = data_columns)
        
        
        # import pdb
        # pdb.set_trace()
   
    # if 'co_investment' in  data[0][-1]:
    #     data[0] 
    
    
    # import pdb
    # pdb.set_trace()
   
    # import pdb
    # pdb.set_trace()
    
    retuned_dt = main( data_dt,coeff_dt )
    if flag:
        retuned_dt['co_inv'] = co_dt['co_inv']
        # import pdb
        # pdb.set_trace()
    print(retuned_dt['Incremental'][0] , "Incremental")
    print(retuned_dt['Base'][0] , "Base")
    print(retuned_dt['Predicted_sales'][0] , "Predicted_sales")
    return retuned_dt

# main_file(file1,file2,retailer,ppg,segment)