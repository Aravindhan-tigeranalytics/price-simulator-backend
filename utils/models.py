import pandas as pd
from utils import util
from numpy import string_
import math
import decimal
class ScenarioPlannerMetricModel:
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
            
class PromoMeta:
    account_name = ''
    corporate_segment = ''
    product_group = ''
    brand_filter = ''
    brand_format_filter=''
    strategic_cell_filter=''
    
    
class UnitModel:
    def __init__(self,
                 predicted_units = 0.0,
                 on_inv_percent = 0.0,
                 list_price = 0.0,
                 tpr_percent = 0.0,
                 off_inv_percent = 0.0,
                 gmac_percent_lsv = 0.0,
                 average_selling_price = 0.0,
                 product_group_weight_in_grams = 0.0,
                 median_base_price_log = 0.0,
                 incremental_unit = 0.0,
                 base_unit = 0.0,
                 promo_elasticity = 0,
                 co_investment = 0
                 ):
        print(co_investment , "coinvestment in the model")
        # print(off_inv_percent , "off inv")
        # print(on_inv_percent , "on inv ")
    
        self.base_unit = base_unit
        self.incremental_unit = incremental_unit
        self.predicted_units = predicted_units
        # import pdb
        # pdb.set_trace()
        # if promo_elasticity:
        self.simulate_predicted_units = self.base_unit * (
            ((1 - ((tpr_percent + co_investment)/100))** promo_elasticity))
        # self.asp = average_selling_price 
        # import pdb
        # pdb.set_trace()
        self.asp =  decimal.Decimal(math.exp(median_base_price_log)) * decimal.Decimal(
            (1 - ((tpr_percent + co_investment)/100)))
        # self.asp2 = decimal.Decimal(math.exp(median_base_price_log)) * decimal.Decimal((1 - (tpr_percent/100)))
        # self.total_rsv_w_o_vat = self.predicted_units * self.promo
        # total_rsv_w_o_vat = total_rsv_w_o_vat   
        self.total_rsv_w_o_vat = self.predicted_units * (self.asp * decimal.Decimal(1 - (20/100)))
        self.promo_asp = 0 if not tpr_percent else util._divide(self.total_rsv_w_o_vat,self.predicted_units)
        self.uplift_lsv = incremental_unit * list_price
        self.uplift_gmac_lsv = self.uplift_lsv * (gmac_percent_lsv/100)
        self.total_lsv = predicted_units * list_price
        self.mars_uplift_on_invoice = self.uplift_lsv * (on_inv_percent / 100)
        self.mars_total_on_invoice = self.total_lsv * (on_inv_percent / 100)
        self.mars_uplift_nrv = self.uplift_lsv - self.mars_uplift_on_invoice
        self.mars_total_nrv = self.total_lsv - self.mars_total_on_invoice
        self.uplift_promo_cost = self.mars_uplift_nrv * (tpr_percent/100)
        self.tpr_budget_roi = self.mars_total_nrv * (tpr_percent/100)
        self.mars_uplift_net_invoice_price = self.mars_uplift_nrv - self.uplift_promo_cost
        self.mars_total_net_invoice_price = self.mars_total_nrv - self.tpr_budget_roi # changed from tpr budget
        self.mars_uplift_off_invoice = self.mars_uplift_net_invoice_price * (off_inv_percent / 100)
        self.mars_total_off_invoice = self.mars_total_net_invoice_price * (off_inv_percent/100)
        self.uplift_trade_expense = self.mars_uplift_off_invoice + self.tpr_budget_roi + self.mars_uplift_on_invoice
        self.total_trade_expense = self.mars_total_on_invoice + self.tpr_budget_roi + self.mars_total_off_invoice
        self.uplift_nsv = self.uplift_lsv - self.uplift_trade_expense
        self.total_nsv = self.total_lsv - self.total_trade_expense
        self.te_per_units = self.total_trade_expense / self.predicted_units
        # import pdb
        # pdb.set_trace()
        self.uplift_royalty = decimal.Decimal(0.5) * self.uplift_nsv
        self.total_uplift_cost = self.uplift_royalty + self.uplift_trade_expense
        # print( self.uplift_gmac_lsv , "::", self.total_uplift_cost , "::result")
        self.roi = util._divide(self.uplift_gmac_lsv,self.total_uplift_cost)
         
        # self.mars_on_invoice = list_price * (on_inv_percent/100)
        # self.gmac_lsv_per_unit = list_price * (gmac_percent_lsv/100)
        # self.mars_nrv = list_price - self.mars_on_invoice
        self.tpr_budget = self.mars_total_nrv * (tpr_percent/100)
        self.mars_total_net_invoice_price = self.mars_total_nrv - self.tpr_budget
        self.mars_cogs_per_unit = list_price - abs(list_price * (gmac_percent_lsv/100))
        self.uplift_cogs = self.incremental_unit * self.mars_cogs_per_unit
        self.uplift_mac = self.uplift_nsv - self.uplift_cogs
        # uplift_mac = uplift_nsv - uplift_cogs
        
        self.total_cogs = self.mars_cogs_per_unit * predicted_units
        # self.total_nsv = self.total_lsv - self.mars_total_on_invoice - self.tpr_budget - self.mars_total_off_invoice
        self.mars_mac = self.total_nsv - self.total_cogs
        # self.mars_net_invoice_price = self.mars_nrv - (self.mars_nrv * (tpr_percent/100))
        # self.mars_off_invoice = self.mars_net_invoice_price*(off_inv_percent/100)
        # self.mars_nsv = self.mars_net_invoice_price - self.mars_off_invoice
        # self.retailer_mark_up = ((average_selling_price/self.mars_nsv) - 1 ) * 100
        # self.total_rsv_w_o_vat = average_selling_price * predicted_units
        # import pdb
        # pdb.set_trace()
        self.total_weight_in_tons = (predicted_units * product_group_weight_in_grams) /1000000
        self.trade_expense = self.mars_total_on_invoice + self.tpr_budget_roi + self.mars_total_off_invoice
        self.retailer_margin =  self.total_rsv_w_o_vat - self.total_nsv
        self.retailer_margin_percent_of_nsv = (self.retailer_margin / self.total_nsv) * 100
        self.retailer_margin_percent_of_rsp = (self.retailer_margin / self.total_rsv_w_o_vat) * 100
        self.mars_mac_percent_of_nsv = (self.mars_mac/self.total_nsv) * 100
        self.te_percent_of_lsv = (self.trade_expense/self.total_lsv) * 100


class TotalUnit:
    total_rsv_w_o_vat = 0
    units = 0
    te= 0
    lsv = 0
    nsv = 0
    mac = 0
    rp = 0
    asp = 0
    avg_promo_selling_price = 0
    roi = 0
    rp_percent = 0
    mac_percent = 0
    volume = 0
    te_per_unit = 0
    te_percent_of_lsv = 0
    base_units = 0
    increment_units = 0
    lift = 0