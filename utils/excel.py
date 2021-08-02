from numpy import save
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet
from rest_framework import response
import xlsxwriter
import datetime
import openpyxl
import decimal
import io
# from openpyxl import worksheet
from datetime import datetime as dt,timedelta
from core import models as model
from utils import models
from . import constants as const
from utils.models import ScenarioPlannerMetricModel,PromoMeta
from utils import util , roi

import utils

def _get_sheet_value(sheet , row , column):
    value = sheet.cell(row = row, column = column).value
    if(isinstance(value,float)):
        value = str(round(value,12))
    return value
def excel(data , output):
    ROW_CONST = 5
    COL_CONST = 1
#    data = da
#     [ {'product_group': 'ORBIT OTC', 'retailer': 'Magnit', 'category': 'Gum', 'product_group_retailer': 'ORBIT OTC', 'current_lpi': 18.02, 'increased_lpi': 18.02, 'current_rsp': 21.33, 'increased_rsp': 21.33, 'current_cogs': 4.881618, 'increased_cogs': 4.881618, 'lpi_increase': 0, 'rsp_increase': 0, 'cogs_increase': 0, 'base_price_elasticity': '-1.98', 'base_price_elasticity_manual': '-1.98', 'net_elasticity': '-1.53', 'competition': 'Not Follows', 'tonnes': 0, 'rsv': 0, 'mac': 0, 'rp': 0, 'nsv': 0, 'te': 0}, {'product_group': 'ORBIT XXL', 'retailer': 'Magnit', 'category': 'Gum', 'product_group_retailer': 'ORBIT XXL', 'current_lpi': 23.74, 'increased_lpi': 23.74, 'current_rsp': 27.44, 'increased_rsp': 27.44, 'current_cogs': 7.197967999999996, 'increased_cogs': 7.197967999999996, 'lpi_increase': 0, 'rsp_increase': 0, 'cogs_increase': 0, 'base_price_elasticity': '-1.08', 'base_price_elasticity_manual': '-1.08', 'net_elasticity': '-1.08', 'competition': 'Not Follows', 'tonnes': 0, 'rsv': 0, 'mac': 0, 'rp': 0, 'nsv': 0, 'te': 0}, {'product_group': 'BIG BARS', 'retailer': 'Magnit', 'category': 'Choco', 'product_group_retailer': 'BIG BARS', 'current_lpi': 32.46, 'increased_lpi': 32.46, 'current_rsp': 39.19, 'increased_rsp': 39.19, 'current_cogs': 12.097842, 'increased_cogs': 12.097842, 'lpi_increase': 0, 'rsp_increase': 0, 'cogs_increase': 0, 'base_price_elasticity': '-1.17', 'base_price_elasticity_manual': '-1.17', 'net_elasticity': '-0.45', 'competition': 'Not Follows', 'tonnes': 0, 'rsv': 0, 'mac': 0, 'rp': 0, 'nsv': 0, 'te': 0}, {'product_group': 'BIG BARS_SNICKERSCRISPER', 'retailer': 'Magnit', 'category': 'Choco', 'product_group_retailer': 'BIG BARS_SNICKERSCRISPER', 'current_lpi': 21.73, 'increased_lpi': 21.73, 'current_rsp': 28.32, 'increased_rsp': 28.32, 'current_cogs': 11.403904, 'increased_cogs': 11.403904, 'lpi_increase': 0, 'rsp_increase': 0, 'cogs_increase': 0, 'base_price_elasticity': '-1.19', 'base_price_elasticity_manual': '-1.19', 'net_elasticity': '-1.21', 'competition': 'Not Follows', 'tonnes': 0, 'rsv': 0, 'mac': 0, 'rp': 0, 'nsv': 0, 'te': 0}, {'product_group': 'BIG BARS_SNICKERSCRISPER_DUO', 'retailer': 'Magnit', 'category': 'Choco', 'product_group_retailer': 'BIG BARS_SNICKERSCRISPER_DUO', 'current_lpi': 17.68, 'increased_lpi': 17.68, 'current_rsp': 23.52, 'increased_rsp': 23.52, 'current_cogs': 7.667816, 'increased_cogs': 7.667816, 'lpi_increase': 0, 'rsp_increase': 0, 'cogs_increase': 0, 'base_price_elasticity': '-1.67', 'base_price_elasticity_manual': '-1.67', 'net_elasticity': '-1.67', 'competition': 'Not Follows', 'tonnes': 0, 'rsv': 0, 'mac': 0, 'rp': 0, 'nsv': 0, 'te': 0}, {'product_group': 'STD BARS', 'retailer': 'Magnit', 'category': 'Choco', 'product_group_retailer': 'STD BARS', 'current_lpi': 19.89, 'increased_lpi': 19.89, 'current_rsp': 23.8, 'increased_rsp': 23.8, 'current_cogs': 6.969456000000001, 'increased_cogs': 6.969456000000001, 'lpi_increase': 0, 'rsp_increase': 0, 'cogs_increase': 0, 'base_price_elasticity': '-1.80', 'base_price_elasticity_manual': '-1.80', 'net_elasticity': '-1.02', 'competition': 'Not Follows', 'tonnes': 0, 'rsv': 0, 'mac': 0, 'rp': 0, 'nsv': 0, 'te': 0}, {'product_group': 'STD BARS_MILKYWAY', 'retailer': 'Magnit', 'category': 'Choco', 'product_group_retailer': 'STD BARS_MILKYWAY', 'current_lpi': 11.19, 'increased_lpi': 11.19, 'current_rsp': 14.76, 'increased_rsp': 14.76, 'current_cogs': 3.6098939999999997, 'increased_cogs': 3.6098939999999997, 'lpi_increase': 0, 'rsp_increase': 0, 'cogs_increase': 0, 'base_price_elasticity': '-1.28', 'base_price_elasticity_manual': '-1.28', 'net_elasticity': '-0.35', 'competition': 'Not Follows', 'tonnes': 0, 'rsv': 0, 'mac': 0, 'rp': 0, 'nsv': 0, 'te': 0}]
    keys = list(data[0].keys())
    values = []
    for d in data:
        values.append(list(d.values()))
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet()
    worksheet.hide_gridlines(2)
    merge_format_date = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'fg_color': 'yellow'})

    merge_format_app = workbook.add_format({
        'bold': 1,
        
        'align': 'center',
        'valign': 'vcenter'
        })
    merge_format_app.set_font_size(20)
    worksheet.set_column('B:D', 12)
    format2 = workbook.add_format({'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'fg_color': 'blue'})

    format1 = workbook.add_format({'bold': 0,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': 1})
    money = workbook.add_format({'num_format': '₽#,##0'})
    # worksheet.write('A2', 'Item', format2)
    # worksheet.write('B2', 'Cost', format2)
    row = ROW_CONST
    col = COL_CONST

    worksheet.merge_range('B2:D2', 'Downloaded on ' +dateformat() , merge_format_date)
    worksheet.merge_range('B3:E3', 'Price Simulator Tool' , merge_format_app)

    # Iterate over the data and write it out row by row.
    for header in keys:
        worksheet.write(row, col,     " ".join(header.split("_")),format2)
        worksheet.set_row(row, 25)
        worksheet.set_column(col,col, 20)
        col+=1
    row+=1
    col =COL_CONST
    for val in values :
        col = COL_CONST
        for v in val:
            worksheet.write(row, col, " ".join(v.split("_")),format1)
            worksheet.set_row(row, 45)
            worksheet.set_column(col,col, 20)
            col+=1
        row+=1

    workbook.close()
def excel_summary(data , output):
    ROW_CONST = 6
    COL_CONST = 4
    # data = {'sc1sdf': {'name': 'sc1sdf', 'header': ['units', 'tonnes', 'lsv', 'rsv', 'nsv', 'cogs', 'nsv_tonnes', 'te', 'te_percent_lsv', 'te_units', 'mac', 'mac_percent_nsv', 'rp', 'rp_percent_rsv'], 'current': ['136.0M ₽', '2.0K ₽', '2.6B ₽', '3.0B ₽', '1.6B ₽', '709.4M ₽', '815.1K ₽', '968.1M ₽', '37.5 ₽', '7.1 ₽', '907.1M ₽', '56.1 ₽', '1.4B ₽', '45.7 ₽'], 'simulated': ['123.4M ₽', '1.8K ₽', '2.7B ₽', '2.8B ₽', '1.7B ₽', '659.0M ₽', '928.8K ₽', '1.0B ₽', '37.5 ₽', '8.1 ₽', '1.0B ₽', '60.7 ₽', '1.2B ₽', '41.0 ₽'], 'Absolute change': ['-12.6M ₽', '-0.2K ₽', '95.6M ₽', '-0.1B ₽', '59.9M ₽', '-50.4M ₽', '-0.3M ₽', '35.7M ₽', '-0.0 ₽', '-2.8 ₽', '110.3M ₽', '4.6 ₽', '-0.2B ₽', '-4.7 ₽'], 'percent change': ['-9.2 %', '-9.0 %', '3.7 %', '-4.5 %', '3.7 %', '-7.1 %', '-0.4 %', '3.7 %', '-0.0 %', '-0.4 %', '12.2 %', '8.2 %', '-14.3 %', '-10.2 %']}, 'otc&xxl': {'name': 'otc&xxl', 'header': ['units', 'tonnes', 'lsv', 'rsv', 'nsv', 'cogs', 'nsv_tonnes', 'te', 'te_percent_lsv', 'te_units', 'mac', 'mac_percent_nsv', 'rp', 'rp_percent_rsv'], 'current': ['266.1M ₽', '10.2K ₽', '5.8B ₽', '6.9B ₽', '4.2B ₽', '1.9B ₽', '406.8K ₽', '1.6B ₽', '28.2 ₽', '6.1 ₽', '2.3B ₽', '55.0 ₽', '2.8B ₽', '40.1 ₽'], 'simulated': ['241.0M ₽', '9.9K ₽', '5.5B ₽', '6.6B ₽', '4.0B ₽', '1.8B ₽', '405.1K ₽', '1.5B ₽', '27.8 ₽', '6.4 ₽', '2.2B ₽', '55.0 ₽', '2.6B ₽', '39.9 ₽'], 'Absolute change': ['-25.1M ₽', '-0.4K ₽', '-0.3B ₽', '-0.3B ₽', '-0.2B ₽', '-69.5M ₽', '453.9K ₽', '-96.0M ₽', '-0.4 ₽', '3.8 ₽', '-92.3M ₽', '-0.1 ₽', '-0.1B ₽', '-0.2 ₽'], 'percent change': ['-9.4 %', '-3.5 %', '-4.4 %', '-4.3 %', '-3.9 %', '-3.7 %', '1.1 %', '-5.9 %', '-1.5 %', '0.6 %', '-4.0 %', '-0.1 %', '-4.8 %', '-0.6 %']}}
    workbook = xlsxwriter.Workbook(output)
    merge_format_date = workbook.add_format({
        'bold': 1,
        'align': 'center',
        'valign': 'vcenter',
         })

    merge_format_app = workbook.add_format({
        'bold': 1,
        
        'align': 'center',
        'valign': 'vcenter'
        })
    merge_format_app.set_font_size(20)
    worksheet = workbook.add_worksheet()
    worksheet.hide_gridlines(2)
    format_header = workbook.add_format({'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})
    format_header.set_font_size(14)
    format_name = workbook.add_format({'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})
    format_name.set_font_size(20)
    format_value = workbook.add_format({
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})

    row = ROW_CONST
    col = COL_CONST
    
    worksheet.merge_range('B2:D2', 'Downloaded on ' +dateformat() , merge_format_date)
    worksheet.merge_range('B3:D3', 'Price Simulator Tool' , merge_format_app)
    worksheet.merge_range('B4:D4', 'Comparison Summary',merge_format_app)
    zip_list = []
    data_val = list(data.values())
    for i in data_val:
        zip_list.append(zip(i["header"],i["current"],i["simulated"],i["Absolute change"],i["percent change"]))
    for idx,z in enumerate(zip_list):
        worksheet.merge_range(row+1,col-4,row+4,col-2 ,data_val[idx]['name'] ,format_name)
        _writeExcel(worksheet,row+1, col-1,"Current Value",format_header)
        _writeExcel(worksheet,row+2, col-1,"Simulated Value",format_header)
        _writeExcel(worksheet,row+3, col-1,"Absolute Change",format_header)
        _writeExcel(worksheet,row+4, col-1,"Percent Change",format_header)
        for header, current , simulate,absc , per in z:
            _writeExcel(worksheet,row, col," ".join(header.split("_")).title(),format_header)
            _writeExcel(worksheet,row+1, col,current,format_value)
            _writeExcel(worksheet,row+2, col,simulate,format_value)
            _writeExcel(worksheet,row+3, col,absc,format_value)
            _writeExcel(worksheet,row+4, col,per,format_value)
            col+=1  
        row+=6
        col = COL_CONST
    
    
    
    workbook.close()

def download_excel_promo(data):
    output = io.BytesIO()
    no_format_header = ['date', 'week']
    currency_header = ['asp', 'total_rsv_w_o_vat', 'promo_asp','total_lsv','total_nsv','mars_mac', 'trade_expense', 'retailer_margin','avg_promo_selling_price','lsv','nsv','mac','te','rp']
    percent_header = [ 'retailer_margin_percent_of_nsv','mars_mac_percent_of_nsv','te_percent_of_lsv'
    ,'rp_percent','mac_percent','roi']

    ROW_CONST = 6
    COL_CONST = 1

    # from scenario_planner import test as t
    # data = t.RESPONSE_PROMO[0]
    # import pdb
    # pdb.set_trace()

    workbook = xlsxwriter.Workbook(output)
    merge_format_date = workbook.add_format({
        'bold': 1,
        'align': 'center',
        'valign': 'vcenter',
         })

    merge_format_app = workbook.add_format({
        'bold': 1,
        'align': 'center',
        'valign': 'vcenter'
        })
    merge_format_app.set_font_size(20)
   
    worksheet = workbook.add_worksheet()
    worksheet.hide_gridlines(2)
    format_header = workbook.add_format({'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})
    format_header.set_font_size(14)
    
    format_name = workbook.add_format({'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})
    format_name.set_font_size(20)
    format_value = workbook.add_format({
        'border': 1,
        'align': 'center',
        'text_wrap': True,
        'valign': 'vcenter'})
    format_value_left = workbook.add_format({
    'border': 1,
    'align': 'left',
    'text_wrap': True,
    'valign': 'vcenter'})
    format_value_left.set_indent(6)
    # summary_value_format = workbook.add_format({
    #     'border': 1,
    #     'align': 'left',
    #     'text_wrap': True,
    #     'valign': 'vcenter'})
    # summary_value_format.set_font_size(14)
    # summary_value_format.set_indent(6)
    summary_value_bold = workbook.add_format({'bold': True})
    summary_value_bold.set_font_size(14)

    row = ROW_CONST
    col = COL_CONST

    format_value_percentage = workbook.add_format({ 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '0.00 %' })

    format_value_currency = workbook.add_format({ 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '[<999950]0.0,"K ₽";[<999950000]0.0,,"M ₽";0.0,,,"B ₽"' })

    format_value_number = workbook.add_format({ 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '[<999950]0.0,"K";[<999950000]0.0,,"M";0.0,,,"B"' })

    summary_value_percentage = workbook.add_format({ 'bold': 1, 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '0.00 %' })
    summary_value_percentage.set_font_size(14)

    summary_value_currency = workbook.add_format({ 'bold': 1, 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '[<999950]0.0,"K ₽";[<999950000]0.0,,"M ₽";0.0,,,"B ₽"' })
    summary_value_currency.set_font_size(14)

    summary_value_number = workbook.add_format({ 'bold': 1, 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '[<999950]0.0,"K";[<999950000]0.0,,"M";0.0,,,"B"' })
    summary_value_number.set_font_size(14)
    
    worksheet.merge_range('B2:D2', 'Downloaded on {}'.format(dateformat()) , merge_format_date)
    worksheet.merge_range('B3:D3', 'Promo Simulator Tool' , merge_format_app)
    worksheet.set_column('B:D', 20)
    worksheet.merge_range('B4:D4', "Account Name : {}".format(data['account_name']),merge_format_app)
    worksheet.merge_range('B5:D5', "Product Group : {}".format(data['product_group']),merge_format_app)
    
    # data_val = data['simulated']['weekly'][0]

    header_key =  ['date','week','promotions', 'predicted_units','base_unit','incremental_unit','total_weight_in_tons','total_lsv','total_nsv','mars_mac_percent_of_nsv','trade_expense','te_percent_of_lsv', 'te_per_units','roi','asp','promo_asp','total_rsv_w_o_vat', 'retailer_margin','retailer_margin_percent_of_nsv', 'mars_mac', ]

    header_title = ['Date','Week','Promotions(Base)','Promotions(Simulated)','Units(Base)','Units(Simulated)','Base units(Base)','Base units(Simulated)','Incremental units(Base)',
    'Incremental units(Simulated)','Volume(Base)','Volume(Simulated)','LSV(Base)','LSV(Simulated)','NSV(Base)','NSV(Simulated)',
    'MAC, %NSV(Base)','MAC, %NSV(Simulated)','Trade expense(Base)','Trade expense(Simulated)','TE, % LSV(Base)','TE, % LSV(Simulated)',
    'TE / Unit(Base)','TE / Unit(Simulated)','ROI(Base)','ROI(Simulated)','ASP(Base)','ASP(Simulated)','Promo ASP(Base)','Promo ASP(Simulated)',
    'RSV w/o VAT(Base)','RSV w/o VAT(Simulated)','Customer Margin(Base)','Customer Margin(Simulated)','Customer Margin,%RSV(Base)',
    'Customer Margin,%RSV(Simulated)','Mars MAC(Base)','Mars MAC(Simulated)']

    # for key in data_val.keys():
    for key in header_title:
        # header_key.append(key)
        _writeExcel(worksheet,row, col,key,format_header)
        col+=1

    col = COL_CONST
    row+=1

    simulated_weekly = data['simulated']['weekly']
    base_weekly = data['base']['weekly']
    bold = workbook.add_format({'bold': True})
    red = workbook.add_format({'color': 'red'})
    for base,simulated in zip(base_weekly,simulated_weekly):
        for k in header_key:
            if k == 'date' or k == 'week':
                value = simulated[k]
                _writeExcel(worksheet,row, col, value ,format_value)
                col+=1
            elif k == 'promotions':
                promotion_value = util.format_promotions(
                    base['flag_promotype_motivation'],
                    base['flag_promotype_n_pls_1'],
                    base['flag_promotype_traffic'],
                    base['promo_depth'],
                    base['co_investment']
                )
                promotion_value_simulated = util.format_promotions(
                    simulated['flag_promotype_motivation'],
                    simulated['flag_promotype_n_pls_1'],
                    simulated['flag_promotype_traffic'],
                    simulated['promo_depth'],
                    simulated['co_investment']
                )
                _writeExcel(worksheet,row, col, promotion_value ,format_value)
                col+=1
                _writeExcel(worksheet,row, col, promotion_value_simulated ,format_value)
                col+=1
            elif k in percent_header:
                _writeExcel(worksheet,row, col, base[k]/100, format_value_percentage)
                col+=1
                _writeExcel(worksheet,row, col, simulated[k]/100, format_value_percentage)
                col+=1
            elif k in currency_header:
                _writeExcel(worksheet,row, col, base[k], format_value_currency)
                col+=1
                _writeExcel(worksheet,row, col, simulated[k], format_value_currency)
                col+=1
            else:
                # diff_value = util.format_value(base[k]-simulated[k], k in percent_header , k in currency_header , k in no_format_header)
                # segments = ["Base: " +str(base_value)+ "\n", bold, "Simulated: "+str(simulated_value) + " " ,red ,"("+diff_value+")" ]
                # worksheet.write_rich_string(row, col, *segments,format_value_left)
                _writeExcel(worksheet,row, col, base[k], format_value_number)
                col+=1
                _writeExcel(worksheet,row, col, simulated[k], format_value_number)
                col+=1
        row+=1
        col = COL_CONST

    simulated_total = data['simulated']['total']
    base_total = data['base']['total']

    total_header = ['units','base_units','increment_units','volume','lsv','nsv','mac_percent','te','te_percent_of_lsv','te_per_unit','roi','asp','avg_promo_selling_price','total_rsv_w_o_vat','rp','rp_percent','mac']
    worksheet.merge_range('B{}:E{}'.format(row+1,row+1), 'Total ' , format_header)
    col = 3
    for k in total_header:
        # base_total_value = util.format_value(base_total[k], k in percent_header , k in currency_header , k in no_format_header)
        # simulated_total_value = util.format_value(simulated_total[k], k in percent_header , k in currency_header , k in no_format_header)
        # diff_total_value = util.format_value(base_total[k]-simulated_total[k], k in percent_header , k in currency_header , k in no_format_header)
        # segments = ["Base: " +str(base_total_value)+ "    " + "\n", summary_value_bold, "Simulated: "+str(simulated_total_value)+ " " ,red ,"("+diff_total_value+")"]
        # worksheet.write_rich_string(row, col, *segments,summary_value_format)
        # _writeExcel(worksheet,row, col,util.format_value(total[k], k in percent_header , k in currency_header , k in no_format_header),format_header)
        if k in percent_header:
            _writeExcel(worksheet,row, col,base_total[k]/100,summary_value_percentage)
            col+=1
            _writeExcel(worksheet,row, col,simulated_total[k]/100,summary_value_percentage)
            col+=1
        elif k in currency_header:
            _writeExcel(worksheet,row, col,base_total[k],summary_value_currency)
            col+=1
            _writeExcel(worksheet,row, col,simulated_total[k],summary_value_currency)
            col+=1
        else:
            _writeExcel(worksheet,row, col,base_total[k],summary_value_number)
            col+=1
            _writeExcel(worksheet,row, col,simulated_total[k],summary_value_number)
            col+=1
    col = COL_CONST
    
    workbook.close()
    output.seek(0)
    return output

def download_excel_promo_compare(data):
    no_format_header = ['date', 'week' ]
    currency_header = ['tpr_budget_roi', 'mars_uplift_net_invoice_price', 'mars_total_net_invoice_price',
                       'mars_uplift_off_invoice', 'mars_total_off_invoice', 'uplift_trade_expense', 
                       'total_trade_expense','uplift_nsv', 'total_nsv', 'uplift_royalty', 'total_uplift_cost', 'roi', 'tpr_budget',
                        'asp', 'total_rsv_w_o_vat', 'promo_asp', 'uplift_lsv','uplift_gmac_lsv', 'total_lsv', 'mars_uplift_on_invoice', 'mars_total_on_invoice',
                       'mars_uplift_nrv', 'mars_total_nrv', 'uplift_promo_cost',
                        'uplift_cogs', 'uplift_mac', 'total_cogs', 'mars_mac', 'total_weight_in_tons', 'trade_expense', 'retailer_margin',]
    percent_header = [ 'retailer_margin_percent_of_nsv', 'retailer_margin_percent_of_rsp', 
                      'mars_mac_percent_of_nsv', 'te_percent_of_lsv', 'mars_cogs_per_unit', 'te_per_units',] 
    output = io.BytesIO()
   
    ROW_CONST = 7
    COL_CONST = 1
    from . import test
    data = test.RESPONSE_PROMO
    # import pdb
    # pdb.set_trace()
    # data = {'sc1sdf': {'name': 'sc1sdf', 'header': ['units', 'tonnes', 'lsv', 'rsv', 'nsv', 'cogs', 'nsv_tonnes', 'te', 'te_percent_lsv', 'te_units', 'mac', 'mac_percent_nsv', 'rp', 'rp_percent_rsv'], 'current': ['136.0M ₽', '2.0K ₽', '2.6B ₽', '3.0B ₽', '1.6B ₽', '709.4M ₽', '815.1K ₽', '968.1M ₽', '37.5 ₽', '7.1 ₽', '907.1M ₽', '56.1 ₽', '1.4B ₽', '45.7 ₽'], 'simulated': ['123.4M ₽', '1.8K ₽', '2.7B ₽', '2.8B ₽', '1.7B ₽', '659.0M ₽', '928.8K ₽', '1.0B ₽', '37.5 ₽', '8.1 ₽', '1.0B ₽', '60.7 ₽', '1.2B ₽', '41.0 ₽'], 'Absolute change': ['-12.6M ₽', '-0.2K ₽', '95.6M ₽', '-0.1B ₽', '59.9M ₽', '-50.4M ₽', '-0.3M ₽', '35.7M ₽', '-0.0 ₽', '-2.8 ₽', '110.3M ₽', '4.6 ₽', '-0.2B ₽', '-4.7 ₽'], 'percent change': ['-9.2 %', '-9.0 %', '3.7 %', '-4.5 %', '3.7 %', '-7.1 %', '-0.4 %', '3.7 %', '-0.0 %', '-0.4 %', '12.2 %', '8.2 %', '-14.3 %', '-10.2 %']}, 'otc&xxl': {'name': 'otc&xxl', 'header': ['units', 'tonnes', 'lsv', 'rsv', 'nsv', 'cogs', 'nsv_tonnes', 'te', 'te_percent_lsv', 'te_units', 'mac', 'mac_percent_nsv', 'rp', 'rp_percent_rsv'], 'current': ['266.1M ₽', '10.2K ₽', '5.8B ₽', '6.9B ₽', '4.2B ₽', '1.9B ₽', '406.8K ₽', '1.6B ₽', '28.2 ₽', '6.1 ₽', '2.3B ₽', '55.0 ₽', '2.8B ₽', '40.1 ₽'], 'simulated': ['241.0M ₽', '9.9K ₽', '5.5B ₽', '6.6B ₽', '4.0B ₽', '1.8B ₽', '405.1K ₽', '1.5B ₽', '27.8 ₽', '6.4 ₽', '2.2B ₽', '55.0 ₽', '2.6B ₽', '39.9 ₽'], 'Absolute change': ['-25.1M ₽', '-0.4K ₽', '-0.3B ₽', '-0.3B ₽', '-0.2B ₽', '-69.5M ₽', '453.9K ₽', '-96.0M ₽', '-0.4 ₽', '3.8 ₽', '-92.3M ₽', '-0.1 ₽', '-0.1B ₽', '-0.2 ₽'], 'percent change': ['-9.4 %', '-3.5 %', '-4.4 %', '-4.3 %', '-3.9 %', '-3.7 %', '1.1 %', '-5.9 %', '-1.5 %', '0.6 %', '-4.0 %', '-0.1 %', '-4.8 %', '-0.6 %']}}
    workbook = xlsxwriter.Workbook(output)
    merge_format_date = workbook.add_format({
        'bold': 1,
        'align': 'center',
        'valign': 'vcenter',
         })

    merge_format_app = workbook.add_format({
        'bold': 1,
        
        'align': 'center',
        'valign': 'vcenter'
        })
    merge_format_app.set_font_size(20)
   
    worksheet = workbook.add_worksheet()
    worksheet.hide_gridlines(2)
    format_header = workbook.add_format({'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})
    format_header.set_font_size(14)
    
    format_name = workbook.add_format({'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})
    format_name.set_font_size(20)
    format_value = workbook.add_format({
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})

    row = ROW_CONST
    col = COL_CONST
    
    
    worksheet.merge_range('B2:D2', 'Downloaded on {}'.format(dateformat()) , merge_format_date)
    worksheet.merge_range('B3:D3', 'Promo Simulator Tool' , merge_format_app)
    worksheet.set_column('B:D', 20)
    worksheet.merge_range('B4:D4', "Account Name : {}".format(data['account_name']),merge_format_app)
    worksheet.merge_range('B5:D5', "Product Group : {}".format(data['product_group']),merge_format_app)
    worksheet.merge_range("B6:D6", "Scenario 1",merge_format_date)
    
    data_val = data['simulated']['weekly'][0]
    header_key = []
    for key in data_val.keys():
        header_key.append(key)
        _writeExcel(worksheet,row, col," ".join(key.split("_")).title(),format_header)
        # _writeExcel(worksheet,row+1, col,data_val[key],format_value)
        col+=1
    print(header_key , "header key ")
    col = COL_CONST
    row+=1
    weekly = data['simulated']['weekly']
     
    for week in weekly:
        for k in header_key:
            _writeExcel(worksheet,row, col,
                        week[k] if k =='date' else util.format_value(
                            week[k], k in percent_header , k in currency_header , k in no_format_header
                            ),
                        format_value)
            col+=1
        row+=1
        col = COL_CONST
    worksheet.merge_range("B{}:D{}".format(row+1 , row+1), "Scenario 2",merge_format_date)
    row+=2
    for week in weekly:
        for k in header_key:
            # week[k].strftime("%b %d %Y")
            _writeExcel(worksheet,row, col,
                        week[k] if k =='date' else util.format_value(
                            week[k], k in percent_header , k in currency_header , k in no_format_header
                            ),
                        format_value)
            col+=1
        row+=1
        col = COL_CONST

         
    workbook.close()
    output.seek(0)
    return output




def download_excel_optimizer(account_name , product_group,data):
    output = io.BytesIO()
   
    ROW_CONST = 6
    COL_CONST = 1
    no_format_header = ['ROI']
    currency_header = ['Avg_PromoSellingPrice','Trade_Expense','MAC','RP','AvgSellingPrice','Sales','GSV', 'NSV']
    percent_header = ['RP_Perc', 'Mac_Perc']

    currency_header_weekly = ['asp', 'total_rsv_w_o_vat', 'promo_asp','total_lsv','total_nsv','mars_mac', 'trade_expense', 'retailer_margin','avg_promo_selling_price','lsv','nsv','mac','te','rp']
    percent_header_weekly = [ 'retailer_margin_percent_of_nsv','mars_mac_percent_of_nsv','te_percent_of_lsv'
    ,'rp_percent','mac_percent','roi']

    # from optimiser import testdata as test
    # data = test.RESPONSE_OPTIMIZER
    
    summary_data = data['summary']
    optimal_data = data['optimal']
    holiday_data = data['holiday']

    simulated_weekly = data['financial_metrics']['simulated']['weekly']
    base_weekly = data['financial_metrics']['base']['weekly']

    workbook = xlsxwriter.Workbook(output)
    merge_format_date = workbook.add_format({
        'bold': 1,
        'align': 'center',
        'valign': 'vcenter',
         })

    merge_format_app = workbook.add_format({
        'bold': 1,
        'align': 'center',
        'valign': 'vcenter'
        })
    merge_format_app.set_font_size(20)
   
    worksheet = workbook.add_worksheet('Summary')
    worksheet.hide_gridlines(2)
    format_header = workbook.add_format({'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})
    format_header.set_font_size(14)
    format_name = workbook.add_format({'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})
    format_name.set_font_size(20)
    format_value = workbook.add_format({
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})

    row = ROW_CONST
    col = COL_CONST
    
    
    worksheet.merge_range('B2:D2', 'Downloaded on {}'.format(dateformat()) , merge_format_date)
    worksheet.merge_range('B3:D3', 'Promo Optimizer Tool' , merge_format_app)
    worksheet.set_column('B:D', 20)
    worksheet.merge_range('B4:D4', "Account Name : {}".format(account_name),merge_format_app)
    worksheet.merge_range('B5:D5', "Product Group : {}".format(product_group),merge_format_app)
    
    data_val = [d['Metric'] for d in summary_data]
    header_key = []
    for val in data_val:
        # header_key.append(key)
        _writeExcel(worksheet,row, col," ".join(val.split("_")).title(),format_header)
        # _writeExcel(worksheet,row+1, col,data_val[key],format_value)
        col+=1
    col = COL_CONST
    row+=1
    _writeExcel(worksheet,row, col-1,'Base Scenario',format_header)
    _writeExcel(worksheet,row+1, col-1,'Recommended Scenario',format_header)
    _writeExcel(worksheet,row+2, col-1,'Change',format_header)
    _writeExcel(worksheet,row+3, col-1,'Delta',format_header)
    for kv in summary_data: 
        _writeExcel(worksheet,row, col,util.format_value(kv['Base_Scenario'], kv['Metric'] in percent_header , kv['Metric'] in currency_header , kv['Metric'] in no_format_header),format_value)
        _writeExcel(worksheet,row+1, col,util.format_value(kv['Recommended_Scenario'], kv['Metric'] in percent_header , kv['Metric'] in currency_header , kv['Metric'] in no_format_header),format_value)
        _writeExcel(worksheet,row+2, col,util.format_value(kv['Change'], kv['Metric'] in percent_header , kv['Metric'] in currency_header , kv['Metric'] in no_format_header),format_value)
        _writeExcel(worksheet,row+3, col,util.format_value(kv['Delta'], kv['Metric'] in percent_header , kv['Metric'] in currency_header , kv['Metric'] in no_format_header),format_value)
        # row+=1
        col+=1

    row = ROW_CONST
    col = COL_CONST
    weekly_worksheet = workbook.add_worksheet('Weekly')
    weekly_worksheet.hide_gridlines(2)
    weekly_worksheet.merge_range('A3:B3', "Optimizer Weekly Data",merge_format_app)
    weekly_worksheet.merge_range('A4:B4', "Account Name : {}".format(account_name),merge_format_app)
    weekly_worksheet.merge_range('A5:B5', "Product Group : {}".format(product_group),merge_format_app)

    row+=1

    format_value_percentage = workbook.add_format({ 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '0.00 %' })

    format_value_currency = workbook.add_format({ 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '[<999950]0.0,"K ₽";[<999950000]0.0,,"M ₽";0.0,,,"B ₽"' })

    format_value_number = workbook.add_format({ 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '[<999950]0.0,"K";[<999950000]0.0,,"M";0.0,,,"B"' })
    
    summary_value_percentage = workbook.add_format({ 'bold': 1, 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '0.00 %' })
    summary_value_percentage.set_font_size(14)

    summary_value_currency = workbook.add_format({ 'bold': 1, 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '[<999950]0.0,"K ₽";[<999950000]0.0,,"M ₽";0.0,,,"B ₽"' })
    summary_value_currency.set_font_size(14)

    summary_value_number = workbook.add_format({ 'bold': 1, 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '[<999950]0.0,"K";[<999950000]0.0,,"M";0.0,,,"B"' })
    summary_value_number.set_font_size(14)
    # optimal_header = ['Date','SI','Baseline_Promo','Baseline_Units','Baseline_Base','Baseline_Incremental','Baseline_ROI','Baseline_Lift','Optimum_Promo','Optimum_Units','Optimum_Base','Optimum_Incremental','Optimum_ROI','Optimum_Lift']
    # if len(holiday_data) > 0:
    #     optimal_header = optimal_header + holiday_data

    header_key =  ['date','week', 'promotions','predicted_units','base_unit','incremental_unit','total_weight_in_tons','total_lsv','total_nsv','mars_mac_percent_of_nsv','trade_expense','te_percent_of_lsv', 'te_per_units','roi','asp','promo_asp','total_rsv_w_o_vat', 'retailer_margin','retailer_margin_percent_of_nsv', 'mars_mac', ]
    optimal_header = ['Date','Week','Promotions(Base)','Promotions(Simulated)','Units(Base)','Units(Simulated)','Base units(Base)','Base units(Simulated)','Incremental units(Base)',
    'Incremental units(Simulated)','Volume(Base)','Volume(Simulated)','LSV(Base)','LSV(Simulated)','NSV(Base)','NSV(Simulated)',
    'MAC, %NSV(Base)','MAC, %NSV(Simulated)','Trade expense(Base)','Trade expense(Simulated)','TE, % LSV(Base)','TE, % LSV(Simulated)',
    'TE / Unit(Base)','TE / Unit(Simulated)','ROI(Base)','ROI(Simulated)','ASP(Base)','ASP(Simulated)','Promo ASP(Base)','Promo ASP(Simulated)',
    'RSV w/o VAT(Base)','RSV w/o VAT(Simulated)','Customer Margin(Base)','Customer Margin(Simulated)','Customer Margin,%RSV(Base)',
    'Customer Margin,%RSV(Simulated)','Mars MAC(Base)','Mars MAC(Simulated)']

    for key in optimal_header:
        _writeExcel(weekly_worksheet,row, col,key,format_header)
        col+=1
    row+=1
    col = COL_CONST

    # number_format_header = ['Baseline_Units','Baseline_Base','Baseline_Incremental','Optimum_Units','Optimum_Base','Optimum_Incremental']
    # for data in optimal_data:
    #     for k in optimal_header:
    #         if k in number_format_header:
    #             value = datetime.datetime.fromtimestamp(int(str(data[k])[0:10])).strftime('%Y-%m-%d') if k =='Date' else data[k]
    #             _writeExcel(weekly_worksheet,row, col, util.format_value(value, k in percent_header , k in currency_header , k in no_format_header) ,format_value)
    #         else:
    #             _writeExcel(weekly_worksheet,row, col, datetime.datetime.fromtimestamp(int(str(data[k])[0:10])).strftime('%Y-%m-%d') if k =='Date' else data[k],format_value)
    #         col+=1
    #     row+=1
    #     col = COL_CONST
    for base,simulated in zip(base_weekly,simulated_weekly):
        for k in header_key:
            if k == 'date' or k == 'week':
                _writeExcel(weekly_worksheet,row, col, datetime.datetime.fromtimestamp(int(str(simulated[k])[0:10])).strftime('%Y-%m-%d') if k =='Date' else simulated[k] ,format_value)
                col+=1
            elif k == 'promotions':
                promotion_value = util.format_promotions(
                    base['flag_promotype_motivation'],
                    base['flag_promotype_n_pls_1'],
                    base['flag_promotype_traffic'],
                    base['promo_depth'],
                    base['co_investment']
                )
                promotion_value_simulated = util.format_promotions(
                    simulated['flag_promotype_motivation'],
                    simulated['flag_promotype_n_pls_1'],
                    simulated['flag_promotype_traffic'],
                    simulated['promo_depth'],
                    simulated['co_investment']
                )
                _writeExcel(weekly_worksheet,row, col, promotion_value ,format_value)
                col+=1
                _writeExcel(weekly_worksheet,row, col, promotion_value_simulated ,format_value)
                col+=1
            elif k in percent_header_weekly:
                _writeExcel(weekly_worksheet,row, col, base[k]/100, format_value_percentage)
                col+=1
                _writeExcel(weekly_worksheet,row, col, simulated[k]/100, format_value_percentage)
                col+=1
            elif k in currency_header_weekly:
                _writeExcel(weekly_worksheet,row, col, base[k], format_value_currency)
                col+=1
                _writeExcel(weekly_worksheet,row, col, simulated[k], format_value_currency)
                col+=1
            else:
                _writeExcel(weekly_worksheet,row, col, base[k], format_value_number)
                col+=1
                _writeExcel(weekly_worksheet,row, col, simulated[k], format_value_number)
                col+=1
        row+=1
        col = COL_CONST
    # row = 7
    # weekly_worksheet.write('A{}'.format(row+1), "Week",format_header)
    # row+=1
    # for i in range(0,len(optimal_data)):
    #     _writeExcel(weekly_worksheet,row, col-1, 'Week-{}'.format(i+1),format_value)
    #     row+=1

    simulated_total = data['financial_metrics']['simulated']['total']
    base_total = data['financial_metrics']['base']['total']

    total_header = ['units','base_units','increment_units','volume','lsv','nsv','mac_percent','te','te_percent_of_lsv','te_per_unit','roi','asp','avg_promo_selling_price','total_rsv_w_o_vat','rp','rp_percent','mac']
    weekly_worksheet.merge_range('B{}:E{}'.format(row+1,row+1), 'Total ' , format_header)
    col = 3
    for k in total_header:
        if k in percent_header_weekly:
            _writeExcel(weekly_worksheet,row, col,base_total[k]/100,summary_value_percentage)
            col+=1
            _writeExcel(weekly_worksheet,row, col,simulated_total[k]/100,summary_value_percentage)
            col+=1
        elif k in currency_header_weekly:
            _writeExcel(weekly_worksheet,row, col,base_total[k],summary_value_currency)
            col+=1
            _writeExcel(weekly_worksheet,row, col,simulated_total[k],summary_value_currency)
            col+=1
        else:
            _writeExcel(weekly_worksheet,row, col,base_total[k],summary_value_number)
            col+=1
            _writeExcel(weekly_worksheet,row, col,simulated_total[k],summary_value_number)
            col+=1
    col = COL_CONST

    workbook.close()
    output.seek(0)
    return output


def _writeExcel(worksheet , row , col , val , _format):
    worksheet.write(row, col,val,_format)
    worksheet.set_row(row, 30)
    worksheet.set_column(col,col, 45)


def dateformat():

    x = datetime.datetime.now()
    return x.strftime("%b %d %Y %H:%M:%S")
    
def read_promo_coeff(file):
    headers = const.COEFF_HEADER
    book = openpyxl.load_workbook(file,data_only=True)
    sheet = book['MODEL_COEFFICIENT']
    columns = sheet.max_column
    rows = sheet.max_row
    # print(columns , rows , "columns and rows")
    col_ = [i for i in range(1,len(headers)+1)]
    row_ =0
    for row in range(row_+2 , rows+1):
        db_meta = model.ModelMeta()
        db_coeff = model.ModelCoefficient()
        for c in range(0,len(col_)):
            
            # obj = {}
            # print(row , c," :rowcvalue")'Account Name' , 'Corporate Segment' , 'PPG'
            cell_obj = sheet.cell(row = row,column = col_[c])
            if(headers[c] in const.PROMO_MODEL_META_MAP):
                # print(headers[c] , const.PROMO_MODEL_META_MAP[headers[c]],cell_obj.value , "generated value")
                setattr(db_meta,const.PROMO_MODEL_META_MAP[headers[c]],cell_obj.value)
            elif(headers[c] in const.PROMO_MODEL_COEFF_MAP):
                setattr(db_coeff,const.PROMO_MODEL_COEFF_MAP[headers[c]],
                        cell_obj.value if cell_obj.value else 0.0)
            
        # print(db_meta.product_group , db_meta.corporate_segment , db_meta.account_name , "from object before save")
        db_meta.slug = util.generate_slug_string(db_meta.account_name,db_meta.corporate_segment,db_meta.product_group)
        if not model.ModelMeta.objects.filter(slug=db_meta.slug).exists():
            db_meta.save()
        
        db_coeff.model_meta = model.ModelMeta.objects.filter(
            account_name=db_meta.account_name,corporate_segment=db_meta.corporate_segment,
            product_group=db_meta.product_group).first()
        db_coeff.save()
    book.close()

def _get_col_map(rows , columns , sheet , headers):
    col_ = {}
    header_found = False
    for row in range(1,rows+1):
        
        for col in range(1,columns+1):
            
            cell_obj = sheet.cell(row = row, column = col)
            if cell_obj.value in headers:
                header_found = True
                # col_.append(col)
                col_[cell_obj.value] = col
        if header_found:
           break 
    return col_

def _get_col(rows , columns , sheet , headers):
    col_ = []
    header_found = False
    for row in range(1,rows+1):
        
        for col in range(1,columns+1):
            
            cell_obj = sheet.cell(row = row, column = col)
            if cell_obj.value in headers:
                header_found = True
                col_.append(col)
        if header_found:
           break 
    return col_
    # print(col_ , "coldddd")
def read_promo_data_bkp(file):
    headers = const.DATA_HEADER
    book = openpyxl.load_workbook(file,data_only=True)
    sheet = book['MODEL_DATA']
    columns = sheet.max_column
    rows = sheet.max_row
    col_map = _get_col_map(rows,columns,sheet,headers)
    row_ =0
    col_ = []
    validation_dict = {}
    for row in range(row_+2 , rows+1):
        db_meta = PromoMeta()
        db_data = model.ModelData()
        db_data.year = sheet.cell(row = row,column = col_map['Year'])
        for c in range(0,len(col_)):
            cell_obj = sheet.cell(row = row,column = col_[c])
            if(headers[c] in const.PROMO_MODEL_META_MAP):
                setattr(db_meta,const.PROMO_MODEL_META_MAP[headers[c]],cell_obj.value)
            elif(headers[c] in const.PROMO_MODEL_DATA_MAP):
                if(isinstance(cell_obj.value,float)):
                    setattr(db_data,const.PROMO_MODEL_DATA_MAP[headers[c]],str(round(cell_obj.value,12)) if cell_obj.value else 0.0)
                else:
                    val = bool(cell_obj.value) if headers[c] == 'Optimiser_flag' else cell_obj.value
                    setattr(db_data,const.PROMO_MODEL_DATA_MAP[headers[c]],val)
        
        if not db_meta.account_name:
            break
        
        slug = util.generate_slug_string(db_meta.account_name,
                                           db_meta.corporate_segment,
                                           db_meta.product_group)
        
        db_data.model_meta = model.ModelMeta.objects.get(    
                account_name=db_meta.account_name,corporate_segment=db_meta.corporate_segment,
            product_group=db_meta.product_group
        )
        db_data.full_clean()
        db_data.save()
        if slug not in validation_dict:
            validation_dict[slug] = {
                'count' : 1,
                'account_name' : db_meta.account_name,
                'corporate_segment' : db_meta.corporate_segment,
                'product_group' : db_meta.product_group
            }
        else:
            validation_dict[slug]['count'] = validation_dict[slug]['count'] + 1
    book.close()
    return validation_dict

def _update_model_data_object(model_data : model.ModelData , sheet:Worksheet,row:int,col_map):
    # cell_obj = sheet.cell(row = row,column = col_[c])
    # cellobj.value
    model_data.year = _get_sheet_value(sheet , row,col_map['Year'])
    model_data.quater = _get_sheet_value(sheet , row,col_map['Quarter'])
    model_data.month = _get_sheet_value(sheet , row,col_map['Month'])
    model_data.period = _get_sheet_value(sheet , row,col_map['Period'])
    model_data.week = _get_sheet_value(sheet , row,col_map['Week'])
    model_data.date = _get_sheet_value(sheet , row,col_map['Date'])
    # model_data.wk_sold_avg_price_byppg = _get_sheet_value(sheet , row,col_map['Year'])
    model_data.promo_depth = _get_sheet_value(sheet , row,col_map['Promo_Depth'])
    model_data.co_investment = _get_sheet_value(sheet , row,col_map['Coinvestment'])
    model_data.average_weight_in_grams = _get_sheet_value(sheet , row,col_map['Average Weight in grams'])
    model_data.weighted_weight_in_grams = _get_sheet_value(sheet , row,col_map['Weighted Weight in grams'])
    model_data.optimiser_flag = bool(_get_sheet_value(sheet , row,col_map['Optimiser_flag']))
    model_data.intercept = _get_sheet_value(sheet , row,col_map['Intercept'])
    model_data.median_base_price_log = _get_sheet_value(sheet , row,col_map['Median_Base_Price_log'])
    model_data.tpr_discount = _get_sheet_value(sheet , row,col_map['TPR_Discount'])
    model_data.tpr_discount_lag1 = _get_sheet_value(sheet , row,col_map['TPR_Discount_lag1'])
    model_data.tpr_discount_lag2 = _get_sheet_value(sheet , row,col_map['TPR_Discount_lag2'])
    model_data.catalogue = _get_sheet_value(sheet , row,col_map['Catalogue'])
    model_data.display = _get_sheet_value(sheet , row,col_map['Display'])
    model_data.acv = _get_sheet_value(sheet , row,col_map['ACV'])
    model_data.si = _get_sheet_value(sheet , row,col_map['SI'])
    model_data.si_month = _get_sheet_value(sheet , row,col_map['SI_month'])
    model_data.si_quarter = _get_sheet_value(sheet , row,col_map['SI_quarter'])
    model_data.c_1_crossretailer_discount = _get_sheet_value(sheet , row,col_map['C_1_crossretailer_discount'])
    model_data.c_1_crossretailer_log_price = _get_sheet_value(sheet , row,col_map['C_1_crossretailer_log_price'])
    model_data.c_1_intra_discount = _get_sheet_value(sheet , row,col_map['C_1_intra_discount'])
    model_data.c_2_intra_discount = _get_sheet_value(sheet , row,col_map['C_2_intra_discount'])
    model_data.c_3_intra_discount = _get_sheet_value(sheet , row,col_map['C_3_intra_discount'])
    model_data.c_4_intra_discount = _get_sheet_value(sheet , row,col_map['C_4_intra_discount'])
    model_data.c_5_intra_discount = _get_sheet_value(sheet , row,col_map['C_5_intra_discount'])
    model_data.c_1_intra_log_price = _get_sheet_value(sheet , row,col_map['C_1_intra_log_price'])
    model_data.c_2_intra_log_price = _get_sheet_value(sheet , row,col_map['C_2_intra_log_price'])
    model_data.c_3_intra_log_price = _get_sheet_value(sheet , row,col_map['C_3_intra_log_price'])
    model_data.c_4_intra_log_price = _get_sheet_value(sheet , row,col_map['C_4_intra_log_price'])
    model_data.c_5_intra_log_price = _get_sheet_value(sheet , row,col_map['C_5_intra_log_price'])
    model_data.category_trend = _get_sheet_value(sheet , row,col_map['Category trend'])
    model_data.trend_month = _get_sheet_value(sheet , row,col_map['Trend_month'])
    model_data.trend_quarter = _get_sheet_value(sheet , row,col_map['Trend_quarter'])
    model_data.trend_year = _get_sheet_value(sheet , row,col_map['Trend_year'])
    model_data.month_no = _get_sheet_value(sheet , row,col_map['month_no'])
    model_data.flag_promotype_motivation = _get_sheet_value(sheet , row,col_map['Flag_promotype_Motivation'])
    model_data.flag_promotype_n_pls_1 = _get_sheet_value(sheet , row,col_map['Flag_promotype_N_pls_1'])
    model_data.flag_promotype_traffic = _get_sheet_value(sheet , row,col_map['Flag_promotype_traffic'])
    model_data.flag_nonpromo_1 = _get_sheet_value(sheet , row,col_map['Flag_nonpromo_1'])
    model_data.flag_nonpromo_2 = _get_sheet_value(sheet , row,col_map['Flag_nonpromo_2'])
    model_data.flag_nonpromo_3 = _get_sheet_value(sheet , row,col_map['Flag_nonpromo_3'])
    model_data.flag_promo_1 = _get_sheet_value(sheet , row,col_map['Flag_promo_1'])
    model_data.flag_promo_2 = _get_sheet_value(sheet , row,col_map['Flag_promo_2'])
    model_data.flag_promo_3 = _get_sheet_value(sheet , row,col_map['Flag_promo_3'])
    model_data.holiday_flag_1 = _get_sheet_value(sheet , row,col_map['Holiday_Flag1'])
    model_data.holiday_flag_2 = _get_sheet_value(sheet , row,col_map['Holiday_Flag2'])
    model_data.holiday_flag_3 = _get_sheet_value(sheet , row,col_map['Holiday_Flag3'])
    model_data.holiday_flag_4 = _get_sheet_value(sheet , row,col_map['Holiday_Flag4'])
    model_data.holiday_flag_5 = _get_sheet_value(sheet , row,col_map['Holiday_Flag5'])
    model_data.holiday_flag_6 = _get_sheet_value(sheet , row,col_map['Holiday_Flag6'])
    model_data.holiday_flag_7 = _get_sheet_value(sheet , row,col_map['Holiday_Flag7'])
    model_data.holiday_flag_8 = _get_sheet_value(sheet , row,col_map['Holiday_Flag8'])
    model_data.holiday_flag_9 = _get_sheet_value(sheet , row,col_map['Holiday_Flag9'])
    model_data.holiday_flag_10 = _get_sheet_value(sheet , row,col_map['Holiday_Flag10'])
    model_data.model_meta = model.ModelMeta.objects.get(
        account_name =  _get_sheet_value(sheet , row,col_map['Account Name']),
        corporate_segment = _get_sheet_value(sheet , row,col_map['Corporate Segment']),
        product_group = _get_sheet_value(sheet , row,col_map['PPG'])
    )

    
    # pass


def read_promo_data(file):
    headers = const.DATA_HEADER
    book = openpyxl.load_workbook(file,data_only=True)
    sheet = book['MODEL_DATA']
    columns = sheet.max_column
    rows = sheet.max_row
     
    col_map = _get_col_map(rows,columns,sheet,headers)
    
    row_ =0
    validation_dict = {}
    for row in range(row_+2 , rows+1):    
        db_data = model.ModelData()
        _update_model_data_object(db_data , sheet , row , col_map)
        if not _get_sheet_value(sheet , row,col_map['Account Name']):
            break
        slug = util.generate_slug_string( _get_sheet_value(sheet , row,col_map['Account Name']),
                                            _get_sheet_value(sheet , row,col_map['Corporate Segment']),
                                           _get_sheet_value(sheet , row,col_map['PPG']))
        db_data.full_clean()
        db_data.save()
        if slug not in validation_dict:
            validation_dict[slug] = {
                'count' : 1,
                'account_name' : _get_sheet_value(sheet , row,col_map['Account Name']),
                'corporate_segment' :  _get_sheet_value(sheet , row,col_map['Corporate Segment']),
                'product_group' : _get_sheet_value(sheet , row,col_map['PPG'])
            }
        else:
            validation_dict[slug]['count'] = validation_dict[slug]['count'] + 1
    print(validation_dict , "Validation dictionary values")
    book.close()
    return validation_dict
def read_roi_data(file):
    # import pdb
    # pdb.set_trace()
    headers = const.ROI_HEADER
    # import pdb
    # pdb.set_trace()
    book = openpyxl.load_workbook(file,data_only=True)
    sheet = book['ROI_Data_All_retailers_flag_N_p']
    columns = sheet.max_column
    rows = sheet.max_row
    col_= []
    row_ =1
    header_found = False
    for row in range(1,rows+1):
        print(row , "row count")
        for col in range(1,columns+1):
            print(col , "column count")
            cell_obj = sheet.cell(row = row, column = col)
            if cell_obj.value in headers:
                header_found = True
                print(cell_obj.value , 'object value')
                # col_taken = True
                col_.append(col)
        if header_found:
           break 
    print(col_ , "coldddd")
    # import pdb
    # pdb.set_trace()
    for row in range(row_+1 , rows+1):
        
        meta , created = model.ModelMeta.objects.get_or_create(
            account_name = _get_sheet_value(sheet ,row , 1),
            corporate_segment = _get_sheet_value(sheet ,row , 2),
            product_group = _get_sheet_value(sheet ,row , 3),
            # brand_filter = _get_sheet_value(sheet ,row , 4),
            # brand_format_filter = _get_sheet_value(sheet ,row , 5),
            # strategic_cell_filter = _get_sheet_value(sheet ,row , 5),
            slug = util.generate_slug_string(
                _get_sheet_value(sheet ,row , 1),
               _get_sheet_value(sheet ,row , 2),
                _get_sheet_value(sheet ,row , 3)
            )
        )
        if created:
            # print(created , "created")
            # print(meta , "meta")
            meta.brand_filter = _get_sheet_value(sheet ,row , 4)
            meta.brand_format_filter =_get_sheet_value(sheet ,row , 5)
            meta.strategic_cell_filter =_get_sheet_value(sheet ,row , 6)
            meta.save()
            # meta.slug = util.generate_slug_string(
            #     _get_sheet_value(sheet ,row , 1),
            #    _get_sheet_value(sheet ,row , 2),
            #     _get_sheet_value(sheet ,row , 3))
        roi = model.ModelROI(
            model_meta = meta,
            neilson_sku_name = _get_sheet_value(sheet ,row , 7),
            date = _get_sheet_value(sheet ,row , 8),
            year = _get_sheet_value(sheet ,row , 9),
            week = _get_sheet_value(sheet ,row , 12),
            activity_name = _get_sheet_value(sheet ,row , 14),
            mechanic = _get_sheet_value(sheet ,row , 15),
            discount_nrv = _get_sheet_value(sheet ,row , 16),
            on_inv = _get_sheet_value(sheet ,row , 18),
            off_inv = _get_sheet_value(sheet ,row , 17),
             gmac = _get_sheet_value(sheet ,row , 20),
            list_price = _get_sheet_value(sheet ,row , 23),
        )
        roi.save() 
    book.close()            
    # print(col_ , "final col_")

def read_excel(loc):
    
    # ScenarioPlannerMetrics.objects.all().delete()
    headers = ['Category' , 'Product Group' , 'Retailer' ,'Brand Filter','Brand Format Filter',
    'Strategic Cell Filter','Year' , 'Date','Base Price Elasticity','Cross Elasticity',
    'Net Elasticity','Base Units','List Price','Retailer Median Base Price',
    'Retailer Median Base Price  w\o VAT','On Inv. %','Off Inv. %','TPR %','GMAC%, LSV',
    'Product Group Weight (grams)']
    taken_header = []
    book = openpyxl.load_workbook(loc,data_only=True)
    shet = book['Scenario Planner']
    columns = shet.max_column
    rows = shet.max_row
    col_ = []
    row_ =0
    col_taken = False
    for row in range(1,rows+1):
        for col in range(1,columns+1):
            cell_obj = shet.cell(row = row, column = col)
            if cell_obj.value in headers and cell_obj.value not in taken_header:
                col_taken = True
                col_.append(col)
                taken_header.append(cell_obj.value)
        if col_taken:
            row_ = row
            break
    # import pdb
    # pdb.set_trace()
    obj = {}
    ob=[]
    week = []
    # import pdb
    # pdb.set_trace()
    weeks = {}
    weeks_date = {}
    for row in range(row_+1 , rows+1):
        obj = {}
        week =1
        # metric = ScenarioPlannerMetrics()
        for c in range(0,len(col_)):
            
            cell_obj = shet.cell(row = row, column = col_[c])
            # print(row , "r0w" , col_[c] , "col c" , headers[c] , ":headers[c]" ,cell_obj.value , "cell_obj.value" )
            _genObj(obj,cell_obj.value,headers[c])
        scm = ScenarioPlannerMetricModel(obj)
        ob.append(scm)
        slug = scm.category + scm.product_group + scm.retailer
        if slug == 'GumORBIT OTCMagnit':
            print(scm.date , ":DATE:" , str(scm.date))
       
        # if slug in weeks:
        #     weeks[slug]['count'] = weeks[slug]['count'] + 1
        # else:
        #     weeks[slug] = {
        #         'count' : 1
        #     }
        if slug in weeks_date:
            weeks_date[slug]['count'] = weeks_date[slug]['count'] + 1
            weeks_date[slug][str(scm.date)] = weeks_date[slug]['count']
        else:
            weeks_date[slug] = {
                'count' : 1 
            }
            weeks_date[slug][str(scm.date)] = weeks_date[slug]['count']
            # {
            #     scm.date.strftime("%d/%m/%Y")  : weeks_date[slug]['count']
            # }
    # print(weeks , "weeks ")
    print(weeks_date , "weeks_date")
  
    _update_date(ob,weeks_date)
    book.close()
    return
      
def _update_date(obj,weeks_date):
    print('updating..')

    # li = util.grouping(obj , max(obj,key=lambda x:x.date).date + timedelta(days=7))
    # import pdb
    # pdb.set_trace()
    for o in obj:
        metric = model.ScenarioPlannerMetrics()
        _updateMetricFromObject(metric , o , weeks_date)
        metric.save()
   
    
def _updateMetricFromObject(metric:model.ScenarioPlannerMetrics , obj : ScenarioPlannerMetricModel,weeks_date):
    # print(obj.year , "object year")
     
    metric.category = obj.category
    metric.product_group = obj.product_group
    metric.retailer = obj.retailer
    slug = obj.category + obj.product_group + obj.retailer
    metric.brand_filter = obj.brand_filter
    metric.brand_format_filter = obj.brand_format_filter
    metric.strategic_cell_filter = obj.brand_format_filter
    metric.year = obj.year
    metric.date = obj.date
    # import pdb
    # pdb.set_trace()
    metric.week = weeks_date[slug][str(obj.date)]
    metric.base_price_elasticity = obj.base_price_elasticity
    metric.cross_elasticity = obj.cross_elasticity
    metric.net_elasticity = obj.net_elasticity
    metric.base_units = obj.base_units
    metric.list_price = obj.list_price
    metric.retailer_median_base_price = obj.retailer_median_base_price
    metric.retailer_median_base_price_w_o_vat = obj.retailer_median_base_price_w_o_vat
    metric.on_inv_percent = obj.on_inv_percent
    metric.off_inv_percent = obj.off_inv_percent
    metric.tpr_percent = obj.tpr_percent 
    metric.gmac_percent_lsv = obj.gmac_percent_lsv
    metric.product_group_weight = obj.product_group_weight

def _genObj(obj,value,header):
    # print(value , "Value " , type(value) , " TYPE VALUE")
     
    if(header == 'Category'):
        obj["category"]= value.strip()
    elif(header == 'Product Group'):
        obj["product_group"]= value.strip()
    elif(header == 'Retailer'):
        obj["retailer"]= value.strip()
    elif(header == 'Brand Filter'):
        obj["brand_filter"]= value.strip()
    elif(header == 'Brand Format Filter'):
        obj["brand_format_filter"]= value.strip()
    elif(header == 'Strategic Cell Filter'):
        obj["strategic_cell_filter"]= value.strip()
    elif(header == 'Year'):
        obj["year"]= value
    elif(header == 'Date'):
        obj["date"] =value
    elif(header == 'Base Price Elasticity'):
        obj["base_price_elasticity"]= round(float(value),3)
    elif(header == 'Cross Elasticity'):
        obj["cross_elasticity"]= round(float(value),3)
    elif(header == 'Net Elasticity'):
        obj["net_elasticity"]= round(float(value),3)
    elif(header == 'Base Units'):
        obj["base_units"]= round(float(value),3)
    elif(header == 'List Price'):
        obj["list_price"]= round(float(value),3)
    elif(header == 'Retailer Median Base Price'):
        obj["retailer_median_base_price"]= round(float(value),3)
    elif(header == 'Retailer Median Base Price  w\o VAT'):
        obj["retailer_median_base_price_w_o_vat"]= round(float(value),3)
    elif(header == 'On Inv. %'):
        obj["on_inv_percent"]= round(float(value) * 100,3) 
    elif(header == 'Off Inv. %'):
        obj["off_inv_percent"]= round(float(value) * 100,3) 
    elif(header == 'TPR %'):
        obj["tpr_percent"]= round(float(value) * 100,3) 
    elif(header == 'GMAC%, LSV'):
        obj["gmac_percent_lsv"]= round(float(value) * 100,3)
    elif(header == 'Product Group Weight (grams)'):
        obj["product_group_weight"]= round(float(value),3)
    

def _updateMetric(metric:model.ScenarioPlannerMetrics , value,header):
    if(header == 'Category'):
        metric.category = value.strip()
    elif(header == 'Product Group'):
        metric.product_group = value.strip()
    elif(header == 'Retailer'):
        metric.retailer = value.strip()
    elif(header == 'Brand Filter'):
        metric.brand_filter = value.strip()
    elif(header == 'Brand Format Filter'):
        metric.brand_format_filter = value.strip()
    elif(header == 'Strategic Cell Filter'):
        metric.strategic_cell_filter = value.strip()
    elif(header == 'Year'):
        metric.year = value
    elif(header == 'Date'):
        metric.date =value
    elif(header == 'Base Price Elasticity'):
        metric.base_price_elasticity = round(float(value),3)
    elif(header == 'Cross Elasticity'):
        metric.cross_elasticity = round(float(value),3)
    elif(header == 'Net Elasticity'):
        metric.net_elasticity = round(float(value),3)
    elif(header == 'Base Units'):
        metric.base_units = round(float(value),3)
    elif(header == 'List Price'):
        metric.list_price = round(float(value),3)
    elif(header == 'Retailer Median Base Price'):
        metric.retailer_median_base_price = round(float(value),3)
    elif(header == 'Retailer Median Base Price  w\o VAT'):
        metric.retailer_median_base_price_w_o_vat = round(float(value),3)
    elif(header == 'On Inv. %'):
        metric.on_inv_percent = round(float(value) * 100,3) 
    elif(header == 'Off Inv. %'):
        metric.off_inv_percent = round(float(value) * 100,3) 
    elif(header == 'TPR %'):
        metric.tpr_percent = round(float(value) * 100,3) 
    elif(header == 'GMAC%, LSV'):
        metric.gmac_percent_lsv = round(float(value) * 100,3)
    elif(header == 'Product Group Weight (grams)'):
        metric.product_group_weight = round(float(value),3)

def _updateMetricDup(metric:model.ScenarioPlannerMetrics , value,header):
    # print(value , "Value " , type(value) , " TYPE VALUE")
    if(header == 'Category'):
        metric.category = value.strip()
    elif(header == 'Product Group'):
        metric.product_group = value.strip()
    elif(header == 'Retailer'):
        metric.retailer = value.strip()
    elif(header == 'Brand Filter'):
        metric.brand_filter = value.strip()
    elif(header == 'Brand Format Filter'):
        metric.brand_format_filter = value.strip()
    elif(header == 'Strategic Cell Filter'):
        metric.strategic_cell_filter = value.strip()
    elif(header == 'Year'):
        metric.year = value + 1
    elif(header == 'Date'):
        metric.date =value
    elif(header == 'Base Price Elasticity'):
        metric.base_price_elasticity = round(float(value),3)
    elif(header == 'Cross Elasticity'):
        metric.cross_elasticity = round(float(value),3)
    elif(header == 'Net Elasticity'):
        metric.net_elasticity = round(float(value),3)
    elif(header == 'Base Units'):
        metric.base_units = round(float(value),3) * 2
    elif(header == 'List Price'):
        metric.list_price = round(float(value),3)
    elif(header == 'Retailer Median Base Price'):
        metric.retailer_median_base_price = round(float(value),3)
    elif(header == 'Retailer Median Base Price  w\o VAT'):
        metric.retailer_median_base_price_w_o_vat = round(float(value),3)
    elif(header == 'On Inv. %'):
        metric.on_inv_percent = round(float(value) * 100,3) 
    elif(header == 'Off Inv. %'):
        metric.off_inv_percent = round(float(value) * 100,3) 
    elif(header == 'TPR %'):
        metric.tpr_percent = round(float(value) * 100,3) 
    elif(header == 'GMAC%, LSV'):
        metric.gmac_percent_lsv = round(float(value) * 100,3)
    elif(header == 'Product Group Weight (grams)'):
        metric.product_group_weight = round(float(value),3)

        
def lift(file1,file2):
    retailer = 'Tander'
    ppg = 'A.Korkunov 192g'
    segment = 'BOXES'
    base_value = roi.main(file1,file2,retailer , ppg , segment)
    # len(base_value)
    # base_value['Incrementa']
    # base_value['Base']
    # print(base_value , "base value")
    data = model.ModelData.objects.select_related('model_meta').filter(
        model_meta__account_name = retailer,
        model_meta__corporate_segment = segment,
        model_meta__product_group = ppg
    ).order_by('week')
    print(len(data) , "len of data")
    # import pdb
    # pdb.set_trace()
    for i in range(0,len(data)):
        query = data.get(week = i+1)
        # import pdb
        # pdb.set_trace()
        query.incremental_unit = decimal.Decimal(round(base_value['Incremental'][i],6))
        query.base_unit =  decimal.Decimal(round(base_value['Base'][i],6))
        query.save()
        print('saved')
    # import pdb
    # pdb.set_trace()
    
def read_coeff_map(file):
    headers = const.COEFF_MAP_HEADER
    book = openpyxl.load_workbook(file,data_only=True)
    sheet = book['COEFF_MAPPING']
    columns = sheet.max_column
    rows = sheet.max_row
    col_= []
    row_ =1
    header_found = False
    for row in range(1,rows+1):
        print(row , "row count")
        for col in range(1,columns+1):
            # print(col , "column count")
            cell_obj = sheet.cell(row = row, column = col)
            if cell_obj.value in headers:
                header_found = True
                # print(cell_obj.value , 'object value')
                # col_taken = True
                col_.append(col)
        if header_found:
           break 
    print(col_ , "coldddd")
    # import pdb
    # pdb.set_trace()
    for row in range(row_+1 , rows+1):
        
        meta , created = model.ModelMeta.objects.get_or_create(
            account_name = _get_sheet_value(sheet ,row , 1),
            corporate_segment = _get_sheet_value(sheet ,row , 2),
            product_group = _get_sheet_value(sheet ,row , 3),
            # brand_filter = _get_sheet_value(sheet ,row , 4),
            # brand_format_filter = _get_sheet_value(sheet ,row , 5),
            # strategic_cell_filter = _get_sheet_value(sheet ,row , 5),
            slug = util.generate_slug_string(
                _get_sheet_value(sheet ,row , 1),
               _get_sheet_value(sheet ,row , 2),
                _get_sheet_value(sheet ,row , 3)
            )
        )
        if created:
            # print(created , "created")
            # print(meta , "meta")
            meta.brand_filter = _get_sheet_value(sheet ,row , 4)
            meta.brand_format_filter =_get_sheet_value(sheet ,row , 5)
            meta.strategic_cell_filter =_get_sheet_value(sheet ,row , 6)
            meta.save()
            # meta.slug = util.generate_slug_string(
            #     _get_sheet_value(sheet ,row , 1),
            #    _get_sheet_value(sheet ,row , 2),
            #     _get_sheet_value(sheet ,row , 3))
        c_map = model.CoeffMap(
            model_meta = meta,
            coefficient_old = _get_sheet_value(sheet ,row , 7),
            coefficient_new = _get_sheet_value(sheet ,row , 8),
            value = _get_sheet_value(sheet ,row , 9),
            # off_inv = _get_sheet_value(sheet ,row , 18),
            #  gmac = _get_sheet_value(sheet ,row , 20),
            # list_price = _get_sheet_value(sheet ,row , 23),
        )
        c_map.save() 
    book.close()

def lift_test():
    from django.db.models.query import Prefetch
    from django.forms.models import model_to_dict
    from . import constants as const
    from django.db.models import F , Value
    from django.db.models.functions import Concat,Abs
    import pandas as pd
    
    
    retailer = "Tander"
    ppg = 'A.Korkunov 192g'
    
    data = model.ModelData.objects.filter(
        model_meta__account_name = retailer,
        model_meta__product_group = ppg
    )
    no_of_weeks = 0
    no_of_waves = 0
    max_promo = 0
    min_promo = 0
    duration_of_waves  =[]
    promos = []
    for d in data:
        waves = []
        if d.tpr_discount:
            # waves/
            no_of_weeks = no_of_weeks +1
        if max_promo > d.tpr_discount:
            max_promo = d.tpr_discount
        if min_promo < d.tpr_discount:
            min_promo = d.tpr_discount
    print(no_of_weeks , "mo of weeks ")
    print(max_promo , "max promo")
    print(min_promo , "min promo")


def download_excel_compare_scenario(data):
    output = io.BytesIO()
    no_format_header = ['date', 'week']
    currency_header = ['asp', 'total_rsv_w_o_vat', 'promo_asp','total_lsv','total_nsv','mars_mac', 'trade_expense',
     'retailer_margin','avg_promo_selling_price','lsv','nsv','mac','te','rp']
    percent_header = [ 'retailer_margin_percent_of_nsv','mars_mac_percent_of_nsv','te_percent_of_lsv'
    ,'rp_percent','mac_percent','roi']

    ROW_CONST = 6
    COL_CONST = 1

    # from scenario_planner import test as t
    # compare_scenario_data = t.RESPONSE_PROMO

    workbook = xlsxwriter.Workbook(output)

    merge_format_date = workbook.add_format({
        'bold': 1,
        'align': 'center',
        'valign': 'vcenter',
         })

    merge_format_app = workbook.add_format({
        'bold': 1,
        'align': 'center',
        'valign': 'vcenter'
    })
    merge_format_app.set_font_size(20)

    format_header = workbook.add_format({'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})
    format_header.set_font_size(14)
    
    format_name = workbook.add_format({'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})
    format_name.set_font_size(20)
    format_value = workbook.add_format({
        'border': 1,
        'align': 'center',
        'text_wrap': True,
        'valign': 'vcenter'})
    format_value_left = workbook.add_format({
    'border': 1,
    'align': 'left',
    'text_wrap': True,
    'valign': 'vcenter'})
    format_value_left.set_indent(6)

    summary_value_bold = workbook.add_format({'bold': True})
    summary_value_bold.set_font_size(14)

    format_value_percentage = workbook.add_format({ 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '0.00 %' })

    format_value_currency = workbook.add_format({ 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '[<999950]0.0,"K ₽";[<999950000]0.0,,"M ₽";0.0,,,"B ₽"' })

    format_value_number = workbook.add_format({ 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '[<999950]0.0,"K";[<999950000]0.0,,"M";0.0,,,"B"' })

    summary_value_percentage = workbook.add_format({ 'bold': 1, 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '0.00 %' })
    summary_value_percentage.set_font_size(14)

    summary_value_currency = workbook.add_format({ 'bold': 1, 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '[<999950]0.0,"K ₽";[<999950000]0.0,,"M ₽";0.0,,,"B ₽"' })
    summary_value_currency.set_font_size(14)

    summary_value_number = workbook.add_format({ 'bold': 1, 'border': 1, 'align': 'center', 'text_wrap': True, 'valign': 'vcenter', 'num_format': '[<999950]0.0,"K";[<999950000]0.0,,"M";0.0,,,"B"' })
    summary_value_number.set_font_size(14)

    header_key =  ['date','week', 'promotions','predicted_units','base_unit','incremental_unit','total_weight_in_tons','total_lsv','total_nsv','mars_mac_percent_of_nsv','trade_expense','te_percent_of_lsv', 'te_per_units','roi','asp','promo_asp','total_rsv_w_o_vat', 'retailer_margin','retailer_margin_percent_of_nsv', 'mars_mac', ]

    header_title = ['Date','Week','Promotion(Base)','Promotion(Simulated)','Units(Base)','Units(Simulated)','Base units(Base)','Base units(Simulated)','Incremental units(Base)',
    'Incremental units(Simulated)','Volume(Base)','Volume(Simulated)','LSV(Base)','LSV(Simulated)','NSV(Base)','NSV(Simulated)',
    'MAC, %NSV(Base)','MAC, %NSV(Simulated)','Trade expense(Base)','Trade expense(Simulated)','TE, % LSV(Base)','TE, % LSV(Simulated)',
    'TE / Unit(Base)','TE / Unit(Simulated)','ROI(Base)','ROI(Simulated)','ASP(Base)','ASP(Simulated)','Promo ASP(Base)','Promo ASP(Simulated)',
    'RSV w/o VAT(Base)','RSV w/o VAT(Simulated)','Customer Margin(Base)','Customer Margin(Simulated)','Customer Margin,%RSV(Base)',
    'Customer Margin,%RSV(Simulated)','Mars MAC(Base)','Mars MAC(Simulated)']

    if len(compare_scenario_data) > 0:
        for data in compare_scenario_data:
            row = ROW_CONST
            col = COL_CONST

            scenario_name = (data['scenario_name'][:29] + '..') if len(data['scenario_name']) > 31 else data['scenario_name']
            worksheet = workbook.add_worksheet(scenario_name)
            worksheet.hide_gridlines(2)

            worksheet.merge_range('B2:D2', 'Downloaded on {}'.format(dateformat()) , merge_format_date)
            worksheet.merge_range('B3:D3', 'Promo Simulator Tool' , merge_format_app)
            worksheet.set_column('B:D', 20)
            worksheet.merge_range('B4:D4', "Account Name : {}".format(data['account_name']),merge_format_app)
            worksheet.merge_range('B5:D5', "Product Group : {}".format(data['product_group']),merge_format_app)


            for key in header_title:
                _writeExcel(worksheet,row, col,key,format_header)
                col+=1

            col = COL_CONST
            row+=1

            simulated_weekly = data['simulated']['weekly']
            base_weekly = data['base']['weekly']

            for base,simulated in zip(base_weekly,simulated_weekly):
                for k in header_key:
                    if k == 'date' or k == 'week':
                        value = simulated[k]
                        _writeExcel(worksheet,row, col, value ,format_value)
                        col+=1
                    elif k == 'promotions':
                        promotion_value = util.format_promotions(
                            base['flag_promotype_motivation'],
                            base['flag_promotype_n_pls_1'],
                            base['flag_promotype_traffic'],
                            base['promo_depth'],
                            base['co_investment']
                        )
                        promotion_value_simulated = util.format_promotions(
                            simulated['flag_promotype_motivation'],
                            simulated['flag_promotype_n_pls_1'],
                            simulated['flag_promotype_traffic'],
                            simulated['promo_depth'],
                            simulated['co_investment']
                        )
                        _writeExcel(worksheet,row, col, promotion_value ,format_value)
                        col+=1
                        _writeExcel(worksheet,row, col, promotion_value_simulated ,format_value)
                        col+=1
                    elif k in percent_header:
                        _writeExcel(worksheet,row, col, base[k]/100, format_value_percentage)
                        col+=1
                        _writeExcel(worksheet,row, col, simulated[k]/100, format_value_percentage)
                        col+=1
                    elif k in currency_header:
                        _writeExcel(worksheet,row, col, base[k], format_value_currency)
                        col+=1
                        _writeExcel(worksheet,row, col, simulated[k], format_value_currency)
                        col+=1
                    else:
                        _writeExcel(worksheet,row, col, base[k], format_value_number)
                        col+=1
                        _writeExcel(worksheet,row, col, simulated[k], format_value_number)
                        col+=1
                row+=1
                col = COL_CONST

            simulated_total = data['simulated']['total']
            base_total = data['base']['total']

            total_header = ['units','base_units','increment_units','volume','lsv','nsv','mac_percent','te','te_percent_of_lsv','te_per_unit','roi','asp','avg_promo_selling_price','total_rsv_w_o_vat','rp','rp_percent','mac']
            worksheet.merge_range('B{}:E{}'.format(row+1,row+1), 'Total ' , format_header)
            col = 3
            for k in total_header:
                if k in percent_header:
                    _writeExcel(worksheet,row, col,base_total[k]/100,summary_value_percentage)
                    col+=1
                    _writeExcel(worksheet,row, col,simulated_total[k]/100,summary_value_percentage)
                    col+=1
                elif k in currency_header:
                    _writeExcel(worksheet,row, col,base_total[k],summary_value_currency)
                    col+=1
                    _writeExcel(worksheet,row, col,simulated_total[k],summary_value_currency)
                    col+=1
                else:
                    _writeExcel(worksheet,row, col,base_total[k],summary_value_number)
                    col+=1
                    _writeExcel(worksheet,row, col,simulated_total[k],summary_value_number)
                    col+=1
            col = COL_CONST
    
    workbook.close()
    output.seek(0)
    return output