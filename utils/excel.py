import xlsxwriter
import datetime

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



    # for item, cost in (expenses):
    #     worksheet.write(row, col,     item,format2)
    #     worksheet.write(row, col + 1, cost , format2)
    #     row += 1

    # worksheet.write(row, 0, 'Total' , bold)
    # worksheet.write(row, 1, '=SUM(B1:B4)' , money)

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
    # format_name.set_font_size(20)

    row = ROW_CONST
    col = COL_CONST
    
    worksheet.merge_range('B2:D2', 'Downloaded on ' +dateformat() , merge_format_date)
    worksheet.merge_range('B3:D3', 'Price Simulator Tool' , merge_format_app)
    worksheet.merge_range('B4:D4', 'Comparison Summary',merge_format_app)
    # for header in keys:
    #     worksheet.write(row, col,     header,format2)
    #     worksheet.set_row(row, 20)
    #     worksheet.set_column(col,col, 15)
    #     col+=1
    zip_list = []
    data_val = list(data.values())
    for i in data_val:
        # print(i['name'] , "i vaue")
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

def _writeExcel(worksheet , row , col , val , _format):
    worksheet.write(row, col,val,_format)
    worksheet.set_row(row, 30)
    worksheet.set_column(col,col, 20)


def dateformat():

    x = datetime.datetime.now()
    return x.strftime("%b %d %Y %H:%M:%S")

# if __name__ == "__main__":
#     excel()
    # dateformat()