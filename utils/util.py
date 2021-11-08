import itertools 
import datetime
import random
import re
from utils import exceptions as ex

# 2021-10-03T18:30:00.000Z

def week_from_date(datestr):
    arr = datestr.split("-")
    a_date = datetime.date(int(arr[0]), int(arr[1]), int(arr[2]))
    return a_date.isocalendar()[1] + 1

def is_date_greater_or_equal(price_date_str,form_date_str):
    format = "%Y-%m-%d"
    if not form_date_str:
        return True
    return datetime.datetime.strptime(convert_timestamp(price_date_str) , format) >= datetime.datetime.strptime(convert_timestamp(form_date_str),format)

def convert_timestamp(str):
    if(str):
        spl = str.split("T")
        if(len(spl) > 0):
            return spl[0]
        return None
    return None
    
def _transform_corporate_segment(segment):
    # import pdb
    # pdb.set_trace()
    
    if segment:
        if segment.lower() == 'gum':
            return 'Gum'
    return 'Choco'

def _get_royalty(segment):
    # print(segment , "royality value...")
    ret = 0.5
    if segment == "Choco":
        ret = 0.0
    print(type(ret))
    return ret

def format_value(val ,is_percent = False , is_currency= False , no_format = False):
    if no_format:
        return val
    currency = "₽"
    percentage = "%"
    value = val
    val = str(val).split(".")[0]
    final = 0
    if is_percent:
        return "{} {}".format("{:.2f}".format(value) , percentage)
    strlen = len(val)
    curr = ""
    if(strlen >=4 and strlen <=6):
        final = value / 1000;
        curr = "K"
    elif (strlen >=7 and strlen <=9):
        final = value / 1000000
        curr = "M"
    elif(strlen >= 10):
        final = value / 1000000000
        curr = "B"
    if is_currency:
        return "{} {} {}".format("{:.2f}".format(final),curr , currency)
    
    return "{} {}".format("{:.2f}".format(final),curr)

def format_promotions(motivation , n_plus_1, traffic , promo_depth , co_inv):
    promo_name = "TPR"
    promo_string = ""

    if motivation:
        promo_name = "Motivation"
    elif n_plus_1:
        promo_name = "N+1"
    elif traffic:
        promo_name = "Traffic"

    if promo_depth:
        promo_string+=promo_name + "-" + str(round(promo_depth,2)) + "%"
  
    if co_inv:
        promo_string+= " (Co-"+str(round(co_inv,2))+"%)"

    if promo_string:
        return promo_string

    return '-'

# def get_key(val):
#     for key, value in my_dict.items():
#          if val == value:
#              return key
 
#     return "key doesn't exist"

def _divide(n1 , n2):
    if not n1 or not n2:
        return 0
    return n1/n2

def _regex(pattern,string):
    return re.compile(pattern).search(string)

def average(n1,n2):
    if not n1:
        return n2
    return (n1 + n2)/2

def _sortList(ut):
    return sorted(ut, key=lambda x: x.date, reverse=False)

def generate_slug_string(s1,s2,s3):
    return "{}-{}-{}".format(remove_duplicate_spaces(s1),remove_duplicate_spaces(s2),remove_duplicate_spaces(s3))

def remove_duplicate_spaces(s):
    return "".join(s.split()).lower()

def grouping(a_list , init_date):
    key_and_group= []
    an_iterator = itertools.groupby(a_list, lambda x : x.category+x.product_group+x.retailer) 
    for key, group in an_iterator: 
        # print(key , "key")
        key_and_group = key_and_group +  printList(_sortList(list(group)),init_date)
        # printDict(key_and_group)
    return key_and_group
   
    # import pdb
    # pdb.set_trace()
    # printDict(key_and_group)
    # print(key_and_group , "key and group")
        # print(key_and_group , "key and grouo")
def printDict(dc):
    print(len(dc) , "DC")
     
     
    # dcl = list(dc.values())
    # import pdb
    # pdb.set_trace()
    # li = []
    # for i in dcl:
    #     li  = li+i
    # return li
    # for k, v in dc.items():
    #     print(k , "key")
    #     # print(v , "Value")
    #     for li in v:
    #         print(li.date , "date")
    #     print("----")

def printList(li,init_date):
    i = init_date
    for l in li:
        l.year = 2021
        l.base_units = l.base_units * random.uniform(2,4)
        # print(l.date, " before update" ,end=" || ")
        l.date = i
        i = i + datetime.timedelta(days=7)
    return li
        # print(l.date , "After update")
    # print("-----------")

def _limit(val , min , max):
    if val < min or val > max:
        return False
    return True

def is_zero_to_hundred(val):
    return _limit(val , 0 ,100)

def is_zero_to_one(val):
    return _limit(val , 0 ,1)

def is_zero_or_one(val):
    return val == 1 or val == 0

def is_zero_or_positive(val):
    return val >= 0

def validate_import_data(validation_dict):
    total = 0 
    print(validation_dict , "validation dict")
    for i in validation_dict.keys():
        total = total + validation_dict[i]['count']
        if validation_dict[i]['count'] != 52:
            raise ex.ImportDataException(
                "Account name {},Corporate segment {}, and Product group {} must have exactly 52 week data but it has {} data".format(
                   validation_dict[i]['account_name'],validation_dict[i]['corporate_segment'],validation_dict[i]['product_group'],
                   validation_dict[i]['count']
                    )
            )
    return total

def format_value(val ,is_percent = False , is_currency= False , no_format = False):
    if no_format:
        return val
    currency = "₽"
    percentage = "%"
    value = val
    val = str(val).split(".")[0]
    final = 0
    if is_percent:
        return "{} {}".format("{:.2f}".format(value) , percentage)
    strlen = len(val)
    curr = ""
    if(strlen >=1 and strlen <=3):
        final = value;
        curr = ""
    if(strlen >=4 and strlen <=6):
        final = value / 1000;
        curr = "K"
    elif (strlen >=7 and strlen <=9):
        final = value / 1000000
        curr = "M"
    elif(strlen >= 10):
        final = value / 1000000000
        curr = "B"
    if is_currency:
        return "{} {} {}".format("{:.2f}".format(final),curr , currency)
    
    return "{} {}".format("{:.2f}".format(final),curr)