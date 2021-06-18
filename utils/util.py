import itertools 
import datetime
import random
import re
from utils import exceptions as ex

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