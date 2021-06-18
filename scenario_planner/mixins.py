import json
from core import models as model
from django.core.exceptions import ObjectDoesNotExist
from utils import units_calculation as uc
from . import query as pd_query
from . import calculations as cal
class CalculationMixin():
    
    '''
    mixin class to transform django ORM query data to dataframe  and calculate financial metrics 
    get_list_value_from_query - return list values from django ORM
    update_from_request - update user input to queried list
    list_to_frame - convert list to data frame to get predicted unit value
    calculate_financial_mertrics - To calculate finacial metic for both simulated and base scenario
    
    '''
    
    # def _get_financial_metics(coeff_list,data_list,roi_list,type):
    #     incremental_split = json.loads(uc.list_to_frame(coeff_list , data_list).to_json(orient="records"))
    #     finalcial_metrics = cal.calculate_financial_mertrics(data_list ,roi_list,
    #                                            incremental_split , type)
    #     return finalcial_metrics
        
        
    def calculate_finacial_metrics(self,value_dict):
        meta = {
        'account_name' : value_dict['account_name'],
        'corporate_segment' : value_dict['corporate_segment'],
        'product_group' : value_dict['product_group']
        }
        
        coeff_list , data_list ,roi_list = pd_query.get_list_value_from_query(model.ModelCoefficient,
                                                                              model.ModelData,
                                                                              model.ModelROI,
                                                                              value_dict['account_name'],
                                           value_dict['product_group'] )
        if len(data_list) == 0:
            raise ObjectDoesNotExist("Account name {} and Product {} does not exists.".format(
                value_dict['account_name'],value_dict['product_group'])
                                     )
      
        simulated_data_list = cal.update_from_request(data_list, value_dict)
        
        base_incremental_split = json.loads(uc.list_to_frame(coeff_list , data_list).to_json(orient="records"))
        simulated_incremental_split = json.loads(uc.list_to_frame(coeff_list , simulated_data_list).to_json(orient="records"))
        base_finalcial_metrics = cal.calculate_financial_mertrics(data_list ,roi_list,
                                               base_incremental_split , 'base')
        simulated_financial_metrics = cal.calculate_financial_mertrics(simulated_data_list ,roi_list,
                                               simulated_incremental_split , 'simulated',value_dict['promo_elasticity'])
        return {**meta,**base_finalcial_metrics , **simulated_financial_metrics}

