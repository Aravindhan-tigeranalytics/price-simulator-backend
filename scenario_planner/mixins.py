import json
from core import models as model
from django.db.models.query import QuerySet
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
    
    
    # def calcu
    
    def calculate_finacial_metrics_from_optimizer(self,optimizer_data : QuerySet[model.OptimizerSave]):
        optimizer_week = list(optimizer_data)
        account_name = optimizer_week[0].model_meta.account_name
        product_group = optimizer_week[0].model_meta.product_group
        meta = {
            "scenario_id" : optimizer_week[0].saved_scenario.id,
            "scenario_name" : optimizer_week[0].saved_scenario.name,
            "account_name" : account_name,
            "product_group" : product_group
        }
        coeff_list , data_list ,roi_list = pd_query.get_list_value_from_query(model.ModelCoefficient,
                                                                              model.ModelData,
                                                                              model.ModelROI,
                                                                              account_name,
                                           product_group )
        simulated_data_list = cal.update_from_optimizer(data_list , optimizer_week)
        
        base_incremental_split = json.loads(uc.list_to_frame(coeff_list , data_list).to_json(orient="records"))
        simulated_incremental_split = json.loads(uc.list_to_frame(coeff_list , simulated_data_list).to_json(orient="records"))
        # import pdb
        # pdb.set_trace()
        base_finalcial_metrics = cal.calculate_financial_mertrics(data_list ,roi_list,
                                               base_incremental_split , 'base')
        simulated_financial_metrics = cal.calculate_financial_mertrics(simulated_data_list ,roi_list,
                                               simulated_incremental_split , 'simulated')
        return {**meta,**base_finalcial_metrics , **simulated_financial_metrics}
        
    
    def calculate_finacial_metrics_from_pricing(self,pricing_week:QuerySet[model.PricingWeek]):
        
        pricing_week = list(pricing_week)
        account_name = pricing_week[0].pricing_save.account_name
        product_group = pricing_week[0].pricing_save.product_group
        scenario_name = pricing_week[0].pricing_save.saved_scenario.name
        scenario_id = pricing_week[0].pricing_save.saved_scenario.id
        scenario_comment =  pricing_week[0].pricing_save.saved_scenario.comments
        pricing_id  = pricing_week[0].pricing_save.id
        # import pdb
        # pdb.set_trace()
        meta = {
        'scenario_id' : scenario_id,
        'pricing_id' : pricing_id,
        'scenario_name' : scenario_name,
        'scenario_comment' : scenario_comment,  
        'account_name' : account_name,
        'corporate_segment' : account_name,
        'product_group' :product_group
        }
        # if promo_save:
        #     meta['promo_id'] = promo_save.id
        coeff_list , data_list ,roi_list = pd_query.get_list_value_from_query(model.ModelCoefficient,
                                                                              model.ModelData,
                                                                              model.ModelROI,
                                                                              account_name,
                                           product_group )
        if len(data_list) == 0:
            raise ObjectDoesNotExist("Account name {} and Product {} does not exists.".format(
                account_name,product_group)
                                     )
      
        simulated_data_list = cal.update_from_pricing(data_list , pricing_week)
        
        base_incremental_split = json.loads(uc.list_to_frame(coeff_list , data_list).to_json(orient="records"))
        
        simulated_incremental_split = json.loads(uc.list_to_frame(coeff_list , simulated_data_list).to_json(orient="records"))
        # print(base_incremental_split , "Base incremental split")
        # print(simulated_incremental_split , "simulated incremental split")
        base_finalcial_metrics = cal.calculate_financial_mertrics(data_list ,roi_list,
                                               base_incremental_split , 'base')
        simulated_financial_metrics = cal.calculate_financial_mertrics_from_pricing(simulated_data_list ,roi_list,
                                               simulated_incremental_split , 'simulated',pricing_week)
        return {**meta,**base_finalcial_metrics , **simulated_financial_metrics}
    
        
    def calculate_finacial_metrics_from_request(self,value_dict):
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
    
    
    def calculate_finacial_metrics(self,promo_week :QuerySet[model.PromoWeek]):
        promo_week = list(promo_week)
        
        account_name = promo_week[0].pricing_save.account_name
        product_group = promo_week[0].pricing_save.product_group
        promo_elasticity =  promo_week[0].pricing_save.promo_elasticity
        scenario_name = promo_week[0].pricing_save.saved_scenario.name
        scenario_id = promo_week[0].pricing_save.saved_scenario.id
        scenario_comment =  promo_week[0].pricing_save.saved_scenario.comments
       
        meta = {
        'scenario_id' : scenario_id,
        'scenario_name' : scenario_name,
        'scenario_comment' : scenario_comment,   
        'account_name' : account_name,
        'corporate_segment' : account_name,
        'product_group' :product_group
        }
        
        
        
        coeff_list , data_list ,roi_list = pd_query.get_list_value_from_query(model.ModelCoefficient,
                                                                              model.ModelData,
                                                                              model.ModelROI,
                                                                              account_name,
                                           product_group )
        if len(data_list) == 0:
            raise ObjectDoesNotExist("Account name {} and Product {} does not exists.".format(
                account_name,product_group)
                                     )
      
        simulated_data_list = cal.update_from_saved_data(data_list, promo_week)
        
        base_incremental_split = json.loads(uc.list_to_frame(coeff_list , data_list).to_json(orient="records"))
        simulated_incremental_split = json.loads(uc.list_to_frame(coeff_list , simulated_data_list).to_json(orient="records"))
        base_finalcial_metrics = cal.calculate_financial_mertrics(data_list ,roi_list,
                                               base_incremental_split , 'base')
        simulated_financial_metrics = cal.calculate_financial_mertrics(simulated_data_list ,roi_list,
                                               simulated_incremental_split , 'simulated',promo_elasticity)
        return {**meta,**base_finalcial_metrics , **simulated_financial_metrics}


    def calculate_finacial_metrics_pricing_promo(self,promo_week :QuerySet[model.PromoWeek] , pricing_week : QuerySet[model.PricingWeek]):
            # import pdb
            # pdb.set_trace()
            
            promo_week = list(promo_week)
            pricing_week = list(pricing_week)
            account_name = promo_week[0].pricing_save.account_name
            product_group = promo_week[0].pricing_save.product_group
            promo_elasticity =  promo_week[0].pricing_save.promo_elasticity
            scenario_name = promo_week[0].pricing_save.saved_scenario.name
            scenario_id = promo_week[0].pricing_save.saved_scenario.id
            scenario_comment =  promo_week[0].pricing_save.saved_scenario.comments
        
            meta = {
            'scenario_id' : scenario_id,
            'scenario_name' : scenario_name,
            'scenario_comment' : scenario_comment,   
            'account_name' : account_name,
            'corporate_segment' : account_name,
            'product_group' :product_group
            }
            
            
            
            coeff_list , data_list ,roi_list = pd_query.get_list_value_from_query(model.ModelCoefficient,
                                                                                model.ModelData,
                                                                                model.ModelROI,
                                                                                account_name,
                                            product_group )
            if len(data_list) == 0:
                raise ObjectDoesNotExist("Account name {} and Product {} does not exists.".format(
                    account_name,product_group)
                                        )
        
            simulated_data_list = cal.update_from_pricing_promo(data_list,pricing_week, promo_week)
            
            base_incremental_split = json.loads(uc.list_to_frame(coeff_list , data_list).to_json(orient="records"))
            simulated_incremental_split = json.loads(uc.list_to_frame(coeff_list , simulated_data_list).to_json(orient="records"))
            base_finalcial_metrics = cal.calculate_financial_mertrics(data_list ,roi_list,
                                                base_incremental_split , 'base')
            simulated_financial_metrics = cal.calculate_financial_mertrics_from_pricing_promo(simulated_data_list ,roi_list,
                                                simulated_incremental_split , 'simulated',pricing_week,promo_elasticity)
            return {**meta,**base_finalcial_metrics , **simulated_financial_metrics}


