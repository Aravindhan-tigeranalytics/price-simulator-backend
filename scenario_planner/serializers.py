from django.db.models import fields, query
from django.db.models.query import Prefetch
from numpy import source
from pulp import constants
from rest_framework import serializers

from utils import exceptions as exception, models
# from core.models import Scenario,ScenarioPlannerMetrics
from core import models as model
from utils import optimizer as optimizer
from . import calculations as cal
from . import fields as field

class FileSerializer(serializers.Serializer):
    # intialize fields
    files = serializers.FileField()
    image = serializers.ImageField()


class SaveScenarioSerializer(serializers.ModelSerializer):
  
    class Meta:
        model = model.SavedScenario
        fields = ('id','scenario_type', 'name', 'comments')
        read_only_fields = ('id',)
        
class SavePromo(serializers.Serializer):
    
    savescenario = SaveScenarioSerializer()
    account_name = serializers.CharField(allow_blank = True)
    product_group = serializers.CharField(allow_blank = True)
    optimizer_data = serializers.CharField(allow_blank = True)


class ScenarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = model.Scenario
        fields = ('id','scenario_type', 'name', 'comments', 'savedump','is_yearly')
        read_only_fields = ('id',)
    # def create(self, validated_data):
    #     print(validated_data , "validated data ")
    #     # import pdb
    #     # pdb.set_trace()
    #     return super().create(validated_data)
    
    # def validate_savedump(self, value):
        # import pdb
        # pdb.set_trace()
        
        # import ast
        # res = ast.literal_eval(value) 
        # if 'account_name' not in res:
        #     raise serializers.ValidationError("Account name sould be present")
      
        
        # return res

    def validate(self,data):
        # import pdb
        # pdb.set_trace()
        # print(data , "DATA")
        if not data['name']:
            raise exception.EmptyException
        return data

class PricingScenarioPlannerMetricsSerializer(serializers.Serializer):
    category = serializers.CharField()
    product_group = serializers.CharField()
    retailer = serializers.CharField()
    brand_filter = serializers.CharField()
    brand_format_filter = serializers.CharField()
    strategic_cell_filter = serializers.CharField()
    week = serializers.CharField()

class ScenarioPlannerMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = model.ScenarioPlannerMetrics
        fields = '__all__'
        
class PricingSaveSerializer(serializers.ModelSerializer):
    # has_promo = serializers.SerializerMethodField('has_promotion')
    class Meta:
        model = model.PricingSave
        fields = ('id','account_name','corporate_segment','product_group','saved_scenario' )
    # def has_promotion(self,obj):
    #     return model.PromoSave.objects.filter(saved_pricing = obj).exists()

class PromoScenarioSavedList(serializers.ModelSerializer):
    class Meta:
        model = model.Scenario
        fields = ('id', 'name', 'comments')
        read_only_fields = ('id',)
        
class ScenarioSavedList(serializers.ModelSerializer):
    meta = serializers.SerializerMethodField('has_pricing')
    class Meta:
        model = model.SavedScenario
        fields = ('id', 'name', 'comments','scenario_type' , 'meta')
        read_only_fields = ('id',)
        
    def get_fields(self , *args,**kwargs):
        return super().get_fields()
    def has_pricing(self,obj):
        
            
        if obj.scenario_type == 'promo':
            promo_save  = obj.promo_saved.get(saved_scenario = obj)
            obj = {
                "retailer" : promo_save.account_name,
                "product_group" : promo_save.product_group,
                "pricing" : False
            }
            if( bool(promo_save.saved_pricing)):
                pricing = model.PricingWeek.objects.filter(pricing_save = promo_save.saved_pricing).first()
                obj['pricing'] = {
                    "lpi" : pricing.lp_increase,
                    "rsp" : pricing.rsp_increase,
                    "cogs" : pricing.cogs_increase,
                    "elasticity" : pricing.base_price_elasticity
                }
            
                    
            return obj
        if obj.scenario_type == 'optimizer':
            # import pdb
            # pdb.set_trace()
            print(obj.name , "name have problem with id " , obj.id)
            optimizer_saved = obj.optimizer_saved.first()
            obj = {
                "retailer" : optimizer_saved.model_meta.account_name,
                "product_group" : optimizer_saved.model_meta.product_group,
                "pricing" : False
            }
            promo_save = optimizer_saved.promo_save
            pricing_save = optimizer_saved.pricing_save
            if promo_save:
                # return bool(promo_save.saved_pricing)
                pricing = model.PricingWeek.objects.filter(pricing_save = promo_save.saved_pricing).first()
                obj['pricing'] = {
                    "lpi" : pricing.lp_increase,
                    "rsp" : pricing.rsp_increase,
                    "cogs" : pricing.cogs_increase,
                    "elasticity" : pricing.base_price_elasticity
                }
            if pricing_save:
                pricing = model.PricingWeek.objects.filter(pricing_save = pricing_save).first()
                obj['pricing'] = {
                "lpi" : pricing.lp_increase,
                "rsp" : pricing.rsp_increase,
                "cogs" : pricing.cogs_increase,
                "elasticity" : pricing.base_price_elasticity
                }
            return obj
        pricing_save = obj.pricing_saved.first()
        obj = {
                "retailer" : pricing_save.account_name,
                "product_group" : pricing_save.product_group,
                "pricing" : False
            }
        pricing = model.PricingWeek.objects.filter(pricing_save = pricing_save).first()
        obj['pricing'] = {
                "lpi" : pricing.lp_increase,
                "rsp" : pricing.rsp_increase,
                "cogs" : pricing.cogs_increase,
                "elasticity" : pricing.base_price_elasticity
                }
        # import pdb
        # pdb.set_trace()
            
        return obj
    def __init__(self, *args, **kwargs):
        # if 'context' in kwargs:
        # import pdb
        # pdb.set_trace()
            
        super().__init__(*args, **kwargs)
  
class PromoSaveSerializer(serializers.ModelSerializer):
    class Meta:
        model = model.PromoSave
        fields = "__all__"

class OptimizerSaveSerializer(serializers.ModelSerializer):
    class Meta:
        model = model.OptimizerSave
        fields = "__all__"
          
# OptimizerSave,PromoSave        
class ScenarioSavedListOptimized(serializers.ModelSerializer):
    meta = serializers.SerializerMethodField('has_pricing')
    # optimizer_saved = OptimizerSaveSerializer(many = True)
    # promo_saved = PromoSaveSerializer(many = True)
    class Meta:
        model = model.SavedScenario
        fields = ('id', 'name', 'comments','scenario_type' , 'meta')
        read_only_fields = ('id',)
        
    @staticmethod
    def setup_eager_loading(queryset):
        """ Perform necessary eager loading of data. """
        # queryset = queryset.prefetch_related('optimizer_saved' , 'promo_saved','optimizer_saved__pricing_save')
        queryset = queryset.prefetch_related(
            Prefetch(
                  'optimizer_saved',
            queryset=model.OptimizerSave.objects.select_related('model_meta' , 'pricing_save').all(),
            to_attr='prefetched_optimizer'
            ),
             Prefetch(
                  'promo_saved',
            queryset=model.PromoSave.objects.select_related('saved_pricing').all(),
            to_attr='prefetched_promo'
            ),
              Prefetch(
                  'pricing_saved',
            queryset=model.PricingSave.objects.all().prefetch_related(
                 Prefetch(
                  'pricing_week',
            queryset=model.PricingWeek.objects.all(),
            to_attr='prefetched_price_week'
            ),
                ),
            to_attr='prefetched_price'
            )
        )
        return queryset
        
    def get_fields(self , *args,**kwargs):
        return super().get_fields()
    def has_pricing(self,obj):
        
            
        if obj.scenario_type == 'promo':
            # import pdb
            # pdb.set_trace()
            # obj.prefetched_promo[0]
            promo_save  = obj.prefetched_promo[0]
            obj = {
                "retailer" : promo_save.account_name,
                "product_group" : promo_save.product_group,
                "pricing" : False
            }
            # if( bool(promo_save.saved_pricing)):
            #     pricing = model.PricingWeek.objects.filter(pricing_save = promo_save.saved_pricing).first()
            #     obj['pricing'] = {
            #         "lpi" : pricing.lp_increase,
            #         "rsp" : pricing.rsp_increase,
            #         "cogs" : pricing.cogs_increase,
            #         "elasticity" : pricing.base_price_elasticity
            #     }
            
                    
            return obj
        if obj.scenario_type == 'optimizer':
            # import pdb
            # pdb.set_trace()
            # print(obj.name , "name have problem with id " , obj.id)
            optimizer_saved = obj.prefetched_optimizer[0]
            obj = {
                "retailer" : optimizer_saved.model_meta.account_name,
                "product_group" : optimizer_saved.model_meta.product_group,
                "pricing" : False
            }
            promo_save = optimizer_saved.promo_save
            pricing_save = optimizer_saved.pricing_save
            # if promo_save:
            #     # return bool(promo_save.saved_pricing)
            #     pricing = model.PricingWeek.objects.filter(pricing_save = promo_save.saved_pricing).first()
            #     obj['pricing'] = {
            #         "lpi" : pricing.lp_increase,
            #         "rsp" : pricing.rsp_increase,
            #         "cogs" : pricing.cogs_increase,
            #         "elasticity" : pricing.base_price_elasticity
            #     }
            # if pricing_save:
            #     pricing = model.PricingWeek.objects.filter(pricing_save = pricing_save).first()
            #     obj['pricing'] = {
            #     "lpi" : pricing.lp_increase,
            #     "rsp" : pricing.rsp_increase,
            #     "cogs" : pricing.cogs_increase,
            #     "elasticity" : pricing.base_price_elasticity
            #     }
            return obj
        # import pdb
        # pdb.set_trace()
        pricing_save = obj.prefetched_price[0]
        pricing = obj.prefetched_price[0].prefetched_price_week[0]
        obj = {
                "retailer" : pricing_save.account_name,
                "product_group" : pricing_save.product_group,
                "pricing" : False
            }
        
        obj['pricing'] = {
                "lpi" : pricing.lp_increase,
                "rsp" : pricing.rsp_increase,
                "cogs" : pricing.cogs_increase,
                "elasticity" : pricing.base_price_elasticity
                }
        # import pdb
        # pdb.set_trace()
            
        return obj
    def __init__(self, *args, **kwargs):
        # if 'context' in kwargs:
        # import pdb
        # pdb.set_trace()
            
        super().__init__(*args, **kwargs)
        
class ScenarioPlannerMetricsSerializerObject(serializers.ModelSerializer):
    my_field = serializers.SerializerMethodField('obj')
    def obj(self,metric):
        # print(metric , "metics additionsl")
        return "hola"
    class Meta:
        model = model.ScenarioPlannerMetrics
        fields = '__all__'
 

class CommentSerializer(serializers.Serializer):
    OBJ_CHOICES = (
        ("MAC", "MAC"), 
        ("RP", "RP"), 
        ("Trade_Expense", "Trade Expense"), 
        ("Units", "Units"), 
    )
    objective_function = serializers.ChoiceField(
                        choices = OBJ_CHOICES)
    max_promotion = serializers.IntegerField(max_value = 53 ,min_value=0,default=23,initial=23)
    min_promotion = serializers.IntegerField(max_value = 53 ,min_value=0,default=16,initial=16)
    
    config_mac = serializers.BooleanField(default=True,initial=True)
    config_rp = serializers.BooleanField(default=True,initial=True)
    config_trade_expense = serializers.BooleanField(default=False,initial=False)
    config_units = serializers.BooleanField(default=False,initial=False)
    config_mac_perc = serializers.BooleanField(default=False,initial=False)
    config_min_length = serializers.BooleanField(default=True,initial=True)
    config_max_length = serializers.BooleanField(default=True,initial=True)
    config_promo_gap = serializers.BooleanField(default=True,initial=True)
    
    param_mac = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_rp = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_trade_expense = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_units = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_mac_perc = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_min_length = serializers.IntegerField(max_value = 10 ,min_value=0,default=2,initial=2)
    param_max_length = serializers.IntegerField(max_value = 10 ,min_value=0,default=5,initial=5)
    param_promo_gap = serializers.IntegerField(max_value = 10 ,min_value=0,default=4,initial=4)
    
    result = serializers.SerializerMethodField('obj')
    def obj(self,ob):
        return optimizer.process(dict(ob))
    
class SimulatedSerializer(serializers.Serializer):
    pass
      
class ModelDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = model.ModelData
        fields = ('model_meta','year' , 'quater','month','period','week','date','promo_depth' ,
                  'co_investment','flag_promotype_motivation','flag_promotype_n_pls_1',
                  'flag_promotype_traffic')
    
class ModelDataSerializerbkp(serializers.ModelSerializer):
    def __init__(self,*args,**kwargs):
        # ex_query = kwargs.pop('extra')
        # if ex_query:
        #     import pdb
        #     pdb.set_trace()
        # self.fields['simulated'] = serializers.SerializerMethodField('obj')
        super().__init__(self,*args,**kwargs)

    class Meta:
        model = model.ModelData
        fields = ('year' , 'quater','month','period','week','date','base')
    base = serializers.SerializerMethodField('obj')

    def obj(self,ob,*args,**kwargs):
        
        return cal.promo_simulator_calculations_test(ob)
    
    
class ModelMetaSerializer(serializers.ModelSerializer):
    prefetched_data = ModelDataSerializer(many=True)
    total_rsv_w_o_vat = serializers.SerializerMethodField('tee')
    volumes_in_tonnes = serializers.SerializerMethodField('tee') 
    te = serializers.SerializerMethodField('tee')
    base_units =  serializers.SerializerMethodField('tee')
    incremental_units =  serializers.SerializerMethodField('tee')
    units = serializers.SerializerMethodField('tee')
    nsv = serializers.SerializerMethodField('tee')
    mac = serializers.SerializerMethodField('tee')
    lsv = serializers.SerializerMethodField('tee')
    rp = serializers.SerializerMethodField('tee')
    average_selling_price = serializers.SerializerMethodField('tee')
    avg_promo_selling_price = serializers.SerializerMethodField('tee')
    rp_percent_of_rsp = serializers.SerializerMethodField('tee')
    mac_percent_of_nsv = serializers.SerializerMethodField('tee')
    te_percent_of_lsv = serializers.SerializerMethodField('tee')
    te_per_unit = serializers.SerializerMethodField('tee')
    roi = serializers.SerializerMethodField('tee')
    lift = serializers.SerializerMethodField('tee')
    # data = ModelDataSerializer(many=True)
    class Meta:
        model = model.ModelMeta
        fields = ('account_name','corporate_segment','product_group','mac','nsv','units','volumes_in_tonnes',
                  'total_rsv_w_o_vat','base_units','incremental_units','roi',
                  'te','lsv','rp','average_selling_price','avg_promo_selling_price','rp_percent_of_rsp',
                  'te_percent_of_lsv','mac_percent_of_nsv','te_per_unit','lift','prefetched_data')
        # 'units','nsv','mac',
    def tee(self,obj):
        return 0
    # def units(self,obj):
    #     return 0
    # def nsv(self,obj):
    #     return 0
    # def mac(self,obj):
    #     return 0
        # read_only_fields = (
        # 'prefetched_data',
        # )
    # def __init__(self,*args,**kwargs):
    #     # ex_query = kwargs.pop('extra')
    #     # prefeteched = args[0].prefetched_data
    #     # # import pdb
    #     # # pdb.set_trace()
    #     # print(prefeteched[0].week , "args[0].prefetched_data")
    #     self.fields['prefetched_data'] = ModelDataSerializer(many=True)
    #     super().__init__(*args,**kwargs)
    
    def get_prefetched_data(self,obj):
        # import pdb
        # pdb.set_trace()
        return 'payload'
    # datas = serializers.SerializerMethodField('obj')
    # def obj(self,ob):
    #     return 3
    #     import pdb
    #     pdb.set_trace()
        
        
class DynamicInputSerializer(serializers.Serializer):
    PROMO_CHOICE = (
        (None,"Choose promo"),
        ("Flag_promotype_Motivation", "Motivation"), 
        ("Flag_promotype_N_pls_1", "N Pls 1"), 
        ("Flag_promotype_traffic", "Traffic"), 
         ("Flag_promotype_traffic", "Promo depth"), 
    )
    promo_depth =serializers.IntegerField(initial = 0.0,default=0.0)
    promo_mechanics = serializers.ChoiceField(choices=PROMO_CHOICE,allow_blank=True)
    co_investment = serializers.IntegerField(initial = 0,default=0)

class ModelMetaExcelUpload(serializers.Serializer):
    simulator_input = serializers.FileField(max_length=None, allow_empty_file=False)

    

class OptimizerMeta(serializers.ModelSerializer):
    class Meta:
        model = model.ModelMeta
        fields = '__all__'
        
        
# class ModelDataSerializer(serializers.ModelSerializer):
#     class Meta:
#         model= model.ModelData
#         fields = "__all__"

class ModelMetaGetSerializer(serializers.Serializer):
    OBJ_CHOICES = (
        ("cell", "Choose Strategic Cell"), 
    )
    OBJ_CHOICES2 = (
       ("brand", "Choose Brand"), 
    )
    OBJ_CHOICES3 = (
       ("format", "Choose Brand Format"), 
    )
    # query = model.ModelMeta.objects.prefetch_related('data').all()
    # account_name = field.ChoiceField(choices=[i + i for i in list(query.values_list('account_name').distinct())])
    # corporate_segment = field.ChoiceField(choices=[i + i for i in list(query.values_list('corporate_segment').distinct())])
    # strategic_cell = serializers.ChoiceField(choices=[i + i for i in list(query.filter(strategic_cell_filter__isnull=False).values_list('strategic_cell_filter').distinct())])
    # brand = serializers.ChoiceField(choices=[i + i for i in list(query.filter(brand_filter__isnull=False).values_list('brand_filter').distinct())])
    # brand_format = serializers.ChoiceField(choices=[i + i for i in list(query.filter(brand_format_filter__isnull=False).values_list('brand_format_filter').distinct())])

    # product_group = field.ChoiceField(choices=[i + i for i in list(query.values_list('product_group').distinct())])
   
    promo_elasticity = serializers.IntegerField(initial = 0,default=0)
    param_depth_all = serializers.IntegerField(initial = 0,default=0)
    # serializers.ChoiceField(
    #                     choices = OBJ_CHOICES)
    
    def __init__(self, *args, **kwargs):
        # import pdb
        # pdb.set_trace()
        query = model.ModelMeta.objects.prefetch_related('data').all()
        user = kwargs.pop("user" , None)
        if user:
            # import pdb
            # pdb.set_trace()
            query = query.filter(id__in = user.allowed_retailers.all())
            print(user , "user in serializer")
        # # import pdb
        # # pdb.set_trace()
        # if query:
        self.fields['account_name'] = field.ChoiceField(choices=[i + i for i in list(query.values_list('account_name').distinct())])
        self.fields['corporate_segment'] = field.ChoiceField(choices=[i + i for i in list(query.values_list('corporate_segment').distinct())])
        self.fields['strategic_cell'] = serializers.ChoiceField(choices=[i + i for i in list(query.filter(strategic_cell_filter__isnull=False).values_list('strategic_cell_filter').distinct())])
        self.fields['brand'] = serializers.ChoiceField(choices=[i + i for i in list(query.filter(brand_filter__isnull=False).values_list('brand_filter').distinct())])
        self.fields['brand_format'] = serializers.ChoiceField(choices=[i + i for i in list(query.filter(brand_format_filter__isnull=False).values_list('brand_format_filter').distinct())])
        self.fields['product_group'] = field.ChoiceField(choices=[i + i for i in list(query.values_list('product_group').distinct())])


        for i in range(1,53):
            self.fields['week-' + str(i)] = DynamicInputSerializer()
                
            
            # self.fields['week-' + str(i)] = DynamicInputSerializer()
        # queryset = kwargs.pop('queryset', None)
        # request = kwargs.get('context', {}).get('request')
        # print(request , "request")
        # str_fields = request.GET.get('fields', '') if request else None
        # print(str_fields , "str fields")
        # fields = str_fields.split(',') if str_fields else None
        # print(fields , "fields")
        # [i + i for i in list(queryset.values_list('account_name').distinct())]
        # print(queryset , "querysetinit")
        # print(args , "querysetinitargs")
        # print(kwargs,"querysetinitkwargs")
        # if(queryset):
        #     print([i + i for i in list(queryset.values_list('account_name').distinct())] , "resultinit")
        #     self.fields['objective_function'].choices = [i + i for i in list(queryset.values_list('account_name').distinct())]
        # # import pdb
        # # pdb.set_trace()
        # print(queryset, "querysetquerysetquerysetquerysetqueryset")
        super().__init__(*args, **kwargs)
        # print(self)
        # import pdb
        # pdb.set_trace()
class ModelMetaGetSerializerTest(serializers.ModelSerializer):
    class Meta:
        model = model.ModelMeta
        fields = ['id','account_name','corporate_segment','product_group','brand_filter','brand_format_filter','strategic_cell_filter']


class MapPricingPromoSerializer(serializers.Serializer):
    save_scenario = SaveScenarioSerializer()
    # scenario_id = serializers.IntegerField()
    pricing_id = serializers.IntegerField()
    promo_details = serializers.CharField()

class CompareScenarioInputSerializer(serializers.Serializer):
    compare_scenario_data = serializers.CharField()