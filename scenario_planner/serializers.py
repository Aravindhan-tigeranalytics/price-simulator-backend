from django.db.models import fields, query
from numpy import source
from rest_framework import serializers
from utils import exceptions as exception
# from core.models import Scenario,ScenarioPlannerMetrics
from core import models as model
from utils import optimizer as optimizer
from . import calculations as cal
from . import fields as field


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
    
    def validate_savedump(self, value):
        
        import ast
        res = ast.literal_eval(value) 
        if 'account_name' not in res:
            raise serializers.ValidationError("Account name sould be present")
      
        
        return res

    def validate(self,data):
        # import pdb
        # pdb.set_trace()
        print(data , "DATA")
        if not data['name']:
            raise exception.EmptyException
        return data

class ScenarioPlannerMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = model.ScenarioPlannerMetrics
        fields = '__all__'

class PromoScenarioSavedList(serializers.ModelSerializer):
    class Meta:
        model = model.Scenario
        fields = ('id', 'name', 'comments')
        read_only_fields = ('id',)

class ScenarioPlannerMetricsSerializerObject(serializers.ModelSerializer):
    my_field = serializers.SerializerMethodField('obj')
    def obj(self,metric):
        print(metric , "metics additionsl")
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
    query = model.ModelMeta.objects.prefetch_related('data').all()
    account_name = field.ChoiceField(choices=[i + i for i in list(query.values_list('account_name').distinct())])
    corporate_segment = field.ChoiceField(choices=[i + i for i in list(query.values_list('corporate_segment').distinct())])
    strategic_cell = serializers.ChoiceField(choices=
                                       OBJ_CHOICES)
    brand = serializers.ChoiceField(choices=
                              OBJ_CHOICES2)
    brand_format = serializers.ChoiceField(choices=
                                     OBJ_CHOICES3)
    product_group = field.ChoiceField(choices=[i + i for i in list(query.values_list('product_group').distinct())])
   
    promo_elasticity = serializers.IntegerField(initial = 0,default=0)
    param_depth_all = serializers.IntegerField(initial = 0,default=0)
    # serializers.ChoiceField(
    #                     choices = OBJ_CHOICES)
    
    def __init__(self, *args, **kwargs):

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
class ModelMetaGetSerializerTest(serializers.Serializer):
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
    # account_name = serializers.ChoiceField(choices = [])
    # corporate_segment = field.ChoiceField(choices=[i + i for i in list(query.values_list('corporate_segment').distinct())])
    strategic_cell = serializers.ChoiceField(choices=
                                       OBJ_CHOICES)
    brand = serializers.ChoiceField(choices=
                              OBJ_CHOICES2)
    brand_format = serializers.ChoiceField(choices=
                                     OBJ_CHOICES3)
    # product_group = field.ChoiceField(choices=[i + i for i in list(query.values_list('product_group').distinct())])
   
    promo_elasticity = serializers.IntegerField(initial = 0,default=0)
    param_depth_all = serializers.IntegerField(initial = 0,default=0)
    
    def __init__(self, *args, **kwargs):
        
        if 'query' in kwargs:
            
            query = kwargs.pop('query')
            print(query , "querty")
            # import pdb
            # pdb.set_trace()
            self.fields['account_name'] = serializers.ChoiceField(choices=
                                       [i + i for i in list(query.values_list('account_name').distinct())])
            
        super().__init__(*args, **kwargs)
