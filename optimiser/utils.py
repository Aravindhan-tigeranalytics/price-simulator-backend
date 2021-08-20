from core import models as model
from django.db.models.query import QuerySet

def update_model_from_optimizer(model_data : QuerySet[model.ModelData] , optimizer_save : QuerySet[model.OptimizerSave]):
    # import pdb
    # pdb.set_trace()
    model_list = [list(i) for i in model_data]
    opt_list = [list(i) for i in optimizer_save]
    
    for i in range(0,len(opt_list)):
      
        opt_list[i].week