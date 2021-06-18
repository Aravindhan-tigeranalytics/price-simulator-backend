from django.apps import AppConfig
# from django.db.models.signals import pre_save
# from . import signals
# from . import models as model


class CoreConfig(AppConfig):
    name = 'core'
    # model = self.get_model()
    
    def ready(self):
    #     ModelMeta = self.get_model('ModelMeta')
    #     pre_save.connect(signals.add_slug_to_article_if_not_exists, sender='core.ModelMeta')
        from . import signals
        
        # from core.signals import add_slug_to_article_if_not_exists
