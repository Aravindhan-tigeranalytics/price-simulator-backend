from django.db import models

class OptimizerManager(models.Manager):
    def get_queryset(self):
        return super(OptimizerManager, self).get_queryset().filter(optimiser_flag=True)