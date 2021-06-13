from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from decimal import Decimal
class TestDb(models.Model):
    percent = models.DecimalField(verbose_name="percent", max_digits=16 , decimal_places=15 , 
                                  validators=[MinValueValidator(Decimal(0.0)), MaxValueValidator(Decimal(1.0))])
    week = models.IntegerField(verbose_name="Week" , validators=[MinValueValidator(1), MaxValueValidator(52)])
    month = models.IntegerField(verbose_name="month" ,  validators=[MinValueValidator(1), MaxValueValidator(12)])