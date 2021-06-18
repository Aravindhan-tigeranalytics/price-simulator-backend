# Generated by Django 3.1.3 on 2021-06-15 06:54

from decimal import Decimal
import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0030_auto_20210615_1027'),
    ]

    operations = [
        migrations.AddField(
            model_name='modeldata',
            name='promo_depth',
            field=models.DecimalField(decimal_places=15, default=0.0, max_digits=20, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0')), django.core.validators.MaxValueValidator(Decimal('100'))], verbose_name='Promo Depth'),
        ),
    ]
