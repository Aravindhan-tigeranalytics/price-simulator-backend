# Generated by Django 3.1.3 on 2021-10-29 12:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0054_pricingsave_promo_save'),
    ]

    operations = [
        migrations.AddField(
            model_name='pricingweek',
            name='base_cogs',
            field=models.DecimalField(decimal_places=3, default=0, max_digits=8, verbose_name='base cogs'),
        ),
        migrations.AddField(
            model_name='pricingweek',
            name='base_list_price',
            field=models.DecimalField(decimal_places=3, default=0, max_digits=8, verbose_name='base list price'),
        ),
        migrations.AddField(
            model_name='pricingweek',
            name='base_retail_price',
            field=models.DecimalField(decimal_places=3, default=0, max_digits=8, verbose_name='base retail price'),
        ),
    ]
