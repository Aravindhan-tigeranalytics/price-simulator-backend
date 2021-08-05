# Generated by Django 3.1.3 on 2021-02-15 07:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0004_auto_20201128_2158'),
    ]

    operations = [
        migrations.CreateModel(
            name='ScenarioPlannerMetrics',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('category', models.CharField(max_length=100, verbose_name='Category')),
                ('product_group', models.CharField(max_length=100, verbose_name='Product Group')),
                ('retailer', models.CharField(max_length=100, verbose_name='Retailer')),
                ('year', models.IntegerField(verbose_name='Year')),
                ('date', models.DateField(verbose_name='Date')),
                ('base_price_elasticity', models.DecimalField(decimal_places=3, max_digits=8, verbose_name='Base Price Elasticity')),
                ('cross_elasticity', models.DecimalField(decimal_places=3, max_digits=8, verbose_name='Cross Elasticity')),
                ('net_elasticity', models.DecimalField(decimal_places=3, max_digits=8, verbose_name='Net Elasticity')),
                ('base_units', models.DecimalField(decimal_places=3, max_digits=12, verbose_name='Base Units')),
                ('list_price', models.DecimalField(decimal_places=3, max_digits=8, verbose_name='List Price')),
                ('retailer_median_base_price', models.DecimalField(decimal_places=3, max_digits=8, verbose_name='Retailer Median Base Price')),
                ('retailer_median_base_price_w_o_vat', models.DecimalField(decimal_places=3, max_digits=8, verbose_name='Retailer Median Base Price w/o VAT')),
                ('on_inv_percent', models.DecimalField(decimal_places=3, max_digits=8, verbose_name='On Inv %')),
                ('off_inv_percent', models.DecimalField(decimal_places=3, max_digits=8, verbose_name='Off Inv %')),
                ('tpr_percent', models.DecimalField(decimal_places=3, max_digits=8, verbose_name='TPR %')),
                ('gmac_percent_lsv', models.DecimalField(decimal_places=3, max_digits=8, verbose_name='GMAC % LSV')),
                ('product_group_weight', models.DecimalField(decimal_places=3, max_digits=8, verbose_name='Product Group Weight (grams)')),
            ],
            options={
                'db_table': 'scenario_planner_metrics',
            },
        ),
    ]