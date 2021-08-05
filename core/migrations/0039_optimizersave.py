# Generated by Django 3.1.3 on 2021-06-28 06:57

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0038_pricingweek_base_price_elasticity'),
    ]

    operations = [
        migrations.CreateModel(
            name='OptimizerSave',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField(verbose_name='Date')),
                ('optimum_promo', models.DecimalField(decimal_places=15, max_digits=30, null=True)),
                ('optimum_units', models.DecimalField(decimal_places=15, max_digits=30, null=True)),
                ('optimum_base', models.DecimalField(decimal_places=15, max_digits=30, null=True)),
                ('optimum_incremental', models.DecimalField(decimal_places=15, max_digits=30, null=True)),
                ('base_promo', models.DecimalField(decimal_places=15, max_digits=30, null=True)),
                ('base_units', models.DecimalField(decimal_places=15, max_digits=30, null=True)),
                ('base_base', models.DecimalField(decimal_places=15, max_digits=30, null=True)),
                ('base_incremental', models.DecimalField(decimal_places=15, max_digits=30, null=True)),
                ('model_meta', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='optimizer', to='core.modelmeta')),
            ],
            options={
                'db_table': 'optimizer_save',
            },
        ),
    ]