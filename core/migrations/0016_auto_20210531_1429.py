# Generated by Django 3.1.3 on 2021-05-31 08:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0015_auto_20210527_1748'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='modelcoefficient',
            name='average_weight_in_grams',
        ),
        migrations.RemoveField(
            model_name='modelcoefficient',
            name='weighted_weight_in_grams',
        ),
        migrations.AddField(
            model_name='modeldata',
            name='average_weight_in_grams',
            field=models.DecimalField(decimal_places=15, default=0.0, max_digits=20),
        ),
        migrations.AddField(
            model_name='modeldata',
            name='weighted_weight_in_grams',
            field=models.DecimalField(decimal_places=15, default=0.0, max_digits=20),
        ),
    ]
