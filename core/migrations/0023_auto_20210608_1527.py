# Generated by Django 3.1.3 on 2021-06-08 09:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0022_auto_20210608_1523'),
    ]

    operations = [
        migrations.AlterField(
            model_name='modelroi',
            name='activity_name',
            field=models.CharField(max_length=500, null=True),
        ),
        migrations.AlterField(
            model_name='modelroi',
            name='mechanic',
            field=models.CharField(max_length=500, null=True),
        ),
        migrations.AlterField(
            model_name='modelroi',
            name='neilson_sku_name',
            field=models.CharField(max_length=500, null=True),
        ),
    ]
