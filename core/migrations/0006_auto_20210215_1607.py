# Generated by Django 3.1.3 on 2021-02-15 10:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0005_scenarioplannermetrics'),
    ]

    operations = [
        migrations.AddField(
            model_name='scenarioplannermetrics',
            name='brand_filter',
            field=models.CharField(default=None, max_length=100, verbose_name='Brand Filter'),
        ),
        migrations.AddField(
            model_name='scenarioplannermetrics',
            name='brand_format_filter',
            field=models.CharField(default=None, max_length=100, verbose_name='Brand Format Filter'),
        ),
        migrations.AddField(
            model_name='scenarioplannermetrics',
            name='strategic_cell_filter',
            field=models.CharField(default=None, max_length=100, verbose_name='Strategic Cell Filter'),
        ),
    ]
