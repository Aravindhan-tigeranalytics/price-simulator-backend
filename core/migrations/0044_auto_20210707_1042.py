# Generated by Django 3.1.3 on 2021-07-07 05:12

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0043_optimizersave_pricing_save'),
    ]

    operations = [
        migrations.AddField(
            model_name='savedscenario',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='savedscenario',
            name='updated_at',
            field=models.DateTimeField(auto_now=True),
        ),
    ]