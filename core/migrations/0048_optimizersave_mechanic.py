# Generated by Django 3.1.3 on 2021-08-12 03:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0047_auto_20210729_1753'),
    ]

    operations = [
        migrations.AddField(
            model_name='optimizersave',
            name='mechanic',
            field=models.CharField(blank=True, default='', max_length=500, null=True),
        ),
    ]
