# Generated by Django 3.1.3 on 2020-11-28 16:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_scenario'),
    ]

    operations = [
        migrations.AddField(
            model_name='scenario',
            name='savedump',
            field=models.TextField(default='', max_length=2000),
        ),
        migrations.AlterField(
            model_name='scenario',
            name='comments',
            field=models.CharField(default='', max_length=500),
        ),
    ]
