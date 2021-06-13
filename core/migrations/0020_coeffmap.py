# Generated by Django 3.1.3 on 2021-06-08 06:44

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0019_auto_20210601_1242'),
    ]

    operations = [
        migrations.CreateModel(
            name='CoeffMap',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('coefficient_old', models.CharField(max_length=100, verbose_name='Account Name')),
                ('coefficient_new', models.CharField(max_length=100, verbose_name='Corporate Segment')),
                ('value', models.DecimalField(decimal_places=15, default=0.0, max_digits=20, null=True)),
                ('model_meta', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='coeff_map', to='core.modelmeta')),
            ],
            options={
                'db_table': 'coeff_map',
            },
        ),
    ]
