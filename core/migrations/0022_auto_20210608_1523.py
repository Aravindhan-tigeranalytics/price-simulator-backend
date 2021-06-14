# Generated by Django 3.1.3 on 2021-06-08 09:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0021_auto_20210608_1516'),
    ]

    operations = [
        migrations.AlterField(
            model_name='modelroi',
            name='list_price',
            field=models.DecimalField(decimal_places=15, default=168.43, max_digits=20, null=True),
        ),
        migrations.AlterField(
            model_name='modelroi',
            name='off_inv',
            field=models.DecimalField(decimal_places=15, default=0.1993, max_digits=20, null=True),
        ),
        migrations.AlterField(
            model_name='modelroi',
            name='on_inv',
            field=models.DecimalField(decimal_places=15, default=0.05, max_digits=20, null=True),
        ),
    ]
