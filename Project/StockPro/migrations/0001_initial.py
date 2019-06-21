# Generated by Django 2.1.5 on 2019-03-05 05:40

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Company',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('stock_code', models.CharField(max_length=20)),
                ('name', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='HistoryData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('data', models.TextField()),
                ('start_date', models.CharField(max_length=30)),
                ('company', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='StockPro.Company')),
            ],
        ),
        migrations.CreateModel(
            name='PredictData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('data', models.TextField()),
                ('start_date', models.CharField(max_length=30)),
                ('company', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='StockPro.Company')),
            ],
        ),
        migrations.CreateModel(
            name='StockIndex',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ri_qi', models.CharField(max_length=30)),
                ('zi_jin', models.IntegerField(default=0)),
                ('qiang_du', models.IntegerField(default=0)),
                ('feng_xian', models.IntegerField(default=0)),
                ('zhuan_qiang', models.IntegerField(default=0)),
                ('chang_yu', models.IntegerField(default=0)),
                ('jin_zi', models.IntegerField(default=0)),
                ('zong_he', models.IntegerField(default=0)),
                ('company', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='StockPro.Company')),
            ],
        ),
    ]
