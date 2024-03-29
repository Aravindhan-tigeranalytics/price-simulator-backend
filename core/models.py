from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, \
    PermissionsMixin
from django.conf import settings
from django.db.models import constraints
# Create your models here.


class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        ''' Create users and save'''
        if not email:
            raise ValueError('Email cannot be empty')
        user = self.model(email=self.normalize_email(email), **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password):
        user = self.create_user(email, password)
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)
        return user


class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(max_length=255, unique=True)
    name = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    USERNAME_FIELD = 'email'
    objects = UserManager()


class Scenario(models.Model):
    name = models.CharField(max_length=255)
    comments = models.CharField(max_length=500, default='')
    savedump = models.TextField(default='')
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE

    )
    is_yearly = models.BooleanField(default=False)


    def __str__(self):
        return self.name

class ScenarioPlannerMetrics(models.Model):
    category = models.CharField(max_length=100,verbose_name="Category")
    product_group = models.CharField(max_length=100,verbose_name="Product Group")
    retailer = models.CharField(max_length=100,verbose_name="Retailer")
    brand_filter = models.CharField(max_length=100,verbose_name="Brand Filter",default=None)
    brand_format_filter = models.CharField(max_length=100,verbose_name="Brand Format Filter",default=None)
    strategic_cell_filter = models.CharField(max_length=100,verbose_name="Strategic Cell Filter",default=None)
    year = models.IntegerField(verbose_name="Year")
    date = models.DateField(verbose_name="Date")
    base_price_elasticity = models.DecimalField(verbose_name="Base Price Elasticity",max_digits=8 , decimal_places=3)
    cross_elasticity = models.DecimalField(verbose_name="Cross Elasticity",max_digits=8 , decimal_places=3)
    net_elasticity = models.DecimalField(verbose_name="Net Elasticity",max_digits=8 , decimal_places=3)
    base_units = models.DecimalField(verbose_name="Base Units",max_digits=12 , decimal_places=3)
    list_price = models.DecimalField(verbose_name="List Price",max_digits=8 , decimal_places=3)
    retailer_median_base_price = models.DecimalField(verbose_name="Retailer Median Base Price",max_digits=8 , decimal_places=3)
    retailer_median_base_price_w_o_vat = models.DecimalField(verbose_name="Retailer Median Base Price w/o VAT",max_digits=8 , decimal_places=3)
    on_inv_percent = models.DecimalField(verbose_name="On Inv %",max_digits=8 , decimal_places=3)
    off_inv_percent = models.DecimalField(verbose_name="Off Inv %",max_digits=8 , decimal_places=3)
    tpr_percent = models.DecimalField(verbose_name="TPR %",max_digits=8 , decimal_places=3)
    gmac_percent_lsv = models.DecimalField(verbose_name="GMAC % LSV",max_digits=8 , decimal_places=3)
    product_group_weight = models.DecimalField(verbose_name="Product Group Weight (grams)",max_digits=8 , decimal_places=3)

    class Meta:
        db_table = "scenario_planner_metrics"
        

class ModelMeta(models.Model):
    account_name = models.CharField(max_length=100,verbose_name="Account Name")
    corporate_segment =  models.CharField(max_length=100,verbose_name="Corporate Segment")
    product_group = models.CharField(max_length=100,verbose_name="Product Group")
    brand_filter = models.CharField(max_length=100,verbose_name="Brand Filter" , default='',null=True)
    brand_format_filter = models.CharField(max_length=100,verbose_name="Brand Format Filter",default='',null=True)
    strategic_cell_filter = models.CharField(max_length=100,verbose_name="Strategic Cell Filter",default='',null=True)
    slug = models.SlugField(db_index=True, max_length=255, unique=True,default='')
    
    def __str__(self):
        return "{}-{}-{}".format(self.account_name,self.corporate_segment,self.product_group)
    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['account_name','corporate_segment','product_group'],name='retailers')
        ]
        # models.UniqueConstraint
        # unique_together = 
        db_table = "model_meta"

class ModelCalculationMetrics(models.Model):
    intercept = models.DecimalField(verbose_name="Intercept", max_digits=20 , decimal_places=15)
    median_base_price_log = models.DecimalField(verbose_name="Median Base Price Log",
                                                max_digits=20 , decimal_places=15)
    tpr_discount = models.DecimalField(verbose_name="TPR Discount", max_digits=20 , decimal_places=15)
    tpr_discount_lag1 = models.DecimalField(max_digits=20 , decimal_places=15)
    tpr_discount_lag2 = models.DecimalField(max_digits=20 , decimal_places=15)
    catalogue = models.DecimalField(max_digits=20 , decimal_places=15)
    display = models.DecimalField(max_digits=20 , decimal_places=15)
    acv = models.DecimalField(max_digits=20 , decimal_places=15)
    si = models.DecimalField(max_digits=20 , decimal_places=15)
    si_month = models.DecimalField(max_digits=20 , decimal_places=15)
    si_quarter = models.DecimalField(max_digits=20 , decimal_places=15)
    c_1_crossretailer_discount = models.DecimalField(max_digits=20 , decimal_places=15)
    c_1_crossretailer_log_price = models.DecimalField(max_digits=20 , decimal_places=15)
    c_1_intra_discount = models.DecimalField(max_digits=20 , decimal_places=15)
    c_2_intra_discount = models.DecimalField(max_digits=20 , decimal_places=15)
    c_3_intra_discount = models.DecimalField(max_digits=20 , decimal_places=15)
    c_4_intra_discount = models.DecimalField(max_digits=20 , decimal_places=15)
    c_5_intra_discount = models.DecimalField(max_digits=20 , decimal_places=15)
    c_1_intra_log_price = models.DecimalField(max_digits=20 , decimal_places=15)
    c_2_intra_log_price = models.DecimalField(max_digits=20 , decimal_places=15)
    c_3_intra_log_price = models.DecimalField(max_digits=20 , decimal_places=15)
    c_4_intra_log_price = models.DecimalField(max_digits=20 , decimal_places=15)
    c_5_intra_log_price = models.DecimalField(max_digits=20 , decimal_places=15)
    category_trend = models.DecimalField(max_digits=20 , decimal_places=15)
    trend_month = models.DecimalField(max_digits=20 , decimal_places=15)
    trend_quarter = models.DecimalField(max_digits=20 , decimal_places=15)
    trend_year = models.DecimalField(max_digits=20 , decimal_places=15)
    month_no = models.DecimalField(max_digits=20 , decimal_places=15)
    flag_promotype_motivation = models.DecimalField(max_digits=20 , decimal_places=15)
    flag_promotype_n_pls_1 = models.DecimalField(max_digits=20 , decimal_places=15)
    flag_promotype_traffic = models.DecimalField(max_digits=20 , decimal_places=15)
    flag_nonpromo_1 = models.DecimalField(max_digits=20 , decimal_places=15)
    flag_nonpromo_2 = models.DecimalField(max_digits=20 , decimal_places=15)
    flag_nonpromo_3 = models.DecimalField(max_digits=20 , decimal_places=15)
    flag_promo_1 = models.DecimalField(max_digits=20 , decimal_places=15)
    flag_promo_2 = models.DecimalField(max_digits=20 , decimal_places=15)
    flag_promo_3 = models.DecimalField(max_digits=20 , decimal_places=15)
    holiday_flag_1 = models.DecimalField(max_digits=20 , decimal_places=15)
    holiday_flag_2 = models.DecimalField(max_digits=20 , decimal_places=15)
    holiday_flag_3 = models.DecimalField(max_digits=20 , decimal_places=15)
    holiday_flag_4 = models.DecimalField(max_digits=20 , decimal_places=15)
    holiday_flag_5 = models.DecimalField(max_digits=20 , decimal_places=15)
    holiday_flag_6 = models.DecimalField(max_digits=20 , decimal_places=15)
    holiday_flag_7 = models.DecimalField(max_digits=20 , decimal_places=15)
    holiday_flag_8 = models.DecimalField(max_digits=20 , decimal_places=15)
    holiday_flag_9 = models.DecimalField(max_digits=20 , decimal_places=15)
    holiday_flag_10 = models.DecimalField(max_digits=20 , decimal_places=15)

    class Meta:
        abstract = True

class ModelData(ModelCalculationMetrics):
    model_meta = models.ForeignKey(
        'core.ModelMeta' , related_name="data" , on_delete=models.CASCADE
    )
    year = models.IntegerField(verbose_name="Year")
    quater = models.IntegerField(verbose_name="Quater")
    month = models.IntegerField(verbose_name="Month")
    period =models.CharField(max_length=10,verbose_name="Period")
    week = models.IntegerField(verbose_name="Week")
    date = models.DateField(verbose_name="Date")
    wk_sold_avg_price_byppg = models.DecimalField(max_digits=20 , decimal_places=15,default=0.0,
                                                  verbose_name='Week Sold Average Price by PPG')
    average_weight_in_grams = models.DecimalField(max_digits=20 , decimal_places=15,default=0.0)
    weighted_weight_in_grams = models.DecimalField(max_digits=20 , decimal_places=15,default=0.0)
    incremental_unit = models.DecimalField(max_digits=30 , decimal_places=15,default=0.0 , null=True)
    base_unit = models.DecimalField(max_digits=30 , decimal_places=15,default=0.0 , null=True)
    # on_inv = models.DecimalField(max_digits=20 , decimal_places=15,default=0.05)
    # off_inv = models.DecimalField(max_digits=20 , decimal_places=15,default=0.1993)
    # list_price = models.DecimalField(max_digits=20 , decimal_places=15,default=168.43)
    # gmac_percent_lsv= models.DecimalField(max_digits=20 , decimal_places=15,default=0.7292)
    class Meta:
        db_table = 'model_data'

class ModelCoefficient(ModelCalculationMetrics):
    model_meta = models.ForeignKey(
        'core.ModelMeta' , related_name="coefficient" , on_delete=models.CASCADE
    )
    wmape = models.DecimalField(max_digits=20 , decimal_places=15)
    rsq = models.DecimalField(max_digits=20 , decimal_places=15)
   
    
    def __str__(self):
        return "{}-{}-{}".format(self.model_meta.account_name,self.model_meta.corporate_segment,self.model_meta.product_group)
    class Meta:
        db_table = 'model_coefficient'

class ModelROI(models.Model):
   
    model_meta = models.ForeignKey(
        'core.ModelMeta' , related_name="roi" , on_delete=models.CASCADE
    )
    year = models.IntegerField(verbose_name="Year" , null=True)
    week = models.IntegerField(verbose_name="Week",null=True)
    on_inv = models.DecimalField(max_digits=20 , decimal_places=15,default=0.05)
    off_inv = models.DecimalField(max_digits=20 , decimal_places=15,default=0.1993)
    list_price = models.DecimalField(max_digits=20 , decimal_places=15,default=168.43)
    gmac= models.DecimalField(max_digits=20 , decimal_places=15,default=0.7292)
    
    
    class Meta:
        db_table = 'model_roi'