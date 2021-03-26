from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, \
    PermissionsMixin
from django.conf import settings
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