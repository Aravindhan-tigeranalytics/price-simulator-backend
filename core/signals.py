# from django.db.models.signals import pre_save
# from django.dispatch import receiver
# # from django.utils.text import slugify
# from . import models as model

# @receiver(pre_save, sender=model.ModelMeta)
# def add_slug_to_article_if_not_exists(sender, instance, *args, **kwargs):
#     print(sender , "sender in")
#     print(instance , "instance in signals")
#     if instance and not instance.slug:
#         slug = "{}-{}-{}".format(instance.account_name,instance.corporate_segment,instance.product_group)
#         instance.slug = slug