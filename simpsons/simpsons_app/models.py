from django.db import models


class Simpsons(models.Model):
    simpson_id = models.AutoField(primary_key=True, default=None)
    name = models.CharField(max_length=225)
    picture = models.ImageField(upload_to='images/', default=None)
