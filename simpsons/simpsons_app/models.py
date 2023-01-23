from django.db import models


class Simpsons(models.Model):
    name = models.CharField(max_length=225)
    picture = models.ImageField(upload_to='images/', default=None)
