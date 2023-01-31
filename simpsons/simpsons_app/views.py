from django.shortcuts import render
from django.http import HttpResponse

from .models import *
from .forms import *

menu = [{'title': 'Pictures uploading', 'url_n': 'upload'},
        {'title': 'Simpsons names', 'url_n': 'name'}]


def index(request):
    return render(request, 'simpsons_app/index.html', {'menu': menu, 'title': 'Main'})


def name(request):
    info = Simpsons.objects.all()
    return render(request, 'simpsons_app/names.html', {'info': info, 'menu': menu, 'title': 'Simpsons names'})


def upload(request):
    if request.method == 'POST':
        form = PictureForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            img_obj = form.instance
            return render(request, 'simpsons_app/upload.html', {'menu': menu, 'title': 'Pictures uploading',
                                                                'form': form, 'img_obj': img_obj})
    else:
        form = PictureForm()
    return render(request, 'simpsons_app/upload.html', {'form': form, 'menu': menu, 'title': 'Simpsons'})
    # return render(request, 'simpsons_app/upload.html',  {'menu': menu, 'title': 'Simpsons'})
