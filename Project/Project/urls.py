"""Project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
import StockPro.views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/',StockPro.views.index,name='index'), # new
    path('predict/',StockPro.views.predict_stock_action,name='predict'),
    path('about/',StockPro.views.about,name='about'),
    path('market/',StockPro.views.market,name='market'),
    path('strategy/',StockPro.views.strategy,name='strategy'),
]
