from django.conf.urls import url

from . import views


urlpatterns = [
    url(r'^classify/', views.ClassifyAPIView.as_view(), name="register"),
]