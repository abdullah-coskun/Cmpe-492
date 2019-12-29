from django.conf.urls import url

from . import views


urlpatterns = [
    url(r'^enrich/', views.EnrichAPIView.as_view(), name="register"),
]