# attendance_api/urls.py
from django.urls import path
from .views import *

urlpatterns = [
    path('register/', RegisterUserAPI.as_view(), name='register_user_api'),
    path('check-in-out/', CheckInAPI.as_view(), name='check_in_out_api'),
    path('stats/user-work/', UserWorkStatsAPI.as_view(), name='user_work_stats_api'),
    path('users/', ListRegisteredUsersAPI.as_view(), name='list_registered_users_api'),
]