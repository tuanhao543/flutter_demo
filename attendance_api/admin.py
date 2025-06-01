# attendance_api/admin.py
from django.contrib import admin
from .models import RegisteredUser, AttendanceLog

admin.site.register(RegisteredUser)
admin.site.register(AttendanceLog)