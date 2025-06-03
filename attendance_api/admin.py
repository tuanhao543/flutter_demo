# attendance_api/admin.py
from django.contrib import admin
from .models import RegisteredUser, AttendanceLog

class AttendanceLogAdmin(admin.ModelAdmin):
    list_display = ('user', 'check_in_time', 'check_out_time', 'latitude', 'longitude') # HIỂN THỊ TRONG LIST
    list_filter = ('user', 'check_in_time')
    search_fields = ('user__name',)

admin.site.register(RegisteredUser)
admin.site.register(AttendanceLog, AttendanceLogAdmin)