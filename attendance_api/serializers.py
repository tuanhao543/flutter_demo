# attendance_api/serializers.py
from rest_framework import serializers
from .models import RegisteredUser, AttendanceLog

class RegisteredUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = RegisteredUser
        fields = ['id', 'name', 'registration_date','is_admin'] # Không expose embedding trực tiếp

class AttendanceLogSerializer(serializers.ModelSerializer):
    user_name = serializers.CharField(source='user.name', read_only=True)

    class Meta:
        model = AttendanceLog
        fields = ['id', 'user', 'user_name', 'check_in_time', 'check_out_time', 'latitude', 'longitude']
        read_only_fields = ['user_name']
        
class AttendanceLogDetailSerializer(serializers.ModelSerializer): # Serializer mới hoặc dùng lại AttendanceLogSerializer
    user_name = serializers.CharField(source='user.name', read_only=True)
    user_id = serializers.IntegerField(source='user.id', read_only=True)

    class Meta:
        model = AttendanceLog
        fields = [
            'id', 
            'user_id', # Thêm user_id để Flutter có thể nhóm
            'user_name', 
            'check_in_time', 
            'check_out_time', 
            'latitude',       # Đảm bảo có
            'longitude'       # Đảm bảo có
        ]
        read_only_fields = ['user_name', 'user_id']
        
class RegisteredUserListSerializer(serializers.ModelSerializer):
    class Meta:
        model = RegisteredUser
        fields = ['id', 'name'] # Chỉ cần id và name cho dropdown