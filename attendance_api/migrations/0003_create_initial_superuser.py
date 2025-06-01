# attendance_api/migrations/000X_create_initial_superuser.py
from django.db import migrations
from django.contrib.auth import get_user_model
import os # Để đọc biến môi trường

def create_superuser(apps, schema_editor):
    User = get_user_model() # Sử dụng get_user_model() để tương thích
    
    # Lấy thông tin superuser từ biến môi trường (AN TOÀN HƠN)
    # Bạn cần đặt các biến này trên Render dashboard
    DJANGO_SUPERUSER_USERNAME = os.environ.get('DJANGO_SUPERUSER_USERNAME', 'admin') 
    DJANGO_SUPERUSER_EMAIL = os.environ.get('DJANGO_SUPERUSER_EMAIL', 'admin@gmail.com')
    DJANGO_SUPERUSER_PASSWORD = os.environ.get('DJANGO_SUPERUSER_PASSWORD') 

    if not DJANGO_SUPERUSER_PASSWORD:
        print("DJANGO_SUPERUSER_PASSWORD environment variable not set. Skipping superuser creation.")
        return # Không tạo nếu không có mật khẩu

    if not User.objects.filter(username=DJANGO_SUPERUSER_USERNAME).exists():
        print(f"Creating superuser: {DJANGO_SUPERUSER_USERNAME}")
        User.objects.create_superuser(
            username=DJANGO_SUPERUSER_USERNAME,
            email=DJANGO_SUPERUSER_EMAIL,
            password=DJANGO_SUPERUSER_PASSWORD
        )
    else:
        print(f"Superuser {DJANGO_SUPERUSER_USERNAME} already exists.")

class Migration(migrations.Migration):

    dependencies = [
        # Thêm dependency vào migration trước đó của app này, hoặc migration cuối cùng của django.contrib.auth
        # Ví dụ: ('your_app_name', '000Y_previous_migration'),
        # Hoặc nếu là migration đầu tiên của app:
        migrations.swappable_dependency(migrations. государств.AUTH_USER_MODEL), # Đảm bảo model User đã được tạo
    ]

    operations = [
        migrations.RunPython(create_superuser),
    ]