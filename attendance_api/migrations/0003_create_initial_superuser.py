# attendance_api/migrations/0003_create_initial_superuser.py
from django.db import migrations
from django.contrib.auth import get_user_model
import os
from django.conf import settings # Thêm import settings

def create_superuser(apps, schema_editor):
    User = get_user_model()
    
    DJANGO_SUPERUSER_USERNAME = os.environ.get('DJANGO_SUPERUSER_USERNAME', 'admin')
    DJANGO_SUPERUSER_EMAIL = os.environ.get('DJANGO_SUPERUSER_EMAIL', 'admin@gamil.com')
    DJANGO_SUPERUSER_PASSWORD = os.environ.get('DJANGO_SUPERUSER_PASSWORD')

    if not DJANGO_SUPERUSER_PASSWORD:
        print("DJANGO_SUPERUSER_PASSWORD environment variable not set. Skipping superuser creation.")
        return

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
        # Nếu migration này là migration đầu tiên của app 'attendance_api' 
        # và bạn muốn nó chạy sau khi các bảng của app 'auth' (chứa model User) đã được tạo:
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        
        # Hoặc, nếu app 'attendance_api' của bạn đã có migration trước đó (ví dụ: 0002_some_other_change.py),
        # bạn nên đặt dependency vào migration đó:
        # ('attendance_api', '0002_some_other_change'), # Thay '0002_some_other_change' bằng tên file migration trước đó
    ]

    operations = [
        migrations.RunPython(create_superuser),
    ]