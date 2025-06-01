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
        ('attendance_api', '0002_registereduser_is_admin'), # <--- SỬA Ở ĐÂY
        # Bạn vẫn có thể giữ dòng này nếu hàm create_superuser của bạn dùng get_user_model()
        # và bạn muốn đảm bảo User model đã sẵn sàng, mặc dù thường thì dependency vào migration
        # trước đó của cùng app là đủ nếu migration đó đã xử lý các model của app.
        # Tuy nhiên, để an toàn và rõ ràng, việc giữ cả hai cũng không sao.
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.RunPython(create_superuser),
    ]