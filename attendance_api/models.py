# attendance_api/models.py
from django.db import models
import uuid # Để tạo ID duy nhất nếu cần

class RegisteredUser(models.Model):
    name = models.CharField(max_length=100, unique=True, help_text="Tên người dùng (không dấu, không cách)")
    average_embedding = models.TextField(help_text="JSON string of the average face embedding vector")
    registration_date = models.DateTimeField(auto_now_add=True)
    is_admin = models.BooleanField(default=False) # <<< THÊM TRƯỜNG NÀY

    def __str__(self):
        return f"{self.name}{' (Admin)' if self.is_admin else ''}"

class AttendanceLog(models.Model):
    # log_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(RegisteredUser, on_delete=models.CASCADE, related_name='attendance_logs')
    check_in_time = models.DateTimeField()
    check_out_time = models.DateTimeField(null=True, blank=True)
    # status = models.CharField(max_length=50, blank=True, null=True) # Optional: 'Checked In', 'Masked'

    def __str__(self):
        return f"{self.user.name} - In: {self.check_in_time} - Out: {self.check_out_time or 'N/A'}"

    class Meta:
        ordering = ['-check_in_time']