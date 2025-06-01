# demo_flutter/settings.py (Hoặc tên project của bạn)

import os
from pathlib import Path
import environ # Đảm bảo bạn đã pip install django-environ

# Khởi tạo django-environ
# Nó sẽ tự động tìm file .env trong thư mục gốc của project nếu có (cho local development)
# Trên Render, nó sẽ đọc từ các biến môi trường của Render.
env = environ.Env(
    # Đặt kiểu và giá trị mặc định cho các biến môi trường
    DEBUG=(bool, False), # Mặc định DEBUG là False cho production
    ALLOWED_HOSTS=(list, []),
    # Các biến cho CSDL nếu không dùng DATABASE_URL
    DB_NAME=(str, 'default_db_name_if_not_set'),
    DB_USER=(str, 'default_db_user_if_not_set'),
    DB_PASSWORD=(str, 'default_db_password_if_not_set'),
    DB_HOST=(str, 'localhost'),
    DB_PORT=(str, '5432'),
    # Các biến AI model paths (có thể để mặc định nếu đường dẫn không thay đổi)
    # Không cần thiết phải đưa tất cả vào env nếu chúng không thay đổi giữa các môi trường
)

# Đọc file .env (nếu có, chủ yếu cho local development)
# File này KHÔNG NÊN commit lên Git nếu chứa thông tin nhạy cảm.
# Render sẽ sử dụng các biến môi trường được thiết lập trên dashboard của nó.
ENV_PATH = os.path.join(Path(__file__).resolve().parent.parent, '.env')
if os.path.exists(ENV_PATH):
    print(f"Reading environment variables from: {ENV_PATH}")
    environ.Env.read_env(ENV_PATH)
else:
    print(f".env file not found at {ENV_PATH}. Using system environment variables or defaults.")


# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/dev/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
# Lấy SECRET_KEY từ biến môi trường. Cung cấp một giá trị mặc định YẾU chỉ cho local dev nếu biến không được set.
# Trên Render, bạn PHẢI đặt biến môi trường SECRET_KEY.
# Chuỗi secret key bạn cung cấp: nn$eq9$^11r8oa2+g3((h#gn!#_-dz@e_y*5-#h*h!aq0g#*n#
SECRET_KEY = env('SECRET_KEY', default='6!ubmyuy@%eygfoinkqsp8yc+kt3e6lxl8n_bfwm@#%!b)d%&l')
# LƯU Ý: Giá trị default ở trên là YẾU và chỉ dùng để code chạy local khi chưa set env.
# Hãy đảm bảo bạn đã đặt SECRET_KEY thực sự (nn$eq9$^11r8oa2+g3((h#gn!#_-dz@e_y*5-#h*h!aq0g#*n#)
# làm biến môi trường trên Render.

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env('DEBUG') # Sẽ lấy từ biến môi trường. Render thường tự đặt là False.

# ALLOWED_HOSTS sẽ được đọc từ biến môi trường, phân tách bằng dấu phẩy.
# Ví dụ trên Render: your-app-name.onrender.com,www.your-app-name.onrender.com
# Mặc định cho phép localhost và 127.0.0.1 cho local development.
ALLOWED_HOSTS = env.list('ALLOWED_HOSTS', default=['localhost', '127.0.0.1','demo-flutter.onrender.com'])
# Nếu bạn deploy lên Render, Render sẽ tự động thêm domain dạng *.onrender.com
# nhưng bạn có thể muốn thêm domain cụ thể của mình vào đây qua biến môi trường trên Render.
# Ví dụ: ALLOWED_HOSTS trên Render có thể là: "my-attendance-api.onrender.com,192.168.1.34" (nếu bạn vẫn muốn test từ IP local)

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'whitenoise.runserver_nostatic', # <<< Thêm cho Whitenoise, phải đứng trước staticfiles
    'django.contrib.staticfiles',    # <<< staticfiles phải sau whitenoise
    'rest_framework',
    'attendance_api',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware', # <<< Thêm Whitenoise
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'demo_flutter.urls' # Thay 'demo_flutter' bằng tên project của bạn nếu khác

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'demo_flutter.wsgi.application' # Thay 'demo_flutter' nếu cần


# Database
# https://docs.djangoproject.com/en/dev/ref/settings/#databases

# Ưu tiên sử dụng DATABASE_URL từ biến môi trường (Render sẽ cung cấp)
# Nếu không có, sử dụng cấu hình mặc định (ví dụ cho local PostgreSQL hoặc SQLite)
DATABASES = {
    'default': env.db_url(
        'DATABASE_URL', # Tên biến môi trường mà Render sẽ cung cấp
        default=f"postgres://{env('DB_USER')}:{env('DB_PASSWORD')}@{env('DB_HOST')}:{env('DB_PORT')}/{env('DB_NAME')}"
        # Hoặc fallback về SQLite cho local dev đơn giản nhất nếu không muốn cài PostgreSQL local:
        # default=f"sqlite:///{os.path.join(BASE_DIR, 'db.sqlite3')}"
    )
}
# Nếu bạn muốn dùng các biến DB_USER, DB_PASSWORD... riêng lẻ thay vì DATABASE_URL,
# thì cấu hình DATABASES như sau:
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.postgresql',
#         'NAME': env('DB_NAME'),
#         'USER': env('DB_USER'),
#         'PASSWORD': env('DB_PASSWORD'),
#         'HOST': env('DB_HOST'),
#         'PORT': env('DB_PORT'),
#     }
# }
# Và bạn cần đặt các biến DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT trên Render.
# Tuy nhiên, dùng DATABASE_URL thường tiện hơn trên Render.


# Password validation
# https://docs.djangoproject.com/en/dev/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',},
]


# Internationalization
# https://docs.djangoproject.com/en/dev/topics/i18n/
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Ho_Chi_Minh'
USE_I18N = True
USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/dev/howto/static-files/
STATIC_URL = 'static/'
# Thư mục mà `collectstatic` sẽ gom tất cả các file tĩnh vào đó
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles_collected')
# Bật Whitenoise để nén và cache file tĩnh hiệu quả
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
# (Tùy chọn) Nếu bạn có thư mục static chung ở gốc project:
# STATICFILES_DIRS = [os.path.join(BASE_DIR, "static")]


# Media files (User uploads)
MEDIA_URL = '/media/'
# Lưu ý quan trọng về MEDIA_ROOT trên các nền tảng PaaS như Render:
# File system trên các nền tảng này thường là ephemeral (tạm thời), nghĩa là file bạn upload
# có thể bị mất sau mỗi lần deploy hoặc khi instance khởi động lại.
# GIẢI PHÁP TỐT NHẤT: Dùng dịch vụ lưu trữ bên ngoài như AWS S3, Google Cloud Storage, Cloudinary.
# Cho demo nhỏ hoặc nếu Render có persistent disk và bạn đã cấu hình:
MEDIA_ROOT = os.path.join(BASE_DIR, 'mediafiles_collected') # Hoặc đường dẫn đến persistent disk


# Default primary key field type
# https://docs.djangoproject.com/en/dev/ref/settings/#default-auto-field
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# Cấu hình cho Django REST framework
REST_FRAMEWORK = {
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.FormParser',
        'rest_framework.parsers.MultiPartParser',
    ],
}

# Đường dẫn đến AI Models (giữ nguyên nếu chúng nằm trong code và được deploy cùng)
AI_MODELS_BASE_PATH = os.path.join(BASE_DIR, 'attendance_api', 'ml_models')
FACE_DETECTOR_PROTOTXT_PATH = os.path.join(AI_MODELS_BASE_PATH, "deploy.prototxt")
FACE_DETECTOR_CAFFEMODEL_PATH = os.path.join(AI_MODELS_BASE_PATH, "res10_300x300_ssd_iter_140000.caffemodel")
# MASK_MODEL_PATH_SETTING = os.path.join(AI_MODELS_BASE_PATH, 'mask_detection.keras')
RECOGNITION_MODEL_PATH_SETTING = os.path.join(AI_MODELS_BASE_PATH, 'face_embedding_model.keras')

# Các hằng số
FACE_CONFIDENCE_THRESHOLD_SETTING = 0.6
RECOGNITION_THRESHOLD_SETTING = 0.55
# MASK_INPUT_SIZE_SETTING = (224, 224)
RECOGNITION_INPUT_SIZE_SETTING = (160, 160)

# (Tùy chọn) Cấu hình CSRF cho API nếu dùng SessionAuthentication
# Nếu API của bạn chủ yếu dùng TokenAuthentication, điều này có thể không cần thiết.
# CSRF_TRUSTED_ORIGINS = env.list('CSRF_TRUSTED_ORIGINS', default=[]) # Ví dụ: ['https://your-app-name.onrender.com']

# (Tùy chọn) Cấu hình CORS nếu frontend Flutter (web) hoặc các domain khác cần gọi API
# pip install django-cors-headers
# INSTALLED_APPS += ['corsheaders']
# MIDDLEWARE = ['corsheaders.middleware.CorsMiddleware'] + MIDDLEWARE
# CORS_ALLOWED_ORIGINS = env.list('CORS_ALLOWED_ORIGINS', default=[]) # Ví dụ: ['http://localhost:3000', 'https://your-flutter-web-domain.com']
# CORS_ALLOW_ALL_ORIGINS = env.bool('CORS_ALLOW_ALL_ORIGINS', default=False) # Cẩn thận khi dùng True