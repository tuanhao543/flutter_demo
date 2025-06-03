# attendance_api/views.py
import os
import cv2
import numpy as np
import pandas as pd # Mặc dù không thấy sử dụng trực tiếp, có thể cần cho một số thư viện khác hoặc logic cũ
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta # Thêm timedelta
import json
import traceback
from django.utils.dateparse import parse_datetime
from django.conf import settings
from django.core.files.storage import FileSystemStorage # Không thấy sử dụng trực tiếp, có thể bỏ nếu không cần
from django.utils import timezone

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .models import RegisteredUser, AttendanceLog # Đảm bảo import đúng model Django
from .serializers import *
from django.db.models import F, ExpressionWrapper, DurationField, Sum, DateField, Min, Max
from django.db.models.functions import TruncDate # Cast không thấy sử dụng trực tiếp


class ListRegisteredUsersAPI(APIView):
    def get(self, request, *args, **kwargs):
        print("\n--- [ListRegisteredUsersAPI] GET Request Received ---")
        try:
            users = RegisteredUser.objects.all().order_by('name')
            # Kiểm tra nếu model AI chưa được load thì không cần thiết ở đây, trừ khi bạn muốn lọc user dựa trên AI
            # if not ensure_models_loaded(): 
            #     return Response({"error": "AI models not loaded, cannot proceed"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

            if not users.exists():
                print("[ListRegisteredUsersAPI] No registered users found in the database.")
                return Response([], status=status.HTTP_200_OK) # Trả về mảng rỗng

            serializer = RegisteredUserListSerializer(users, many=True)
            print(f"[ListRegisteredUsersAPI] Returning {len(users)} users.")
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            error_msg = "An unexpected error occurred while fetching registered users."
            print(f"[ListRegisteredUsersAPI] Unexpected Exception: {e}")
            print(traceback.format_exc())
            return Response({"error": error_msg, "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        

# --- L2NormalizationLayer (Giữ lại nếu model embedding của bạn thực sự cần nó) ---
class L2NormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L2NormalizationLayer, self).__init__(**kwargs)
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)
    def get_config(self):
        config = super(L2NormalizationLayer, self).get_config()
        return config
    # Bổ sung compute_output_shape nếu bạn gặp lỗi liên quan đến shape
    def compute_output_shape(self, input_shape):
        return input_shape
# -------------------------------------------------------------------------------------



# Biến toàn cục cho models, khởi tạo là None
face_detector_model = None
recognition_model_instance = None
# Sử dụng dict để lưu trạng thái chi tiết hơn, giúp debug và tránh thử tải lại liên tục nếu lỗi
models_load_status = {"loaded": False, "error_message": None, "attempted": False}

# Lock để tránh tải model đồng thời nếu có nhiều worker (ít quan trọng với workers=1, nhưng để thực hành tốt)
# from threading import Lock
# model_load_lock = Lock() # Nếu bạn tăng số worker lên > 1, hãy bật lại Lock

def ensure_models_loaded():
    global face_detector_model, recognition_model_instance, models_load_status
    
    # with model_load_lock: # Nếu dùng Lock
    if models_load_status["loaded"]:
        return True
    
    # Nếu đã thử tải và lỗi, không thử lại ngay trong cùng request, trả về lỗi
    if models_load_status["attempted"] and models_load_status["error_message"]:
        print(f"Models previously failed to load: {models_load_status['error_message']}")
        return False

    print("Attempting to lazy load AI Models...")
    models_load_status["attempted"] = True # Đánh dấu đã cố gắng tải

    try:
        # Load face detector
        print("Lazy Loading Face Detector...")
        if not os.path.exists(settings.FACE_DETECTOR_PROTOTXT_PATH) or \
           not os.path.exists(settings.FACE_DETECTOR_CAFFEMODEL_PATH):
            raise FileNotFoundError("Face Detector model files (prototxt or caffemodel) missing.")
        face_detector_model = cv2.dnn.readNetFromCaffe(settings.FACE_DETECTOR_PROTOTXT_PATH, settings.FACE_DETECTOR_CAFFEMODEL_PATH)
        print("Face Detector lazy loaded.")

        # Load recognition model
        print("Lazy Loading Recognition Model...")
        if not os.path.exists(settings.RECOGNITION_MODEL_PATH_SETTING):
            raise FileNotFoundError(f"Recognition Model file missing: {settings.RECOGNITION_MODEL_PATH_SETTING}")
        
        custom_objects = {}
        # Kiểm tra xem L2NormalizationLayer có được định nghĩa không và có cần cho model không
        if 'L2NormalizationLayer' in globals() and hasattr(tf.keras.models, 'load_model'): # Thêm kiểm tra
             custom_objects['L2NormalizationLayer'] = L2NormalizationLayer

        recognition_model_instance = tf.keras.models.load_model(
            settings.RECOGNITION_MODEL_PATH_SETTING,
            custom_objects=custom_objects if custom_objects else None,
            compile=False # Quan trọng: compile=False để tải nhanh hơn nếu không cần training
        )
        print("Recognition Model lazy loaded.")
        
        models_load_status["loaded"] = True
        models_load_status["error_message"] = None # Xóa lỗi cũ nếu tải thành công
        print("AI Models lazy loaded successfully.")
        return True

    except Exception as e:
        error_msg = f"CRITICAL ERROR: Could not lazy load AI models: {e}"
        print(error_msg)
        print(traceback.format_exc())
        models_load_status["loaded"] = False
        models_load_status["error_message"] = str(e)
        # Đảm bảo reset các model về None nếu tải lỗi
        face_detector_model = None
        recognition_model_instance = None
        return False

# --- Hàm Tiện ích AI ---
def preprocess_facenet_api(image_array):
    image_array = tf.cast(image_array, tf.float32)
    image_array = (image_array - 127.5) / 128.0 # Chuẩn hóa cho FaceNet
    return image_array

def detect_faces_api(image_np):
    if face_detector_model is None: # Kiểm tra model đã được tải chưa
        print("Face detector model is not loaded. Call ensure_models_loaded() first or check logs.")
        return [] # Trả về rỗng nếu model chưa sẵn sàng
        
    (h, w) = image_np.shape[:2]
    blob = cv2.dnn.blobFromImage(image_np, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector_model.setInput(blob)
    detections = face_detector_model.forward()
    faces_found = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > settings.FACE_CONFIDENCE_THRESHOLD_SETTING:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Đảm bảo tọa độ hợp lệ và kích thước crop dương
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            if startX < endX and startY < endY: # Crop phải có kích thước
                face_crop = image_np[startY:endY, startX:endX]
                if face_crop.size > 0: # Đảm bảo crop không rỗng
                    faces_found.append({'box': (startX, startY, endX, endY), 'crop': face_crop, 'confidence': confidence})
    # Sắp xếp các khuôn mặt tìm thấy theo kích thước (khuôn mặt lớn nhất lên đầu)
    faces_found.sort(key=lambda f: (f['box'][2]-f['box'][0])*(f['box'][3]-f['box'][1]), reverse=True)
    return faces_found


def get_embedding_api(face_crop_np):
    if recognition_model_instance is None: # Kiểm tra model
        print("Recognition model is not loaded. Call ensure_models_loaded() first or check logs.")
        return None # Trả về None nếu model chưa sẵn sàng
    try:
        face_resized = cv2.resize(face_crop_np, settings.RECOGNITION_INPUT_SIZE_SETTING) # (160, 160)
        
        # Đảm bảo ảnh có 3 channel (BGR)
        if face_resized.ndim == 2: # Ảnh xám
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2BGR)
        elif face_resized.shape[2] == 4: # Ảnh BGRA (có kênh alpha)
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGRA2BGR)

        face_array = img_to_array(face_resized) # Chuyển sang RGB theo mặc định của Keras
        face_array = np.expand_dims(face_array, axis=0) # Thêm batch dimension
        face_array = preprocess_facenet_api(face_array) # Chuẩn hóa
        
        # Sử dụng predict trực tiếp, không cần session với TF2
        embedding = recognition_model_instance.predict(face_array, verbose=0)[0] # verbose=0 để không in log predict
        return embedding
    except Exception as e:
        print(f"Error in get_embedding_api: {e}")
        traceback.print_exc()
        return None

# --- API Views ---
class RegisterUserAPI(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if not ensure_models_loaded():
            # models_load_status["error_message"] sẽ chứa lỗi chi tiết
            return Response({"error": f"AI models could not be loaded on server. Details: {models_load_status['error_message']}"}, 
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
        # Kiểm tra lại một lần nữa phòng trường hợp logic trong ensure_models_loaded phức tạp
        if face_detector_model is None or recognition_model_instance is None:
             return Response({"error": "AI models are not available after attempting to load. Please check server logs."}, 
                             status=status.HTTP_503_SERVICE_UNAVAILABLE)

        name = request.data.get('name')
        images = request.FILES.getlist('images')

        if not name or not images:
            return Response({"error": "Name and images are required."}, status=status.HTTP_400_BAD_REQUEST)

        if RegisteredUser.objects.filter(name=name).exists():
            return Response({"error": f"User with name '{name}' already exists."}, status=status.HTTP_400_BAD_REQUEST)

        user_embeddings = []
        # FileSystemStorage và lưu ảnh là tùy chọn, nếu chỉ cần embedding thì không cần
        # fs = FileSystemStorage() 
        # user_image_dir_relative = os.path.join('user_images_registration', name) 
        # actual_user_image_dir_on_server = os.path.join(settings.MEDIA_ROOT, user_image_dir_relative)
        # os.makedirs(actual_user_image_dir_on_server, exist_ok=True)
        # saved_image_paths_for_response = []

        for idx, image_file in enumerate(images):
            try:
                image_stream = image_file.read()
                # cv2.imdecode cần một mảng numpy từ buffer
                image_np_arr = np.frombuffer(image_stream, np.uint8)
                image_np = cv2.imdecode(image_np_arr, cv2.IMREAD_COLOR) # Đọc ảnh màu

                if image_np is None: 
                    print(f"Could not decode image {idx} for user {name}. File: {image_file.name}")
                    continue
                
                detected_faces = detect_faces_api(image_np)
                if detected_faces: # Lấy khuôn mặt lớn nhất
                    face_crop = detected_faces[0]['crop']
                    embedding = get_embedding_api(face_crop)
                    if embedding is not None:
                        user_embeddings.append(embedding)
                    else:
                        print(f"Could not get embedding for image {idx} of user {name}")
                else:
                    print(f"No face detected in image {idx} for user {name}")
            except Exception as e:
                print(f"Error processing image {idx} for {name}: {e}")
                traceback.print_exc()
                # Có thể trả về lỗi chi tiết hơn cho client nếu cần, hoặc chỉ log server
                # return Response({"error": f"Error processing image {idx}: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                continue # Bỏ qua ảnh lỗi và tiếp tục với các ảnh khác

        if not user_embeddings:
            return Response({"error": "No valid faces found or embeddings extracted from the uploaded images."}, 
                            status=status.HTTP_400_BAD_REQUEST)

        avg_embedding = np.mean(user_embeddings, axis=0)
        avg_embedding_str = json.dumps(avg_embedding.tolist()) # Chuyển sang list rồi mới JSON dump

        user = RegisteredUser.objects.create(name=name, average_embedding=avg_embedding_str)
        serializer = RegisteredUserSerializer(user) # Đảm bảo serializer có is_admin
        
        return Response({
            "message": f"User '{name}' registered successfully with {len(user_embeddings)} valid face(s).",
            "user": serializer.data,
        }, status=status.HTTP_201_CREATED)


class CheckInAPI(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        print("\n--- [CheckInAPI] POST Request Received ---")
        print(f"[CheckInAPI] Request Content-Type: {request.content_type}")
        print(f"[CheckInAPI] Request Data (form fields): {request.data}") # Sẽ chứa latitude, longitude nếu có
        print(f"[CheckInAPI] Request FILES (uploaded files): {request.FILES}")

        # 1. Kiểm tra model AI đã sẵn sàng chưa
        if not ensure_models_loaded(): # Hàm này bạn đã có để lazy load model
            print(f"[CheckInAPI] AI models failed to load. Error: {models_load_status.get('error_message', 'Unknown error')}")
            return Response({"error": f"AI models could not be loaded. Details: {models_load_status.get('error_message', 'Please check server logs')}"}, 
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
        if face_detector_model is None or recognition_model_instance is None:
             print("[CheckInAPI] AI models are still None after attempting to load. Critical error.")
             return Response({"error": "AI models are not available. Please check server logs."}, 
                             status=status.HTTP_503_SERVICE_UNAVAILABLE)

        # 2. Lấy file ảnh từ request
        image_file_obj = request.FILES.get('image')
        if not image_file_obj:
            print("[CheckInAPI] Validation Error: 'image' file is missing.")
            return Response({"error": "'image' file is required for check-in."}, status=status.HTTP_400_BAD_REQUEST)
        
        print(f"[CheckInAPI] Received image file: {image_file_obj.name}")

        # 3. Lấy dữ liệu kinh độ, vĩ độ từ request.data
        latitude_str = request.data.get('latitude')
        longitude_str = request.data.get('longitude')
        
        latitude = None
        longitude = None

        if latitude_str:
            try:
                latitude = float(latitude_str)
            except ValueError:
                print(f"[CheckInAPI] Invalid latitude value received: '{latitude_str}'. Will be stored as null.")
                # Không trả lỗi, chỉ ghi nhận và lưu là null
        if longitude_str:
            try:
                longitude = float(longitude_str)
            except ValueError:
                print(f"[CheckInAPI] Invalid longitude value received: '{longitude_str}'. Will be stored as null.")
        
        print(f"[CheckInAPI] Received Location - Lat: {latitude}, Lng: {longitude}")

        try:
            # 4. Xử lý ảnh và nhận diện
            image_stream = image_file_obj.read()
            image_np_arr = np.frombuffer(image_stream, np.uint8)
            image_np = cv2.imdecode(image_np_arr, cv2.IMREAD_COLOR)

            if image_np is None:
                print("[CheckInAPI] Invalid image format or corrupted image.")
                return Response({"error": "Invalid image format or corrupted image."}, status=status.HTTP_400_BAD_REQUEST)

            detected_faces = detect_faces_api(image_np) # Hàm tiện ích của bạn
            if not detected_faces:
                print("[CheckInAPI] No face detected in the image.")
                return Response({"message": "No face detected in the image.", "user_name": "N/A", "is_admin": False}, status=status.HTTP_200_OK) 

            main_face_crop = detected_faces[0]['crop']
            current_embedding = get_embedding_api(main_face_crop) # Hàm tiện ích của bạn

            if current_embedding is None:
                print("[CheckInAPI] Could not extract face embedding from the detected face.")
                return Response({"error": "Could not extract face embedding.", "is_admin": False}, 
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            print("[CheckInAPI] Embedding extracted from uploaded image.")

            # 5. So sánh với người dùng đã đăng ký
            all_users = RegisteredUser.objects.all()
            if not all_users.exists():
                print("[CheckInAPI] No users registered in the system.")
                return Response({"message": "No users registered in the system yet.", "user_name": "Unknown", "is_admin": False}, 
                                status=status.HTTP_200_OK)

            known_db_embeddings = []
            known_db_users_map = {} # Map index với user object để truy xuất dễ dàng
            
            for user_profile in all_users:
                try:
                    stored_embedding_list = json.loads(user_profile.average_embedding)
                    known_db_embeddings.append(np.array(stored_embedding_list))
                    known_db_users_map[len(known_db_embeddings)-1] = user_profile
                except json.JSONDecodeError:
                    print(f"[CheckInAPI] Error decoding embedding for user {user_profile.name}. Skipping this user for recognition.")
                    continue
            
            if not known_db_embeddings:
                print("[CheckInAPI] No valid user embeddings found in the database.")
                return Response({"message": "No valid user embeddings found in the database.", "user_name": "Unknown", "is_admin": False}, 
                                status=status.HTTP_200_OK)
            print(f"[CheckInAPI] Comparing with {len(known_db_embeddings)} known user embeddings.")

            similarities = cosine_similarity([current_embedding], np.array(known_db_embeddings))[0]
            best_match_idx = np.argmax(similarities)
            max_similarity = similarities[best_match_idx]
            print(f"[CheckInAPI] Max similarity: {max_similarity:.4f} at index {best_match_idx}")

            best_match_user = None
            if max_similarity >= settings.RECOGNITION_THRESHOLD_SETTING and best_match_idx in known_db_users_map:
                best_match_user = known_db_users_map[best_match_idx]
                print(f"[CheckInAPI] User recognized: {best_match_user.name} (ID: {best_match_user.id})")
            else:
                print(f"[CheckInAPI] User not recognized or similarity ({max_similarity:.4f}) below threshold ({settings.RECOGNITION_THRESHOLD_SETTING}).")
                return Response({
                    "message": "User not recognized or similarity below threshold.",
                    "user_name": "Unknown",
                    "is_admin": False,
                    "similarity": float(max_similarity)
                }, status=status.HTTP_200_OK)

            # 6. Ghi log chấm công (check-in hoặc check-out) với thông tin vị trí
            now = timezone.now() # Sử dụng timezone.now() để có thời gian aware
            last_log = AttendanceLog.objects.filter(user=best_match_user, check_out_time__isnull=True).order_by('-check_in_time').first()
            
            attendance_status_msg = ""
            log_entry = None # Để tham chiếu đến log entry mới hoặc được cập nhật

            if last_log: # Nếu có log check-in chưa check-out -> thực hiện check-out
                last_log.check_out_time = now
                # Quyết định xem có cập nhật vị trí khi check-out không.
                # Thông thường, vị trí check-in quan trọng hơn.
                # Nếu muốn cập nhật cả vị trí check-out:
                # last_log.latitude = latitude
                # last_log.longitude = longitude
                last_log.save()
                log_entry = last_log
                attendance_status_msg = "Check-out successful"
                print(f"[CheckInAPI] Check-out recorded for {best_match_user.name} at {now}. Log ID: {last_log.id}")
            else: # Nếu không có log nào đang mở -> thực hiện check-in mới
                log_entry = AttendanceLog.objects.create(
                    user=best_match_user, 
                    check_in_time=now,
                    latitude=latitude,    # Lưu vị trí khi check-in
                    longitude=longitude   # Lưu vị trí khi check-in
                )
                attendance_status_msg = "Check-in successful"
                print(f"[CheckInAPI] Check-in recorded for {best_match_user.name} at {now} with Location (Lat: {latitude}, Lng: {longitude}). Log ID: {log_entry.id}")
            
            # 7. Trả về response thành công
            return Response({
                "message": attendance_status_msg,
                "user_name": best_match_user.name,
                "user_id": best_match_user.id,
                "is_admin": best_match_user.is_admin,
                "similarity": float(max_similarity), # Có thể giữ lại để debug
                "timestamp": now.isoformat(), # Thời gian server xử lý
                # Tùy chọn: Trả về cả kinh độ, vĩ độ đã lưu nếu cần
                # "latitude": log_entry.latitude if log_entry else None,
                # "longitude": log_entry.longitude if log_entry else None,
            }, status=status.HTTP_200_OK)

        except Exception as e:
            # Xử lý các lỗi không mong muốn khác
            print(f"[CheckInAPI] Unexpected Error during processing: {e}")
            print(traceback.format_exc()) # In đầy đủ traceback để debug
            return Response({"error": "An unexpected error occurred during the check-in/out process.", "is_admin": False, "details": str(e)}, 
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class UserWorkStatsAPI(APIView):
    def get(self, request, *args, **kwargs):
        print("\n[UserWorkStatsAPI] Received GET request for detailed daily stats")
        try:
            user_id_str = request.query_params.get('user_id')
            month_str = request.query_params.get('month')
            year_str = request.query_params.get('year')
            days_str = request.query_params.get('days_limit')

            print(f"[UserWorkStatsAPI] Query params: user_id={user_id_str}, month={month_str}, year={year_str}, days_limit={days_str}")

            if not user_id_str: # User ID là bắt buộc cho API này
                 return Response({"error": "User ID is required for detailed daily statistics.", "details": "Please provide 'user_id' query parameter."}, 
                                 status=status.HTTP_400_BAD_REQUEST)
            try:
                target_user_id = int(user_id_str)
                if not RegisteredUser.objects.filter(id=target_user_id).exists():
                    return Response({"error": f"User with ID {target_user_id} does not exist."}, status=status.HTTP_404_NOT_FOUND)
            except ValueError:
                return Response({"error": "Invalid User ID format. Must be an integer."}, status=status.HTTP_400_BAD_REQUEST)

            # Xác định khoảng thời gian
            if month_str and year_str:
                try:
                    month = int(month_str)
                    year = int(year_str)
                    if not (1 <= month <= 12):
                        raise ValueError("Month must be between 1 and 12.")
                    # Tạo ngày đầu tháng và đầu tháng sau (đã aware timezone)
                    start_date = timezone.make_aware(datetime(year, month, 1), timezone.get_current_timezone())
                    if month == 12:
                        end_date_exclusive = timezone.make_aware(datetime(year + 1, 1, 1), timezone.get_current_timezone())
                    else:
                        end_date_exclusive = timezone.make_aware(datetime(year, month + 1, 1), timezone.get_current_timezone())
                except ValueError as e:
                    return Response({"error": f"Invalid month/year: {e}"}, status=status.HTTP_400_BAD_REQUEST)
            elif days_str:
                try:
                    days_limit = int(days_str)
                    if days_limit <= 0:
                        raise ValueError("Days limit must be greater than 0.")
                    # Tính từ đầu ngày hôm nay lùi lại N ngày
                    end_of_today_exclusive = (timezone.now() + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                    start_date = end_of_today_exclusive - timedelta(days=days_limit)
                    end_date_exclusive = end_of_today_exclusive
                except ValueError as e:
                    return Response({"error": f"Invalid days_limit: {e}"}, status=status.HTTP_400_BAD_REQUEST)
            else: # Mặc định 7 ngày gần nhất nếu không có month/year hoặc days_limit
                end_of_today_exclusive = (timezone.now() + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                start_date = end_of_today_exclusive - timedelta(days=7)
                end_date_exclusive = end_of_today_exclusive

            print(f"[UserWorkStatsAPI] Date range (UTC): {start_date.astimezone(timezone.utc)} to {end_date_exclusive.astimezone(timezone.utc)} for User ID: {target_user_id}")

            # Query logs đã hoàn thành (có cả check-in và check-out)
            logs_query = AttendanceLog.objects.filter(
                user_id=target_user_id,
                check_in_time__gte=start_date,
                check_in_time__lt=end_date_exclusive, # check_in_time phải nhỏ hơn ngày kết thúc
                check_out_time__isnull=False,          # Phải có check-out
                user__isnull=False                     # Đảm bảo user tồn tại (dù ForeignKey thường đã đảm bảo)
            ).annotate(
                work_duration_calc=ExpressionWrapper( # Tính thời gian làm việc cho mỗi log
                    F('check_out_time') - F('check_in_time'),
                    output_field=DurationField()
                )
            ).filter( # Loại bỏ các log có thời gian làm việc âm hoặc bằng 0 (nếu có lỗi dữ liệu)
                work_duration_calc__gt=timedelta(seconds=0) 
            )

            if not logs_query.exists():
                print("[UserWorkStatsAPI] No attendance data found for the user and criteria.")
                return Response([], status=status.HTTP_200_OK) # Trả về mảng rỗng

            # Nhóm theo ngày, tính tổng thời gian, giờ vào sớm nhất, giờ ra muộn nhất
            daily_summary_qs = logs_query.annotate(
                date=TruncDate('check_in_time', output_field=DateField()) # Truncate check_in_time về ngày
            ).values(
                'date',
                'user__name' 
            ).annotate(
                total_duration_seconds=Sum(ExpressionWrapper(F('work_duration_calc'), output_field=DurationField())), # Sum duration
                earliest_check_in_of_day=Min('check_in_time'), # Giờ vào sớm nhất trong ngày
                latest_check_out_of_day=Max('check_out_time')    # Giờ ra muộn nhất trong ngày
            ).order_by('date') # Sắp xếp theo ngày tăng dần

            print(f"[UserWorkStatsAPI] Found {len(daily_summary_qs)} summary entries from DB.")

            chart_data = []
            local_tz = timezone.get_current_timezone() # Lấy timezone hiện tại của server (đã set trong settings.py)

            for entry in daily_summary_qs:
                if entry['total_duration_seconds'] is None: # Bỏ qua nếu không có duration
                    continue
                
                total_hours = entry['total_duration_seconds'].total_seconds() / 3600.0
                
                # Chuyển đổi thời gian sang múi giờ local của server và định dạng HH:MM
                earliest_in_local = entry['earliest_check_in_of_day'].astimezone(local_tz)
                latest_out_local = entry['latest_check_out_of_day'].astimezone(local_tz)

                chart_data.append({
                    "date": entry['date'].isoformat(), # Ngày dạng YYYY-MM-DD
                    "hours": round(total_hours, 2),
                    "user_name": entry['user__name'], # Tên user
                    "check_in_time_of_day": earliest_in_local.strftime('%H:%M'), # Giờ vào:phút
                    "check_out_time_of_day": latest_out_local.strftime('%H:%M') # Giờ ra:phút
                })
            
            print(f"[UserWorkStatsAPI] Processed chart_data (first 2 entries if available): {chart_data[:2]}")
            return Response(chart_data, status=status.HTTP_200_OK)

        except ValueError as ve: # Bắt lỗi cụ thể cho việc parse số
            error_msg = f"Invalid input format: {ve}"
            print(f"[UserWorkStatsAPI] ValueError: {error_msg}")
            traceback.print_exc()
            return Response({"error": error_msg, "details": str(ve)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            error_msg = "An unexpected error occurred while generating statistics."
            print(f"[UserWorkStatsAPI] Unexpected Exception: {e}")
            print(traceback.print_exc()) # In đầy đủ traceback để debug
            return Response({"error": error_msg, "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
class UserAttendanceLogsAPI(APIView):
    def get(self, request, *args, **kwargs):
        print("\n--- [UserAttendanceLogsAPI] GET Request Received ---")
        try:
            user_id_str = request.query_params.get('user_id')
            date_from_str = request.query_params.get('date_from') # YYYY-MM-DD
            date_to_str = request.query_params.get('date_to')     # YYYY-MM-DD

            print(f"[UserAttendanceLogsAPI] Query params: user_id={user_id_str}, date_from={date_from_str}, date_to={date_to_str}")

            if not user_id_str:
                return Response({"error": "User ID is required."}, status=status.HTTP_400_BAD_REQUEST)
            if not date_from_str or not date_to_str:
                return Response({"error": "Both date_from and date_to are required."}, status=status.HTTP_400_BAD_REQUEST)

            try:
                target_user_id = int(user_id_str)
                # parse_datetime an toàn hơn vì nó có thể xử lý cả date và datetime
                # nhưng chúng ta chỉ quan tâm đến phần ngày
                date_from = parse_datetime(date_from_str + "T00:00:00Z") # Thêm giờ, phút, giây và UTC
                date_to = parse_datetime(date_to_str + "T23:59:59.999999Z") 

                if date_from is None or date_to is None:
                    raise ValueError("Invalid date format. Use YYYY-MM-DD.")

            except ValueError as ve:
                return Response({"error": f"Invalid parameter format: {ve}"}, status=status.HTTP_400_BAD_REQUEST)

            if not RegisteredUser.objects.filter(id=target_user_id).exists():
                return Response({"error": f"User with ID {target_user_id} does not exist."}, status=status.HTTP_404_NOT_FOUND)

            print(f"[UserAttendanceLogsAPI] Fetching logs for User ID: {target_user_id} from {date_from} to {date_to}")

            logs = AttendanceLog.objects.filter(
                user_id=target_user_id,
                check_in_time__gte=date_from,
                check_in_time__lte=date_to # Bao gồm cả ngày kết thúc
            ).order_by('check_in_time') # Sắp xếp theo thời gian check-in

            if not logs.exists():
                print("[UserAttendanceLogsAPI] No attendance logs found for the criteria.")
                return Response([], status=status.HTTP_200_OK)

            serializer = AttendanceLogDetailSerializer(logs, many=True)
            print(f"[UserAttendanceLogsAPI] Returning {len(logs)} attendance logs.")
            return Response(serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            error_msg = "An unexpected error occurred while fetching attendance logs."
            print(f"[UserAttendanceLogsAPI] Unexpected Exception: {e}")
            print(traceback.format_exc())
            return Response({"error": error_msg, "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)