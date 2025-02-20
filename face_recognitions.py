#\\\\\\\\\\\\\\\\\\\\\\\\      MODEL     /////////////////////////////////
"""
from django.contrib.auth.models import AbstractUser
from django.db.models import CharField, ImageField, DateTimeField, Model, ForeignKey, CASCADE, TextChoices
from django.db.models.fields import IntegerField, TextField
from django.core.validators import RegexValidator

import os
import face_recognition
class Employee(AbstractUser):
    phone = CharField(max_length=23, default='', validators=[RegexValidator(regex=r'^\+?\d{9,15}$')])
    image = ImageField(upload_to='image/')
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)
    encoding=TextField(null=True,blank=True)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)  # Avval modelni saqlaymiz

        if self.image:  # Agar rasm mavjud bo‘lsa
            image_path = self.image.path  # Rasmning to‘liq yo‘li
            if os.path.exists(image_path):  # Fayl mavjudligini tekshiramiz
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    self.encoding = ",".join(map(str, encodings[0]))  # Encodingni string ko‘rinishida saqlash
                    super().save(update_fields=['encoding'])


class Cameras(Model):
    location = CharField(max_length=255)
    ip_address = CharField(max_length=255)
    created_at = DateTimeField(auto_now_add=True)


class Attendance(Model):
    class TYPE(TextChoices):
        IN = 'in', 'In',
        OUT = 'out', 'Out',

    employee = ForeignKey('apps.Employee', CASCADE, related_name='attendances')
    camera = ForeignKey('apps.Cameras', CASCADE, related_name='attendances')
    timestamp = DateTimeField(auto_now_add=True)
    entry_type = CharField(max_length=255, choices=TYPE.choices)


class WorkSessions(Model):
    class STATUS(TextChoices):
        ACTIVE = 'active', 'Active'
        COMPLETED = 'completed', 'Completed'

    employee = ForeignKey('apps.Employee', CASCADE, related_name='work_sessions')
    check_in = DateTimeField()  # `auto_now_add=True` olib tashlandi
    check_out = DateTimeField(null=True, blank=True)
    duration = IntegerField(default=0, null=True, blank=True)
    status = CharField(max_length=244, choices=STATUS.choices, default=STATUS.COMPLETED)

"""
from datetime import datetime
import cv2
import face_recognition
import numpy as np
import psycopg2
import os

# PostgreSQL bazasiga ulanish
try:
    conn = psycopg2.connect(
        dbname="face",
        user="maxsud",
        password="1",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    # print("✅ Baza bilan ulanish muvaffaqiyatli!")
except Exception as e:
    # print(f"❌ Baza bilan ulanishda xatolik: {e}")
    exit()

# Kameralarni ochish
def find_available_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(cap)
        else:
            break
    return available_cameras

# Barcha ishlayotgan kameralarni olish
cameras = find_available_cameras()

# Barcha xodimlarning yuzlarini bazadan yuklash
cursor.execute("SELECT username, image FROM apps_employee")
known_faces = cursor.fetchall()

known_encodings = []
known_names = []

for name, image_path in known_faces:
    if not image_path:
        continue  # Agar rasm yo'q bo'lsa, o'tkazib yuboriladi

    full_image_path = os.path.join("/Users/admin/PycharmProjects/sdcsdcsdcs/media", image_path)

    if not os.path.exists(full_image_path):
        # print(f"❌ Rasm topilmadi: {full_image_path}")
        continue

    try:
        image = face_recognition.load_image_file(full_image_path)
        encoding = face_recognition.face_encodings(image)

        if encoding:
            known_encodings.append(encoding[0])
            known_names.append(name)
        else:
            print(f"⚠️ Yuz topilmadi: {name}")
    except Exception as e:
        print(f"❌ Rasmni yuklashda xatolik: {e}")

def register_attendance(employee_name, camera_id):
    try:
        cursor.execute("SELECT id FROM apps_employee WHERE username = %s", (employee_name,))
        employee = cursor.fetchone()
        if not employee:
            # print(f"❌ {employee_name} topilmadi!")
            return

        employee_id = employee[0]
        now = datetime.now()

        # Oxirgi tashrifni tekshirish
        cursor.execute("""
            SELECT entry_type FROM apps_attendance 
            WHERE employee_id = %s ORDER BY timestamp DESC LIMIT 1
        """, (employee_id,))
        last_entry = cursor.fetchone()

        if camera_id == 1 and (not last_entry or last_entry[0] != 'in'):
            entry_type = 'in'
        elif camera_id == 2 and last_entry and last_entry[0] == 'in':
            entry_type = 'out'
        else:
            return

        cursor.execute(
            "INSERT INTO apps_attendance (employee_id, camera_id, timestamp, entry_type) VALUES (%s, %s, %s, %s)",
            (employee_id, camera_id, now, entry_type)
        )
        conn.commit()
        # print(f"✅ {entry_type.upper()} saqlandi: {employee_name} ({now}) | Kamera ID: {camera_id}")

        # Work sessionni yangilash
        register_work_session(employee_id, entry_type)

    except Exception as e:
        conn.rollback()
        # print(f"❌ Bazaga yozishda xatolik: {e}")

def register_work_session(employee_id, entry_type):
    now = datetime.now()
    try:
        if entry_type == 'in':
            # Avval mavjud yozuv bor yoki yo‘qligini tekshiramiz
            cursor.execute("""
                SELECT id FROM apps_worksessions 
                WHERE employee_id = %s AND DATE(check_in) = CURRENT_DATE
            """, (employee_id,))
            existing_session = cursor.fetchone()

            if not existing_session:
                cursor.execute("""
                    INSERT INTO apps_worksessions (employee_id, check_in, status) 
                    VALUES (%s, %s, 'active')
                """, (employee_id, now))




        elif entry_type == 'out':

            # Oxirgi 'in' vaqtini olish

            cursor.execute("""

                        SELECT timestamp FROM apps_attendance 
    WHERE employee_id = %s AND entry_type = 'in' 
    ORDER BY timestamp DESC LIMIT 1;

                    """, (employee_id,))

            last_check_in = cursor.fetchone()

            if last_check_in:

                last_check_in = last_check_in[0]

                cursor.execute("""

                            UPDATE apps_worksessions 

                            SET 

                                duration = COALESCE(duration, 0) + 

                                    EXTRACT(EPOCH FROM (%s - %s)), 

                                check_out = %s,
                                status = 'completed'

                            WHERE employee_id = %s 

                            AND DATE(check_in) = CURRENT_DATE

                            RETURNING check_out, duration;

                        """, ( now,last_check_in,now, employee_id))

                result = cursor.fetchone()

                # print(f"✅ Check-out: {result[0]}, Duration: {result[1]}")

            else:

                print("⚠️ Xatolik: Oxirgi 'in' vaqti topilmadi!")

        conn.commit()
        # print(f"✅ Work session yangilandi: {employee_id}, {entry_type} ({now})")

    except Exception as e:
        conn.rollback()
        # print(f"❌ Work session yozishda xatolik: {e}")


def process_frame(img, camera_id):
    small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_img)
    face_encodings = face_recognition.face_encodings(rgb_small_img, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        box_color = (0, 0, 255)  # Qizil - taninmadi

        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if any(matches):
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                box_color = (0, 255, 0)  # Yashil - tanildi
                register_attendance(name, camera_id)

        # Yuz joylashuvini qayta hisoblash
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(img, (left, top), (right, bottom), box_color, 2)
        cv2.rectangle(img, (left, top - 30), (right, top), box_color, cv2.FILLED)
        cv2.putText(img, name, (left + 5, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return img


while True:
    for i, cap in enumerate(cameras):
        isTrue, img = cap.read()
        if isTrue:
            img = process_frame(img, i + 1)
            cv2.imshow(f'Camera {i + 1}', img)

    # Chiqish tugmasi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kameralarni yopish
for cap in cameras:
    cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()










