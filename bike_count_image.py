from ultralytics import YOLO
import cv2

# بارگذاری مدل YOLOv8 آماده
model = YOLO("yolov8n.pt")

# مسیر عکس
image_path = "test.jpg"  # مطمئن شو که این عکس داخل پوشه پروژه هست
image = cv2.imread(image_path)

# اجرای مدل روی عکس
results = model(image)[0]

# جدا کردن اشیاء تشخیص‌داده‌شده
boxes = results.boxes

# لیست جدا برای افراد و دوچرخه‌ها
persons = []
bicycles = []

for box in boxes:
    cls = int(box.cls[0])
    if cls == 0:       # کلاس 0 = شخص
        persons.append(box)
    elif cls == 1:     # کلاس 1 = دوچرخه
        bicycles.append(box)

# شمارش افرادی که روی دوچرخه هستن
count = 0

for person in persons:
    px1, py1, px2, py2 = person.xyxy[0]
    for bike in bicycles:
        bx1, by1, bx2, by2 = bike.xyxy[0]

        # محاسبه مرکز بدن فرد
        px_center = (px1 + px2) / 2
        py_center = (py1 + py2) / 2

        # اگه مرکز بدن فرد داخل محدوده دوچرخه باشه
        if bx1 < px_center < bx2 and by1 < py_center < by2:
            count += 1
            break

print(f"تعداد افرادی که روی دوچرخه هستن: {count}")
# این کد برای تشخیص افراد سوار بر دوچرخه هستن 