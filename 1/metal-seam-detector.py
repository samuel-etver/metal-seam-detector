import cv2
import numpy as np

# Укажите имя обрабатываемого файла
IMAGE_NAME = 'x2.jpg'

# 1. Загрузка изображения
img = cv2.imread(IMAGE_NAME)
if img is None:
    print(f"Ошибка: Не удалось открыть файл {IMAGE_NAME}")
else:
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. АВТОПОДБОР ЯДРА ДЛЯ ШИРОКИХ И ТОНКИХ ЛИНИЙ
    # Оцениваем текстуру в центре, чтобы понять ширину стыка
    roi_test = gray[:, int(width * 0.45):int(width * 0.55)]
    kernel_size = 35 if np.std(roi_test) > 20 else 15

    # 3. МОРФОЛОГИЧЕСКАЯ ФИЛЬТРАЦИЯ (Black Hat)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blurred = cv2.GaussianBlur(blackhat, (9, 9), 0)

    # 4. НАСТРОЙКА ЗОНЫ ПОИСКА (центральные 30% ширины кадра)
    roi_left = int(width * 0.35)
    roi_right = int(width * 0.65)

    # Разметка на 30 сегментов по высоте для точного отслеживания наклона
    num_segments = 30
    segment_height = height // num_segments
    points = []

    # 5. ПОСТРОЧНЫЙ СБОР КООРДИНАТ СТЫКА
    for i in range(num_segments):
        y_start = i * segment_height
        y_end = (i + 1) * segment_height
        y_center = (y_start + y_end) // 2
        
        roi_segment = blurred[y_start:y_end, roi_left:roi_right]
        profile = np.mean(roi_segment, axis=0)
        
        if np.max(profile) > 8:  
            max_idx = np.argmax(profile)
            final_x = roi_left + max_idx
            
            if roi_left <= final_x <= roi_right:
                points.append([final_x, y_center])

    # 6. НАДЕЖНОЕ ПОСТРОЕНИЕ ЛИНИИ (Метод Huber)
    if len(points) >= 6:
        pts_array = np.array(points, dtype=np.float32)
        
        # Находим идеальный вектор прямой, игнорируя локальные блики и капли
        [vx, vy, x0, y0] = cv2.fitLine(pts_array, cv2.DIST_HUBER, 0, 0.01, 0.01)
        
        vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
        
        if np.abs(vy) > 0.001:
            x_top = int(x0 + (0 - y0) * (vx / vy))
            x_bottom = int(x0 + (height - y0) * (vx / vy))
            angle = np.abs(np.arctan2(vy, vx) * 180 / np.pi)
            
            # Проверка допуска геометрии (85 - 95 градусов)
            if 85 <= angle <= 95:
                print(f"Анализ завершен автоматически (Размер ядра: {kernel_size})")
                print(f" -> Координата X вверху: {x_top} px")
                print(f" -> Координата X внизу: {x_bottom} px")
                print(f" -> Угол линии: {angle:.2f}°")
                
                # Отрисовка зеленой оси строго по центру зазора
                cv2.line(img, (x_top, 0), (x_bottom, height), (0, 255, 0), 3)
                
                # Отрисовка опорных контрольных точек (красные маркеры)
                for pt in points:
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
            else:
                print(f"Линия найдена, но угол ({angle:.1f}°) вне допуска 85°-95°.")
    else:
        print("Не удалось надежно выделить ось стыка. Недостаточно опорных точек.")

    # 7. Вывод на экран
    cv2.imshow('Automated Universal Detection', cv2.resize(img, (width // 2, height // 2)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

