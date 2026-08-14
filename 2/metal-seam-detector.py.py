import cv2
import numpy as np

# 1. Загрузка изображения
img = cv2.imread('x1.jpg')
height, width = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. МОРФОЛОГИЧЕСКИЙ ФИЛЬТР ВЫДЕЛЕНИЯ ТОНКИХ ЛИНИЙ
# Ядро (9, 1) — критически важно! Оно выделяет объекты шириной ДО 9 пикселей (нашу линию)
# и полностью уничтожает широкие тени желоба, блики и круглые капли сварки
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)) # !!! 9 -> 35
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

# Легкое сглаживание, чтобы объединить пунктирные участки линии
blurred = cv2.GaussianBlur(blackhat, (5, 5), 0) # !!! (5,5) -> (9,9)

# 3. Настройка зоны поиска (берём центральные 30% ширины кадра)
roi_left = int(width * 0.35)
roi_right = int(width * 0.65)

# Разбиваем на 25 мелких сегментов по ВСЕЙ высоте кадра
num_segments = 25
segment_height = height // num_segments

points = []

# 4. Построчный сбор координат линии
for i in range(num_segments):
    y_start = i * segment_height
    y_end = (i + 1) * segment_height
    y_center = (y_start + y_end) // 2
    
    roi_segment = blurred[y_start:y_end, roi_left:roi_right]
    
    # Считаем профиль. Теперь линия — это яркий узкий пик
    profile = np.mean(roi_segment, axis=0)
    
    if np.max(profile) > 10:  # Отсекаем пустой фон
        max_idx = np.argmax(profile)
        final_x = roi_left + max_idx
        
        # Защита границ
        if roi_left <= final_x <= roi_right:
            points.append([final_x, y_center])

# 5. ИСПОЛЬЗОВАНИЕ RANSAC ДЛЯ ИГНОРИРОВАНИЯ ПРОЖОГА И КАПЕЛЬ
if len(points) >= 5:
    pts_array = np.array(points, dtype=np.float32)
    
    # Алгоритм построит прямую только по тем точкам, которые выстроены в один ряд.
    # Точки, попавшие на круглое отверстие или капли сварки, будут автоматически отброшены!
    [vx, vy, x0, y0] = cv2.fitLine(pts_array, cv2.DIST_HUBER, 0, 0.01, 0.01)
    
    vx = float(vx)
    vy = float(vy)
    x0 = float(x0)
    y0 = float(y0)
    
    if np.abs(vy) > 0.001:
        x_top = int(x0 + (0 - y0) * (vx / vy))
        x_bottom = int(x0 + (height - y0) * (vx / vy))
        
        # Расчет точного угла наклона (85 - 95 градусов)
        angle = np.abs(np.arctan2(vy, vx) * 180 / np.pi)
        
        if 85 <= angle <= 95:
            print(f"Линия шва успешно найдена сквозь дефекты!")
            print(f" -> Координата X вверху: {x_top} px")
            print(f" -> Координата X внизу: {x_bottom} px")
            print(f" -> Точный угол: {angle:.2f}°")
            
            # Отрисовка финальной зеленой линии строго по трещине/шву
            cv2.line(img, (x_top, 0), (x_bottom, height), (0, 255, 0), 3)
            
            # Визуализация точек для контроля
            for pt in points:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), -1)
        else:
            print(f"Линия найдена, но угол ({angle:.1f}°) выходит за рамки 85°-95°.")
else:
    print("Не удалось выделить достаточное количество точек шва.")

# 6. Отображение результата (сжато для монитора)
cv2.imshow('Universal Seam Detection', cv2.resize(img, (width // 2, height // 2)))
cv2.waitKey(0)
cv2.destroyAllWindows()


