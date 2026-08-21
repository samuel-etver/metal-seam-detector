import cv2
import gxipy as gx

device_manager = gx.DeviceManager()
dev_num, dev_info_list = device_manager.update_device_list()

if dev_num == 0:
    print("No Camera")
    exit()

cam = device_manager.open_device_by_sn(dev_info_list[0]["sn"])
cam.BalanceWhiteAuto.set(gx.GxAutoEntry.CONTINUOUS)
cam.stream_on()

# --- НАСТРОЙКИ СЕТКИ ---
STEP = 100            # Шаг сетки в пикселях
LINE_COLOR = (0, 255, 0)  # Зеленый цвет линий (BGR)
LINE_THICKNESS = 1        # Толщина линии в 1 пиксель

try:
    while True:
        raw_image = cam.data_stream[0].get_image()
        if raw_image is None:
            continue
        if raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
            continue
            
        rgb_image = raw_image.convert("RGB")
        if rgb_image is None:
            continue
            
        numpy_image = rgb_image.get_numpy_array()
        bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        
        # --- МАСШТАБИРОВАНИЕ ЗАКОММЕНТИРОВАНО ---
        # small_image = cv2.resize(bgr_image, (0, 0), fx=0.5, fy=0.5)
        
        h, w, _ = bgr_image.shape
        
        # --- ОТРИСОВКА ВЕРТИКАЛЬНЫХ ЛИНИЙ С ШАГОМ 100 PX ---
        # Цикл идет от 100 до ширины кадра (w) с шагом в 100 пикселей
        for x_coord in range(STEP, w, STEP):
            # Рисуем вертикальную линию толщиной 1 пиксель
            cv2.line(bgr_image, (x_coord, 0), (x_coord, h), LINE_COLOR, LINE_THICKNESS)
            
            # Подписываем координату пикселя (размер уменьшен, чтобы не перегружать экран)
            cv2.putText(bgr_image, str(x_coord), (x_coord + 4, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, LINE_COLOR, 1)

        cv2.imshow("MER2-302-37gC-P Stream", bgr_image)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cam.stream_off()
    cam.close_device()
    cv2.destroyAllWindows()

