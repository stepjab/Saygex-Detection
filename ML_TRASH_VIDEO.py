import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
from datetime import datetime
from PySide6.QtCore import QThread, Signal

class VideoProcessor(QThread):
    garbageDetected = Signal(str, str)

    def __init__(self, video_path, model_path1, model_path2):
        super().__init__()
        self.video_path = video_path
        self.model1 = YOLO(model_path1)
        self.model2 = YOLO(model_path2)
        self.tracker = Sort()
        self.detected_objects = set()

    def run(self):
        cap = cv2.VideoCapture(self.video_path) # ГОЙДААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААа
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Определите максимальную площадь объекта (0.5% от площади кадра)
        max_object_area = frame_width * frame_height * 0.005

        # Initialize variables for motion detection
        prev_frame = None
        motion_threshold = 50  # Adjust this threshold based on your needs

        frame_interval = 20  # Process every 20th frame
        frame_count = 0  # Frame counter

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Initialize prev_frame with the first frame
                if prev_frame is None:
                    prev_frame = gray
                    frame_count += 1
                    continue

                # Compute the absolute difference between the current frame and previous frame
                frame_diff = cv2.absdiff(prev_frame, gray)
                _, motion_mask = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)

                if frame_count % frame_interval == 0:
                    results1 = self.model1(frame, conf=0.25, save=False, imgsz=640)
                    results2 = self.model2(frame, conf=0.25, save=False, imgsz=640)

                    boxes1 = results1[0].boxes.data.cpu().numpy()
                    boxes2 = results2[0].boxes.data.cpu().numpy()
                    combined_boxes = np.vstack((boxes1, boxes2))

                    unique_boxes = self.remove_duplicates(combined_boxes)

                    annotated_frame = frame.copy()
                    detections = []

                    if len(unique_boxes) > 0:
                        for box in unique_boxes:
                            x1, y1, x2, y2, conf, cls = box
                            width = x2 - x1
                            height = y2 - y1
                            object_area = width * height

                            # Игнорировать объекты, занимающие более 0.5% площади кадра
                            if object_area > max_object_area:
                                continue

                            # Check if the object is NOT moving
                            object_region = motion_mask[int(y1):int(y2), int(x1):int(x2)]
                            if cv2.countNonZero(object_region) == 0:
                                label = f'{"TRASH"} {conf:.2f}'
                                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                                detections.append([x1, y1, x2, y2, conf])

                    # Update tracker
                    if detections:
                        detections = np.array(detections)
                        trackers = self.tracker.update(detections)

                        # Check for new detections
                        for track in trackers:
                            x1, y1, x2, y2, obj_id = map(int, track)

                            if obj_id not in self.detected_objects:
                                self.detected_objects.add(obj_id)
                                time_str = datetime.now().strftime("%H:%M")
                                platform_str = "Платформа №X"  # Replace with actual data
                                self.garbageDetected.emit(time_str, platform_str)

                    cv2.imshow("YOLO Inference", annotated_frame)

                prev_frame = gray
                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    def remove_duplicates(self, boxes, iou_threshold=0.5, score_threshold=0.2):
        indices = cv2.dnn.NMSBoxes(boxes[:, :4], boxes[:, 4], score_threshold, iou_threshold)
        if len(indices) > 0:
            return boxes[indices.flatten()]
        else:
            return np.array([])