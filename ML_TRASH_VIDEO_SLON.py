import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO models
model1 = YOLO('D:/ProjectVC/norm.pt')
model2 = YOLO('D:/ProjectVC/yolov8m-seg.pt')

# Open the video file
video_path = "Платформа-2.mkv"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fps = int(fps)

# Define the codec and create VideoWriter object
output_path = "predict_video.mp4"  # Specify the output file path and name
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 output (common codec)
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Function to remove duplicate detections
def remove_duplicates(boxes, iou_threshold=0.5, score_threshold=0.2):
    indices = cv2.dnn.NMSBoxes(boxes[:, :4], boxes[:, 4], score_threshold, iou_threshold)  # Added score_threshold and nms_threshold
    if len(indices) > 0:
        return boxes[indices.flatten()]
    else:
        return np.array([])

# Определите интервал кадров
frame_interval = 20  # Обрабатывать каждый 20-й кадр
frame_count = 0  # Счетчик кадров

# Определите максимальную площадь объекта (0.5% от площади кадра)
max_object_area = frame_width * frame_height * 0.005

# Initialize variables for motion detection
prev_frame = None
motion_threshold = 50  # Adjust this threshold based on your needs

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
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

        # Обрабатываем только кадры, которые соответствуют интервалу
        if frame_count % frame_interval == 0:
            # Run YOLO inference on the frame using the first model
            results1 = model1(frame, conf=0.20, save=False, imgsz=1088)

            # Run YOLO inference on the frame using the second model
            results2 = model2(frame, conf=0.25, save=False, imgsz=1088)

            # Combine the results from both models
            boxes1 = results1[0].boxes.data.cpu().numpy()
            boxes2 = results2[0].boxes.data.cpu().numpy()
            combined_boxes = np.vstack((boxes1, boxes2))

            # Remove duplicate detections
            unique_boxes = remove_duplicates(combined_boxes)

            # Visualize the combined results on the frame
            annotated_frame = frame.copy()
            if len(unique_boxes) > 0:  # Check if there are any boxes to draw
                for box in unique_boxes:
                    x1, y1, x2, y2, conf, cls = box
                    width = x2 - x1
                    height = y2 - y1
                    object_area = width * height

                    # Игнорировать объекты, занимающие более 0.5% площади кадра
                    if object_area > max_object_area:
                        continue

                    # Check if the object is moving
                    object_region = motion_mask[int(y1):int(y2), int(x1):int(x2)]
                    if cv2.countNonZero(object_region) == 0:
                        label = f'{"TRASH"} {conf:.2f}'
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Display the annotated frame
            cv2.imshow("YOLO Inference", annotated_frame)

            # Write the annotated frame to the output video
            out.write(annotated_frame)

        # Update the previous frame
        prev_frame = gray

        # Увеличиваем счетчик кадров
        frame_count += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
