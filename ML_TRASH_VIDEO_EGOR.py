import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO models
model1 = YOLO('D:/ProjectVC/norm.pt')
model2 = YOLO('D:/ProjectVC/yolov8m-seg.pt')

# Open the video file
video_path = "Платформа Мусор-2.mkv"
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
    indices = cv2.dnn.NMSBoxes(boxes[:, :4], boxes[:, 4], score_threshold, iou_threshold)
    if len(indices) > 0:
        return boxes[indices.flatten()]
    else:
        return np.array([])

# Определите интервал кадров
frame_interval = 30  # Обрабатывать каждый 40-й кадр
frame_count = 0  # Счетчик кадров

# Определите максимальную площадь объекта (0.5% от площади кадра)
max_object_area = frame_width * frame_height * 0.005

# Initialize a dictionary to store object positions
object_positions = {}
stationary_threshold = 2  # Number of frames to consider an object stationary
coordinate_tolerance = 200  # Tolerance for coordinate changes

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
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

                    label = f'{"TRASH"} {conf:.2f}'
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Store the object position
                    object_id = (int(x1), int(y1), int(x2), int(y2))
                    if object_id not in object_positions:
                        object_positions[object_id] = []
                    object_positions[object_id].append((frame_count, (x1, y1, x2, y2)))

                    # Check if the object has not moved for `stationary_threshold` frames
                    if len(object_positions[object_id]) > stationary_threshold:
                        positions = object_positions[object_id][-stationary_threshold:]
                        # Calculate the average position change
                        avg_position_change = np.mean([np.linalg.norm(np.array(pos[1]) - np.array(positions[0][1])) for pos in positions])
                        if avg_position_change < coordinate_tolerance:  # Threshold for considering the object stationary
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                            cv2.putText(annotated_frame, "Stationary", (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display the annotated frame
            cv2.imshow("YOLO Inference", annotated_frame)

            # Write the annotated frame to the output video
            out.write(annotated_frame)

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

