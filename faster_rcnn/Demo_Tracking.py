from sort import *
import torch
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from sort import Sort


def detect_cv2_camera(model_path='model_weights_final.pth', confidence_threshold=0.8, video_path='/Users/aleksandrvedernikov/Desktop/people.mp4'):
    # Initialize tracker
    mot_tracker = Sort()

    # Load the Faster R-CNN model with custom weights
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Use CUDA if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Capture video from file or webcam
    cap = cv2.VideoCapture(video_path)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame or video ended")
                break

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_image = F.to_tensor(rgb_image).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(tensor_image)

            boxes = prediction[0]['boxes'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()
            selected_indices = scores >= confidence_threshold
            selected_boxes = boxes[selected_indices]

            # Continue if no boxes are detected
            if len(selected_boxes) == 0:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            trackers = mot_tracker.update(selected_boxes)

            for track in trackers:
                bbox = track[:4]
                track_id = int(track[4])
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                cv2.putText(frame, str(track_id), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_cv2_camera()