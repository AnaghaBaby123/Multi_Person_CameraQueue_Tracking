import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchreid
from torchvision import transforms
import torch

def extract_embedding(crop):
    with torch.no_grad():
        img = Image.fromarray(crop)
        input_tensor = transform(img).unsqueeze(0)
        emb = reid_model(input_tensor).squeeze().numpy()
        return emb / np.linalg.norm(emb)

def match_embedding(embedding, threshold=0.7):
    best_id = None
    best_sim = -1
    for gid, vectors in global_id_map.items():
        for vec in vectors:
            sim = np.dot(embedding, vec)
            if sim > best_sim:
                best_sim = sim
                best_id = gid
    return best_id if best_sim > threshold else None

def assign_global_id(cam_id, track_id, crop):
    global next_global_id
    embedding = extract_embedding(crop)
    matched_id = match_embedding(embedding)

    if matched_id is not None:
        global_id_map[matched_id].append(embedding)
    else:
        matched_id = next_global_id
        global_id_map[matched_id] = [embedding]
        next_global_id += 1

    track_to_global[(cam_id, track_id)] = matched_id
    return matched_id

def detect_track_global(frame, tracker, cam_id, active_gids):
    #print(frame)
    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        if cls_id == 0: # its a person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        x1, y1, x2, y2 = map(int, [l, t, l + w, t + h])

        # Shrink width
        box_width = x2 - x1
        shrink = int(0.8 * box_width)
        x2 -= shrink

        #Crop upper 60%
        box_height = y2 - y1
        upper_y2 = y1 + int(0.6 * box_height)

        # Clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        upper_y2 = min(frame.shape[0], upper_y2)

        # Extract crop
        crop = frame[y1:upper_y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
            continue

        # Assign global ID
        global_id = assign_global_id(cam_id, track_id, crop)
        active_gids.add(global_id)

        # Draw box and ID
        cv2.rectangle(frame, (x1, y1), (x2, upper_y2), (0, 255, 0), 2)
        if person_left:
            time = average_waiting_time + global_id * average_waiting_time
            time -= average_waiting_time * person_left # Adjust time for people who left
        else:
            time = average_waiting_time + global_id * average_waiting_time
        cv2.putText(frame, f"GID {global_id}, time {time}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame


model = YOLO("yolov8n.pt")

tracker1 = DeepSort(max_age=30)
tracker2 = DeepSort(max_age=30)

reid_model = torchreid.models.build_model('osnet_x0_25', num_classes=1000, pretrained=True)
torchreid.utils.load_pretrained_weights(reid_model, r"C:\Users\anagha\Downloads\osnet_x0_25_imagenet.pth")
reid_model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Global tracking data
global_id_map = {}              # {global_id: [embeddings]}
track_to_global = {}            # {(cam_id, track_id): global_id}
next_global_id = 0

average_waiting_time = 2  # seconds
cap1 = cv2.VideoCapture(r"C:\Users\anagha\Documents\queue\Retail.mp4")
cap2 = cv2.VideoCapture(r"C:\Users\anagha\Documents\queue\Retail.mp4")
person_left = 0
prev_count = None 

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 and not ret2:
        break

    active_gids = set()  # Global IDs seen in this frame
    
    if ret1:
        frame1 = cv2.resize(frame1, (640, 360))
        frame1 = detect_track_global(frame1, tracker1, 'C1', active_gids)
    else:
        frame1 = cv2.imread("black.jpg")

    if ret2:
        frame2 = cv2.resize(frame2, (640, 360))
        frame2 = detect_track_global(frame2, tracker2, 'C2', active_gids)
    else:
        frame2 = cv2.imread("black.jpg")

    # Remove global IDs not seen in current frame
    all_gids = list(global_id_map.keys())
    for gid in all_gids:
        if gid not in active_gids:
            del global_id_map[gid]
            
            

    # Display global ID count
    total_count = len(global_id_map)
    if prev_count is None:# Handle first frame separately
        prev_count = total_count
        person_left = 0
    else:
        if prev_count - total_count == 1:
            person_left +=1 #tracking the number of people left
        else:
            pass

        prev_count = total_count  # update after comparison

    frame_height = frame1.shape[0]
    cv2.putText(frame1, f"Count: {total_count}", (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    frame_height = frame2.shape[0]
    cv2.putText(frame2, f"Count: {total_count}", (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    combined = cv2.hconcat([frame1, frame2])
    cv2.imshow("Global ID Tracking", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
