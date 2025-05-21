# Unified Video Person Counting with Stable Global IDs
# Uses YOLOv8 for detection, Deep SORT for tracking, and ReID for stable global IDs.

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchreid
from torchvision import transforms
import torch
import random
import torchvision.transforms as T

# --- Configuration ---
DETECTION_MODEL = 'yolov8m.pt'
REID_BACKBONE   = 'osnet_x0_25'
REID_MODEL_PATH = 'reid_osnet.pt'
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_WIDTH       = 640   # width for resizing
IMG_HEIGHT      = 800   # height for resizing
print(DEVICE)
CONF_THRESHOLD  = 0.2
IOU_THRESHOLD   = 0.5
REID_THRESHOLD  = 0.82
MIN_BOX_AREA    = 8000
MAX_AGE         = 30

# Global mappings
global_id_map   = {}    # {gid: [embeddings]}
track_to_global = {}    # {(cam_id, track_id): gid}
next_global_id  = 0
id_color_map    = {}

# --- Load models and transforms ---
model       = YOLO(DETECTION_MODEL)
model.to(DEVICE)

tracker1    = DeepSort(max_age=MAX_AGE)
tracker2    = DeepSort(max_age=MAX_AGE)

reid_model = torchreid.models.build_model(
    name='osnet_ain_x1_0',
    num_classes=9,
    loss='softmax',
    pretrained=False,
    use_gpu=torch.cuda.is_available()
)
reid_model.load_state_dict(torch.load(REID_MODEL_PATH, map_location=DEVICE))
reid_model.to(DEVICE).eval()
transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Helper functions ---
def extract_embedding(crop_bgr):
    with torch.no_grad():
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(crop_rgb)
        x = transform(img).unsqueeze(0).to(DEVICE)
        feat = reid_model(x).squeeze(0)
        vec = feat.cpu().numpy()
        return vec / np.linalg.norm(vec)

def get_color_for_id(gid):
    if gid not in id_color_map:
        id_color_map[gid] = tuple(random.randint(100,255) for _ in range(3))
    return id_color_map[gid]

def match_embedding(emb):
    best_gid, best_sim = None, -1.0
    for gid, vecs in global_id_map.items():
        max_sim = max(np.dot(emb, v) for v in vecs)
        if max_sim > best_sim:
            best_gid, best_sim = gid, max_sim
    return best_gid if best_sim > REID_THRESHOLD else None

def assign_global_id(cam_id, track_id, crop):
    global next_global_id
    key = (cam_id, track_id)
    emb = extract_embedding(crop)
    gid = match_embedding(emb)
    if gid is None:
        gid = next_global_id
        global_id_map[gid] = []
        next_global_id += 1
    global_id_map[gid].append(emb)
    track_to_global[key] = gid
    return gid

# now include confidence display in bounding box

def detect_track_global(frame, tracker, cam_id):
    results = model(frame, imgsz=IMG_WIDTH, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)[0]
    dets = []
    for box in results.boxes:
        if int(box.cls[0]) != 0:
            continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        w, h = x2 - x1, y2 - y1
        # filter by area, minimum width/height, and aspect ratio
        if w * h < MIN_BOX_AREA or w < 80 or h < 200 :
            continue
        dets.append(([x1,y1,w,h], conf, 'person'))
    # pass filtered detections into tracker
    tracks = tracker.update_tracks(dets, frame=frame)
    active = set()
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1,y1,x2,y2 = map(int, track.to_ltrb())
        # refine head region (top 60%)
        w = x2 - x1
        pad = int(w * 0.1)
        x1 += pad; x2 -= pad
        h = y2 - y1
        y2_head = y1 + int(h * 0.6)
        x1,y1 = max(0,x1), max(0,y1)
        x2 = min(frame.shape[1], x2)
        y2_head = min(frame.shape[0], y2_head)
        crop = frame[y1:y2_head, x1:x2]
        if crop.size == 0:
            continue
        gid = assign_global_id(cam_id, track.track_id, crop)
        active.add(gid)
        col = get_color_for_id(gid)
        # draw head bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2_head), col, 2)
        # draw GID and original detection confidence
        label = f"GID {gid} "
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return frame, active


# --- Video Processing Mode ---
cap1 = cv2.VideoCapture('video/cam1_fps.mp4')
cap2 = cv2.VideoCapture('video/cam2_fps.mp4')
# Print properties
w1, h1, f1 = cap1.get(cv2.CAP_PROP_FRAME_WIDTH), cap1.get(cv2.CAP_PROP_FRAME_HEIGHT), cap1.get(cv2.CAP_PROP_FPS)
print(f"Camera1: {int(w1)}x{int(h1)} @ {f1:.2f}FPS")
w2, h2, f2 = cap2.get(cv2.CAP_PROP_FRAME_WIDTH), cap2.get(cv2.CAP_PROP_FRAME_HEIGHT), cap2.get(cv2.CAP_PROP_FPS)
print(f"Camera2: {int(w2)}x{int(h2)} @ {f2:.2f}FPS")
# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter('output1.mp4', fourcc, f1, (IMG_WIDTH*2, IMG_HEIGHT))

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 and not ret2:
        break
    active = set()
    # Cam1
    if ret1:
        f1 = cv2.resize(f1, (IMG_WIDTH, IMG_HEIGHT))
        f1, a1 = detect_track_global(f1, tracker1, 'C1')
        active |= a1
    else:
        f1 = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)
    # Cam2
    if ret2:
        f2 = cv2.resize(f2, (IMG_WIDTH, IMG_HEIGHT))
        f2, a2 = detect_track_global(f2, tracker2, 'C2')
        active |= a2
    else:
        f2 = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)
   
    global_id_map = {gid:embs for gid,embs in global_id_map.items() if gid in active}
    track_to_global = {k:v for k,v in track_to_global.items() if v in global_id_map}
    # Annotate counts
    count = len(global_id_map)
    for frm in (f1, f2):
        cv2.putText(frm, f"Count: {count}", (10, IMG_HEIGHT-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    # concat/write/show
    out = cv2.hconcat([f1, f2])
    out_writer.write(out)
    cv2.imshow('Global ID Tracking', out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cleanup
cap1.release()
cap2.release()
out_writer.release()
cv2.destroyAllWindows()

