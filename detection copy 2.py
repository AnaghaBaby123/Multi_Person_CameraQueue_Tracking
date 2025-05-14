# Unified Video/Image Person Counting with Stable Global IDs
# Uses YOLOv8 for detection, Deep SORT for short-term tracking,
# and a centroid-based ReID matcher that reuses per-track IDs.

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchreid
from torchvision import transforms
import torch
import random

# --- Configuration ---
DETECTION_MODEL  = 'yolov8m.pt'   # balanced midsize detection
REID_BACKBONE    = 'osnet_x0_25'  # fast ReID
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE         = 640            # YOLO inference and frame resize
CONF_THRESHOLD   = 0.3            # YOLO confidence
IOU_THRESHOLD    = 0.5            # NMS IoU (default)
REID_THRESHOLD   = 0.78           # cosine similarity for matching
MIN_BOX_AREA     = 3000           # drop tiny boxes (~55x55)
MAX_AGE          = 30             # Deep SORT max_age
global_id_map = {}
track_to_global = {}  # (cam_id, track_id) -> gid
next_global_id = 0
id_color_map = {}


def extract_embedding(crop_bgr):
    """
    Convert BGR crop -> RGB -> tensor, feed through ReID model, return L2-normalized vector
    """
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
    """
    Compare emb against each global_id centroid; return gid if highest sim > threshold
    """
    # best_gid, best_sim = None, -1.0
    # for gid, data in global_id_map.items():
    #     centroid = data['centroid']
    #     sim = np.dot(emb, centroid)
    #     if sim > best_sim:
    #         best_sim, best_gid = sim, gid
    # if best_sim > REID_THRESHOLD:
    #     return best_gid
    # return None
    # best_id = None
    # best_sim = -1
    # for gid, vectors in global_id_map.items():
    #     for vec in vectors:
    #         sim = np.dot(emb, vec)
    #         if sim > best_sim:
    #             best_sim = sim
    #             best_id = gid
    # if best_sim > REID_THRESHOLD: 
    #     print(f"Best match: {best_id}, Similarity: {best_sim}")
    #     return best_id
    # else:
    #     return None
    sims = []
    for gid, vecs in global_id_map.items():
        max_sim = max(np.dot(emb, v) for v in vecs)
        sims.append((gid, max_sim))
    if not sims:
        return None
    # pick highest similarity
    best_gid, best_sim = max(sims, key=lambda x: x[1])
    return best_gid if best_sim > REID_THRESHOLD else None

def assign_global_id(cam_id, track_id, crop_bgr):
    """
    For a given track, reuse existing mapping if available; otherwise match by embedding or create new ID.
    """
    global next_global_id
    key = (cam_id, track_id)
    # reuse existing ID for this track
    if key in track_to_global:
        return track_to_global[key]

    # emb = extract_embedding(crop_bgr)
    # gid = match_embedding(emb)
    # if gid is None:
    #     gid = next_global_id
    #     global_id_map[gid] = {'centroid': emb, 'count': 1}
    #     next_global_id += 1
    # else:
    #     # update running centroid
    #     data = global_id_map[gid]
    #     c0, n0 = data['centroid'], data['count']
    #     new_centroid = (c0 * n0 + emb) / (n0 + 1)
    #     global_id_map[gid] = {'centroid': new_centroid / np.linalg.norm(new_centroid), 'count': n0 + 1}

    # track_to_global[key] = gid
    # return gid
    
    emb = extract_embedding(crop_bgr)
    matched_id = match_embedding(emb)

    if matched_id is not None:
        global_id_map[matched_id].append(emb)
    else:
        matched_id = next_global_id
        global_id_map[matched_id] = [emb]
        next_global_id += 1

    track_to_global[key] = matched_id
    return matched_id


def detect_track_global(frame, tracker, cam_id):
    """
    Detect people, track with Deep SORT, assign stable global IDs, and annotate frame.
    Returns annotated frame and set of active_gids.
    """
    # YOLO detection
    results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)[0]
    dets = []
    for box in results.boxes:
        if int(box.cls[0]) != 0:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area < MIN_BOX_AREA:
            continue
        dets.append(([x1, y1, x2 - x1, y2 - y1], float(box.conf[0]), 'person'))
    # update tracker
    tracks = tracker.update_tracks(dets, frame=frame)

    active_gids = set()
    for track in tracks:
        x1,y1,x2,y2 = map(int, track.to_ltrb())
        # refine crop: shrink width 10%, upper 60%
        w, h = x2-x1, y2-y1
        pad = int(w * 0.1)
        x1+=pad; x2-=pad
        y2_crop = y1 + int(h * 0.73)
        x1,y1 = max(0,x1), max(0,y1)
        x2 = min(frame.shape[1], x2)
        y2_crop = min(frame.shape[0], y2_crop)
        crop = frame[y1:y2_crop, x1:x2]
        if crop.size==0: continue
        gid = assign_global_id(cam_id, track.track_id, crop)
        active_gids.add(gid)
        # draw
        color = get_color_for_id(gid)
        cv2.rectangle(frame, (x1,y1), (x2,y2_crop), color, 2)
        cv2.putText(frame, f"GID {gid}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, active_gids

# --- Initialization ---
model = YOLO(DETECTION_MODEL)
tracker1 = DeepSort(max_age=MAX_AGE)
tracker2 = DeepSort(max_age=MAX_AGE)
reid_model = torchreid.models.build_model(REID_BACKBONE, num_classes=751, pretrained=True)
torchreid.utils.load_pretrained_weights(reid_model, r"C:\Users\anagha\Downloads\osnet_x0_25_imagenet.pth")
reid_model.eval()
transform = transforms.Compose([
    transforms.Resize((256,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- Example: Image Analysis Mode ---
paths = ['cam1.png','cam2.png']
frames, actives = [], set()
for i,p in enumerate(paths,1):
    img = cv2.imread(p)
    img = cv2.resize(img, (IMG_SIZE, int(IMG_SIZE*img.shape[0]/img.shape[1])))
    res, a = detect_track_global(img, tracker1 if i==1 else tracker2, f'C{i}')
    frames.append(res)
    actives |= a
# cleanup stale IDs
global_id_map = {k:v for k,v in global_id_map.items() if k in actives}
# annotate total count
count = len(global_id_map)
for f in frames:
    h=f.shape[0]; cv2.putText(f, f"Count: {count}", (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
out = cv2.hconcat(frames)
cv2.imshow('Result', out)
cv2.waitKey(0)
cv2.destroyAllWindows()