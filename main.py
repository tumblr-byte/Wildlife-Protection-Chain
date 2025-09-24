# import libraries for file handling, video processing, deep learning, image transforms, and object detection
import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
from ultralytics import YOLO
from PIL import Image
import numpy as np

#loads your model
model = YOLO("yolo12n.pt")  # pretrained YOLO12 model used to detect general environment objects
model2 = YOLO("best.pt")  # custom YOLO12 model trained on species types (rhino, elephant, tiger)
species_envo = YOLO("envo_best.pt")  # custom YOLO12 model trained on threat (weapon/fire) detection


# pretrained ResNet18 model where `best_train.pt` is trained to classify species condition (normal or injured)
species_condition_model = models.resnet18(pretrained=True)
in_features = species_condition_model.fc.in_features
species_condition_model.fc = nn.Linear(in_features, 2)
species_condition_model.load_state_dict(torch.load("best_train.pt", map_location="cpu"))
species_condition_model.eval()

# class labels for species, species condition, and detected threats/environment objects
species_class = ["rhino", "elephant", "tiger"]
condition_class = ["injured", "normal"]
envo_class = {0: "Automatic Rifle", 1: "Bazooka", 2: "Grenade Launcher", 3: "Handgun", 4: "Knife", 5: "Shotgun", 6: "SMG", 7: "Sniper", 8: "Sword", 9: "smoke", 10: "fire"}
more_envo_class = {0: "person", 2: "car", 7: "truck", -1: "plastic"}

device = "cuda" if torch.cuda.is_available() else "cpu"
species_condition_model.to(device)

DETECTION_SKIP = 5
CONDITION_SKIP = 15
THREAT_SKIP = 10

# detect species and get their info
def detect_all_species(frame):
    results = model2(frame, conf=0.8, verbose=False)
    detections = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf = float(box.conf.cpu().numpy()[0])
            if conf < 0.8:
                continue
            cls_id = int(box.cls.cpu().numpy()[0])
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            cropped = frame[y1:y2, x1:x2]
            name = species_class[cls_id]

            detections.append({
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "name": name,
                "cropped": cropped,
                "center": ((x1 + x2) // 2, (y1 + y2) // 2)
            })

    return detections


# check whether the detected species is normal or injured
def get_condition(cropped):
    if cropped.size == 0:
        return "normal"

    image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    transform = T.Compose([T.Resize((224,224)), T.ToTensor()])
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = species_condition_model(image)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return condition_class[pred]

# check for threats in the environment
def get_threats(frame):
    threats = []

    results1 = model(frame, conf=0.6, verbose=False)
    for result in results1:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            if conf >= 0.6 and cls_id in more_envo_class:
                threats.append(more_envo_class[cls_id])

    results2 = species_envo(frame, conf=0.6, verbose=False)
    for result in results2:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            if conf >= 0.6 and cls_id in envo_class:
                threats.append(envo_class[cls_id])

    return list(set(threats)) if threats else ["None"]

# calculate Euclidean distance between two points (used for tracking species across frames)
def calculate_distance(center1, center2):
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

# simple tracking by matching detections frame-to-frame based on distance
def simple_tracking(current_detections, previous_tracks, max_distance=100):
    tracks = []
    used_track_ids = set()

    for detection in current_detections:
        best_match = None
        min_distance = float('inf')

        for track_id, prev_info in previous_tracks.items():
            if track_id in used_track_ids:
                continue

            if prev_info["name"] == detection["name"]:
                distance = calculate_distance(detection["center"], prev_info["center"])
                if distance < max_distance and distance < min_distance:
                    min_distance = distance
                    best_match = track_id

        if best_match:
            track_id = best_match
            used_track_ids.add(track_id)
        else:
            track_id = f"species_{len(previous_tracks) + len(tracks) + 1}"

        tracks.append({
            "track_id": track_id,
            "detection": detection
        })

    return tracks

# process input video, perform detection, classification, tracking, and save results
def process_video(video_path, out_csv="output.csv", out_video="output_tracked.mp4"):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    saved_species = set()
    results_list = []
    previous_tracks = {}
    last_conditions = {}
    last_threats = ["None"]
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        current_detections = []
        if frame_count % DETECTION_SKIP == 0:
            current_detections = detect_all_species(frame)

        if current_detections:
            tracks = simple_tracking(current_detections, previous_tracks)

            previous_tracks = {}
            for track in tracks:
                track_id = track["track_id"]
                detection = track["detection"]

                previous_tracks[track_id] = {
                    "name": detection["name"],
                    "center": detection["center"],
                    "bbox": detection["bbox"],
                    "cropped": detection["cropped"]
                }

            if frame_count % CONDITION_SKIP == 0:
                for track in tracks:
                    track_id = track["track_id"]
                    detection = track["detection"]
                    condition = get_condition(detection["cropped"])
                    last_conditions[track_id] = condition

            if frame_count % THREAT_SKIP == 0:
                last_threats = get_threats(frame)

            for track in tracks:
                track_id = track["track_id"]
                detection = track["detection"]

                if track_id not in saved_species:
                    condition = last_conditions.get(track_id, "normal")
                    threat_str = ",".join(last_threats)

                    results_list.append({
                        "track_id": track_id,
                        "timestamp": round(timestamp, 2),
                        "species_type": detection["name"],
                        "condition": condition,
                        "threat": threat_str
                    })
                    saved_species.add(track_id)

                x1, y1, x2, y2 = detection["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{detection['name']} {track_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    df = pd.DataFrame(results_list)
    df.to_csv(out_csv, index=False)
    print(f"Processing complete. Results saved to {out_csv}")
    print(f"Tracked video saved to {out_video}")

# path to the input video file to be processed
video_path = "/content/12953739_1920_1080_60fps.mp4" 
process_video(video_path, out_csv="output.csv", out_video="output_tracked7.mp4")


