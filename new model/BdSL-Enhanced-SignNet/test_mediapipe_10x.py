#!/usr/bin/env python3
"""
Simple MediaPipe 0.10.x test script
Tests pose, hand, and face landmark extraction on a single video
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# Initialize models with new API
pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_poses=1,
)
pose = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
)
hands = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)

face_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    refine_landmarks=True,
    num_faces=1,
)
face = mp.tasks.vision.FaceLandmarker.create_from_options(face_options)

print("✅ Models initialized successfully")

# Test on first video
video_path = Path(
    "/home/raco/Repos/bangla-sign-language-recognition/Data/raw_s01/অবাক__S01__sess01__rep01__neutral.mp4"
)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print("❌ Cannot open video")
    exit(1)

print(f"\n📹 Processing: {video_path.name}")

# Process first 10 frames as test
frame_count = 0
pose_count = 0
left_hand_count = 0
right_hand_count = 0
face_count = 0

while frame_count < 10:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Extract Pose
    pose_result = pose.detect(mp_image)
    if pose_result.pose_landmarks:
        pose_count += 1

    # Extract Hands
    hand_result = hands.detect(mp_image)
    if hand_result.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
            handedness = (
                hand_result.handedness[i][0].display_name
                if hand_result.handedness
                else "Unknown"
            )
            if "Left" in handedness:
                left_hand_count += 1
            else:
                right_hand_count += 1

    # Extract Face
    face_result = face.detect(mp_image)
    if face_result.face_landmarks:
        face_count += 1

    print(
        f"Frame {frame_count}: pose={len(pose_result.pose_landmarks) if pose_result.pose_landmarks else 0}, "
        f"hands={len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0}, "
        f"face={len(face_result.face_landmarks) if face_result.face_landmarks else 0}"
    )

cap.release()

print(f"\n📊 Summary ({frame_count} frames):")
print(
    f"   Pose detections: {pose_count}/{frame_count} ({pose_count / frame_count * 100:.1f}%)"
)
print(
    f"   Left hand: {left_hand_count}/{frame_count} ({left_hand_count / frame_count * 100:.1f}%)"
)
print(
    f"   Right hand: {right_hand_count}/{frame_count} ({right_hand_count / frame_count * 100:.1f}%)"
)
print(f"   Face: {face_count}/{frame_count} ({face_count / frame_count * 100:.1f}%)")
print("\n✅ Test completed!")
