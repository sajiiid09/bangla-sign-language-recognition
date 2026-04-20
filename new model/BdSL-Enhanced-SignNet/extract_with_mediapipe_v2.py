#!/usr/bin/env python3
"""
MediaPipe Landmark Extraction - Compatible with 0.10.x
Uses the new task-based API for pose, hand, and face detection
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    print(f"✅ MediaPipe version: {mp.__version__}")
except ImportError as e:
    print(f"❌ MediaPipe import error: {e}")
    sys.exit(1)


class MediaPipeExtractorV2:
    """Extractor using MediaPipe 0.10.x task-based API."""
    
    def __init__(self):
        """Initialize MediaPipe models using task API."""
        print("🔧 Initializing MediaPipe models...")
        
        # Create PoseLandmarker
        base_options_pose = python.BaseOptions(model_asset_path=None)
        options_pose = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)
        
        # Create HandLandmarker
        base_options_hand = python.BaseOptions(model_asset_path=None)
        options_hand = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)
        
        # Create FaceLandmarker
        base_options_face = python.BaseOptions(model_asset_path=None)
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_face)
        
        print("✅ MediaPipe models initialized")
    
    def extract_from_video(self, video_path: Path) -> dict:
        """Extract landmarks from a video file."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"❌ Cannot open video: {video_path}")
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30  # default
            
            pose_sequence = []
            left_hand_sequence = []
            right_hand_sequence = []
            face_sequence = []
            
            frame_idx = 0
            pose_detected = 0
            left_detected = 0
            right_detected = 0
            face_detected = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Calculate timestamp in milliseconds
                timestamp_ms = int((frame_idx / fps) * 1000)
                
                # Process pose
                pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                    pose_lm = pose_result.pose_landmarks[0]
                    pose_arr = np.array([[lm.x, lm.y, lm.z] for lm in pose_lm], dtype=np.float32)
                    pose_sequence.append(pose_arr)
                    pose_detected += 1
                else:
                    pose_sequence.append(np.zeros((33, 3), dtype=np.float32))
                
                # Process hands
                hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                left_hand_arr = None
                right_hand_arr = None
                
                if hand_result.hand_landmarks and hand_result.handedness:
                    for idx, hand_lm in enumerate(hand_result.hand_landmarks):
                        hand_arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm], dtype=np.float32)
                        # Check handedness
                        handedness = hand_result.handedness[idx][0].category_name
                        if handedness == "Left":
                            left_hand_arr = hand_arr
                            left_detected += 1
                        else:
                            right_hand_arr = hand_arr
                            right_detected += 1
                
                left_hand_sequence.append(
                    left_hand_arr if left_hand_arr is not None else np.zeros((21, 3), dtype=np.float32)
                )
                right_hand_sequence.append(
                    right_hand_arr if right_hand_arr is not None else np.zeros((21, 3), dtype=np.float32)
                )
                
                # Process face
                face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
                if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
                    face_lm = face_result.face_landmarks[0]
                    face_arr = np.array([[lm.x, lm.y, lm.z] for lm in face_lm], dtype=np.float32)
                    face_sequence.append(face_arr)
                    face_detected += 1
                else:
                    face_sequence.append(np.zeros((478, 3), dtype=np.float32))
                
                frame_idx += 1
            
            cap.release()
            
            if frame_idx == 0:
                return None
            
            # Convert to arrays
            pose_array = np.array(pose_sequence, dtype=np.float32)
            left_array = np.array(left_hand_sequence, dtype=np.float32)
            right_array = np.array(right_hand_sequence, dtype=np.float32)
            face_array = np.array(face_sequence, dtype=np.float32)
            
            # Normalize
            pose_array = self._normalize_pose(pose_array)
            left_array = self._normalize_hand(left_array)
            right_array = self._normalize_hand(right_array)
            face_array = self._normalize_face(face_array)
            
            # Pad or crop to 150 frames
            max_len = 150
            seq_len = min(pose_array.shape[0], max_len)
            
            if pose_array.shape[0] < max_len:
                pad_len = max_len - pose_array.shape[0]
                pose_array = np.pad(pose_array, ((0, pad_len), (0, 0), (0, 0)), mode='constant')
                left_array = np.pad(left_array, ((0, pad_len), (0, 0), (0, 0)), mode='constant')
                right_array = np.pad(right_array, ((0, pad_len), (0, 0), (0, 0)), mode='constant')
                face_array = np.pad(face_array, ((0, pad_len), (0, 0), (0, 0)), mode='constant')
            elif pose_array.shape[0] > max_len:
                start = (pose_array.shape[0] - max_len) // 2
                pose_array = pose_array[start:start + max_len]
                left_array = left_array[start:start + max_len]
                right_array = right_array[start:start + max_len]
                face_array = face_array[start:start + max_len]
            
            return {
                'pose_sequence': pose_array,
                'left_hand': left_array,
                'right_hand': right_array,
                'face': face_array,
                'raw_length': seq_len,
                'stats': {
                    'total_frames': frame_idx,
                    'pose_detected': pose_detected,
                    'left_hand_detected': left_detected,
                    'right_hand_detected': right_detected,
                    'face_detected': face_detected,
                    'pose_pct': (pose_detected / frame_idx) * 100 if frame_idx > 0 else 0,
                    'left_pct': (left_detected / frame_idx) * 100 if frame_idx > 0 else 0,
                    'right_pct': (right_detected / frame_idx) * 100 if frame_idx > 0 else 0,
                    'face_pct': (face_detected / frame_idx) * 100 if frame_idx > 0 else 0,
                }
            }
            
        except Exception as e:
            print(f"❌ Error processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _normalize_pose(self, pose_array: np.ndarray) -> np.ndarray:
        """Normalize pose landmarks using shoulder reference."""
        if pose_array.shape[0] == 0:
            return pose_array
        
        # Shoulder indices: left=11, right=12
        left_shoulder = pose_array[:, 11, :2]
        right_shoulder = pose_array[:, 12, :2]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder, axis=-1, keepdims=True)
        shoulder_width = np.maximum(shoulder_width, 0.1)
        
        pose_array[:, :, :2] = (pose_array[:, :, :2] - shoulder_center[:, None, :]) / shoulder_width[:, None, :]
        return pose_array
    
    def _normalize_hand(self, hand_array: np.ndarray) -> np.ndarray:
        """Normalize hand landmarks using wrist reference."""
        if hand_array.shape[0] == 0:
            return hand_array
        
        wrist = hand_array[:, 0, :2]
        hand_array[:, :, :2] = hand_array[:, :, :2] - wrist[:, None, :]
        
        # Scale by hand span
        palm_points = hand_array[:, [0, 5, 9, 13, 17], :2]
        palm_span = np.std(palm_points, axis=1).mean(axis=-1, keepdims=True)
        palm_span = np.maximum(palm_span, 0.01)
        hand_array[:, :, :2] = hand_array[:, :, :2] / palm_span[:, None, :]
        
        return hand_array
    
    def _normalize_face(self, face_array: np.ndarray) -> np.ndarray:
        """Normalize face landmarks."""
        if face_array.shape[0] == 0:
            return face_array
        
        # Center and scale face
        face_center = face_array[:, :, :2].mean(axis=1, keepdims=True)
        face_array[:, :, :2] = face_array[:, :, :2] - face_center
        
        face_std = np.std(face_array[:, :, :2], axis=1, keepdims=True)
        face_std = np.maximum(face_std, 0.01)
        face_array[:, :, :2] = face_array[:, :, :2] / face_std
        
        return face_array


def main():
    parser = argparse.ArgumentParser(description='Extract landmarks using MediaPipe 0.10.x')
    parser.add_argument('--input_dir', type=str, required=True, help='Input video directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for .npz files')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(input_dir.glob(f'**/*{ext}')))
    
    print(f"\n{'='*70}")
    print(f"📹 Found {len(video_files)} video files")
    print(f"📁 Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # Initialize extractor
    extractor = MediaPipeExtractorV2()
    
    # Process videos
    success_count = 0
    fail_count = 0
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        # Check if already processed
        output_filename = video_path.stem + '.npz'
        output_path = output_dir / output_filename
        
        if output_path.exists():
            continue
        
        # Extract landmarks
        result = extractor.extract_from_video(video_path)
        
        if result is not None:
            # Save to .npz
            np.savez_compressed(
                output_path,
                pose_sequence=result['pose_sequence'],
                left_hand=result['left_hand'],
                right_hand=result['right_hand'],
                face=result['face'],
                raw_length=result['raw_length'],
                stats=result['stats']
            )
            success_count += 1
        else:
            fail_count += 1
            print(f"⚠️  Failed: {video_path.name}")
    
    print(f"\n{'='*70}")
    print(f"✅ Successfully processed: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"📁 Output saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
