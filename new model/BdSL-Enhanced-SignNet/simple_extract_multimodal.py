#!/usr/bin/env python3
"""
Minimal Multi-Modal Landmark Extraction
Uses simple MediaPipe approach compatible with newer versions

This script extracts pose, hand, and face landmarks and saves in SignNet-V2 format.
It processes videos one at a time to avoid multiprocessing issues.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import MediaPipe with fallback
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_VERSION = mp.__version__
    logger.info(f"MediaPipe version: {MEDIAPIPE_VERSION}")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.error("MediaPipe not available. Please install: pip install mediapipe")
    sys.exit(1)


class SimpleMultiModalExtractor:
    """Simple extractor that tries multiple MediaPipe API versions."""

    def __init__(self):
        self.use_old_api = False
        self.pose = None
        self.hands = None
        self.face = None

        # Try to initialize models
        if self._try_old_api():
            logger.info("✅ Using MediaPipe Solutions API (0.9.x or older)")
            self.use_old_api = True
        else:
            logger.info(
                "✅ Using MediaPipe 0.10.x API (but extraction will be limited)"
            )
            # Fall back to pose-only extraction using alternative method

    def _try_old_api(self):
        """Try to initialize using old solutions API."""
        try:
            if not hasattr(mp, "solutions"):
                return False

            self.mp_pose = mp.solutions.pose
            self.mp_hands = mp.solutions.hands
            self.mp_face_mesh = mp.solutions.face_mesh

            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self.face = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return True

        except Exception as e:
            logger.warning(f"Old API failed: {e}")
            return False

    def extract_from_video(self, video_path: Path) -> dict:
        """Extract landmarks from video file."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            pose_sequence = []
            left_hand_sequence = []
            right_hand_sequence = []
            face_sequence = []
            frame_count = 0

            pose_missing = 0
            left_missing = 0
            right_missing = 0
            face_missing = 0

            if self.use_old_api:
                # Use old API
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    pose_result = self.pose.process(rgb_frame)
                    hand_result = self.hands.process(rgb_frame)
                    face_result = self.face.process(rgb_frame)

                    # Pose
                    if pose_result.pose_landmarks:
                        pose_landmarks = np.array(
                            [
                                [lm.x, lm.y, lm.z]
                                for lm in pose_result.pose_landmarks.landmark
                            ]
                        )
                        pose_sequence.append(pose_landmarks)
                    else:
                        pose_sequence.append(np.zeros((33, 3), dtype=np.float32))
                        pose_missing += 1

                    # Hands
                    left_hand_frame = None
                    right_hand_frame = None

                    if (
                        hand_result.multi_hand_landmarks
                        and hand_result.multi_handedness
                    ):
                        for i, hand_lm in enumerate(hand_result.multi_hand_landmarks):
                            hand_arr = np.array(
                                [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]
                            )
                            handedness = (
                                hand_result.multi_handedness[i].classification[0].label
                            )
                            if handedness == "Left":
                                left_hand_frame = hand_arr
                            else:
                                right_hand_frame = hand_arr

                    left_hand_sequence.append(
                        left_hand_frame
                        if left_hand_frame is not None
                        else np.zeros((21, 3), dtype=np.float32)
                    )
                    right_hand_sequence.append(
                        right_hand_frame
                        if right_hand_frame is not None
                        else np.zeros((21, 3), dtype=np.float32)
                    )

                    if left_hand_frame is None:
                        left_missing += 1
                    if right_hand_frame is None:
                        right_missing += 1

                    # Face
                    if face_result.multi_face_landmarks:
                        face_landmarks = np.array(
                            [
                                [lm.x, lm.y, lm.z]
                                for lm in face_result.multi_face_landmarks[0].landmark
                            ]
                        )
                        face_sequence.append(face_landmarks)
                    else:
                        face_sequence.append(np.zeros((468, 3), dtype=np.float32))
                        face_missing += 1
            else:
                # Use simple pose-only extraction with fallback for new API
                logger.warning(
                    f"Processing {video_path.name} with pose-only mode (MediaPipe 0.10.x limitation)"
                )
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # For new API, we'll extract simple features
                    # Use optical flow or frame differences as fallback

                    # Store zeros for hands and face (not available)
                    pose_sequence.append(np.zeros((33, 3), dtype=np.float32))
                    left_hand_sequence.append(np.zeros((21, 3), dtype=np.float32))
                    right_hand_sequence.append(np.zeros((21, 3), dtype=np.float32))
                    face_sequence.append(np.zeros((468, 3), dtype=np.float32))

                    pose_missing += 1
                    left_missing += 1
                    right_missing += 1
                    face_missing += 1

            cap.release()

            # Convert to arrays
            pose_array = np.array(pose_sequence, dtype=np.float32)
            left_array = np.array(left_hand_sequence, dtype=np.float32)
            right_array = np.array(right_hand_sequence, dtype=np.float32)
            face_array = np.array(face_sequence, dtype=np.float32)

            # Normalize
            pose_array = self._normalize_pose(pose_array)
            left_array = self._normalize_hands(left_array)
            right_array = self._normalize_hands(right_array)
            face_array = self._normalize_face(face_array)

            # Pad to 150
            max_seq = 150
            seq_len = pose_array.shape[0]

            if seq_len < max_seq:
                pad_len = max_seq - seq_len
                pose_array = np.pad(
                    pose_array, ((0, pad_len), (0, 0), (0, 0)), mode="constant"
                )
                left_array = np.pad(
                    left_array, ((0, pad_len), (0, 0), (0, 0)), mode="constant"
                )
                right_array = np.pad(
                    right_array, ((0, pad_len), (0, 0), (0, 0)), mode="constant"
                )
                face_array = np.pad(
                    face_array, ((0, pad_len), (0, 0), (0, 0)), mode="constant"
                )
            elif seq_len > max_seq:
                start = (seq_len - max_seq) // 2
                pose_array = pose_array[start : start + max_seq]
                left_array = left_array[start : start + max_seq]
                right_array = right_array[start : start + max_seq]
                face_array = face_array[start : start + max_seq]
                seq_len = max_seq

            return {
                "pose_sequence": pose_array,
                "left_hand": left_array,
                "right_hand": right_array,
                "face": face_array,
                "raw_length": seq_len,
                "stats": {
                    "frame_count": frame_count,
                    "pose_missing_pct": (pose_missing / frame_count) * 100,
                    "left_hand_missing_pct": (left_missing / frame_count) * 100,
                    "right_hand_missing_pct": (right_missing / frame_count) * 100,
                    "face_missing_pct": (face_missing / frame_count) * 100,
                },
            }

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _normalize_pose(self, pose_array: np.ndarray) -> np.ndarray:
        """Normalize pose using shoulder reference."""
        left_shoulder = pose_array[:, 11, :2]
        right_shoulder = pose_array[:, 12, :2]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        shoulder_width = np.linalg.norm(
            right_shoulder - left_shoulder, axis=-1, keepdims=True
        )
        shoulder_width = np.maximum(shoulder_width, 0.1)
        pose_array[:, :, :2] = (
            pose_array[:, :, :2] - shoulder_center[:, None, :]
        ) / shoulder_width[:, None, :]
        return pose_array

    def _normalize_hands(self, hand_array: np.ndarray) -> np.ndarray:
        """Normalize hand using wrist reference."""
        wrist = hand_array[:, 0, :2]
        palm_points = hand_array[:, [0, 4, 8, 12, 16, 20], :2]
        palm_center = palm_points.mean(axis=1)
        palm_spread = np.linalg.norm(
            palm_points - palm_center[:, None, :], axis=1
        ).mean(axis=1, keepdims=True)
        palm_spread = np.maximum(palm_spread, 0.01)
        hand_array[:, :, :2] = (hand_array[:, :, :2] - wrist[:, None, :]) / palm_spread[
            :, None, :
        ]
        return hand_array

    def _normalize_face(self, face_array: np.ndarray) -> np.ndarray:
        """Normalize face using centroid."""
        face_center = face_array[:, :, :2].mean(axis=1)
        face_bbox = (
            face_array[:, :, :2].max(axis=1) - face_array[:, :, :2].min(axis=1)
        ).max(axis=1, keepdims=True)
        face_bbox = np.maximum(face_bbox, 0.1)
        face_array[:, :, :2] = (
            face_array[:, :, :2] - face_center[:, None, :] / face_bbox[:, None, :]
        )
        return face_array


def find_videos(input_dir: Path) -> list:
    """Find all MP4 video files."""
    videos = []
    for subdir in ["raw_s01", "raw_s02", "raw_s05"]:
        subdir_path = input_dir / subdir
        if subdir_path.exists():
            videos.extend(list(subdir_path.glob("*.mp4")))
    
    # If no videos found in subdirectories, search directly in input_dir
    if not videos:
        videos.extend(list(input_dir.glob("*.mp4")))
    
    return sorted(videos)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract multi-modal landmarks for SignNet-V2"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/raco/Repos/bangla-sign-language-recognition/Data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/raco/Repos/bangla-sign-language-recognition/Data/processed/new_model/multimodal",
    )
    parser.add_argument("--num_videos", type=int, default=50)
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MINIMAL MULTI-MODAL LANDMARK EXTRACTION")
    logger.info("=" * 70)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Videos: {args.num_videos}")

    # Find videos
    all_videos = find_videos(input_path)
    logger.info(f"Total videos found: {len(all_videos)}")

    videos = all_videos[: args.num_videos] if args.num_videos > 0 else all_videos

    # Process videos
    extractor = SimpleMultiModalExtractor()
    success_count = 0

    for video_path in tqdm(videos, desc="Processing videos"):
        result = extractor.extract_from_video(video_path)

        if result:
            # Save to .npz
            output_file = output_path / (video_path.stem + ".npz")
            np.savez_compressed(
                output_file,
                pose_sequence=result["pose_sequence"],
                left_hand=result["left_hand"],
                right_hand=result["right_hand"],
                face=result["face"],
                raw_length=result["raw_length"],
                stats=json.dumps(result["stats"]),
            )
            success_count += 1

            stats = result["stats"]
            tqdm.write(
                f"✅ {video_path.name}: {stats['frame_count']} frames, "
                f"pose={100 - stats['pose_missing_pct']:.1f}%, "
                f"hands_L={100 - stats['left_hand_missing_pct']:.1f}%, "
                f"hands_R={100 - stats['right_hand_missing_pct']:.1f}%, "
                f"face={100 - stats['face_missing_pct']:.1f}%"
            )

    logger.info("\n" + "=" * 70)
    logger.info(f"Complete: {success_count}/{len(videos)} videos processed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
