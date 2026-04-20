#!/usr/bin/env python3
"""
Multi-Modal Landmark Extraction for SignNet-V2
==============================================

Extracts pose, hand, and face landmarks from Bengali Sign Language videos
using MediaPipe. Saves in SignNet-V2 compatible .npz format.

Usage:
    python extract_multimodal_landmarks.py \
        --input_dir /path/to/raw/videos \
        --output_dir /path/to/multimodal/npz \
        --num_videos 50 \
        --num_workers 4
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("landmark_extraction.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class MultiModalLandmarkExtractor:
    """Extract pose, hand, and face landmarks from videos using MediaPipe."""

    def __init__(self, max_num_hands=2, refine_landmarks=True):
        """
        Initialize MediaPipe models (using new MediaPipe 0.10.x API).

        Args:
            max_num_hands: Maximum number of hands to detect
            refine_landmarks: Whether to refine hand/face landmarks
        """
        # Initialize PoseLandmarker
        self.pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="pose_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(
            self.pose_options
        )

        # Initialize HandLandmarker
        self.hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(
            self.hand_options
        )

        # Initialize FaceLandmarker
        self.face_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            refine_landmarks=refine_landmarks,
            num_faces=1,
        )
        self.face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(
            self.face_options
        )

        logger.info("✅ MediaPipe models initialized (using 0.10.x API)")
        logger.info(f"   Hands: max_num_hands={max_num_hands}")
        logger.info(f"   Face Mesh: refine_landmarks={refine_landmarks}")

    def process_video(self, video_path: Path, output_path: Path) -> Dict:
        """
        Extract landmarks from a single video file.

        Args:
            video_path: Path to input video (.mp4)
            output_path: Path to save output (.npz)

        Returns:
            dict with processing results and statistics
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return {"status": "error", "error": "Cannot open video"}

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Storage for landmarks
            pose_sequence = []
            left_hand_sequence = []
            right_hand_sequence = []
            face_sequence = []
            frame_count = 0

            pose_missing = 0
            left_hand_missing = 0
            right_hand_missing = 0
            face_missing = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process with MediaPipe
                pose_results = self.pose.process(rgb_frame)
                hands_results = self.hands.process(rgb_frame)
                face_results = self.face_mesh.process(rgb_frame)

                # Extract Pose (33 landmarks)
                if pose_results.pose_landmarks:
                    pose_landmarks = np.array(
                        [
                            [lm.x, lm.y, lm.z]
                            for lm in pose_results.pose_landmarks.landmark
                        ]
                    )
                    pose_sequence.append(pose_landmarks)
                else:
                    pose_sequence.append(np.zeros((33, 3), dtype=np.float32))
                    pose_missing += 1

                # Extract Hands (21 landmarks per hand)
                left_hand_frame = None
                right_hand_frame = None

                if (
                    hands_results.multi_hand_landmarks
                    and hands_results.multi_handedness
                ):
                    for i, hand_landmarks in enumerate(
                        hands_results.multi_hand_landmarks
                    ):
                        handedness = (
                            hands_results.multi_handedness[i].classification[0].label
                        )
                        hand_array = np.array(
                            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                        )

                        if handedness == "Left":
                            left_hand_frame = hand_array
                        else:
                            right_hand_frame = hand_array

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
                    left_hand_missing += 1
                if right_hand_frame is None:
                    right_hand_missing += 1

                # Extract Face Mesh (468 landmarks)
                if face_results.multi_face_landmarks:
                    face_landmarks = np.array(
                        [
                            [lm.x, lm.y, lm.z]
                            for lm in face_results.multi_face_landmarks[0].landmark
                        ]
                    )
                    face_sequence.append(face_landmarks)
                else:
                    face_sequence.append(np.zeros((468, 3), dtype=np.float32))
                    face_missing += 1

            cap.release()

            # Convert to numpy arrays
            pose_array = np.array(pose_sequence, dtype=np.float32)
            left_hand_array = np.array(left_hand_sequence, dtype=np.float32)
            right_hand_array = np.array(right_hand_sequence, dtype=np.float32)
            face_array = np.array(face_sequence, dtype=np.float32)

            # Calculate statistics
            stats = {
                "frame_count": frame_count,
                "pose_missing_pct": (pose_missing / frame_count) * 100,
                "left_hand_missing_pct": (left_hand_missing / frame_count) * 100,
                "right_hand_missing_pct": (right_hand_missing / frame_count) * 100,
                "face_missing_pct": (face_missing / frame_count) * 100,
                "fps": fps,
                "resolution": (width, height),
            }

            # Normalize and save
            self._normalize_and_save(
                pose_array,
                left_hand_array,
                right_hand_array,
                face_array,
                output_path,
                stats,
            )

            logger.info(
                f"✅ {video_path.name}: {frame_count} frames, "
                f"pose={100 - stats['pose_missing_pct']:.1f}%, "
                f"hands_L={100 - stats['left_hand_missing_pct']:.1f}%, "
                f"hands_R={100 - stats['right_hand_missing_pct']:.1f}%, "
                f"face={100 - stats['face_missing_pct']:.1f}%"
            )

            return {"status": "success", "stats": stats}

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            import traceback

            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    def _normalize_and_save(
        self,
        pose_array: np.ndarray,
        left_hand_array: np.ndarray,
        right_hand_array: np.ndarray,
        face_array: np.ndarray,
        output_path: Path,
        stats: Dict,
    ):
        """
        Normalize landmarks and save to .npz file.

        Normalization method: Center and scale using reference points
        - Pose: Center at shoulder midpoint, scale by shoulder width
        - Hands: Center at wrist, scale by palm spread
        - Face: Center at face centroid, scale by face bounding box
        """
        # Normalize Pose (shoulder-based)
        left_shoulder = pose_array[:, 11, :2]  # (frames, 2)
        right_shoulder = pose_array[:, 12, :2]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        shoulder_width = np.linalg.norm(
            right_shoulder - left_shoulder, axis=-1, keepdims=True
        )
        shoulder_width = np.maximum(shoulder_width, 0.1)  # Avoid division by zero

        # Center and scale pose
        pose_array[:, :, :2] = (
            pose_array[:, :, :2] - shoulder_center[:, None, :]
        ) / shoulder_width[:, None, :]

        # Normalize Hands (wrist-based)
        for hand_array, wrist_idx in [(left_hand_array, 0), (right_hand_array, 0)]:
            wrist = hand_array[:, wrist_idx, :2]
            # Calculate scale using palm spread (wrist + finger tips)
            palm_points = hand_array[:, [0, 4, 8, 12, 16, 20], :2]
            palm_center = palm_points.mean(axis=1)
            palm_spread = np.linalg.norm(
                palm_points - palm_center[:, None, :], axis=1
            ).mean(axis=1, keepdims=True)
            palm_spread = np.maximum(palm_spread, 0.01)

            # Center and scale hand
            hand_array[:, :, :2] = (
                hand_array[:, :, :2] - wrist[:, None, :]
            ) / palm_spread[:, None, :]

        # Normalize Face (centroid-based)
        face_center_2d = face_array[:, :, :2].mean(axis=1)
        face_bbox_size = face_array[:, :, :2].ptp(axis=1).max(axis=1, keepdims=True)
        face_bbox_size = np.maximum(face_bbox_size, 0.1)

        face_array[:, :, :2] = (
            face_array[:, :, :2] - face_center_2d[:, None, :]
        ) / face_bbox_size[:, None, :]

        # Pad/Truncate to max sequence length (150)
        max_seq_length = 150
        seq_length = pose_array.shape[0]

        if seq_length > max_seq_length:
            # Center crop
            start = (seq_length - max_seq_length) // 2
            pose_array = pose_array[start : start + max_seq_length]
            left_hand_array = left_hand_array[start : start + max_seq_length]
            right_hand_array = right_hand_array[start : start + max_seq_length]
            face_array = face_array[start : start + max_seq_length]
            seq_length = max_seq_length
        elif seq_length < max_seq_length:
            # Zero pad at end
            pad_length = max_seq_length - seq_length
            pose_array = np.pad(
                pose_array, ((0, pad_length), (0, 0), (0, 0)), mode="constant"
            )
            left_hand_array = np.pad(
                left_hand_array, ((0, pad_length), (0, 0), (0, 0)), mode="constant"
            )
            right_hand_array = np.pad(
                right_hand_array, ((0, pad_length), (0, 0), (0, 0)), mode="constant"
            )
            face_array = np.pad(
                face_array, ((0, pad_length), (0, 0), (0, 0)), mode="constant"
            )

        # Save to .npz
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            pose_sequence=pose_array,
            left_hand=left_hand_array,
            right_hand=right_hand_array,
            face=face_array,
            raw_length=seq_length,
            stats=json.dumps(stats),
        )

        logger.debug(f"Saved multimodal landmarks to {output_path}")
        logger.debug(
            f"  Pose: {pose_array.shape}, Hands: {left_hand_array.shape}, Face: {face_array.shape}"
        )


def process_single_video(args: Tuple[Path, Path]) -> Dict:
    """Wrapper function for multiprocessing."""
    video_path, output_dir = args
    extractor = MultiModalLandmarkExtractor()

    # Generate output filename
    output_filename = video_path.stem + ".npz"
    output_path = output_dir / output_filename

    return extractor.process_video(video_path, output_path)


def find_videos(input_dir: Path) -> List[Path]:
    """Find all .mp4 video files in input directory."""
    video_files = list(input_dir.glob("*.mp4"))
    logger.info(f"Found {len(video_files)} video files in {input_dir}")
    return sorted(video_files)


def main():
    parser = argparse.ArgumentParser(
        description="Extract multi-modal landmarks for SignNet-V2"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/raco/Repos/bangla-sign-language-recognition/Data",
        help="Directory containing raw video files (default: Data/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/raco/Repos/bangla-sign-language-recognition/Data/processed/new_model/multimodal",
        help="Output directory for .npz files (default: Data/processed/new_model/multimodal)",
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=50,
        help="Number of videos to process (default: 50, use -1 for all)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--pose_complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe Pose model complexity (default: 1)",
    )
    parser.add_argument(
        "--max_num_hands",
        type=int,
        default=2,
        help="Maximum number of hands to detect (default: 2)",
    )

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MULTI-MODAL LANDMARK EXTRACTION FOR SignNet-V2")
    logger.info("=" * 70)
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Number of videos to process: {args.num_videos}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Pose complexity: {args.pose_complexity}")
    logger.info(f"Max hands: {args.max_num_hands}")
    logger.info("=" * 70)

    # Find all videos
    all_videos = []
    
    # Try to find videos in subdirectories first
    for subdir in ["raw_s01", "raw_s02", "raw_s05"]:
        subdir_path = input_path / subdir
        if subdir_path.exists():
            all_videos.extend(find_videos(subdir_path))
    
    # If no videos found in subdirectories, search directly in input_path
    if not all_videos:
        all_videos.extend(find_videos(input_path))

    if not all_videos:
        logger.error("No video files found!")
        return

    logger.info(f"Total videos found: {len(all_videos)}")

    # Select subset if specified
    if args.num_videos > 0 and args.num_videos < len(all_videos):
        videos = all_videos[: args.num_videos]
        logger.info(f"Processing first {args.num_videos} videos")
    else:
        videos = all_videos
        logger.info(f"Processing all {len(videos)} videos")

    # Process videos with multiprocessing
    logger.info(f"\nStarting extraction with {args.num_workers} workers...")
    logger.info("This may take a while. Please be patient.\n")

    success_count = 0
    error_count = 0

    # Prepare arguments for multiprocessing
    process_args = [(video, output_path) for video in videos]

    # Use Pool for parallel processing
    with Pool(processes=args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_video, process_args),
                total=len(videos),
                desc="Extracting landmarks",
                unit="video",
            )
        )

    # Count successes and errors
    for result in results:
        if result["status"] == "success":
            success_count += 1
        else:
            error_count += 1
            logger.error(f"Failed: {result.get('error', 'Unknown error')}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Videos processed: {len(videos)}")
    logger.info(f"Successes: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Success rate: {(success_count / len(videos) * 100):.1f}%")
    logger.info(f"Output directory: {output_path}")
    logger.info("=" * 70)

    # Save summary
    summary_path = output_path / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total_videos": len(videos),
                "success_count": success_count,
                "error_count": error_count,
                "success_rate": success_count / len(videos),
                "input_directory": str(input_path),
                "output_directory": str(output_path),
                "pose_complexity": args.pose_complexity,
                "max_num_hands": args.max_num_hands,
            },
            f,
            indent=2,
        )

    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
