"""
collect_dataset.py
Standalone command-line tool to record hand-landmark sequences for training.

For each gesture in config.GESTURES it will:
  1. Create dataset/<gesture>/<user_id>/ if it does not exist
  2. Open the webcam
  3. Wait for the user to press 's' to start a take
  4. Record config.SEQUENCE_LENGTH consecutive frames of 126-d landmark
     vectors (both hands, zero-padded if only one hand is visible)
  5. Save the sequence as a .npy file AND save landmark CSV + a
     representative snapshot image for auditing
  6. Optionally generate N augmented copies of every real take so the
     training set is larger without needing more physical recording time
  7. Log every sample into the SQLite database via database.py

Usage:
    python collect_dataset.py --gesture hello --user user1 --takes 30
    python collect_dataset.py --all --user user1 --takes 20
    python collect_dataset.py --list
"""

import os
import sys
import csv
import argparse
import numpy as np
import cv2

import config
import database
from utils.camera import HandTracker, VideoStream
from utils.helpers import extract_hand_landmarks, normalize_sequence_length, \
    augment_sequence, ensure_dir, timestamp_str


def gesture_dir(gesture, user_id):
    return ensure_dir(os.path.join(config.DATASET_DIR, gesture, user_id))


def save_sequence(gesture, user_id, sequence, snapshot_frame=None, augmented=False):
    """Persist one recorded (or augmented) sequence to disk + database."""
    out_dir = gesture_dir(gesture, user_id)
    tag = "aug" if augmented else "raw"
    stamp = timestamp_str()
    base_name = f"{tag}_{stamp}_{np.random.randint(1000, 9999)}"

    npy_path = os.path.join(out_dir, base_name + ".npy")
    np.save(npy_path, sequence)

    csv_path = os.path.join(out_dir, base_name + ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for hand in ("L", "R"):
            for i in range(config.NUM_HAND_LANDMARKS):
                header += [f"{hand}{i}_x", f"{hand}{i}_y", f"{hand}{i}_z"]
        writer.writerow(header)
        for frame in sequence:
            writer.writerow(list(frame))

    if snapshot_frame is not None and not augmented:
        img_path = os.path.join(out_dir, base_name + ".jpg")
        cv2.imwrite(img_path, snapshot_frame)

    database.insert_dataset_sample(gesture, user_id, npy_path, len(sequence))
    return npy_path


def record_takes(gesture, user_id, num_takes, camera_source=0, num_augmented_per_take=3):
    """Interactive webcam recording loop for a single gesture."""
    tracker = HandTracker()
    stream = VideoStream(source=camera_source)

    if not stream.is_opened():
        print(f"[ERROR] Could not open camera source {camera_source}. "
              f"Check that a webcam is connected and not in use by another app.")
        return 0

    print(f"\n=== Recording gesture: '{gesture}' for user '{user_id}' ===")
    print(f"Target takes: {num_takes}. Press 's' to start a take, 'q' to quit early.\n")

    completed = 0
    while completed < num_takes:
        ok, frame = stream.read()
        if not ok:
            print("[ERROR] Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)
        results = tracker.process(frame)
        display = tracker.draw(frame.copy(), results)

        cv2.putText(display, f"Gesture: {gesture}  Take: {completed+1}/{num_takes}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press 's' to start take, 'q' to quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Dataset Collection", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quit requested.")
            break

        if key == ord('s'):
            sequence = []
            snapshot = None
            frame_count = 0
            while frame_count < config.SEQUENCE_LENGTH:
                ok, frame = stream.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                results = tracker.process(frame)
                vec = extract_hand_landmarks(
                    results.multi_hand_landmarks, results.multi_handedness
                )
                sequence.append(vec)
                if frame_count == config.SEQUENCE_LENGTH // 2:
                    snapshot = tracker.draw(frame.copy(), results)

                display = tracker.draw(frame.copy(), results)
                cv2.putText(display, f"RECORDING {frame_count+1}/{config.SEQUENCE_LENGTH}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Dataset Collection", display)
                cv2.waitKey(1)
                frame_count += 1

            sequence = normalize_sequence_length(sequence)
            save_sequence(gesture, user_id, sequence, snapshot_frame=snapshot, augmented=False)

            # Generate augmented copies to enlarge the effective dataset
            for _ in range(num_augmented_per_take):
                aug_seq = augment_sequence(sequence)
                save_sequence(gesture, user_id, aug_seq, augmented=True)

            completed += 1
            print(f"  Take {completed}/{num_takes} saved "
                  f"(+{num_augmented_per_take} augmented copies).")

    tracker.close()
    stream.release()
    cv2.destroyAllWindows()
    return completed


def main():
    parser = argparse.ArgumentParser(description="Sign language dataset collector")
    parser.add_argument("--gesture", type=str, help="Single gesture label to record")
    parser.add_argument("--all", action="store_true", help="Record every gesture in config.GESTURES")
    parser.add_argument("--user", type=str, default="user1", help="User/session id")
    parser.add_argument("--takes", type=int, default=30, help="Number of raw takes per gesture")
    parser.add_argument("--augment", type=int, default=3, help="Augmented copies generated per raw take")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--list", action="store_true", help="List configured gestures and exit")
    args = parser.parse_args()

    database.init_db()

    if args.list:
        print("Configured gestures:")
        for g in config.GESTURES:
            print(f"  - {g}")
        return

    if not args.gesture and not args.all:
        print("Specify --gesture <name> or --all. Use --list to see configured gestures.")
        sys.exit(1)

    gestures_to_record = config.GESTURES if args.all else [args.gesture]

    for gesture in gestures_to_record:
        if gesture not in config.GESTURES:
            print(f"[WARN] '{gesture}' is not in config.GESTURES; recording anyway.")
        record_takes(gesture, args.user, args.takes, args.camera, args.augment)

    print("\nDataset collection complete. Run train_model.py next.")


if __name__ == "__main__":
    main()
