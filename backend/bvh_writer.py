import numpy as np
from datetime import datetime

# MediaPipe landmark indices we care about for BVH skeleton
# We map these to a simplified humanoid skeleton
LANDMARK_INDICES = {
    "Hips":         (23, 24),   # average of left and right hip
    "Spine":        (11, 12),   # average of left and right shoulder (approximate)
    "Neck":         (0,),       # nose as head proxy
    "LeftArm":      (11,),      # left shoulder
    "LeftForeArm":  (13,),      # left elbow
    "LeftHand":     (15,),      # left wrist
    "RightArm":     (12,),      # right shoulder
    "RightForeArm": (14,),      # right elbow
    "RightHand":    (16,),      # right wrist
    "LeftUpLeg":    (23,),      # left hip
    "LeftLeg":      (25,),      # left knee
    "LeftFoot":     (27,),      # left ankle
    "RightUpLeg":   (24,),      # right hip
    "RightLeg":     (26,),      # right knee
    "RightFoot":    (28,),      # right ankle
}

# BVH skeleton hierarchy as a string template
BVH_HIERARCHY = """HIERARCHY
ROOT Hips
{{
    OFFSET 0.00 0.00 0.00
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {{
        OFFSET 0.00 10.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Neck
        {{
            OFFSET 0.00 10.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            End Site
            {{
                OFFSET 0.00 5.00 0.00
            }}
        }}
    }}
    JOINT LeftArm
    {{
        OFFSET -5.00 8.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftForeArm
        {{
            OFFSET -8.00 0.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftHand
            {{
                OFFSET -8.00 0.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {{
                    OFFSET -4.00 0.00 0.00
                }}
            }}
        }}
    }}
    JOINT RightArm
    {{
        OFFSET 5.00 8.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightForeArm
        {{
            OFFSET 8.00 0.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightHand
            {{
                OFFSET 8.00 0.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {{
                    OFFSET 4.00 0.00 0.00
                }}
            }}
        }}
    }}
    JOINT LeftUpLeg
    {{
        OFFSET -4.00 0.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftLeg
        {{
            OFFSET 0.00 -10.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftFoot
            {{
                OFFSET 0.00 -10.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {{
                    OFFSET 0.00 -4.00 0.00
                }}
            }}
        }}
    }}
    JOINT RightUpLeg
    {{
        OFFSET 4.00 0.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightLeg
        {{
            OFFSET 0.00 -10.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightFoot
            {{
                OFFSET 0.00 -10.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {{
                    OFFSET 0.00 -4.00 0.00
                }}
            }}
        }}
    }}
}}
"""

# Order of joints in the motion data (must match hierarchy above)
JOINT_ORDER = [
    "Hips",         # 6 channels (pos + rot)
    "Spine",        # 3 channels
    "Neck",         # 3 channels
    "LeftArm",      # 3 channels
    "LeftForeArm",  # 3 channels
    "LeftHand",     # 3 channels
    "RightArm",     # 3 channels
    "RightForeArm", # 3 channels
    "RightHand",    # 3 channels
    "LeftUpLeg",    # 3 channels
    "LeftLeg",      # 3 channels
    "LeftFoot",     # 3 channels
    "RightUpLeg",   # 3 channels
    "RightLeg",     # 3 channels
    "RightFoot",    # 3 channels
]


def get_landmark_position(landmarks, indices):
    """
    Get 3D position for a joint.
    If multiple indices given, average them (e.g. hips = avg of left+right hip).
    Returns (x, y, z) in a scaled coordinate system.
    """
    positions = []
    for idx in indices:
        lm = landmarks[idx]
        # Scale normalized coords to BVH units (multiply by 100 for cm-like scale)
        # Flip Y axis because MediaPipe Y increases downward, BVH Y increases upward
        positions.append([lm["x"] * 100, -lm["y"] * 100, lm["z"] * 100])

    avg = np.mean(positions, axis=0)
    return avg


def compute_rotation(parent_pos, child_pos):
    """
    Compute simple Euler angles (in degrees) to point from parent to child.
    This is a simplified rotation — good enough for BVH visualization.
    Returns (rz, rx, ry) in degrees.
    """
    direction = child_pos - parent_pos
    norm = np.linalg.norm(direction)

    if norm < 1e-6:
        return 0.0, 0.0, 0.0

    direction = direction / norm

    # Compute angles from direction vector
    rx = np.degrees(np.arcsin(-direction[1]))
    ry = np.degrees(np.arctan2(direction[0], direction[2]))
    rz = 0.0  # roll not computable from single direction vector

    return rz, rx, ry


def landmarks_to_frame(landmarks):
    """
    Convert a set of MediaPipe landmarks to one BVH motion frame.
    Returns a list of floats representing all channel values for this frame.
    """
    # Get positions for all joints
    positions = {}
    for joint, indices in LANDMARK_INDICES.items():
        positions[joint] = get_landmark_position(landmarks, indices)

    frame_data = []

    for joint in JOINT_ORDER:
        pos = positions[joint]

        if joint == "Hips":
            # Root joint: position + rotation
            frame_data.extend([pos[0], pos[1], pos[2]])

            # Rotation relative to spine
            rz, rx, ry = compute_rotation(positions["Hips"], positions["Spine"])
            frame_data.extend([rz, rx, ry])

        elif joint == "Spine":
            rz, rx, ry = compute_rotation(positions["Spine"], positions["Neck"])
            frame_data.extend([rz, rx, ry])

        elif joint == "Neck":
            frame_data.extend([0.0, 0.0, 0.0])

        elif joint == "LeftArm":
            rz, rx, ry = compute_rotation(positions["LeftArm"], positions["LeftForeArm"])
            frame_data.extend([rz, rx, ry])

        elif joint == "LeftForeArm":
            rz, rx, ry = compute_rotation(positions["LeftForeArm"], positions["LeftHand"])
            frame_data.extend([rz, rx, ry])

        elif joint == "LeftHand":
            frame_data.extend([0.0, 0.0, 0.0])

        elif joint == "RightArm":
            rz, rx, ry = compute_rotation(positions["RightArm"], positions["RightForeArm"])
            frame_data.extend([rz, rx, ry])

        elif joint == "RightForeArm":
            rz, rx, ry = compute_rotation(positions["RightForeArm"], positions["RightHand"])
            frame_data.extend([rz, rx, ry])

        elif joint == "RightHand":
            frame_data.extend([0.0, 0.0, 0.0])

        elif joint == "LeftUpLeg":
            rz, rx, ry = compute_rotation(positions["LeftUpLeg"], positions["LeftLeg"])
            frame_data.extend([rz, rx, ry])

        elif joint == "LeftLeg":
            rz, rx, ry = compute_rotation(positions["LeftLeg"], positions["LeftFoot"])
            frame_data.extend([rz, rx, ry])

        elif joint == "LeftFoot":
            frame_data.extend([0.0, 0.0, 0.0])

        elif joint == "RightUpLeg":
            rz, rx, ry = compute_rotation(positions["RightUpLeg"], positions["RightLeg"])
            frame_data.extend([rz, rx, ry])

        elif joint == "RightLeg":
            rz, rx, ry = compute_rotation(positions["RightLeg"], positions["RightFoot"])
            frame_data.extend([rz, rx, ry])

        elif joint == "RightFoot":
            frame_data.extend([0.0, 0.0, 0.0])

    return frame_data


class BVHWriter:
    """
    Collects trusted frames and writes them to a BVH file.
    Only trusted frames (uncertainty.trusted == True) are written.
    This directly implements Paper 2's risk-coverage principle:
    discard unreliable frames to get cleaner animation data.
    """

    def __init__(self, fps=30):
        self.fps = fps
        self.frames = []        # list of frame data arrays
        self.skipped = 0        # count of untrusted frames skipped

    def add_frame(self, landmarks, trusted=True):
        """
        Add a frame to the BVH recording.
        If trusted=False, frame is skipped (Paper 2 risk-coverage).
        """
        if not trusted:
            self.skipped += 1
            return

        if landmarks is None:
            self.skipped += 1
            return

        frame_data = landmarks_to_frame(landmarks)
        self.frames.append(frame_data)

    def save(self, filepath=None):
        """
        Save all collected frames to a BVH file.
        Returns the filepath where it was saved.
        """
        if not self.frames:
            print("No frames to save.")
            return None

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"../output/recording_{timestamp}.bvh"

        frame_time = 1.0 / self.fps

        with open(filepath, "w") as f:
            # Write hierarchy
            f.write(BVH_HIERARCHY)

            # Write motion section
            f.write("MOTION\n")
            f.write(f"Frames: {len(self.frames)}\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")

            # Write each frame as one line of space-separated floats
            for frame in self.frames:
                line = " ".join(f"{v:.6f}" for v in frame)
                f.write(line + "\n")

        print(f"\nBVH saved: {filepath}")
        print(f"  Frames recorded : {len(self.frames)}")
        print(f"  Frames skipped  : {self.skipped}")
        print(f"  Duration        : {len(self.frames) / self.fps:.2f} seconds")
        print(f"  Coverage        : {len(self.frames) / (len(self.frames) + self.skipped) * 100:.1f}% trusted frames used")

        return filepath

    def reset(self):
        """Clear all recorded frames to start a new recording."""
        self.frames = []
        self.skipped = 0


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import cv2
    from pose import PoseExtractor
    from uncertainty import UncertaintyScorer

    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open source: {source}")
        sys.exit(1)

    extractor = PoseExtractor()
    scorer = UncertaintyScorer()
    writer = BVHWriter(fps=30)

    print("Recording BVH. Press S to stop and save, Q to quit without saving.\n")

    recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, annotated = extractor.process_frame(frame)
        result = scorer.score(landmarks)

        # Display status
        color = (0, 255, 0) if result["trusted"] else (0, 0, 255)
        label = "TRUSTED" if result["trusted"] else "UNTRUSTED"
        rec_label = "● REC" if recording else "○ Press R to record"

        cv2.putText(annotated, rec_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if recording else (200, 200, 200), 2)
        cv2.putText(annotated, f"{label}  uncertainty={result['total']:.3f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated, f"Frames: {len(writer.frames)}  Skipped: {writer.skipped}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, "R=Record  S=Save  Q=Quit",
                    (10, annotated.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if recording:
            writer.add_frame(landmarks, trusted=result["trusted"])

        cv2.imshow("BVH Recorder", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recording = not recording
            if recording:
                writer.reset()
                print("Recording started...")
            else:
                print("Recording paused.")
        elif key == ord('s'):
            print("Saving...")
            writer.save()
            recording = False
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()