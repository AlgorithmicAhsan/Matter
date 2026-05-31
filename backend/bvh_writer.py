import numpy as np
from datetime import datetime

# MediaPipe world_landmark indices → BVH skeleton joints.
# world_landmarks are metric (meters), Y-up, centered on hip midpoint.
# This is far more reliable than image-space landmarks (no Y-flip needed,
# Z comes from the 3D pose model rather than 2D depth estimation).
LANDMARK_INDICES = {
    "Hips":        (23, 24),   # midpoint of left + right hip
    "Spine":       (11, 12),   # midpoint of shoulders (spine approximation)
    "Neck":        (0,),       # nose as head proxy
    "LeftArm":     (11,),      # left shoulder
    "LeftForeArm": (13,),      # left elbow
    "LeftHand":    (15,),      # left wrist
    "RightArm":    (12,),      # right shoulder
    "RightForeArm":(14,),      # right elbow
    "RightHand":   (16,),      # right wrist
    "LeftUpLeg":   (23,),      # left hip
    "LeftLeg":     (25,),      # left knee
    "LeftFoot":    (27,),      # left ankle
    "RightUpLeg":  (24,),      # right hip
    "RightLeg":    (26,),      # right knee
    "RightFoot":   (28,),      # right ankle
}

BVH_HIERARCHY = """HIERARCHY
ROOT Hips
{
    OFFSET 0.00 0.00 0.00
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
        OFFSET 0.00 10.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Neck
        {
            OFFSET 0.00 10.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            End Site
            {
                OFFSET 0.00 5.00 0.00
            }
        }
    }
    JOINT LeftArm
    {
        OFFSET -5.00 8.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftForeArm
        {
            OFFSET -8.00 0.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftHand
            {
                OFFSET -8.00 0.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET -4.00 0.00 0.00
                }
            }
        }
    }
    JOINT RightArm
    {
        OFFSET 5.00 8.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightForeArm
        {
            OFFSET 8.00 0.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightHand
            {
                OFFSET 8.00 0.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET 4.00 0.00 0.00
                }
            }
        }
    }
    JOINT LeftUpLeg
    {
        OFFSET -4.00 0.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftLeg
        {
            OFFSET 0.00 -10.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftFoot
            {
                OFFSET 0.00 -10.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET 0.00 -4.00 0.00
                }
            }
        }
    }
    JOINT RightUpLeg
    {
        OFFSET 4.00 0.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightLeg
        {
            OFFSET 0.00 -10.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightFoot
            {
                OFFSET 0.00 -10.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET 0.00 -4.00 0.00
                }
            }
        }
    }
}
"""

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

# T-pose rest direction for each bone = direction from parent joint to child joint.
# Derived from the OFFSET of the child joint in BVH_HIERARCHY, then normalized.
# Example: LeftForeArm has OFFSET -8,0,0 from LeftArm → rest direction is (-1,0,0).
# These are the directions the bones point in the rest pose BEFORE any motion is applied.
REST_DIRECTIONS = {
    "Hips":        np.array([ 0.0,  1.0,  0.0]),  # Hips → Spine (offset 0,10,0)
    "Spine":       np.array([ 0.0,  1.0,  0.0]),  # Spine → Neck (offset 0,10,0)
    "LeftArm":     np.array([-1.0,  0.0,  0.0]),  # LeftArm → LeftForeArm (offset -8,0,0)
    "LeftForeArm": np.array([-1.0,  0.0,  0.0]),  # LeftForeArm → LeftHand (offset -8,0,0)
    "RightArm":    np.array([ 1.0,  0.0,  0.0]),  # RightArm → RightForeArm (offset 8,0,0)
    "RightForeArm":np.array([ 1.0,  0.0,  0.0]),  # RightForeArm → RightHand (offset 8,0,0)
    "LeftUpLeg":   np.array([ 0.0, -1.0,  0.0]),  # LeftUpLeg → LeftLeg (offset 0,-10,0)
    "LeftLeg":     np.array([ 0.0, -1.0,  0.0]),  # LeftLeg → LeftFoot (offset 0,-10,0)
    "RightUpLeg":  np.array([ 0.0, -1.0,  0.0]),  # RightUpLeg → RightLeg (offset 0,-10,0)
    "RightLeg":    np.array([ 0.0, -1.0,  0.0]),  # RightLeg → RightFoot (offset 0,-10,0)
}

# Joints with no child to point toward — always output zero rotation.
TERMINAL_JOINTS = {"Neck", "LeftHand", "RightHand", "LeftFoot", "RightFoot"}


def _get_position(world_landmarks, indices):
    """
    Average world landmark positions for a joint (used when a joint maps to
    multiple landmarks, e.g. Hips = midpoint of left+right hip).
    Scaled to cm to match the bone OFFSET units in BVH_HIERARCHY.
    """
    pts = []
    for idx in indices:
        lm = world_landmarks[idx]
        pts.append([lm["x"] * 100.0, lm["y"] * 100.0, lm["z"] * 100.0])
    return np.mean(pts, axis=0)


def _rotation_to_euler_zxy(rest_dir, bone_vec):
    """
    Compute the rotation that takes rest_dir to the direction of bone_vec,
    then decompose it to ZXY Euler angles in degrees.

    ZXY is the BVH channel order: Zrotation Xrotation Yrotation.
    R = Ry · Rx · Rz  (last channel applied last = Y applied last).

    This gives the LOCAL rotation of the bone relative to its T-pose direction,
    which is exactly what BVH motion data stores.
    """
    rest = np.asarray(rest_dir, dtype=float)
    actual_norm = np.linalg.norm(bone_vec)
    if actual_norm < 1e-6:
        return 0.0, 0.0, 0.0
    actual = bone_vec / actual_norm

    dot = float(np.clip(np.dot(rest, actual), -1.0, 1.0))

    if dot > 0.9999:   # already aligned — no rotation
        return 0.0, 0.0, 0.0

    if dot < -0.9999:  # 180° flip — pick a stable perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(rest[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(rest, perp)
        axis /= np.linalg.norm(axis)
        angle = np.pi
    else:
        axis = np.cross(rest, actual)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot)

    # Rodrigues' rotation matrix
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    R = np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c  ],
    ])

    # ZXY Euler decomposition (R = Ry·Rx·Rz):
    #   R[1][2] = -sin(rx)
    #   R[0][2] = sin(ry)*cos(rx)
    #   R[2][2] = cos(ry)*cos(rx)
    #   R[1][0] = cos(rx)*sin(rz)
    #   R[1][1] = cos(rx)*cos(rz)
    rx = np.degrees(np.arcsin(np.clip(-R[1, 2], -1.0, 1.0)))
    if abs(R[1, 2]) < 0.9999:  # no gimbal lock
        ry = np.degrees(np.arctan2(R[0, 2], R[2, 2]))
        rz = np.degrees(np.arctan2(R[1, 0], R[1, 1]))
    else:                       # gimbal lock: rz is indeterminate, set to 0
        ry = 0.0
        rz = np.degrees(np.arctan2(-R[0, 1], R[0, 0]))

    return rz, rx, ry


def _bone_rot(joint, parent_pos, child_pos):
    """Wrapper: compute rotation for a named bone from world positions."""
    return _rotation_to_euler_zxy(REST_DIRECTIONS[joint], child_pos - parent_pos)


def landmarks_to_frame(world_landmarks):
    """
    Convert one set of MediaPipe world_landmarks to a BVH motion frame.

    Each bone's rotation is the angle needed to rotate the bone FROM its
    T-pose rest direction TO the actual direction observed in this frame.
    This is exactly what BVH rotation channels represent.

    Returns a flat list of floats: 6 values for Hips + 3 for each other joint
    = 48 values total, matching the channel count in BVH_HIERARCHY.
    """
    pos = {joint: _get_position(world_landmarks, indices)
           for joint, indices in LANDMARK_INDICES.items()}

    frame_data = []
    for joint in JOINT_ORDER:
        if joint in TERMINAL_JOINTS:
            frame_data.extend([0.0, 0.0, 0.0])
            continue

        if joint == "Hips":
            # Root: world position (world_landmarks are hip-centered so this is
            # always near 0,0,0 — no absolute room-scale tracking from MediaPipe)
            frame_data.extend([0.0, 0.0, 0.0])
            frame_data.extend(_bone_rot("Hips", pos["Hips"], pos["Spine"]))
        elif joint == "Spine":
            frame_data.extend(_bone_rot("Spine", pos["Spine"], pos["Neck"]))
        elif joint == "LeftArm":
            frame_data.extend(_bone_rot("LeftArm", pos["LeftArm"], pos["LeftForeArm"]))
        elif joint == "LeftForeArm":
            frame_data.extend(_bone_rot("LeftForeArm", pos["LeftForeArm"], pos["LeftHand"]))
        elif joint == "RightArm":
            frame_data.extend(_bone_rot("RightArm", pos["RightArm"], pos["RightForeArm"]))
        elif joint == "RightForeArm":
            frame_data.extend(_bone_rot("RightForeArm", pos["RightForeArm"], pos["RightHand"]))
        elif joint == "LeftUpLeg":
            frame_data.extend(_bone_rot("LeftUpLeg", pos["LeftUpLeg"], pos["LeftLeg"]))
        elif joint == "LeftLeg":
            frame_data.extend(_bone_rot("LeftLeg", pos["LeftLeg"], pos["LeftFoot"]))
        elif joint == "RightUpLeg":
            frame_data.extend(_bone_rot("RightUpLeg", pos["RightUpLeg"], pos["RightLeg"]))
        elif joint == "RightLeg":
            frame_data.extend(_bone_rot("RightLeg", pos["RightLeg"], pos["RightFoot"]))

    return frame_data


class BVHWriter:
    """
    Collects trusted frames and writes them to a BVH file.
    Only trusted frames (uncertainty.trusted == True) are written —
    this implements Paper 2's risk-coverage principle.

    Usage:
        writer = BVHWriter(fps=30)
        writer.add_frame(landmarks, world_landmarks, trusted=True)
        writer.save("output/take1.bvh")
    """

    def __init__(self, fps=30):
        self.fps = fps
        self.frames = []
        self.skipped = 0

    def add_frame(self, landmarks, world_landmarks, trusted=True):
        """
        Add one frame to the recording.
        landmarks       : image-space landmarks (used only for the None check)
        world_landmarks : metric 3D landmarks — these drive the actual BVH data
        trusted         : if False the frame is silently dropped (Paper 2 coverage)
        """
        if not trusted or landmarks is None or world_landmarks is None:
            self.skipped += 1
            return
        self.frames.append(landmarks_to_frame(world_landmarks))

    def save(self, filepath=None):
        """Write all collected frames to a BVH file. Returns the filepath."""
        if not self.frames:
            print("No frames to save.")
            return None

        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"../output/recording_{ts}.bvh"

        frame_time = 1.0 / self.fps

        with open(filepath, "w") as f:
            f.write(BVH_HIERARCHY)
            f.write("MOTION\n")
            f.write(f"Frames: {len(self.frames)}\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")
            for frame in self.frames:
                f.write(" ".join(f"{v:.6f}" for v in frame) + "\n")

        total = len(self.frames) + self.skipped
        print(f"\nBVH saved: {filepath}")
        print(f"  Frames recorded : {len(self.frames)}")
        print(f"  Frames skipped  : {self.skipped}")
        print(f"  Duration        : {len(self.frames) / self.fps:.2f} seconds")
        print(f"  Coverage        : {len(self.frames) / max(total, 1) * 100:.1f}% trusted frames used")

        return filepath

    def reset(self):
        """Clear all recorded frames to start a new recording."""
        self.frames = []
        self.skipped = 0


# ── Standalone test ───────────────────────────────────────────────────────────
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
    recording = False

    print("BVH Recorder | R=Record/Pause  S=Save  Q=Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, annotated, world_landmarks = extractor.process_frame(frame)
        result = scorer.score(landmarks)

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
            writer.add_frame(landmarks, world_landmarks, trusted=result["trusted"])

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
