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
        # Convert MediaPipe world space → BVH space by negating BOTH X and Y:
        #   • Y: MediaPipe Y increases downward; BVH Y increases upward.
        #   • X: MediaPipe's left-body landmarks sit on +X, but our skeleton anchors
        #        the "Left" joints on -X. Without this flip each limb would start at
        #        one hip/shoulder and cross to the opposite side ("opening inward").
        # Negating two axes (not one or three) keeps the coordinate system right-
        # handed, so the figure is not mirrored — only correctly re-oriented.
        pts.append([-lm["x"] * 100.0, -lm["y"] * 100.0, lm["z"] * 100.0])
    return np.mean(pts, axis=0)


# Each non-terminal bone points from its own joint toward this child joint.
# The observed (parent → child) vector is what we align the rest direction to.
BONE_CHILD = {
    "Hips":         "Spine",
    "Spine":        "Neck",
    "LeftArm":      "LeftForeArm",
    "LeftForeArm":  "LeftHand",
    "RightArm":     "RightForeArm",
    "RightForeArm": "RightHand",
    "LeftUpLeg":    "LeftLeg",
    "LeftLeg":      "LeftFoot",
    "RightUpLeg":   "RightLeg",
    "RightLeg":     "RightFoot",
}

# Parent bone in the hierarchy. A joint's LOCAL rotation is its world rotation
# expressed relative to its parent's world rotation. Root (Hips) has no parent.
PARENT_BONE = {
    "Hips":         None,
    "Spine":        "Hips",
    "LeftArm":      "Hips",
    "LeftForeArm":  "LeftArm",
    "RightArm":     "Hips",
    "RightForeArm": "RightArm",
    "LeftUpLeg":    "Hips",
    "LeftLeg":      "LeftUpLeg",
    "RightUpLeg":   "Hips",
    "RightLeg":     "RightUpLeg",
}


def _swing_matrix(rest_dir, bone_vec):
    """
    World-space shortest-arc (swing) rotation matrix that maps rest_dir onto the
    direction of bone_vec. No roll/twist is recovered — positions alone cannot
    determine a bone's rotation about its own axis, so we use the minimal rotation.
    Returns a 3x3 numpy array (identity if the bone has zero length).
    """
    rest = np.asarray(rest_dir, dtype=float)
    n = np.linalg.norm(bone_vec)
    if n < 1e-6:
        return np.eye(3)
    actual = bone_vec / n

    dot = float(np.clip(np.dot(rest, actual), -1.0, 1.0))
    if dot > 0.999999:                 # already aligned
        return np.eye(3)

    if dot < -0.999999:                # 180° flip — choose any stable perpendicular
        perp = np.array([1.0, 0.0, 0.0]) if abs(rest[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(rest, perp)
        axis /= np.linalg.norm(axis)
        angle = np.pi
    else:
        axis = np.cross(rest, actual)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot)

    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c  ],
    ])


def _matrix_to_euler_zxy(R):
    """
    Decompose a rotation matrix into BVH ZXY Euler angles (degrees), matching the
    channel order 'Zrotation Xrotation Yrotation', i.e. R = Rz · Rx · Ry — the
    convention used by Blender's and Three.js's BVH importers.

    From R = Rz·Rx·Ry:
        R[2][1] =  sin(rx)
        R[2][0] = -cos(rx)·sin(ry),  R[2][2] = cos(rx)·cos(ry)
        R[0][1] = -cos(rx)·sin(rz),  R[1][1] = cos(rx)·cos(rz)
    Returns (rz, rx, ry) in degrees — the order the channels are written.
    """
    sx = float(np.clip(R[2, 1], -1.0, 1.0))
    rx = np.arcsin(sx)
    if abs(sx) < 0.999999:             # no gimbal lock
        ry = np.arctan2(-R[2, 0], R[2, 2])
        rz = np.arctan2(-R[0, 1], R[1, 1])
    else:                              # gimbal lock (rx = ±90°): fold y into z
        ry = 0.0
        rz = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees(rz), np.degrees(rx), np.degrees(ry)


def landmarks_to_frame(world_landmarks):
    """
    Convert one set of MediaPipe world_landmarks to a BVH motion frame.

    For each bone we first build its WORLD rotation (shortest arc from the T-pose
    rest direction to the observed parent→child direction). We then convert that
    to a LOCAL rotation relative to the parent bone — because BVH rotations
    compound down the hierarchy, a joint must store only the rotation *added* on
    top of its parent. Finally each local rotation is decomposed to ZXY Euler.

    Returns a flat list of 48 floats: 6 channels for Hips (pos + rot) plus 3 for
    each of the other 14 joints, matching the channel layout in BVH_HIERARCHY.
    """
    pos = {joint: _get_position(world_landmarks, indices)
           for joint, indices in LANDMARK_INDICES.items()}

    # World-space rotation matrix for every non-terminal bone.
    world_rot = {
        bone: _swing_matrix(REST_DIRECTIONS[bone], pos[child] - pos[bone])
        for bone, child in BONE_CHILD.items()
    }

    frame_data = []
    for joint in JOINT_ORDER:
        if joint in TERMINAL_JOINTS:
            # End-effectors have no child to aim at and no children of their own,
            # so their rotation is irrelevant — write zeros.
            frame_data.extend([0.0, 0.0, 0.0])
            continue

        if joint == "Hips":
            # Root translation: world_landmarks are hip-centered, so the hips are
            # always at the origin — no global motion is available to record.
            frame_data.extend([0.0, 0.0, 0.0])
            local = world_rot["Hips"]
        else:
            parent = PARENT_BONE[joint]
            # local = parent_world⁻¹ · bone_world   (rotation matrices: inverse = transpose)
            local = world_rot[parent].T @ world_rot[joint]

        frame_data.extend(_matrix_to_euler_zxy(local))

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
