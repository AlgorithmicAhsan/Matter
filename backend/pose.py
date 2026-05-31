import cv2
import mediapipe as mp
import sys

# MediaPipe Pose connections - pairs of landmark indices that should be connected
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),       # face left
    (0, 4), (4, 5), (5, 6), (6, 8),       # face right
    (11, 12),                               # shoulders
    (11, 13), (13, 15),                    # left arm
    (12, 14), (14, 16),                    # right arm
    (11, 23), (12, 24),                    # torso sides
    (23, 24),                              # hips
    (23, 25), (25, 27),                    # left leg
    (24, 26), (26, 28),                    # right leg
    (15, 17), (15, 19), (15, 21),          # left hand
    (16, 18), (16, 20), (16, 22),          # right hand
    (27, 29), (27, 31), (29, 31),          # left foot
    (28, 30), (28, 32), (30, 32),          # right foot
]

class PoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=1,       # 0=Lite, 1=Full, 2=Heavy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Face Mesh — detected in parallel to the body pose (see process_face).
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,    # adds iris points → 478 landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Light contour connections (eyes, brows, lips, face oval, irises) —
        # used to draw the face the same way POSE_CONNECTIONS draws the body.
        self.FACE_CONNECTIONS = self.mp_face.FACEMESH_CONTOURS

        # Hands — detected in parallel to the body pose (see process_hands).
        # Gives 21 articulated landmarks per hand (full fingers), unlike the
        # ~3 coarse hand points the body Pose model provides.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.HAND_CONNECTIONS = self.mp_hands.HAND_CONNECTIONS

    def process_frame(self, frame):
        """
        Main method — call this once per frame.
        Returns (landmarks, annotated_frame)
        landmarks = list of 33 dicts {x, y, z, visibility} or None
        annotated_frame = frame with skeleton drawn manually using OpenCV
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None, annotated, None

        # Extract landmarks
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })

        # Convert normalized coords to pixel coords
        points = []
        for lm in landmarks:
            px = int(lm["x"] * w)
            py = int(lm["y"] * h)
            points.append((px, py))

        # Draw connections manually
        for start_idx, end_idx in POSE_CONNECTIONS:
            if (landmarks[start_idx]["visibility"] > 0.3 and
                    landmarks[end_idx]["visibility"] > 0.3):
                cv2.line(annotated, points[start_idx], points[end_idx],
                         (0, 255, 0), 2)

        # Draw landmark dots
        for i, (px, py) in enumerate(points):
            if landmarks[i]["visibility"] > 0.3:
                cv2.circle(annotated, (px, py), 4, (0, 0, 255), -1)

        # World landmarks — metric 3D coords (meters) centered on hip midpoint.
        # Z is far more reliable here than image-space Z because it comes from the
        # 3D pose model's skeleton estimate, not 2D foreshortening guesses.
        world_landmarks = None
        if results.pose_world_landmarks:
            world_landmarks = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in results.pose_world_landmarks.landmark
            ]

        return landmarks, annotated, world_landmarks

    def process_face(self, frame, annotated=None):
        """
        Detect facial landmarks (MediaPipe Face Mesh) in parallel to the body
        pose. Mirrors process_frame: returns the landmark list and, when an
        `annotated` frame is supplied, draws the face contours onto it.

        Returns a list of 478 dicts {x, y, z} or None. Face Mesh landmarks have
        no per-point visibility, unlike pose landmarks.
        """
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        face_landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in face.landmark]

        # Draw face contours manually — same style as the pose skeleton above.
        if annotated is not None:
            points = [(int(lm["x"] * w), int(lm["y"] * h)) for lm in face_landmarks]
            for start_idx, end_idx in self.FACE_CONNECTIONS:
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(annotated, points[start_idx], points[end_idx],
                             (255, 180, 0), 1)

        return face_landmarks

    def process_hands(self, frame, annotated=None):
        """
        Detect hand landmarks (MediaPipe Hands) in parallel to the body pose.
        Mirrors process_frame / process_face: returns the landmarks and, when an
        `annotated` frame is supplied, draws each hand skeleton onto it.

        Returns {"left": [...21], "right": [...21]} (each a list of {x, y, z}),
        with a side set to None when that hand isn't visible — or None if no
        hand is detected at all. Hand landmarks have no per-point visibility.

        NOTE: MediaPipe's Left/Right labels assume a mirrored (selfie) image.
        The raw OpenCV frame is NOT mirrored, so these labels may be swapped
        relative to the user's real hands — we'll account for that when mapping.
        """
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        out = {"left": None, "right": None}
        for lm_list, handedness in zip(results.multi_hand_landmarks,
                                       results.multi_handedness):
            label = handedness.classification[0].label.lower()  # 'left' / 'right'
            pts   = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in lm_list.landmark]
            out[label] = pts

            # Draw the hand skeleton — same manual style as the body skeleton.
            if annotated is not None:
                px = [(int(p["x"] * w), int(p["y"] * h)) for p in pts]
                for start_idx, end_idx in self.HAND_CONNECTIONS:
                    if start_idx < len(px) and end_idx < len(px):
                        cv2.line(annotated, px[start_idx], px[end_idx],
                                 (0, 200, 255), 2)
                for (x, y) in px:
                    cv2.circle(annotated, (x, y), 3, (255, 0, 200), -1)

        if out["left"] is None and out["right"] is None:
            return None
        return out


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open source: {source}")
        sys.exit(1)

    extractor = PoseExtractor()
    print("Running pose extraction. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or failed to read frame.")
            break

        landmarks, annotated_frame, _ = extractor.process_frame(frame)

        if landmarks:
            print("First 3 landmarks:")
            for i, lm in enumerate(landmarks[:3]):
                print(f"  [{i}] x={lm['x']:.3f} y={lm['y']:.3f} z={lm['z']:.3f} vis={lm['visibility']:.3f}")
        else:
            print("No pose detected")

        cv2.imshow("Pose Extraction", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()