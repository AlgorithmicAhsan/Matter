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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

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
            return None, annotated

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

        return landmarks, annotated


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

        landmarks, annotated_frame = extractor.process_frame(frame)

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