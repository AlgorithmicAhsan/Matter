import cv2
import mediapipe as mp
import sys

mpHolistic = mp.solutions.holistic
mpDrawing  = mp.solutions.drawing_utils
mpStyles   = mp.solutions.drawing_styles


class PoseExtractor:
    def __init__(self):
        self.holistic = mpHolistic.Holistic(
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def process_frame(self, frame):
        """
        Returns (result_dict, annotated_frame)
        result_dict keys: pose2d, pose3d, face, leftHand, rightHand
        Each value is a list of {x,y,z,visibility} dicts or None.
        annotated_frame has skeleton drawn on it.
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)

        def lmList(landmarks, include3d=False):
            if not landmarks:
                return None
            out = []
            for lm in landmarks.landmark:
                entry = {"x": lm.x, "y": lm.y, "z": lm.z}
                if hasattr(lm, "visibility"):
                    entry["visibility"] = lm.visibility
                else:
                    entry["visibility"] = 1.0
                out.append(entry)
            return out

        # Build 3D pose from pose_world_landmarks
        pose3d = None
        if results.pose_world_landmarks:
            pose3d = []
            for lm in results.pose_world_landmarks.landmark:
                pose3d.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility if hasattr(lm, "visibility") else 1.0,
                })

        resultDict = {
            "pose2d":     lmList(results.pose_landmarks),
            "pose3d":     pose3d,
            "face":       lmList(results.face_landmarks),
            "leftHand":   lmList(results.left_hand_landmarks),
            "rightHand":  lmList(results.right_hand_landmarks),
        }

        # Draw overlays on annotated frame
        mpDrawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mpHolistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mpStyles.get_default_pose_landmarks_style(),
        )
        mpDrawing.draw_landmarks(
            annotated,
            results.face_landmarks,
            mpHolistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mpStyles.get_default_face_mesh_tesselation_style(),
        )
        mpDrawing.draw_landmarks(
            annotated,
            results.left_hand_landmarks,
            mpHolistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mpStyles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mpStyles.get_default_hand_connections_style(),
        )
        mpDrawing.draw_landmarks(
            annotated,
            results.right_hand_landmarks,
            mpHolistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mpStyles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mpStyles.get_default_hand_connections_style(),
        )

        return resultDict, annotated
