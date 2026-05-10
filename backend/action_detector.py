"""
Action Detector - Maps pose landmarks to discrete actions
Currently a placeholder - will be filled with gesture recognition logic
"""

import numpy as np
from typing import Optional, List, Dict
from actions import ActionType


class PoseActionDetector:
    """
    Analyzes pose landmarks and detects which actions are being performed.
    This is where we'll map skeletal features to game actions.
    """
    
    def __init__(self):
        # Thresholds and parameters for gesture detection
        self.detection_params = {
            # Movement detection
            'movement_threshold': 0.05,  # normalized coordinate change
            'turn_threshold': 15.0,      # degrees
            
            # Jump detection (vertical acceleration of hips)
            'jump_acceleration_threshold': 0.3,
            
            # Crouch detection (hip height relative to shoulders)
            'crouch_ratio_threshold': 0.7,
            
            # Sprint detection (high step frequency + forward lean)
            'sprint_step_frequency': 2.0,  # steps per second
            'sprint_lean_angle': 10.0,     # degrees forward
        }
        
        # Historical data for temporal features
        self.landmark_history: List[Optional[List[Dict]]] = []
        self.history_length = 10  # Keep last N frames
        
        # Previous frame landmarks for delta computation
        self.prev_landmarks: Optional[List[Dict]] = None
    
    def detect_actions(self, landmarks: Optional[List[Dict]]) -> List[ActionType]:
        """
        Main detection method.
        
        Args:
            landmarks: List of 33 MediaPipe pose landmarks {x, y, z, visibility}
        
        Returns:
            List of detected actions
        """
        if landmarks is None or len(landmarks) != 33:
            return [ActionType.IDLE]
        
        # Update history
        self._update_history(landmarks)
        
        detected_actions = []
        
        # ── PLACEHOLDER DETECTION LOGIC ──
        # TODO: Implement actual gesture recognition
        # For now, return IDLE
        
        # Example detection patterns to implement:
        # - Forward movement: forward lean + alternating leg movement
        # - Backward movement: backward lean + alternating leg movement
        # - Strafe: lateral hip movement without shoulder rotation
        # - Turn: shoulder rotation without hip translation
        # - Jump: rapid upward acceleration of center of mass
        # - Crouch: reduced hip height relative to shoulders
        # - Sprint: high step frequency + exaggerated arm swing
        
        detected_actions.append(ActionType.IDLE)
        
        self.prev_landmarks = landmarks
        return detected_actions
    
    def _update_history(self, landmarks: Optional[List[Dict]]):
        """Maintain sliding window of landmark history."""
        self.landmark_history.append(landmarks)
        if len(self.landmark_history) > self.history_length:
            self.landmark_history.pop(0)
    
    # ── HELPER METHODS FOR FEATURE EXTRACTION ──
    
    def _get_landmark(self, landmarks: List[Dict], idx: int) -> np.ndarray:
        """Extract landmark as numpy array [x, y, z]."""
        lm = landmarks[idx]
        return np.array([lm['x'], lm['y'], lm['z']])
    
    def _compute_center_of_mass(self, landmarks: List[Dict]) -> np.ndarray:
        """Compute approximate center of mass (average of hip landmarks)."""
        left_hip = self._get_landmark(landmarks, 23)
        right_hip = self._get_landmark(landmarks, 24)
        return (left_hip + right_hip) / 2.0
    
    def _compute_body_lean(self, landmarks: List[Dict]) -> float:
        """
        Compute forward/backward lean angle in degrees.
        Positive = forward, Negative = backward
        """
        # Use shoulder-to-hip vector
        left_shoulder = self._get_landmark(landmarks, 11)
        left_hip = self._get_landmark(landmarks, 23)
        
        vec = left_hip - left_shoulder
        # Project onto sagittal plane (x-y)
        angle_rad = np.arctan2(vec[0], -vec[1])  # -y because y increases downward
        return np.rad2deg(angle_rad)
    
    def _compute_hip_height(self, landmarks: List[Dict]) -> float:
        """Get normalized hip height (y-coordinate)."""
        return self._compute_center_of_mass(landmarks)[1]
    
    def _compute_shoulder_rotation(self, landmarks: List[Dict]) -> float:
        """
        Compute shoulder rotation angle relative to hips.
        Returns angle in degrees.
        """
        left_shoulder = self._get_landmark(landmarks, 11)
        right_shoulder = self._get_landmark(landmarks, 12)
        left_hip = self._get_landmark(landmarks, 23)
        right_hip = self._get_landmark(landmarks, 24)
        
        shoulder_vec = right_shoulder - left_shoulder
        hip_vec = right_hip - left_hip
        
        # Compute angle between vectors in XZ plane
        shoulder_angle = np.arctan2(shoulder_vec[2], shoulder_vec[0])
        hip_angle = np.arctan2(hip_vec[2], hip_vec[0])
        
        relative_angle = shoulder_angle - hip_angle
        return np.rad2deg(relative_angle)
    
    def _detect_step_frequency(self) -> float:
        """
        Detect step frequency from alternating ankle heights.
        Returns steps per second.
        """
        if len(self.landmark_history) < 5:
            return 0.0
        
        # TODO: Implement peak detection on ankle height oscillation
        return 0.0
    
    # ── INDIVIDUAL ACTION DETECTORS ──
    # These will be called by detect_actions() once implemented
    
    def _detect_forward_movement(self, landmarks: List[Dict]) -> bool:
        """Detect forward walking/running."""
        # Check for:
        # 1. Forward lean (5-15 degrees)
        # 2. Alternating leg movement (step detection)
        # 3. Consistent forward velocity of center of mass
        return False
    
    def _detect_backward_movement(self, landmarks: List[Dict]) -> bool:
        """Detect backward movement."""
        # Similar to forward but with backward lean
        return False
    
    def _detect_strafe_left(self, landmarks: List[Dict]) -> bool:
        """Detect left strafing."""
        # Check for lateral hip movement without shoulder rotation
        return False
    
    def _detect_strafe_right(self, landmarks: List[Dict]) -> bool:
        """Detect right strafing."""
        return False
    
    def _detect_turn_left(self, landmarks: List[Dict]) -> bool:
        """Detect left turning."""
        # Check shoulder rotation without significant hip translation
        return False
    
    def _detect_turn_right(self, landmarks: List[Dict]) -> bool:
        """Detect right turning."""
        return False
    
    def _detect_jump(self, landmarks: List[Dict]) -> bool:
        """Detect jump initiation."""
        # Check for rapid upward acceleration of hips
        # Also check for knee/ankle extension pattern
        return False
    
    def _detect_crouch(self, landmarks: List[Dict]) -> bool:
        """Detect crouching."""
        hip_height = self._compute_hip_height(landmarks)
        
        # Compare to shoulder height
        left_shoulder = self._get_landmark(landmarks, 11)
        shoulder_height = left_shoulder[1]
        
        # If hips are much closer to shoulders than normal, it's a crouch
        # (This is a simplified check)
        height_ratio = (shoulder_height - hip_height)
        return height_ratio < self.detection_params['crouch_ratio_threshold']
    
    def _detect_sprint(self, landmarks: List[Dict]) -> bool:
        """Detect sprinting (fast forward movement)."""
        # Check for:
        # 1. High step frequency
        # 2. Exaggerated forward lean
        # 3. Large arm swing amplitude
        return False


# ── Landmark Indices Reference ──────────────────────────────────────────────
"""
MediaPipe Pose Landmark Indices (33 total):

0: nose
1: left_eye_inner
2: left_eye
3: left_eye_outer
4: right_eye_inner
5: right_eye
6: right_eye_outer
7: left_ear
8: right_ear
9: mouth_left
10: mouth_right

11: left_shoulder
12: right_shoulder
13: left_elbow
14: right_elbow
15: left_wrist
16: right_wrist
17: left_pinky
18: right_pinky
19: left_index
20: right_index
21: left_thumb
22: right_thumb

23: left_hip
24: right_hip
25: left_knee
26: right_knee
27: left_ankle
28: right_ankle
29: left_heel
30: right_heel
31: left_foot_index
32: right_foot_index
"""


if __name__ == "__main__":
    # Quick test
    detector = PoseActionDetector()
    
    # Create dummy landmarks
    dummy_landmarks = [
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}
        for _ in range(33)
    ]
    
    actions = detector.detect_actions(dummy_landmarks)
    print(f"Detected actions: {[a.name for a in actions]}")
    
    # Test helper methods
    print(f"\nCenter of mass: {detector._compute_center_of_mass(dummy_landmarks)}")
    print(f"Body lean: {detector._compute_body_lean(dummy_landmarks):.2f}°")
    print(f"Hip height: {detector._compute_hip_height(dummy_landmarks):.3f}")