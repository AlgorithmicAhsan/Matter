"""
Action Detector - Maps MediaPipe pose landmarks to discrete ActionTypes.

Design principles:
  1. Body-relative geometry   — all features are ratios/angles, never raw coords,
                                so they work regardless of camera distance or height.
  2. Calibration              — first CALIBRATION_FRAMES frames of a stable pose are
                                averaged to build a personal neutral reference.
  3. Temporal voting          — each feature votes over a sliding window; an action
                                fires only when it wins a majority, killing flicker.
  4. Confidence gating        — landmarks below a visibility threshold are ignored;
                                if key landmarks are invisible the detector degrades
                                gracefully to IDLE.
  5. Priority / exclusion     — JUMP overrides CROUCH. Movement and rotation are
                                handled continuously via compute_pose_transform().

MediaPipe coordinate conventions (normalised):
  x  : 0 (left edge) -> 1 (right edge)  -- mirrored relative to person in webcam
  y  : 0 (top)       -> 1 (bottom)       -- y increases downward
  z  : negative = closer to camera       -- unreliable for depth, used sparingly
  visibility : 0 -> 1
"""

import collections
import math
import numpy as np
from typing import Optional, List, Dict
from actions import ActionType


# -- Landmark index constants --------------------------------------------------
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28
NOSE                   = 0


# -- Tunable thresholds (all documented) --------------------------------------
class DetectorConfig:

    # Calibration
    CALIBRATION_FRAMES         = 45      # ~1.5 s at 30 fps - person must stand still
    CALIB_STILLNESS_THRESHOLD  = 0.02    # max mean position delta to accept a frame

    # Visibility gates
    VIS_MIN                    = 0.50    # per-landmark visibility minimum
    KEY_VIS_MIN                = 0.40    # gate for structural landmarks (hips/shoulders)

    # Temporal voting windows (frames) — only used for CROUCH / JUMP
    VOTE_WINDOW                = 14      # majority = > window//2 + 1 (longer = less jitter)
    JUMP_VOTE_WINDOW           = 4       # 4-frame window; jump needs only 1 positive vote

    # -- Crouch ---------------------------------------------------------------
    CROUCH_HIP_DROP_RATIO      = 0.22    # drop > 22% of torso length -> crouch (raised to kill jitter)

    # -- Jump -----------------------------------------------------------------
    JUMP_HIP_RISE_RATIO        = 0.15    # hip rises > 15% of torso in <=4 frames
    JUMP_COOLDOWN_FRAMES       = 15      # frames to suppress repeat jumps (~0.5s at 30fps)

    # -- Rotation (shoulder angle delta) --------------------------------------
    ROTATION_EMA               = 0.20    # EMA smoothing: lower = smoother, more lag
    ROTATION_SCALE             = 0.9     # amplify the smoothed delta
    ROTATION_DEADZONE          = 0.018   # ignore deltas smaller than this — raised to kill arm-raise noise

    # -- Hip X mirroring → world units ----------------------------------------
    # Avatar is 150 units tall, world is 1000 units wide.
    # A shoulder-width lateral move maps to HIP_X_WORLD_SCALE units.
    HIP_X_WORLD_SCALE          = 120.0

    # -- Walk speed mapping ---------------------------------------------------
    WALK_SPEED_SCALE           = 900.0   # osc_amplitude * this = units/s
    WALK_SPEED_MIN             = 25.0
    WALK_SPEED_MAX             = 180.0

    # -- Anti-phase walking ---------------------------------------------------
    ANKLE_OSC_THRESHOLD        = 0.10    # min avg peak-to-peak amplitude
    WALK_ANTIPHASE_CORR        = -0.50   # correlation must be below this
    ANKLE_MIN_STD              = 0.008   # BOTH ankles must move this much — kills single-leg triggers

    # -- Walk direction (nose-hip delta, normalised) ---------------------------
    LEAN_FORWARD_THRESHOLD     = -0.04
    LEAN_BACKWARD_THRESHOLD    =  0.04

    # -- Strafe (lateral walking) -----------------------------------------------
    # Hip X velocity (normalized image coords per frame) to trigger strafe.
    # Paired with a minimum ankle std so pure swaying doesn't count.
    STRAFE_VELOCITY_THRESHOLD  = 0.005


class PoseActionDetector:
    """
    Analyses MediaPipe pose landmarks each frame.

    Two entry points:
      detect_actions()         → [CROUCH] / [JUMP] / [IDLE]  (discrete, voted)
      compute_pose_transform() → rotation delta, hip X, walking  (continuous)
    """

    def __init__(self, config: DetectorConfig = None):
        self.cfg = config or DetectorConfig()

        # -- Calibration state -------------------------------------------------
        self.calibrated           = False
        self._calib_buffer: List[List[Dict]] = []

        # Neutral reference (populated after calibration)
        self.neutral_shoulder_width = None
        self.neutral_shoulder_y     = None
        self.neutral_shoulder_x     = None
        self.neutral_hip_y          = None
        self.neutral_hip_x          = None
        self.neutral_torso_length   = None
        self.neutral_nose_hip_dy    = None   # nose_y - hip_y at rest

        # -- History buffers ---------------------------------------------------
        W = max(self.cfg.VOTE_WINDOW, 30)
        self._hip_y_buf      = collections.deque(maxlen=W)
        self._hip_x_buf      = collections.deque(maxlen=W)
        self._l_ankle_y_buf  = collections.deque(maxlen=W)
        self._r_ankle_y_buf  = collections.deque(maxlen=W)
        self._l_ankle_x_buf  = collections.deque(maxlen=W)
        self._r_ankle_x_buf  = collections.deque(maxlen=W)
        self._l_wrist_y_buf  = collections.deque(maxlen=W)
        self._r_wrist_y_buf  = collections.deque(maxlen=W)

        # Smoothed world-space Z for shoulder landmarks (for rotation + lean).
        self._smooth_ls_z    = None
        self._smooth_rs_z    = None

        # -- Per-action vote queues (CROUCH + JUMP only) -----------------------
        VW = self.cfg.VOTE_WINDOW
        JW = self.cfg.JUMP_VOTE_WINDOW
        self._votes: Dict[ActionType, collections.deque] = {
            ActionType.CROUCH: collections.deque(maxlen=VW),
            ActionType.JUMP:   collections.deque(maxlen=JW),
        }

        self._jump_cooldown = 0   # frames remaining

        # -- Rotation tracking -------------------------------------------------
        self._prev_shoulder_angle   = None
        self._smooth_rotation_delta = 0.0

    # =========================================================================
    # Public API
    # =========================================================================

    def detect_actions(self, landmarks: Optional[List[Dict]]) -> List[ActionType]:
        """
        Returns [CROUCH] and/or [JUMP] only.
        Movement and rotation are handled by compute_pose_transform().
        """
        if landmarks is None or len(landmarks) != 33:
            self._cast_all_votes(0)
            return [ActionType.IDLE]

        if not self._key_landmarks_visible(landmarks):
            self._cast_all_votes(0)
            return [ActionType.IDLE]

        if not self.calibrated:
            self._accumulate_calibration(landmarks)
            return [ActionType.IDLE]

        self._update_buffers(landmarks)
        self._vote_frame(landmarks)

        active = self._tally_votes()
        
        # Jump needs only 1 positive vote — it's too fast for majority
        jump_q = self._votes[ActionType.JUMP]
        if jump_q and sum(jump_q) >= 1:
            active.add(ActionType.JUMP)
        
        active = self._apply_priority(active)

        # Jump cooldown
        if self._jump_cooldown > 0:
            self._jump_cooldown -= 1
            active.discard(ActionType.JUMP)
        elif ActionType.JUMP in active:
            self._jump_cooldown = self.cfg.JUMP_COOLDOWN_FRAMES
            # Flush the rotation EMA so accumulated z-wobble during the jump
            # doesn't bleed into post-landing rotation.
            self._smooth_rotation_delta = 0.0

        return list(active) if active else [ActionType.IDLE]

    def compute_pose_transform(self, landmarks: Optional[List[Dict]],
                               world_landmarks: Optional[List[Dict]] = None) -> Optional[dict]:
        """
        Returns continuous transform values for direct avatar control:
          rotation_delta   – radians to add to avatar Y rotation this frame
          hip_x_world      – world-space X to set directly (not velocity)
          is_walking        – True when anti-phase ankle oscillation detected
          walk_speed        – units/s to move (0 when not walking)
          walk_direction_z  – 1 = forward, -1 = backward
          walk_direction_x  – +1 = strafe right, -1 = strafe left, 0 = none
        Returns None if not calibrated or landmarks invalid.
        """
        if not self.calibrated or landmarks is None or len(landmarks) < 33:
            return None
        if not self._key_landmarks_visible(landmarks):
            return None

        # ── Rotation: frame-to-frame shoulder angle delta ─────────────────────
        # Prefer world landmarks for Z — they are metric 3D estimates from the
        # full 3D pose model, far more stable than image-space Z which is inferred
        # from 2D foreshortening and is the primary source of rotation jitter.
        if world_landmarks and len(world_landmarks) >= 33:
            ls_rot = world_landmarks[L_SHOULDER]
            rs_rot = world_landmarks[R_SHOULDER]
            dx = ls_rot['x'] - rs_rot['x']
            dz = ls_rot['z'] - rs_rot['z']
        else:
            # Fallback: smooth image-space Z aggressively before use
            ls = landmarks[L_SHOULDER]
            rs = landmarks[R_SHOULDER]
            Z_ALPHA = 0.05
            self._smooth_ls_z = ls['z'] if self._smooth_ls_z is None else self._smooth_ls_z + Z_ALPHA * (ls['z'] - self._smooth_ls_z)
            self._smooth_rs_z = rs['z'] if self._smooth_rs_z is None else self._smooth_rs_z + Z_ALPHA * (rs['z'] - self._smooth_rs_z)
            dx = ls['x'] - rs['x']
            dz = self._smooth_ls_z - self._smooth_rs_z

        current_angle = math.atan2(dz, dx)

        if self._prev_shoulder_angle is None:
            self._prev_shoulder_angle = current_angle
            raw_delta = 0.0
        else:
            raw_delta = current_angle - self._prev_shoulder_angle
            # Wrap to [-π, π] to handle angle discontinuity
            if raw_delta >  math.pi: raw_delta -= 2 * math.pi
            if raw_delta < -math.pi: raw_delta += 2 * math.pi
            
            # Suppress ALL rotation during jump cooldown — not just large spikes.
            # Small z-axis wobbles during a jump accumulate in the EMA and cause
            # a persistent rotation offset after landing.
            if self._jump_cooldown > 0:
                raw_delta = 0.0
                current_angle = self._prev_shoulder_angle  # hold ref, don't advance

            self._prev_shoulder_angle = current_angle

        # EMA smooth to remove z-axis noise while preserving intentional turns
        a = self.cfg.ROTATION_EMA
        self._smooth_rotation_delta = (
            a * raw_delta + (1 - a) * getattr(self, '_smooth_rotation_delta', 0.0)
        )
        rotation_delta = self._smooth_rotation_delta * self.cfg.ROTATION_SCALE
        
        if abs(rotation_delta) < self.cfg.ROTATION_DEADZONE:
            rotation_delta = 0.0

        # ── Hip X & Y: direct position mirroring ──────────────────────────────
        hip_x = (landmarks[L_HIP]['x'] + landmarks[R_HIP]['x']) / 2
        hip_y = (landmarks[L_HIP]['y'] + landmarks[R_HIP]['y']) / 2
        
        # X mapping: Image X is mirrored, we negate it
        hip_x_offset = self.neutral_hip_x - hip_x
        hip_x_world  = hip_x_offset * self.cfg.HIP_X_WORLD_SCALE

        # Y mapping: only lower the avatar when the person is clearly crouching.
        # Apply a dead-band (10% of torso) so normal posture noise doesn't float the avatar.
        hip_y_raw    = hip_y - self.neutral_hip_y          # positive = hips dropped
        dead_band    = self.neutral_torso_length * 0.10
        hip_y_offset = max(0.0, hip_y_raw - dead_band)     # only the amount past the dead-band
        hip_y_world  = -hip_y_offset * self.cfg.HIP_X_WORLD_SCALE * 1.5

        # ── Walking: anti-phase ankle oscillation ─────────────────────────────
        is_walking, osc_amp = self._ankles_alternating()

        walk_speed      = 0.0
        walk_direction_z = 1   # forward; backward detection is a future improvement
        if is_walking:
            raw_speed  = osc_amp * self.cfg.WALK_SPEED_SCALE
            walk_speed = float(np.clip(raw_speed, self.cfg.WALK_SPEED_MIN, self.cfg.WALK_SPEED_MAX))

        walk_direction_x = self._detect_strafe()

        return {
            'rotation_delta':   float(rotation_delta),
            'hip_x_world':      float(hip_x_world),
            'hip_y_world':      float(hip_y_world),
            'is_walking':       bool(is_walking),
            'walk_speed':       float(walk_speed),
            'walk_direction_z': int(walk_direction_z),
            'walk_direction_x': int(walk_direction_x),
        }

    def is_calibrated(self) -> bool:
        return self.calibrated

    def calibration_progress(self) -> float:
        """Returns 0.0 -> 1.0."""
        return min(len(self._calib_buffer) / self.cfg.CALIBRATION_FRAMES, 1.0)

    def reset_calibration(self):
        self.calibrated = False
        self._calib_buffer.clear()
        self._prev_shoulder_angle   = None
        self._smooth_rotation_delta = 0.0

    # =========================================================================
    # Calibration
    # =========================================================================

    def _accumulate_calibration(self, landmarks: List[Dict]):
        """Accept only still frames during calibration."""
        if self._calib_buffer:
            delta = self._mean_position_delta(self._calib_buffer[-1], landmarks)
            if delta > self.cfg.CALIB_STILLNESS_THRESHOLD:
                progress = len(self._calib_buffer)
                print(f"[CALIB] Frame rejected — movement {delta:.5f} > {self.cfg.CALIB_STILLNESS_THRESHOLD} | Progress: {progress}/{self.cfg.CALIBRATION_FRAMES}")
                return

        self._calib_buffer.append(landmarks)
        progress = len(self._calib_buffer)
        print(f"[CALIB] Frame accepted | Progress: {progress}/{self.cfg.CALIBRATION_FRAMES} ({100*progress/self.cfg.CALIBRATION_FRAMES:.1f}%)")

        if len(self._calib_buffer) >= self.cfg.CALIBRATION_FRAMES:
            self._finalise_calibration()

    def _finalise_calibration(self):
        buf = self._calib_buffer

        def avg(idx, coord):
            return float(np.mean([f[idx][coord] for f in buf]))

        ls_x = avg(L_SHOULDER, 'x');  rs_x = avg(R_SHOULDER, 'x')
        ls_y = avg(L_SHOULDER, 'y');  rs_y = avg(R_SHOULDER, 'y')
        lh_x = avg(L_HIP, 'x');       rh_x = avg(R_HIP, 'x')
        lh_y = avg(L_HIP, 'y');       rh_y = avg(R_HIP, 'y')
        la_y = avg(L_ANKLE, 'y');     ra_y = avg(R_ANKLE, 'y')
        nose_y = avg(NOSE, 'y')

        self.neutral_shoulder_width = abs(rs_x - ls_x)
        self.neutral_shoulder_y     = (ls_y + rs_y) / 2
        self.neutral_shoulder_x     = (ls_x + rs_x) / 2
        self.neutral_hip_y          = (lh_y + rh_y) / 2
        self.neutral_hip_x          = (lh_x + rh_x) / 2
        self.neutral_ankle_y        = (la_y + ra_y) / 2
        self.neutral_torso_length   = max(self.neutral_hip_y - self.neutral_shoulder_y, 0.01)
        self.neutral_nose_hip_dy    = nose_y - self.neutral_hip_y

        self.calibrated = True
        print(
            f"[ActionDetector] Calibrated — "
            f"shoulder_w={self.neutral_shoulder_width:.3f}  "
            f"torso={self.neutral_torso_length:.3f}  "
            f"hip_y={self.neutral_hip_y:.3f}"
        )

    # =========================================================================
    # Buffer updates
    # =========================================================================

    def _update_buffers(self, lm: List[Dict]):
        hip_y = (lm[L_HIP]['y'] + lm[R_HIP]['y']) / 2
        hip_x = (lm[L_HIP]['x'] + lm[R_HIP]['x']) / 2
        self._hip_y_buf.append(hip_y)
        self._hip_x_buf.append(hip_x)

        if self._vis(lm, L_ANKLE):
            self._l_ankle_y_buf.append(lm[L_ANKLE]['y'])
            self._l_ankle_x_buf.append(lm[L_ANKLE]['x'])
        if self._vis(lm, R_ANKLE):
            self._r_ankle_y_buf.append(lm[R_ANKLE]['y'])
            self._r_ankle_x_buf.append(lm[R_ANKLE]['x'])
        if self._vis(lm, L_WRIST): self._l_wrist_y_buf.append(lm[L_WRIST]['y'])
        if self._vis(lm, R_WRIST): self._r_wrist_y_buf.append(lm[R_WRIST]['y'])

    # =========================================================================
    # Per-frame voting (CROUCH + JUMP only)
    # =========================================================================

    def _vote_frame(self, lm: List[Dict]):
        """Cast votes for CROUCH and JUMP only. Movement handled by compute_pose_transform."""
        torso         = self.neutral_torso_length
        hip_y         = (lm[L_HIP]['y'] + lm[R_HIP]['y']) / 2
        hip_y_delta_n = (hip_y - self.neutral_hip_y) / torso

        # -- CROUCH -----------------------------------------------------------
        self._votes[ActionType.CROUCH].append(
            int(hip_y_delta_n > self.cfg.CROUCH_HIP_DROP_RATIO)
        )

        # -- JUMP -------------------------------------------------------------
        is_jump = False
        hip_rise = 0.0
        if len(self._hip_y_buf) >= self.cfg.JUMP_VOTE_WINDOW:
            recent   = list(self._hip_y_buf)[-self.cfg.JUMP_VOTE_WINDOW:]
            hip_rise = (recent[0] - recent[-1]) / torso
            
            # To prevent standing up from a crouch from triggering a jump:
            # 1. Hip must rise fast enough.
            # 2. Hip must clear neutral standing height.
            # 3. Ankles must lift off the ground (if visible).
            is_above_neutral = hip_y < (self.neutral_hip_y - 0.05 * torso)
            
            ankle_y = (lm[L_ANKLE]['y'] + lm[R_ANKLE]['y']) / 2
            # Require ankles to be at least 15% torso above their resting floor position.
            ankles_lifted = ankle_y < (self.neutral_ankle_y - 0.15 * torso)
            
            # We use JUMP_HIP_RISE_RATIO = 0.25 to prevent explosive stand-ups
            is_fast_enough = hip_rise > 0.25
            
            is_jump = is_fast_enough and is_above_neutral and ankles_lifted
        self._votes[ActionType.JUMP].append(int(is_jump))

    # =========================================================================
    # Vote tallying
    # =========================================================================

    def _tally_votes(self) -> set:
        active = set()
        for action, queue in self._votes.items():
            if not queue:
                continue
            if sum(queue) >= len(queue) // 2 + 1:
                active.add(action)
        return active

    def _cast_all_votes(self, value: int):
        for q in self._votes.values():
            q.append(value)

    # =========================================================================
    # Priority / exclusion rules
    # =========================================================================

    def _apply_priority(self, active: set) -> set:
        # JUMP overrides CROUCH
        if ActionType.JUMP in active:
            active.discard(ActionType.CROUCH)
        return active

    # =========================================================================
    # Feature helpers
    # =========================================================================

    def _ankles_alternating(self):
        """
        Returns (is_walking, oscillation_amplitude).
        Walking = ankles oscillate in anti-phase (left up while right down).
        Standing still = ankles move in sync or barely at all.
        Uses 16 frames (~0.53s) to reliably cover one full step cycle even at
        slow cadence, compared to the old 8-frame window.
        """
        if len(self._l_ankle_y_buf) < 16 or len(self._r_ankle_y_buf) < 16:
            return False, 0.0

        la = np.array(list(self._l_ankle_y_buf)[-16:])
        ra = np.array(list(self._r_ankle_y_buf)[-16:])

        la_std = la.std()
        ra_std = ra.std()
        # Both ankles must show meaningful movement — kills single-leg raises
        if la_std < self.cfg.ANKLE_MIN_STD or ra_std < self.cfg.ANKLE_MIN_STD:
            return False, 0.0

        la_n = la - la.mean()
        ra_n = ra - ra.mean()
        corr = float(np.dot(la_n, ra_n) / (la_std * ra_std * len(la)))
        osc  = float((la.max() - la.min() + ra.max() - ra.min()) / 2)

        is_walking = corr < self.cfg.WALK_ANTIPHASE_CORR and osc > self.cfg.ANKLE_OSC_THRESHOLD
        return is_walking, osc

    def _detect_strafe(self) -> int:
        """
        Returns +1 (strafe avatar-right), -1 (strafe avatar-left), or 0 (no strafe).

        Method: track hip X velocity over the last 8 frames. A sustained lateral
        shift paired with any ankle movement = sidestep / strafe.

        Image X is mirrored for a front-facing webcam, so:
          hip_x decreasing → person moving to their right → avatar strafe right (+1)
          hip_x increasing → person moving to their left  → avatar strafe left  (-1)
        """
        if len(self._hip_x_buf) < 8:
            return 0

        # Require at least one ankle to be moving (rule out pure camera sway)
        la_std = np.std(list(self._l_ankle_y_buf)[-8:]) if len(self._l_ankle_y_buf) >= 8 else 0.0
        ra_std = np.std(list(self._r_ankle_y_buf)[-8:]) if len(self._r_ankle_y_buf) >= 8 else 0.0
        if max(la_std, ra_std) < self.cfg.ANKLE_MIN_STD:
            return 0

        recent = list(self._hip_x_buf)[-8:]
        vel = (recent[-1] - recent[0]) / 8   # normalized image X per frame

        if abs(vel) < self.cfg.STRAFE_VELOCITY_THRESHOLD:
            return 0

        # Negate: image X decreasing = person went right = avatar strafe right (+1)
        return -1 if vel > 0 else 1

    def _oscillation(self, buf_a: collections.deque, buf_b: collections.deque) -> float:
        """Average peak-to-peak amplitude across two signal buffers."""
        scores = []
        for buf in (buf_a, buf_b):
            if len(buf) >= 6:
                arr = np.array(buf)
                scores.append(arr.max() - arr.min())
        return float(np.mean(scores)) if scores else 0.0

    def _vis(self, lm: List[Dict], idx: int, threshold: float = None) -> bool:
        t = threshold if threshold is not None else self.cfg.VIS_MIN
        return lm[idx]['visibility'] >= t

    def _key_landmarks_visible(self, lm: List[Dict]) -> bool:
        for idx in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, NOSE]:
            if not self._vis(lm, idx, self.cfg.KEY_VIS_MIN):
                return False
        return True

    def _mean_position_delta(self, a: List[Dict], b: List[Dict]) -> float:
        deltas = [
            ((la['x']-lb['x'])**2 + (la['y']-lb['y'])**2) ** 0.5
            for la, lb in zip(a, b)
        ]
        return float(np.mean(deltas))


# =============================================================================
# Quick CLI test
# =============================================================================

if __name__ == "__main__":
    import sys, cv2
    from pose import PoseExtractor

    source = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source {source}"); exit(1)

    extractor = PoseExtractor()
    detector  = PoseActionDetector()
    cfg       = DetectorConfig()

    print(f"Stand still for ~{cfg.CALIBRATION_FRAMES/30:.0f}s to calibrate. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        landmarks, annotated = extractor.process_frame(frame)
        actions = detector.detect_actions(landmarks)

        if not detector.calibrated:
            label = f"CALIBRATING {detector.calibration_progress()*100:.0f}%"
            color = (0, 200, 255)
        else:
            label = " | ".join(a.value for a in actions)
            color = (0, 255, 80)

        cv2.putText(annotated, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow("Action Detector", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()