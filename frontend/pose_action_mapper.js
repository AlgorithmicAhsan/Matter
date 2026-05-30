/**
 * PoseActionMapper.js
 * --------------------
 * Decouples MediaPipe pose landmarks from the ActionController. Detects a small
 * set of SIMPLE HAND GESTURES and translates them into predefined locomotion
 * moves. Used ONLY by the Gesture tab — Live/Video modes never instantiate it.
 *
 * Gesture → Move map (this is the source of truth the UI "Moves Mapping" menu
 * mirrors — keep them in sync):
 *
 *   GESTURE                                  →  MOVE
 *   ──────────────────────────────────────     ───────────────────────
 *   Right hand raised above shoulder         →  Walk forward
 *   Left hand raised above shoulder          →  Walk backward
 *   BOTH hands raised above shoulders        →  Sprint forward
 *   BOTH hands raised high (above head)      →  Jump   (one-shot)
 *   Right arm extended out to the side       →  Turn right
 *   Left arm extended out to the side        →  Turn left
 *   BOTH hands dropped low (near hips)        →  Crouch
 *
 * The first 4 (forward/backward/sprint/crouch) are ported faithfully from the
 * original Matter-abd mapper. Jump + the two turn gestures are added so the
 * obstacle course (jump gap + zig-zag walls) is completable with gestures alone.
 *
 * Detection uses normalized MediaPipe coords: x∈[0,1] (→ right in image),
 * y∈[0,1] (→ down). Thresholds are relative to the user's own shoulders/torso
 * so they hold regardless of distance from the camera.
 */

class PoseActionMapper {
    /**
     * @param {ActionController} actionCtrl
     */
    constructor(actionCtrl) {
        this.actionCtrl = actionCtrl;

        // Landmark indices (MediaPipe Pose)
        this.LANDMARKS = {
            NOSE:           0,
            LEFT_SHOULDER:  11,
            RIGHT_SHOULDER: 12,
            LEFT_WRIST:     15,
            RIGHT_WRIST:    16,
        };

        // ── Tunables ────────────────────────────────────────────────────────
        this.UP_MARGIN     = 0.12;  // wrist this far above shoulder = "hand up"
        this.LOW_MARGIN    = 0.35;  // wrist this far below shoulder = "hand low"
        this.SIDE_FACTOR    = 1.15; // wrist out by >factor×shoulderSpan = "arm out"
        this.HORIZ_BAND    = 0.18;  // |wrist.y - shoulder.y| within this = "horizontal"

        // Edge-trigger latch for the one-shot jump gesture.
        this._jumpArmed = false;
    }

    /**
     * Main entry point — call every frame with the latest pose data.
     * @param {Array|null} landmarks  MediaPipe pose landmarks (33)
     * @param {boolean} poseMode      true only while gesture/pose tracking is ON
     */
    update(landmarks, poseMode) {
        if (!poseMode || !landmarks) {
            this._cleanup();
            return;
        }

        const L  = this.LANDMARKS;
        const ls = landmarks[L.LEFT_SHOULDER];
        const rs = landmarks[L.RIGHT_SHOULDER];
        const lw = landmarks[L.LEFT_WRIST];
        const rw = landmarks[L.RIGHT_WRIST];
        const nose = landmarks[L.NOSE];

        // Safety check — need shoulders + wrists to detect anything.
        if (!ls || !rs || !lw || !rw) { this._cleanup(); return; }

        // ── Reference measurements ──────────────────────────────────────────
        const shoulderSpan = Math.max(Math.abs(ls.x - rs.x), 0.05);
        // "High" line: above the head. Use the nose when available, otherwise
        // approximate a head line a bit above the shoulders.
        const highY = nose ? nose.y : (Math.min(ls.y, rs.y) - 0.18);

        // ── 1. Gesture booleans ─────────────────────────────────────────────
        const leftHandUp   = lw.y < (ls.y - this.UP_MARGIN);
        const rightHandUp  = rw.y < (rs.y - this.UP_MARGIN);
        const bothHandsUp  = leftHandUp && rightHandUp;

        const leftHandHigh  = lw.y < highY;
        const rightHandHigh = rw.y < highY;
        const bothHandsHigh = leftHandHigh && rightHandHigh;   // → JUMP

        const bothHandsLow = (lw.y > (ls.y + this.LOW_MARGIN)) &&
                             (rw.y > (rs.y + this.LOW_MARGIN));

        // Arm extended out to the side, roughly horizontal, and NOT raised up.
        const rightArmOut = Math.abs(rw.x - rs.x) > shoulderSpan * this.SIDE_FACTOR &&
                            Math.abs(rw.y - rs.y) < this.HORIZ_BAND &&
                            !rightHandUp && !bothHandsLow;
        const leftArmOut  = Math.abs(lw.x - ls.x) > shoulderSpan * this.SIDE_FACTOR &&
                            Math.abs(lw.y - ls.y) < this.HORIZ_BAND &&
                            !leftHandUp && !bothHandsLow;

        const A = ActionType;

        // ── 2. JUMP (one-shot, rising edge of "both hands high") ────────────
        if (bothHandsHigh) {
            if (!this._jumpArmed) {
                this.actionCtrl.activate(A.JUMP);   // consumed by ActionController
                this._jumpArmed = true;
            }
        } else {
            this._jumpArmed = false;
        }

        // ── 3. SPRINT (both hands up, but not the higher "jump" pose) ───────
        if (bothHandsUp && !bothHandsHigh) this.actionCtrl.activate(A.SPRINT);
        else                               this.actionCtrl.deactivate(A.SPRINT);

        // ── 4. FORWARD / BACKWARD ───────────────────────────────────────────
        if (bothHandsUp || rightHandUp) {
            this.actionCtrl.activate(A.MOVE_FORWARD);
            this.actionCtrl.deactivate(A.MOVE_BACKWARD);
        } else if (leftHandUp) {
            this.actionCtrl.activate(A.MOVE_BACKWARD);
            this.actionCtrl.deactivate(A.MOVE_FORWARD);
        } else {
            this.actionCtrl.deactivate(A.MOVE_FORWARD);
            this.actionCtrl.deactivate(A.MOVE_BACKWARD);
        }

        // ── 5. TURN left / right (arm extended to the side) ─────────────────
        if (rightArmOut) this.actionCtrl.activate(A.TURN_RIGHT);
        else             this.actionCtrl.deactivate(A.TURN_RIGHT);

        if (leftArmOut)  this.actionCtrl.activate(A.TURN_LEFT);
        else             this.actionCtrl.deactivate(A.TURN_LEFT);

        // ── 6. CROUCH (both hands low, neither raised) ──────────────────────
        const shouldCrouch = bothHandsLow && !leftHandUp && !rightHandUp;
        if (shouldCrouch) this.actionCtrl.activate(A.CROUCH);
        else              this.actionCtrl.deactivate(A.CROUCH);
    }

    /** Resets all gesture-driven actions. Call when gestures stop / no landmarks. */
    _cleanup() {
        const A = ActionType;
        const ACTIONS = [
            A.MOVE_FORWARD, A.MOVE_BACKWARD, A.SPRINT, A.CROUCH,
            A.TURN_LEFT, A.TURN_RIGHT, A.JUMP,
        ];
        for (const action of ACTIONS) this.actionCtrl.deactivate(action);
        this._jumpArmed = false;
    }
}
