/**
 * PoseActionMapper.js
 * --------------------
 * Decouples MediaPipe pose + hand landmarks from the ActionController. Detects a
 * small set of SIMPLE GESTURES and translates them into predefined locomotion
 * moves. Used ONLY by the Gesture tab — Live/Video modes never instantiate it.
 *
 * Gesture → Move map (this is the source of truth the UI "Moves Mapping" menu
 * mirrors — keep them in sync):
 *
 *   GESTURE                                  →  MOVE
 *   ──────────────────────────────────────     ───────────────────────
 *   Right hand raised above shoulder         →  Walk forward
 *   Left hand raised above shoulder          →  Crouch-walk forward (duck + move)
 *   BOTH hands raised above shoulders        →  Sprint forward
 *   BOTH hands raised high (above head)      →  Jump   (one-shot)
 *   Right thumbs-up (only the thumb)         →  Turn right
 *   Left thumbs-up  (only the thumb)         →  Turn left
 *
 * Turns use a thumbs-up: the thumb extended UP while the other four fingers are
 * curled. A fully open hand does NOT turn. A thumbs-up hand is also excluded
 * from the "hand raised" walk/crouch detection so the two never fight.
 *
 * Pose detection uses normalized MediaPipe coords: x∈[0,1] (→ right in image),
 * y∈[0,1] (→ down). Thresholds are relative to the user's own shoulders so they
 * hold regardless of distance from the camera.
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
        this.UP_MARGIN   = 0.12;  // wrist this far above shoulder = "hand up"

        // Turning is DISCRETE: one fixed step, then a cooldown (extra turn input
        // during the cooldown is discarded) so a held thumb doesn't spin wildly.
        this.TURN_STEP_DEG    = 35;    // degrees per accepted turn
        this.TURN_COOLDOWN_MS = 1000;  // min time between turns
        this._lastTurnTime    = 0;

        // Edge-trigger latch for the one-shot jump gesture.
        this._jumpArmed = false;
    }

    /**
     * Main entry point — call every frame with the latest tracking data.
     * @param {Array|null}  landmarks      MediaPipe pose landmarks (33)
     * @param {object|null} handLandmarks  { left:[21]|null, right:[21]|null }
     * @param {boolean}     poseMode       true only while gesture tracking is ON
     */
    update(landmarks, handLandmarks, poseMode) {
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

        const A = ActionType;

        // ── Thumbs-up (turn) — detected from the hand landmarks ─────────────
        const rightThumbUp = this._isThumbsUp(handLandmarks && handLandmarks.right);
        const leftThumbUp  = this._isThumbsUp(handLandmarks && handLandmarks.left);

        // ── Reference measurements ──────────────────────────────────────────
        // "High" line: above the head. Use the nose when available, else a line
        // a bit above the shoulders.
        const highY = nose ? nose.y : (Math.min(ls.y, rs.y) - 0.18);

        // ── Gesture booleans ────────────────────────────────────────────────
        // A thumbs-up hand is excluded from "hand raised" so turning never also
        // triggers walk/crouch.
        const rightHandUp = (rw.y < (rs.y - this.UP_MARGIN)) && !rightThumbUp;
        const leftHandUp  = (lw.y < (ls.y - this.UP_MARGIN)) && !leftThumbUp;
        const bothHandsUp = leftHandUp && rightHandUp;
        const leftOnly    = leftHandUp && !rightHandUp;
        const rightOnly   = rightHandUp && !leftHandUp;

        const bothHandsHigh = (lw.y < highY) && (rw.y < highY) &&
                              !leftThumbUp && !rightThumbUp;   // → JUMP

        // ── 1. JUMP (one-shot, rising edge of "both hands high") ────────────
        if (bothHandsHigh) {
            if (!this._jumpArmed) {
                this.actionCtrl.activate(A.JUMP);   // consumed by ActionController
                this._jumpArmed = true;
            }
        } else {
            this._jumpArmed = false;
        }

        // ── 2. SPRINT (both hands up, but not the higher "jump" pose) ───────
        if (bothHandsUp && !bothHandsHigh) this.actionCtrl.activate(A.SPRINT);
        else                               this.actionCtrl.deactivate(A.SPRINT);

        // ── 3. CROUCH-WALK (left only) / FORWARD (right or both) ────────────
        // Left hand raised → duck AND keep moving forward so you can advance
        // under the low barrier. Right hand / both hands → walk forward upright.
        if (leftOnly) {
            this.actionCtrl.activate(A.CROUCH);
            this.actionCtrl.activate(A.MOVE_FORWARD);
        } else {
            this.actionCtrl.deactivate(A.CROUCH);
            if (bothHandsUp || rightOnly) this.actionCtrl.activate(A.MOVE_FORWARD);
            else                          this.actionCtrl.deactivate(A.MOVE_FORWARD);
        }
        // Walk-backward is not gesture-mapped.
        this.actionCtrl.deactivate(A.MOVE_BACKWARD);

        // ── 4. TURN (thumbs-up) — discrete: one step, then a 1s cooldown ────
        // Right thumb and left thumb apply opposite rotation steps; the sign is
        // chosen so the on-screen turn matches the menu (right→right, left→left).
        // Only one thumb counts; both raised = no turn.
        const now = (typeof performance !== 'undefined') ? performance.now() : Date.now();
        const oneThumb = (rightThumbUp ? 1 : 0) + (leftThumbUp ? 1 : 0) === 1;
        if (oneThumb && (now - this._lastTurnTime) >= this.TURN_COOLDOWN_MS) {
            const dir = rightThumbUp ? +1 : -1;   // sign flipped from the old mapping
            this.actionCtrl.applyTurnStep(dir * this.TURN_STEP_DEG);
            this._lastTurnTime = now;
        }
        // Gesture turning is discrete — never hold the continuous turn actions.
        this.actionCtrl.deactivate(A.TURN_LEFT);
        this.actionCtrl.deactivate(A.TURN_RIGHT);
    }

    /**
     * Thumbs-up detection for ONE hand (21 MediaPipe hand landmarks).
     * Requires: thumb extended + pointing up, AND the other four fingers curled.
     * A fully open hand (all fingers up) returns false.
     */
    _isThumbsUp(hand) {
        if (!hand || hand.length < 21) return false;

        const w  = hand[0];                                   // wrist
        const d2 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y); // 2D dist (z is noisy)

        // Thumb extended: tip (4) is farther from the wrist than the IP joint (3).
        const thumbExtended = d2(hand[4], w) > d2(hand[3], w) * 1.05;
        // Thumb pointing up: tip clearly above the wrist and the index knuckle.
        const thumbUp = (hand[4].y < w.y - 0.04) && (hand[4].y < hand[5].y);

        // The other four fingers must be CURLED — tip closer to the wrist than
        // its PIP joint. (Index 8/6, Middle 12/10, Ring 16/14, Pinky 20/18.)
        const folded = (tip, pip) => d2(hand[tip], w) < d2(hand[pip], w);
        const fingersFolded =
            folded(8, 6) && folded(12, 10) && folded(16, 14) && folded(20, 18);

        return thumbExtended && thumbUp && fingersFolded;
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
