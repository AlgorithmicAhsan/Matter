/**
 * PoseActionMapper.js
 * Decouples MediaPipe landmarks from the ActionController.
 * Detects gestures and translates them into predefined actions.
 */

class PoseActionMapper {
    /**
     * @param {ActionController} actionCtrl 
     */
    constructor(actionCtrl) {
        this.actionCtrl = actionCtrl;
        
        // Landmark indices (MediaPipe Pose)
        this.LANDMARKS = {
            LEFT_SHOULDER:  11,
            RIGHT_SHOULDER: 12,
            LEFT_WRIST:     15,
            RIGHT_WRIST:    16
        };
    }

    /**
     * Main entry point called on every frame with pose data.
     */
    update(landmarks, poseMode) {
        if (!poseMode || !landmarks) {
            this._cleanup();
            return;
        }

        const ls = landmarks[this.LANDMARKS.LEFT_SHOULDER];
        const rs = landmarks[this.LANDMARKS.RIGHT_SHOULDER];
        const lw = landmarks[this.LANDMARKS.LEFT_WRIST];
        const rw = landmarks[this.LANDMARKS.RIGHT_WRIST];

        // Safety check
        if (!ls || !rs || !lw || !rw) return;

        // 1. Gesture Detection Logic
        // ─────────────────────────────────────────────────────────────────────
        
        const leftHandUp    = lw.y < (ls.y - 0.12);
        const rightHandUp   = rw.y < (rs.y - 0.12);
        const bothHandsUp   = leftHandUp && rightHandUp;
        
        // Crouch detection: hands down far below shoulder level (at or below hips)
        const bothHandsLow  = (lw.y > (ls.y + 0.35)) && (rw.y > (rs.y + 0.35));

        // 2. Action Mapping (Source-based)
        // ─────────────────────────────────────────────────────────────────────

        // SPRINT
        if (bothHandsUp) this.actionCtrl.activate(ActionType.SPRINT, 'pose');
        else             this.actionCtrl.deactivate(ActionType.SPRINT, 'pose');

        // MOVEMENT (Forward / Backward)
        if (bothHandsUp || rightHandUp) {
            this.actionCtrl.activate(ActionType.MOVE_FORWARD, 'pose');
            this.actionCtrl.deactivate(ActionType.MOVE_BACKWARD, 'pose');
        } 
        else if (leftHandUp) {
            this.actionCtrl.activate(ActionType.MOVE_BACKWARD, 'pose');
            this.actionCtrl.deactivate(ActionType.MOVE_FORWARD, 'pose');
        } 
        else {
            this.actionCtrl.deactivate(ActionType.MOVE_FORWARD, 'pose');
            this.actionCtrl.deactivate(ActionType.MOVE_BACKWARD, 'pose');
        }

        // CROUCH
        const shouldCrouch = bothHandsLow && !leftHandUp && !rightHandUp;
        if (shouldCrouch) this.actionCtrl.activate(ActionType.CROUCH, 'pose');
        else              this.actionCtrl.deactivate(ActionType.CROUCH, 'pose');
    }

    /**
     * Resets all gesture-driven actions.
     */
    _cleanup() {
        const ACTIONS = [
            ActionType.MOVE_FORWARD, ActionType.MOVE_BACKWARD,
            ActionType.SPRINT, ActionType.CROUCH
        ];
        for (const action of ACTIONS) {
            this.actionCtrl.deactivate(action, 'pose');
        }
    }
}
