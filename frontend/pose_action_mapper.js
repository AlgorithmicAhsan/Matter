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
            RIGHT_WRIST:    16,
            LEFT_HIP:       23,
            RIGHT_HIP:      24
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
        const lh = landmarks[this.LANDMARKS.LEFT_HIP];
        const rh = landmarks[this.LANDMARKS.RIGHT_HIP];

        // Safety check
        if (!ls || !rs || !lw || !rw || !lh || !rh) return;

        // 1. Gesture Detection Logic
        // ─────────────────────────────────────────────────────────────────────
        
        // Vertical thresholds
        const leftHandUp    = lw.y < (ls.y - 0.12); // Wrist significantly above shoulder
        const rightHandUp   = rw.y < (rs.y - 0.12);
        const bothHandsUp   = leftHandUp && rightHandUp;
        
        // Crouch detection: hands down near or below hip level
        const bothHandsLow  = (lw.y > (lh.y - 0.05)) && (rw.y > (rh.y - 0.05));

        // 2. Action Mapping (Priority Logic)
        // ─────────────────────────────────────────────────────────────────────

        // SPRINT
        if (bothHandsUp) {
            this.actionCtrl.activate(ActionType.SPRINT);
        } else {
            this.actionCtrl.deactivate(ActionType.SPRINT);
        }

        // MOVEMENT (Forward / Backward)
        if (bothHandsUp || rightHandUp) {
            // Both up OR just right up triggers forward (Both up = Sprint Forward)
            this.actionCtrl.activate(ActionType.MOVE_FORWARD);
            this.actionCtrl.deactivate(ActionType.MOVE_BACKWARD);
        } 
        else if (leftHandUp) {
            // Only left up triggers backward
            this.actionCtrl.activate(ActionType.MOVE_BACKWARD);
            this.actionCtrl.deactivate(ActionType.MOVE_FORWARD);
        } 
        else {
            // Neither raised
            this.actionCtrl.deactivate(ActionType.MOVE_FORWARD);
            this.actionCtrl.deactivate(ActionType.MOVE_BACKWARD);
        }

        // CROUCH
        // Only crouch if hands are low and we aren't trying to walk/run
        if (bothHandsLow && !leftHandUp && !rightHandUp) {
            this.actionCtrl.activate(ActionType.CROUCH);
        } else {
            this.actionCtrl.deactivate(ActionType.CROUCH);
        }
    }

    /**
     * Resets all gesture-driven actions.
     */
    _cleanup() {
        this.actionCtrl.deactivate(ActionType.MOVE_FORWARD);
        this.actionCtrl.deactivate(ActionType.MOVE_BACKWARD);
        this.actionCtrl.deactivate(ActionType.SPRINT);
        this.actionCtrl.deactivate(ActionType.CROUCH);
    }
}
