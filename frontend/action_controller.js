/**
 * ActionController.js
 * JS port of backend/actions.py — handles WASD / Space / Shift / Ctrl keys
 * and drives the avatar's world position + Y-rotation in Three.js.
 *
 * Architecture:
 *   KeyboardController  →  ActionController  →  AvatarController (position/rotation)
 *
 * The AvatarController then calls avatar.setWorldTransform(pos, rot) each frame.
 */

// ── ActionType enum ─────────────────────────────────────────────────────────
const ActionType = Object.freeze({
    IDLE:           'idle',
    MOVE_FORWARD:   'move_forward',
    MOVE_BACKWARD:  'move_backward',
    STRAFE_LEFT:    'strafe_left',
    STRAFE_RIGHT:   'strafe_right',
    TURN_LEFT:      'turn_left',
    TURN_RIGHT:     'turn_right',
    JUMP:           'jump',
    CROUCH:         'crouch',
    SPRINT:         'sprint',
});

// ── ActionConfig ─────────────────────────────────────────────────────────────
class ActionConfig {
    constructor(overrides = {}) {
        this.walkSpeed       = overrides.walkSpeed       ?? 80.0;  // units/s (avatar = 150u tall)
        this.sprintSpeed     = overrides.sprintSpeed     ?? 180.0;
        this.strafeSpeed     = overrides.strafeSpeed     ?? 60.0;
        this.backwardSpeed   = overrides.backwardSpeed   ?? 55.0;
        this.turnSpeed       = overrides.turnSpeed       ?? 160.0; // deg/s
        this.jumpVelocity    = overrides.jumpVelocity    ?? 350.0;
        this.jumpCooldown    = overrides.jumpCooldown    ?? 0.5;   // s
        this.gravity         = overrides.gravity         ?? -800.0;
        this.crouchSpeedMult = overrides.crouchSpeedMult ?? 0.45;
        this.sprintMult      = overrides.sprintMult      ?? 2.2;
        this.groundY         = overrides.groundY         ?? 0.0;
    }
}

// ── ActionController ─────────────────────────────────────────────────────────
class ActionController {
    constructor(config = new ActionConfig()) {
        this.config  = config;

        // World state
        this.position     = new THREE.Vector3(0, 0, 0);
        this.rotation     = 0;      // Y-axis angle in radians
        this.verticalVel  = 0;      // for jump physics
        this.isGrounded   = true;
        this.isCrouching  = false;
        this.isSprinting  = false;
        this.jumpCooldown = 0;

        // Active actions
        this.active = new Set();

        // Locomotion state for animation blending: 'idle' | 'walk' | 'run' | 'crouch' | 'jump'
        this.locomotionState = 'idle';

        this.poseMode       = false;   // when true, keyboard is ignored
        this._poseWalkSpeed = 0;       // units/s set by pose transform
        this._poseHipX      = undefined; // hip X position from pose
    }

    // ── Input ────────────────────────────────────────────────────────────────
    activate(action)   { this.active.add(action); }
    deactivate(action) { this.active.delete(action); }
    has(action)        { return this.active.has(action); }

    // ── Update ───────────────────────────────────────────────────────────────
    /**
     * @param {number} dt  delta time in seconds
     * @returns {{ position: THREE.Vector3, rotationRad: number, locomotionState: string }}
     */
    update(dt) {
        const cfg = this.config;

        // Jump cooldown
        if (this.jumpCooldown > 0) this.jumpCooldown = Math.max(0, this.jumpCooldown - dt);

        // Sprint / crouch modifiers
        this.isSprinting = this.has(ActionType.SPRINT) && !this.isCrouching;
        this.isCrouching = this.has(ActionType.CROUCH) && !this.has(ActionType.SPRINT);

        // Base horizontal speed
        let speed = cfg.walkSpeed;
        if (this.isSprinting) speed = cfg.sprintSpeed;
        if (this.isCrouching) speed *= cfg.crouchSpeedMult;

        // Forward / backward
        let localZ = 0;
        // In pose mode, use the detected walk speed instead of fixed walkSpeed
        const fwdSpeed = (this.poseMode && this._poseWalkSpeed > 0)
            ? this._poseWalkSpeed : speed;
        const bkdSpeed = (this.poseMode && this._poseWalkSpeed > 0)
            ? this._poseWalkSpeed * 0.7 : cfg.backwardSpeed * (this.isCrouching ? cfg.crouchSpeedMult : 1);

        if (this.has(ActionType.MOVE_FORWARD))  localZ += fwdSpeed;
        if (this.has(ActionType.MOVE_BACKWARD)) localZ -= bkdSpeed;

        // Strafe — always apply; in pose mode driven by walk_direction_x from backend
        let localX = 0;
        if (this.has(ActionType.STRAFE_RIGHT)) localX += cfg.strafeSpeed;
        if (this.has(ActionType.STRAFE_LEFT))  localX -= cfg.strafeSpeed;

        // Rotation
        let rotDelta = 0;
        if (this.has(ActionType.TURN_LEFT))  rotDelta += THREE.MathUtils.degToRad(cfg.turnSpeed) * dt;
        if (this.has(ActionType.TURN_RIGHT)) rotDelta -= THREE.MathUtils.degToRad(cfg.turnSpeed) * dt;
        this.rotation += rotDelta;

        // Jump
        if (this.has(ActionType.JUMP) && this.isGrounded && this.jumpCooldown <= 0) {
            this.verticalVel  = cfg.jumpVelocity;
            this.isGrounded   = false;
            this.jumpCooldown = cfg.jumpCooldown;
            this.deactivate(ActionType.JUMP); // one-shot
        }

        // Vertical physics
        if (!this.isGrounded) {
            this.verticalVel    += cfg.gravity * dt;
            this.position.y     += this.verticalVel * dt;
            
            // Only land if we are falling downwards!
            // (If we jump while crouched, position.y might be < 0 for a few frames)
            if (this.verticalVel <= 0 && this.position.y <= cfg.groundY) {
                this.position.y  = cfg.groundY;
                this.verticalVel = 0;
                this.isGrounded  = true;
            }
        }

        // Convert local velocity → world space (rotate around Y)
        const sin = Math.sin(this.rotation);
        const cos = Math.cos(this.rotation);
        const worldX = localX * cos + localZ * sin;
        const worldZ = -localX * sin + localZ * cos;

        this.position.x += worldX * dt;
        this.position.z += worldZ * dt;

        // Determine locomotion state — crouch triggers even when standing still
        const moving     = localX !== 0 || localZ > 0;
        const movingBack = localZ < 0;
        const poseStrafe = this.poseMode && (this.has(ActionType.STRAFE_LEFT) || this.has(ActionType.STRAFE_RIGHT));
        if (!this.isGrounded)                    this.locomotionState = 'jump';
        else if (this.isCrouching)               this.locomotionState = 'crouch';
        else if (this.isSprinting && moving)     this.locomotionState = 'run';
        else if (moving || movingBack || poseStrafe) this.locomotionState = 'walk';
        else                                     this.locomotionState = 'idle';

        // MUST be here — after all integration
        if (this.poseMode && this._poseHipX !== undefined) {
            this.position.x = this._poseHipX;
            // Only drop the Y position if grounded (to avoid interfering with physics jumps)
            if (this.isGrounded && this._poseHipY !== undefined) {
                this.position.y = cfg.groundY + this._poseHipY;
            }
        }

        return {
            position:      this.position.clone(),
            rotationRad:   this.rotation,
            locomotionState: this.locomotionState,
        };
    }

    getStateDict() {
        return {
            position:       [this.position.x, this.position.y, this.position.z],
            rotationDeg:    THREE.MathUtils.radToDeg(this.rotation),
            active:         [...this.active],
            isGrounded:     this.isGrounded,
            isCrouching:    this.isCrouching,
            isSprinting:    this.isSprinting,
            locomotionState: this.locomotionState,
        };
    }

    /**
     * Called by the WebSocket handler with the active_actions list from the backend.
     * Replaces all pose-owned actions; keyboard actions are unaffected.
     * @param {string[]} activeActions  e.g. ["move_forward", "sprint"]
     */
    setPoseActions(activeActions) {
        const POSE_OWNED = new Set([
            ActionType.MOVE_FORWARD, ActionType.MOVE_BACKWARD,
            ActionType.STRAFE_LEFT,  ActionType.STRAFE_RIGHT,
            ActionType.TURN_LEFT,    ActionType.TURN_RIGHT,
            ActionType.CROUCH,       ActionType.SPRINT,
            ActionType.JUMP,
        ]);

        const incoming = new Set(activeActions);

        for (const action of POSE_OWNED) {
            if (incoming.has(action)) {
                this.activate(action);
            } else {
                this.deactivate(action);
            }
        }
    }

    /**
     * Called each frame when pose_mode is ON.
     * Directly applies rotation delta, sets X from hip mirroring,
     * and activates/deactivates walking based on anti-phase ankle detection.
     * @param {object} transform  payload from backend compute_pose_transform()
     */
    applyPoseTransform(transform) {
        if (!transform) return;

        // Rotation: accumulate delta directly (no ambiguity, tracks exact movement)
        this.rotation += transform.rotation_delta;

        // X and Y position: store hip positions to apply after world space calculation
        this._poseHipX = transform.hip_x_world;
        this._poseHipY = transform.hip_y_world;

        // Forward/backward walking from anti-phase ankle oscillation
        this._poseWalkSpeed = transform.is_walking ? transform.walk_speed : 0;

        if (transform.is_walking) {
            const dirZ = transform.walk_direction_z ?? transform.walk_direction ?? 1;
            if (dirZ >= 0) {
                this.active.add(ActionType.MOVE_FORWARD);
                this.active.delete(ActionType.MOVE_BACKWARD);
            } else {
                this.active.add(ActionType.MOVE_BACKWARD);
                this.active.delete(ActionType.MOVE_FORWARD);
            }
        } else {
            this.active.delete(ActionType.MOVE_FORWARD);
            this.active.delete(ActionType.MOVE_BACKWARD);
        }

        // Strafe: lateral hip velocity detected by backend
        const dirX = transform.walk_direction_x ?? 0;
        if (dirX > 0) {
            this.active.add(ActionType.STRAFE_RIGHT);
            this.active.delete(ActionType.STRAFE_LEFT);
        } else if (dirX < 0) {
            this.active.add(ActionType.STRAFE_LEFT);
            this.active.delete(ActionType.STRAFE_RIGHT);
        } else {
            this.active.delete(ActionType.STRAFE_LEFT);
            this.active.delete(ActionType.STRAFE_RIGHT);
        }
    }
}

// ── KeyboardController ────────────────────────────────────────────────────────
class KeyboardController {
    /**
     * @param {ActionController} actionController
     * @param {function(string, boolean): void} onKeyChange  optional callback(key, isDown)
     */
    constructor(actionController, onKeyChange = null) {
        this.ctrl           = actionController;
        this.onKeyChange    = onKeyChange;
        this.pressedKeys    = new Set();

        // Standard FPS bindings — arrow keys also supported
        this.bindings = {
            'w':          ActionType.MOVE_FORWARD,
            'arrowup':    ActionType.MOVE_FORWARD,
            's':          ActionType.MOVE_BACKWARD,
            'arrowdown':  ActionType.MOVE_BACKWARD,
            // A/D turn the character to face that direction, then W moves
            'a':          ActionType.TURN_LEFT,
            'arrowleft':  ActionType.TURN_LEFT,
            'd':          ActionType.TURN_RIGHT,
            'arrowright': ActionType.TURN_RIGHT,
            'q':          ActionType.STRAFE_LEFT,   // Q/E for strafing if needed
            'e':          ActionType.STRAFE_RIGHT,
            ' ':          ActionType.JUMP,
            'shift':      ActionType.SPRINT,
            'control':    ActionType.CROUCH,
        };

        this._onKeyDown = this._onKeyDown.bind(this);
        this._onKeyUp   = this._onKeyUp.bind(this);
        window.addEventListener('keydown', this._onKeyDown);
        window.addEventListener('keyup',   this._onKeyUp);
    }

    _onKeyDown(e) {
        if (this.ctrl.poseMode) return;   // ← Skip keyboard input in pose mode
        // Prevent browser from stealing game keys (incl. Ctrl which triggers browser shortcuts)
        if ([' ', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Control'].includes(e.key)) {
            e.preventDefault();
        }

        const key = e.key.toLowerCase();
        if (this.pressedKeys.has(key)) return; // suppress key-repeat
        this.pressedKeys.add(key);

        const action = this.bindings[key];
        if (action) {
            this.ctrl.activate(action);
            if (this.onKeyChange) this.onKeyChange(key, true);
        }
    }

    _onKeyUp(e) {
        // NOTE: do NOT guard keyUp with poseMode — always let key releases through
        // to prevent stuck actions when switching modes while holding a key
        const key = e.key.toLowerCase();
        this.pressedKeys.delete(key);

        const action = this.bindings[key];
        if (action) {
            this.ctrl.deactivate(action);
            if (this.onKeyChange) this.onKeyChange(key, false);
        }
    }

    clearKeys() {
        for (const key of this.pressedKeys) {
            const action = this.bindings[key];
            if (action) this.ctrl.deactivate(action);
        }
        this.pressedKeys.clear();
    }

    destroy() {
        window.removeEventListener('keydown', this._onKeyDown);
        window.removeEventListener('keyup',   this._onKeyUp);
    }

    /** Human-readable list of currently held keys (for HUD). */
    getHeldKeys() {
        return [...this.pressedKeys];
    }
}

// ── HUD Overlay ────────────────────────────────────────────────────────────────
class KeyboardHUD {
    constructor() {
        this.el = document.getElementById('keyboard-hud');
        if (!this.el) return;
        this._keys = {};
    }

    update(heldKeys) {
        if (!this.el) return;

        const keyLabels = {
            'w': 'W', 's': 'S', 'a': 'A', 'd': 'D',
            'q': 'Q', 'e': 'E', ' ': 'SPC',
            'shift': '⇧', 'control': 'CTRL',
            'arrowup': '↑', 'arrowdown': '↓',
            'arrowleft': '←', 'arrowright': '→',
        };

        const held = new Set(heldKeys);
        this.el.querySelectorAll('[data-key]').forEach(btn => {
            const k = btn.getAttribute('data-key');
            btn.classList.toggle('active', held.has(k));
        });
    }
}

/**
 * PoseModeController
 * Manages the pose-mode toggle button and calibration progress bar in the UI.
 * Uses window.posiSimApp.ws directly so it always talks to the live socket,
 * even if the connection was recreated after a disconnect.
 */
class PoseModeController {
    constructor(wsSend) {
        // wsSend kept only for legacy compat — internal sends go through _wsSend()
        this._legacySend = wsSend;
        this._poseMode   = false;
        this._calibrated = false;
        this._timer      = null;    // setTimeout handle while countdown runs
        this.isPending   = false;   // true while countdown is ticking

        this._btn      = document.getElementById('pose-mode-btn');
        this._bar      = document.getElementById('calib-bar-fill');
        this._barWrap  = document.getElementById('calib-bar-wrap');
        this._label    = document.getElementById('calib-label');

        if (this._btn) {
            this._btn.addEventListener('click', () => this._toggle());
        }
    }

    // Always grabs the current live WebSocket — no stale-closure risk.
    _wsSend(msg) {
        const app = window.posiSimApp;
        if (app && app.ws && app.ws.readyState === WebSocket.OPEN) {
            app.ws.send(JSON.stringify(msg));
            return true;
        }
        console.warn('[PoseModeCtrl] WebSocket not open — could not send:', msg.action);
        return false;
    }

    _toggle() {
        if (!this._poseMode) {
            // ── Turning ON ──────────────────────────────────────────────────
            // Flip local state immediately so the button shows "Starting…"
            this._poseMode = true;
            this.isPending = true;
            this._updateBtn();

            // Show countdown overlay
            window.posiSimApp?._showCountdown('Stand still — calibration starts in…');

            // After the countdown, tell the backend to start
            this._timer = setTimeout(() => {
                this._timer    = null;
                this.isPending = false;
                console.log('[PoseModeCtrl] Countdown finished — sending set_pose_mode ON');
                const sent = this._wsSend({ action: 'set_pose_mode', enabled: true });
                if (!sent) {
                    // WS was closed — revert so the user can try again
                    this._poseMode = false;
                }
                this._updateBtn();
            }, 5000);

        } else {
            // ── Turning OFF ─────────────────────────────────────────────────
            if (this._timer !== null) {
                clearTimeout(this._timer);
                this._timer    = null;
                this.isPending = false;
            }
            this._poseMode = false;
            this._wsSend({ action: 'set_pose_mode', enabled: false });
            this._updateBtn();
        }
    }

    /** Called each frame when a WS payload arrives. */
    onFrame(poseMode, calibrated, calibProgress) {
        // While countdown is running the backend still reports pose_mode=false — ignore it.
        if (!this.isPending) {
            this._poseMode = poseMode;
        }
        this._calibrated = calibrated;
        this._updateBtn();
        this._updateCalibBar(calibrated, calibProgress);
    }

    _updateBtn() {
        if (!this._btn) return;
        if (this._poseMode) {
            let label;
            if (this.isPending)          label = '⏳ Starting…';
            else if (this._calibrated)   label = '🎥 Pose ON';
            else                         label = '⏳ Calibrating…';
            this._btn.textContent = label;
            this._btn.classList.add('active');
        } else {
            this._btn.textContent = '🎥 Pose OFF';
            this._btn.classList.remove('active');
        }
    }

    _updateCalibBar(calibrated, progress) {
        if (!this._barWrap) return;
        if (calibrated || !this._poseMode) {
            this._barWrap.style.display = 'none';
            return;
        }
        this._barWrap.style.display = 'block';
        if (this._bar)   this._bar.style.width = (progress * 100).toFixed(0) + '%';
        if (this._label) this._label.textContent =
            `Stand still — calibrating ${(progress * 100).toFixed(0)}%`;
    }

    /** Immediately turn pose off — used when switching app modes. */
    _forceOff() {
        if (this._timer !== null) { clearTimeout(this._timer); this._timer = null; }
        this.isPending   = false;
        this._poseMode   = false;
        this._wsSend({ action: 'set_pose_mode', enabled: false });
        this._updateBtn();
    }

    resetCalibration() {
        this._wsSend({ action: 'reset_calibration' });
    }
}
