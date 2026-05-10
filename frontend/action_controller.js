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
        if (this.has(ActionType.MOVE_FORWARD))  localZ += speed;
        if (this.has(ActionType.MOVE_BACKWARD)) localZ -= cfg.backwardSpeed * (this.isCrouching ? cfg.crouchSpeedMult : 1);

        // Strafe
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
            if (this.position.y <= cfg.groundY) {
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
        if (!this.isGrounded)              this.locomotionState = 'jump';
        else if (this.isCrouching)         this.locomotionState = 'crouch';  // standing OR moving
        else if (this.isSprinting && moving) this.locomotionState = 'run';
        else if (moving || movingBack)     this.locomotionState = 'walk';
        else                               this.locomotionState = 'idle';

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
        const key = e.key.toLowerCase();
        this.pressedKeys.delete(key);

        const action = this.bindings[key];
        if (action) {
            this.ctrl.deactivate(action);
            if (this.onKeyChange) this.onKeyChange(key, false);
        }
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
