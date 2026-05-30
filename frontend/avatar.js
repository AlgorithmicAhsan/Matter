class AvatarRenderer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);

        this.camera = new THREE.PerspectiveCamera(
            50,
            this.container.clientWidth / this.container.clientHeight,
            0.1, 2000
        );
        this.camera.position.set(0, 160, 320);
        this.camera.lookAt(0, 80, 0);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        // Prevent renderer.setSize from fixing the canvas to pixel dimensions via
        // inline style — let CSS (width:100%; height:100%) size the display instead.
        // The render buffer resolution is set correctly above; only display differs.
        this.renderer.domElement.style.width  = '';
        this.renderer.domElement.style.height = '';
        this.renderer.shadowMap.enabled = true;
        this.renderer.outputEncoding = THREE.sRGBEncoding; // critical for GLTF materials
        this.renderer.toneMapping = THREE.NoToneMapping;
        this.container.appendChild(this.renderer.domElement);

        this._setupLighting();
        this._buildEnvironment();

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 80, 0);
        this.controls.update();

        this.model       = null;
        this.bones       = {};
        this.restPose    = {};
        this._boneToChildDir = {}; // leg bone "toward child" directions, filled at load time
        this._modelYOffset = 0; // FIX: store load-time Y offset separately

        this.worldPosition   = new THREE.Vector3();
        this.worldRotation   = 0;
        this.locomotionState = 'idle';
        this._animClock      = 0;
        this._lastTime       = performance.now();
        this._landmarkTimestamp = 0;
        this._smoothLm = null;
        
        this.boneMap = {
            leftArm:       'mixamorigLeftArm',
            rightArm:      'mixamorigRightArm',
            leftForeArm:   'mixamorigLeftForeArm',
            rightForeArm:  'mixamorigRightForeArm',
            leftUpLeg:     'mixamorigLeftUpLeg',
            rightUpLeg:    'mixamorigRightUpLeg',
            leftLeg:       'mixamorigLeftLeg',
            rightLeg:      'mixamorigRightLeg',
            leftFoot:      'mixamorigLeftFoot',
            rightFoot:     'mixamorigRightFoot',
            spine:         'mixamorigSpine',
            spine1:        'mixamorigSpine1',
            spine2:        'mixamorigSpine2',
            hips:          'mixamorigHips',
            // Clavicles — lift when arm is raised
            leftClavicle:  'mixamorigLeftShoulder',
            rightClavicle: 'mixamorigRightShoulder',
            // Head — driven by face landmarks (nod / turn / tilt)
            neck:          'mixamorigNeck',
            head:          'mixamorigHead',
        };

        // ── Head tracking (face nod/turn/tilt → Head bone) ──────────────────
        // Smoothed Euler angles applied on top of the head's rest pose.
        // Signs flip direction (mirror feel); gains scale; clamps cap range.
        this._headAngles = { pitch: 0, yaw: 0, roll: 0 };
        this._headSign   = { pitch: 1, yaw: -1, roll: -1 };
        this._headGain   = { pitch: 1.0, yaw: 1.0, roll: 1.0 };
        this._headClamp  = { pitch: 0.6, yaw: 0.7, roll: 0.5 }; // radians
        this._headAlpha  = 0.4;  // bone slerp responsiveness

        // ── Adaptive camera framing ─────────────────────────────────────────
        // Zoom/pan the camera to fit only the body region currently visible in
        // the live feed (face-only → head close-up, full body → wide shot).
        // Only the orbit DISTANCE + target are managed; the user's orbit angle
        // is preserved. Heights are in avatar units (model ≈ 150 tall, feet=0).
        this._autoFrame           = true;
        this._focusOffset         = 80;   // current target height above feet
        this._focusOffsetTarget   = 80;
        this._focusDistance       = 320;  // current orbit distance
        this._focusDistanceTarget = 320;

        // ── Finger mapping (hand landmarks → finger bone curl) ───────────────
        // Each finger: the 4 MediaPipe landmark indices + the 3 drivable Mixamo
        // bone suffixes (tip bone Finger4 isn't rotated).
        this._FINGERS = [
            { lms: [1,  2,  3,  4],  bones: ['Thumb1',  'Thumb2',  'Thumb3'  ] },
            { lms: [5,  6,  7,  8],  bones: ['Index1',  'Index2',  'Index3'  ] },
            { lms: [9,  10, 11, 12], bones: ['Middle1', 'Middle2', 'Middle3' ] },
            { lms: [13, 14, 15, 16], bones: ['Ring1',   'Ring2',   'Ring3'   ] },
            { lms: [17, 18, 19, 20], bones: ['Pinky1',  'Pinky2',  'Pinky3'  ] },
        ];
        this._fingerCurl     = {};   // per-bone EMA-smoothed curl angle
        this._fingerBendAxis = new THREE.Vector3(0, 0, 1); // 4-finger local flex axis
        this._fingerBendSign = { left: 1, right: 1 };      // flip per hand if curling backwards
        // Thumb sits in a different plane → its own local flex axis + sign.
        this._thumbBendAxis  = new THREE.Vector3(0, 1, 0);
        this._thumbBendSign  = { left: 1, right: 1 };
        this._fingerGain     = 1.25; // amplify curl slightly
        this._handSwap       = false; // swap detected↔avatar hands if mirrored

        // Leg swing sign — +1 or -1, auto-detected after model loads
        // (some Mixamo exports have inverted local X for leg bones)
        this._legSign = 1;

        // Pre-allocated scratch objects — reused every frame to avoid GC pressure.
        // rotQ is used in updatePose scope; q0/q1 are used inside _aimBone/_applyClavicle.
        this._scratch = {
            v0: new THREE.Vector3(), v1: new THREE.Vector3(),
            v2: new THREE.Vector3(), v3: new THREE.Vector3(),
            q0: new THREE.Quaternion(), q1: new THREE.Quaternion(),
            rotQ: new THREE.Quaternion(),
        };

        // Per-landmark EMA smoothing alphas — higher = more responsive, less smooth.
        // These smooth the raw landmark positions; bone slerp is a separate alpha.
        this._lmAlpha = new Array(33).fill(0.30);
        for (let i = 11; i <= 16; i++) this._lmAlpha[i] = 0.50; // arms — fast
        for (let i = 23; i <= 28; i++) this._lmAlpha[i] = 0.40; // legs
        for (let i = 29; i <= 32; i++) this._lmAlpha[i] = 0.35; // feet

        // When true, landmark EMA is bypassed (alpha=1) so pre-computed video
        // poses are applied instantly with no lag vs the skeleton overlay.
        this._bypassEMA = false;

        // 'direct' = live landmark-to-bone translation
        // 'gesture' = action/gesture-driven procedural animation
        this.translationMode = 'direct';

        this._loadModel();
        this._animate();
        window.addEventListener('resize', () => this._onWindowResize());
    }

    setTranslationMode(mode) {
        this.translationMode = mode;
        if (mode !== 'direct') {
            // Reset landmark timestamp so hasLivePose immediately goes false
            this._landmarkTimestamp = 0;
            this._smoothLm = null;
        }
        console.log('[AvatarRenderer] Translation mode set to:', mode);
    }

    /** Call with true during video playback, false for live webcam. */
    setBypassEMA(enabled) {
        this._bypassEMA = enabled;
        if (enabled) this._smoothLm = null; // clear stale EMA state
    }

    _setupLighting() {
        // Very bright ambient — GLTF PBR needs this
        this.scene.add(new THREE.AmbientLight(0xffffff, 4.0));

        const key = new THREE.DirectionalLight(0xffffff, 5.0);
        key.position.set(3, 10, 6);
        key.castShadow = true;
        key.shadow.mapSize.setScalar(1024);
        key.shadow.camera.near = 1;
        key.shadow.camera.far  = 600;
        key.shadow.camera.left = key.shadow.camera.bottom = -200;
        key.shadow.camera.right = key.shadow.camera.top   =  200;
        this.scene.add(key);

        const fill = new THREE.DirectionalLight(0xffeedd, 3.0);
        fill.position.set(-4, 6, -4);
        this.scene.add(fill);

        const top = new THREE.DirectionalLight(0xffffff, 2.0);
        top.position.set(0, 20, 0);
        this.scene.add(top);
    }

    _buildEnvironment() {
        const grid = new THREE.GridHelper(1000, 40, 0x334477, 0x223366);
        this.scene.add(grid);

        const floor = new THREE.Mesh(
            new THREE.PlaneGeometry(1000, 1000),
            new THREE.MeshLambertMaterial({ color: 0x111133 })
        );
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        this.scene.add(floor);

        // Keep references so the gesture tab can hide the plain grid/floor while
        // the obstacle course (which has its own floor) is shown, then restore it.
        this._defaultEnv = { grid, floor };
    }

    /**
     * Show or hide the default flat grid + floor. The gesture tab hides them
     * while the obstacle course is active so the two floors don't overlap.
     */
    setDefaultEnvVisible(visible) {
        if (!this._defaultEnv) return;
        this._defaultEnv.grid.visible  = visible;
        this._defaultEnv.floor.visible = visible;
    }

    /**
     * Force the adaptive-camera targets. The gesture tab uses this to lock a
     * steady follow distance so the avatar stays framed while it walks the
     * obstacle course (setAutoFrame is not called in gesture mode).
     */
    setFollowFraming(offset, distance) {
        this._focusOffsetTarget   = offset;
        this._focusDistanceTarget = distance;
    }

    _loadModel() {
        const loader = new THREE.GLTFLoader();
        loader.load('/models/avatar.glb', (gltf) => {
            this.model = gltf.scene;
            this.model.rotation.y = Math.PI; // face camera

            // Scale to ~150 units
            const box1  = new THREE.Box3().setFromObject(this.model);
            const size  = box1.getSize(new THREE.Vector3());
            const scale = 150 / size.y;
            this.model.scale.setScalar(scale);

            // Centre + ground the feet
            const box2   = new THREE.Box3().setFromObject(this.model);
            const centre = box2.getCenter(new THREE.Vector3());
            this.model.position.x = -centre.x;
            this.model.position.z = -centre.z;

            // FIX: store Y offset — setWorldTransform will ADD this, not override it
            this._modelYOffset    = -box2.min.y;
            this.model.position.y = this._modelYOffset;

            this.model.traverse((child) => {
                if (child.isMesh || child.isSkinnedMesh) {
                    child.castShadow    = true;
                    child.receiveShadow = true;
                    // FIX: override bounding sphere so model never disappears on zoom-out
                    if (child.geometry) {
                        child.geometry.boundingSphere = new THREE.Sphere(
                            new THREE.Vector3(0, 75, 0), 300
                        );
                    }
                    child.frustumCulled = false;
                }
                if (child.isBone) this.bones[child.name] = child;
            });

            this.scene.add(this.model);

            // Capture rest pose AFTER model is in scene
            Object.keys(this.bones).forEach(n => {
                this.restPose[n] = this.bones[n].quaternion.clone();
            });

            // Pre-compute the actual "toward child" direction for each leg bone,
            // expressed in that bone's parent-local space at rest.
            // childBone.position = child joint offset in CURRENT bone's local space.
            // Applying the current bone's rest quaternion rotates it into PARENT-local space.
            // This is the correct restDir for _aimBone — no assumption about which local
            // axis points toward the child (avoids antiparallel singularity for any rig).
            this._boneToChildDir = {};
            const legChildPairs = [
                [this.boneMap.leftUpLeg,  this.boneMap.leftLeg],
                [this.boneMap.leftLeg,    this.boneMap.leftFoot],
                [this.boneMap.rightUpLeg, this.boneMap.rightLeg],
                [this.boneMap.rightLeg,   this.boneMap.rightFoot],
            ];
            for (const [parentName, childName] of legChildPairs) {
                const childBone  = this.bones[childName];
                const parentRest = this.restPose[parentName];
                if (childBone && parentRest && childBone.position.lengthSq() > 0) {
                    this._boneToChildDir[parentName] =
                        childBone.position.clone().normalize().applyQuaternion(parentRest);
                }
            }
            console.log('Leg restDirs:', JSON.stringify(
                Object.fromEntries(Object.entries(this._boneToChildDir)
                    .map(([k,v]) => [k, [+v.x.toFixed(3), +v.y.toFixed(3), +v.z.toFixed(3)]]))
            ));

            const overlay = document.getElementById('loading-overlay');
            if (overlay) overlay.style.display = 'none';

            console.log('Avatar loaded. Bones:', Object.keys(this.bones).length);
        },
        (p) => {
            if (p.total > 0) {
                const el = document.getElementById('loading-pct');
                if (el) el.textContent = Math.round(p.loaded / p.total * 100) + '%';
            }
        },
        (e) => console.error('Avatar error:', e));
    }

    // Called every frame by ActionController
    setWorldTransform(position, rotationRad, locomotionState) {
        this.worldPosition.copy(position);
        this.worldRotation   = rotationRad;
        this.locomotionState = locomotionState || 'idle';

        if (this.model) {
            this.model.position.x = position.x;
            // FIX: always ADD the load-time Y offset on top of world Y
            this.model.position.y = position.y + this._modelYOffset;
            this.model.position.z = position.z;
            this.model.rotation.y = Math.PI + rotationRad;
        }

        // ── Adaptive framing ────────────────────────────────────────────────
        // Smooth the focus params toward their targets (set by setAutoFrame).
        this._focusOffset   += (this._focusOffsetTarget   - this._focusOffset)   * 0.08;
        this._focusDistance += (this._focusDistanceTarget - this._focusDistance) * 0.08;

        const desiredTarget = this._scratch.v2.set(
            position.x, position.y + this._focusOffset, position.z
        );
        this.controls.target.lerp(desiredTarget, 0.1);

        if (this._autoFrame) {
            // Keep the user's current orbit direction; only override the distance.
            const dir = this._scratch.v0.subVectors(this.camera.position, this.controls.target);
            const len = dir.length();
            if (len > 1e-3) {
                dir.multiplyScalar(this._focusDistance / len);
                const desiredPos = this._scratch.v1.copy(this.controls.target).add(dir);
                this.camera.position.lerp(desiredPos, 0.08);
            }
        }
    }

    // Pick a camera framing from the body region visible in the live feed.
    // landmarks = MediaPipe pose (33, with visibility); faceLandmarks = face mesh.
    setAutoFrame(landmarks, faceLandmarks) {
        if (!this._autoFrame) return;

        const vis     = (i) => landmarks && landmarks[i] && landmarks[i].visibility > 0.5;
        const hasFace = !!faceLandmarks || vis(0);

        let offset, distance;
        if (vis(25) || vis(26) || vis(27) || vis(28)) { offset = 80;  distance = 320; } // legs  → full body
        else if (vis(23) || vis(24))                  { offset = 100; distance = 235; } // hips  → upper body
        else if (vis(11) || vis(12))                  { offset = 125; distance = 150; } // shldr → head + shoulders
        else if (hasFace)                             { offset = 145; distance = 95;  } // face  → close-up
        else return;  // nothing detected — hold the last framing

        this._focusOffsetTarget   = offset;
        this._focusDistanceTarget = distance;
    }

    // Called when MediaPipe landmarks arrive.
    // worldLandmarks: optional pose_world_landmarks (metric 3D, hip-centred).
    // When provided, their Z replaces image-space Z for leg bones — giving proper
    // forward/backward knee depth during walking steps.
    updatePose(landmarks, worldLandmarks = null) {
        if (this.translationMode !== 'direct') return;
        if (!this.model || !landmarks || landmarks.length < 29) return;
        if (!Object.keys(this.bones).length) return;

        this._landmarkTimestamp = performance.now();

        // ── Per-region EMA smoothing ────────────────────────────────────────
        // _bypassEMA=true in video playback: landmarks are already clean and
        // EMA introduces lag that desynchronises avatar from the skeleton overlay.
        if (!this._smoothLm) {
            this._smoothLm = landmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z }));
        } else if (this._bypassEMA) {
            for (let i = 0; i < landmarks.length; i++) {
                this._smoothLm[i].x = landmarks[i].x;
                this._smoothLm[i].y = landmarks[i].y;
                this._smoothLm[i].z = landmarks[i].z;
            }
        } else {
            for (let i = 0; i < landmarks.length; i++) {
                const a = this._lmAlpha[i] ?? 0.12;
                this._smoothLm[i].x += a * (landmarks[i].x - this._smoothLm[i].x);
                this._smoothLm[i].y += a * (landmarks[i].y - this._smoothLm[i].y);
                this._smoothLm[i].z += a * (landmarks[i].z - this._smoothLm[i].z);
            }
        }

        const j = this._smoothLm;

        // ── Body-relative coordinate frame ──────────────────────────────────
        // Center all positions on the hip midpoint and scale by torso length.
        // This keeps the mapping stable regardless of where the user stands in
        // the camera frame or how far they are from the camera.
        const hipMidX = (j[23].x + j[24].x) / 2;
        const hipMidY = (j[23].y + j[24].y) / 2;
        const shdMidY = (j[11].y + j[12].y) / 2;
        // Torso length = hip-to-shoulder distance in Y (most stable axis).
        const torsoLen = Math.max(Math.abs(hipMidY - shdMidY), 0.05);

        // World torso length for normalising world-space Z to the same scale.
        // World landmarks are in metres; a typical torso is ~0.45–0.55 m.
        let worldTorsoLen = 0.5;
        if (worldLandmarks && worldLandmarks.length >= 25) {
            const wShY = (worldLandmarks[11].y + worldLandmarks[12].y) / 2;
            const wHpY = (worldLandmarks[23].y + worldLandmarks[24].y) / 2;
            worldTorsoLen = Math.max(Math.abs(wHpY - wShY), 0.1);
        }

        // Convert one landmark index to a body-relative 3-D vector.
        // MediaPipe: x→right (0-1), y→down (0-1), z→depth (neg = closer to cam).
        // Output:    x→right, y→up, z→forward (away from camera, scaled).
        const bv = (idx, stripZ = false) => {
            const lmk = j[idx];
            return new THREE.Vector3(
                (lmk.x - hipMidX) / torsoLen,
                -(lmk.y - hipMidY) / torsoLen,
                stripZ ? 0 : (-lmk.z / torsoLen)
            );
        };

        // Leg/foot variant: use world Z when available (far more stable for the
        // forward/backward swing of knees and ankles during a walking step).
        const bvLeg = (imgIdx, wIdx) => {
            const lmk = j[imgIdx];
            const z = (worldLandmarks && wIdx < worldLandmarks.length)
                ? (-worldLandmarks[wIdx].z / worldTorsoLen)
                : (-lmk.z / torsoLen);
            return new THREE.Vector3(
                (lmk.x - hipMidX) / torsoLen,
                -(lmk.y - hipMidY) / torsoLen,
                z
            );
        };

        // ── Avatar-rotation compensation ────────────────────────────────────
        // Pre-rotate each body-relative direction by worldRotation so _aimBone's
        // parent-quaternion removal cancels correctly at any facing angle.
        const rotQ = this._scratch.rotQ.setFromAxisAngle(
            new THREE.Vector3(0, 1, 0), this.worldRotation
        );
        const lv = (idx, stripZ = false) => bv(idx, stripZ).applyQuaternion(rotQ);

        // ── Arms ────────────────────────────────────────────────────────────
        // Slerp alpha here controls how fast the bone tracks the (already-smoothed)
        // landmark target. Keep it high (0.5+) for responsive tracking.
        this._aimBone(this.boneMap.leftArm,      lv(11), lv(13), 0.55);
        this._aimBone(this.boneMap.leftForeArm,  lv(13), lv(15), 0.55);
        this._aimBone(this.boneMap.rightArm,     lv(12), lv(14), 0.55);
        this._aimBone(this.boneMap.rightForeArm, lv(14), lv(16), 0.55);

        // ── Clavicle compensation ───────────────────────────────────────────
        this._applyClavicle(this.boneMap.leftClavicle,  lv(11), lv(13), false);
        this._applyClavicle(this.boneMap.rightClavicle, lv(12), lv(14), true);

        // ── Legs ────────────────────────────────────────────────────────────
        // bvLeg uses world Z (if available) for the forward/backward depth of
        // each joint — this makes knees visibly swing forward during a step.
        const lvLeg = (imgIdx, wIdx) => bvLeg(imgIdx, wIdx).applyQuaternion(rotQ);

        this._aimBone(this.boneMap.leftUpLeg,  lvLeg(23, 23), lvLeg(25, 25), 0.45, true);
        this._aimBone(this.boneMap.leftLeg,    lvLeg(25, 25), lvLeg(27, 27), 0.45, true);
        this._aimBone(this.boneMap.rightUpLeg, lvLeg(24, 24), lvLeg(26, 26), 0.45, true);
        this._aimBone(this.boneMap.rightLeg,   lvLeg(26, 26), lvLeg(28, 28), 0.45, true);

        // ── Feet ─────────────────────────────────────────────────────────────
        if (landmarks.length > 32) {
            this._aimBone(this.boneMap.leftFoot,  lvLeg(29, 29), lvLeg(31, 31), 0.35, true);
            this._aimBone(this.boneMap.rightFoot, lvLeg(30, 30), lvLeg(32, 32), 0.35, true);
        }

        // ── Spine ───────────────────────────────────────────────────────────
        // Strip Z: MediaPipe depth (Z) for torso is noisy in video and causes
        // the spine to lean forward/backward spuriously.  XY tilt is reliable.
        const spineHipX = (lv(23).x + lv(24).x) / 2;
        const spineHipY = (lv(23).y + lv(24).y) / 2;
        const spineShdX = (lv(11).x + lv(12).x) / 2;
        const spineShdY = (lv(11).y + lv(12).y) / 2;
        const spineHipVec = this._scratch.v2.set(spineHipX, spineHipY, 0);
        const spineShdVec = this._scratch.v3.set(spineShdX, spineShdY, 0);
        this._aimBone(this.boneMap.spine, spineHipVec, spineShdVec, 0.20);

    }

    // Called with MediaPipe Face Mesh landmarks — drives the head bone's
    // nod (pitch) / turn (yaw) / tilt (roll). Independent of body pose.
    updateFace(faceLandmarks) {
        const headBone = this.bones[this.boneMap.head];
        const rest     = headBone && this.restPose[this.boneMap.head];
        if (!headBone || !rest) return;

        // No face detected this frame → ease the head back to its rest pose.
        if (!faceLandmarks || faceLandmarks.length < 468) {
            headBone.quaternion.slerp(rest, 0.1);
            return;
        }

        const L = faceLandmarks;
        // View-space vectors: x→right, y→up, z→toward camera.
        const V = (i) => new THREE.Vector3(L[i].x, -L[i].y, -L[i].z);

        // Build an orthonormal head basis from stable facial reference points.
        //   right: right cheek → left cheek   up: chin → forehead
        const right = V(454).sub(V(234)).normalize();
        let   up    = V(10).sub(V(152)).normalize();
        const fwd   = new THREE.Vector3().crossVectors(right, up).normalize();
        up.crossVectors(fwd, right).normalize();

        // Orientation matrix → Euler. A face looking straight at the camera
        // yields ~identity, so the angles self-zero at neutral (no calibration).
        const m = new THREE.Matrix4().makeBasis(right, up, fwd);
        const e = new THREE.Euler().setFromRotationMatrix(m, 'XYZ');

        const sgn = this._headSign, gain = this._headGain, clmp = this._headClamp;
        const tgtPitch = THREE.MathUtils.clamp(e.x * gain.pitch * sgn.pitch, -clmp.pitch, clmp.pitch);
        const tgtYaw   = THREE.MathUtils.clamp(e.y * gain.yaw   * sgn.yaw,   -clmp.yaw,   clmp.yaw);
        const tgtRoll  = THREE.MathUtils.clamp(e.z * gain.roll  * sgn.roll,  -clmp.roll,  clmp.roll);

        // EMA-smooth the angles to kill landmark jitter.
        const a = this._bypassEMA ? 1 : 0.5;
        this._headAngles.pitch += a * (tgtPitch - this._headAngles.pitch);
        this._headAngles.yaw   += a * (tgtYaw   - this._headAngles.yaw);
        this._headAngles.roll  += a * (tgtRoll  - this._headAngles.roll);

        const offset = this._scratch.q1.setFromEuler(new THREE.Euler(
            this._headAngles.pitch, this._headAngles.yaw, this._headAngles.roll, 'XYZ'
        ));
        this._scratch.q0.copy(rest).multiply(offset);
        headBone.quaternion.slerp(this._scratch.q0, this._headAlpha);
    }

    // Called with MediaPipe hand landmarks: { left:[...21]|null, right:[...21]|null }.
    // Drives finger curl on both hands. Independent of body pose.
    updateHands(handData) {
        if (!this.model || !Object.keys(this.bones).length) return;
        this._applyHand('left',  handData && handData.left);
        this._applyHand('right', handData && handData.right);
    }

    // Map one detected hand onto an avatar hand and curl its fingers.
    _applyHand(detectedSide, lms) {
        // Detected MediaPipe side → avatar side (optional mirror swap).
        const swap = this._handSwap;
        const avatarSide = (detectedSide === 'left')
            ? (swap ? 'Right' : 'Left')
            : (swap ? 'Left'  : 'Right');

        if (!lms || lms.length < 21) { this._relaxHand(avatarSide); return; }

        const side = avatarSide.toLowerCase();
        const V = (i) => new THREE.Vector3(lms[i].x, lms[i].y, lms[i].z);
        const wrist = V(0);

        for (const f of this._FINGERS) {
            // Thumb uses its own flex axis/sign — different plane from the fingers.
            const isThumb = f.bones[0].startsWith('Thumb');
            const axis = isThumb ? this._thumbBendAxis : this._fingerBendAxis;
            const sign = isThumb ? (this._thumbBendSign[side]  ?? 1)
                                 : (this._fingerBendSign[side] ?? 1);

            // 5 points (wrist + 4 joints) → 4 segments → 3 joint bend angles.
            const P = [wrist, V(f.lms[0]), V(f.lms[1]), V(f.lms[2]), V(f.lms[3])];
            const seg = [];
            for (let i = 0; i < 4; i++) {
                seg.push(P[i + 1].clone().sub(P[i]).normalize());
            }
            for (let i = 0; i < 3; i++) {
                // Bend angle is rotation-invariant: works at any hand orientation.
                const d = THREE.MathUtils.clamp(seg[i].dot(seg[i + 1]), -1, 1);
                const angle = Math.acos(d);  // ≥ 0, grows as the joint curls
                this._applyCurl('mixamorig' + avatarSide + 'Hand' + f.bones[i], angle, sign, axis);
            }
        }
    }

    // Apply a single finger-joint curl on top of its rest pose (EMA-smoothed).
    _applyCurl(boneName, angle, sign, axis) {
        const bone = this.bones[boneName];
        const rest = this.restPose[boneName];
        if (!bone || !rest) return;

        const prev = this._fingerCurl[boneName] ?? 0;
        const sm   = prev + 0.5 * (angle * this._fingerGain - prev);
        this._fingerCurl[boneName] = sm;

        const q = this._scratch.q1.setFromAxisAngle(axis, sm * sign);
        this._scratch.q0.copy(rest).multiply(q);
        bone.quaternion.slerp(this._scratch.q0, 0.5);
    }

    // Ease a hand's fingers back to rest when it isn't detected.
    _relaxHand(avatarSide) {
        for (const f of this._FINGERS) {
            for (const b of f.bones) {
                const name = 'mixamorig' + avatarSide + 'Hand' + b;
                const bone = this.bones[name], rest = this.restPose[name];
                if (bone && rest) bone.quaternion.slerp(rest, 0.2);
                this._fingerCurl[name] = 0;
            }
        }
    }

    // Lift the clavicle bone proportionally when the arm is raised above horizontal.
    // isRight=true flips the lift axis for the right side.
    _applyClavicle(clavName, shoulderVec, elbowVec, isRight) {
        const bone = this.bones[clavName];
        const rest = this.restPose[clavName];
        if (!bone || !rest) return;

        const dir = this._scratch.v0.subVectors(elbowVec, shoulderVec).normalize();
        // dir.y: +1 = arm fully raised, 0 = horizontal, -1 = arm down.
        const elevation = Math.max(0, dir.y);
        const liftAngle = elevation * 0.30;

        const zSign = isRight ? -1 : 1;
        const offsetQ = this._scratch.q1.setFromEuler(
            new THREE.Euler(0, 0, zSign * liftAngle)
        );
        // Apply offset on top of rest pose (rest × offset)
        this._scratch.q0.copy(rest).multiply(offsetQ);
        bone.quaternion.slerp(this._scratch.q0, 0.35);
    }

    _applyLocomotionAnim(dt) {
        this._animClock += dt;
        const t     = this._animClock;
        const state = this.locomotionState;
        const s     = this._legSign;
        const hasLivePose = (performance.now() - this._landmarkTimestamp) < 500;
        // When live landmarks are driving bones directly, procedural animation
        // must not fight them. In gesture mode, livePoseActive is always false.
        const livePoseActive = hasLivePose && this.translationMode === 'direct';

        // Blend a local Euler offset ON TOP of the bone's rest quaternion
        const blendBone = (name, ex, ey, ez, alpha = 0.3) => {
            const bone = this.bones[name];
            const rest = this.restPose[name];
            if (!bone || !rest) return;
            const offset = new THREE.Quaternion().setFromEuler(new THREE.Euler(ex, ey, ez));
            bone.quaternion.slerp(rest.clone().multiply(offset), alpha);
        };

        const returnToRest = (name, alpha = 0.06) => {
            const bone = this.bones[name];
            const rest = this.restPose[name];
            if (!bone || !rest) return;
            bone.quaternion.slerp(rest, alpha);
        };

        if (state === 'idle') {
            const b = Math.sin(t * 1.1) * 0.012;
            blendBone(this.boneMap.spine1, b, 0, 0, 0.04);
            
            // Bring arms down to sides (Natural pose) — only if no live landmark data
            if (!livePoseActive) {
                blendBone(this.boneMap.leftArm,  0, 0,  1.5, 0.08);
                blendBone(this.boneMap.rightArm, 0, 0,  1.5, 0.08); 
                blendBone(this.boneMap.leftForeArm, 0.2, 0, 0, 0.08);
                blendBone(this.boneMap.rightForeArm, 0.2, 0, 0, 0.08);
            }

            if (!livePoseActive) {
                [this.boneMap.leftUpLeg,    this.boneMap.rightUpLeg,
                 this.boneMap.leftLeg,      this.boneMap.rightLeg,
                 this.boneMap.leftFoot,     this.boneMap.rightFoot,
                 this.boneMap.spine,
                 this.boneMap.leftClavicle, this.boneMap.rightClavicle,
                ].forEach(n => returnToRest(n));
            }

        } else if (state === 'walk' || state === 'run') {
            const freq  = state === 'run' ? 4.0 : 2.5;
            const swing = state === 'run' ? 0.65 : 0.4;
            const knee  = state === 'run' ? 0.70 : 0.40;
            const foot  = state === 'run' ? 0.30 : 0.15;
            const arm   = state === 'run' ? 0.6 : 0.35;
            const alpha = 0.4;

            const phase = Math.sin(t * freq);

            // ── Legs: procedural only when no live pose data ──
            if (!livePoseActive) {
                blendBone(this.boneMap.leftUpLeg,   s *  phase * swing, 0, 0, alpha);
                blendBone(this.boneMap.rightUpLeg,  s * -phase * swing, 0, 0, alpha);

                const lKnee = Math.max(0, -phase) * knee;
                const rKnee = Math.max(0,  phase) * knee;
                blendBone(this.boneMap.leftLeg,  lKnee, 0, 0, alpha);
                blendBone(this.boneMap.rightLeg, rKnee, 0, 0, alpha);

                blendBone(this.boneMap.leftFoot,  -lKnee * foot, 0, 0, alpha);
                blendBone(this.boneMap.rightFoot, -rKnee * foot, 0, 0, alpha);
            }

            // ── Arms: Swing fore/aft — only if no live landmark data ──
            if (!livePoseActive) {
                // Pulling both arms forward: negative offsets on both sides
                // since positive Y rotation was biased backward on this rig.
                const armDown = 1.5;
                const lSwing = (phase * arm) - 0.25; 
                const rSwing = (phase * arm) - 0.15; 

                blendBone(this.boneMap.leftArm,   0,  lSwing, armDown, 0.25);
                blendBone(this.boneMap.rightArm,  0,  rSwing, armDown, 0.25);
                
                // Forearms slightly bent
                blendBone(this.boneMap.leftForeArm,  0.3, 0, 0, 0.1);
                blendBone(this.boneMap.rightForeArm, 0.3, 0, 0, 0.1);
            }

            // Slight counter-rotation in spine
            blendBone(this.boneMap.spine1, 0, -phase * 0.05, 0, 0.1);

        } else if (state === 'crouch') {
            blendBone(this.boneMap.spine,      0.15, 0, 0,     0.15);
            blendBone(this.boneMap.leftUpLeg,  0.35, 0,  0.04, 0.15);
            blendBone(this.boneMap.rightUpLeg, 0.35, 0, -0.04, 0.15);
            blendBone(this.boneMap.leftLeg,    1.20, 0, 0,     0.15);
            blendBone(this.boneMap.rightLeg,   1.20, 0, 0,     0.15);
            blendBone(this.boneMap.leftFoot,  -0.50, 0, 0,     0.15);
            blendBone(this.boneMap.rightFoot, -0.50, 0, 0,     0.15);
            if (!livePoseActive) {
                blendBone(this.boneMap.leftArm,  0, 0, 1.5, 0.15);
                blendBone(this.boneMap.rightArm, 0, 0, 1.5, 0.15);
            }

        } else if (state === 'jump') {
            if (!livePoseActive) {
                blendBone(this.boneMap.leftArm,    0, 0, 0.6, 0.25);
                blendBone(this.boneMap.rightArm,   0, 0, 0.6, 0.25);
                blendBone(this.boneMap.leftUpLeg,  -0.4, 0, 0, 0.25);
                blendBone(this.boneMap.rightUpLeg, -0.4, 0, 0, 0.25);
                blendBone(this.boneMap.leftLeg,    0.7, 0, 0, 0.25);
                blendBone(this.boneMap.rightLeg,   0.7, 0, 0, 0.25);
            }
        }
    }


    _aimBone(boneName, fromVec, toVec, alpha = 0.35, useRestDir = false) {
        const bone = this.bones[boneName];
        const rest = this.restPose[boneName];
        if (!bone || !rest) return;

        // Reuse scratch objects — no heap allocation per call.
        const sc = this._scratch;
        const dir = sc.v0.subVectors(toVec, fromVec).normalize();
        if (dir.lengthSq() < 0.000001) return;

        const parentQ = sc.q0;
        parentQ.identity();
        if (bone.parent) bone.parent.getWorldQuaternion(parentQ);
        const localDir = sc.v1.copy(dir).applyQuaternion(parentQ.invert());

        const targetQ = sc.q1;
        if (useRestDir) {
            // Use pre-computed child-direction if available (leg bones).
            // childBone.position expressed in parent space via rest quaternion —
            // this is the actual "toward child" axis, avoiding (0,1,0) assumption
            // which hits antiparallel singularity on many Mixamo rigs.
            const precomputed = this._boneToChildDir && this._boneToChildDir[boneName];
            const restDir = precomputed
                ? sc.v2.copy(precomputed)
                : sc.v2.set(0, 1, 0).applyQuaternion(rest).normalize();
            targetQ.setFromUnitVectors(restDir, localDir.normalize());
            targetQ.multiply(rest);
        } else {
            targetQ.setFromUnitVectors(sc.v3.set(0, 1, 0), localDir);
        }

        bone.quaternion.slerp(targetQ, alpha);
    }

    _animate = () => {
        requestAnimationFrame(this._animate);
        const now = performance.now();
        const dt  = Math.min((now - this._lastTime) / 1000, 0.05);
        this._lastTime = now;
        if (this.model) this._applyLocomotionAnim(dt);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    _onWindowResize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
        // setSize re-writes inline pixel styles — clear them again so CSS flex
        // controls display size (avoids canvas overflowing in "both" view).
        this.renderer.domElement.style.width  = '';
        this.renderer.domElement.style.height = '';
    }
}