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
        this._modelYOffset = 0; // FIX: store load-time Y offset separately

        this.worldPosition   = new THREE.Vector3();
        this.worldRotation   = 0;
        this.locomotionState = 'idle';
        this._animClock      = 0;
        this._lastTime       = performance.now();
        this._landmarkTimestamp = 0;
        this._smoothLm = null;
        
        this.boneMap = {
            leftArm:      'mixamorigLeftArm',
            rightArm:     'mixamorigRightArm',
            leftForeArm:  'mixamorigLeftForeArm',
            rightForeArm: 'mixamorigRightForeArm',
            leftUpLeg:    'mixamorigLeftUpLeg',
            rightUpLeg:   'mixamorigRightUpLeg',
            leftLeg:      'mixamorigLeftLeg',
            rightLeg:     'mixamorigRightLeg',
            leftFoot:     'mixamorigLeftFoot',
            rightFoot:    'mixamorigRightFoot',
            spine:        'mixamorigSpine',
            spine1:       'mixamorigSpine1',
            spine2:       'mixamorigSpine2',
            hips:         'mixamorigHips',
        };

        // Leg swing sign — +1 or -1, auto-detected after model loads
        // (some Mixamo exports have inverted local X for leg bones)
        this._legSign = 1;

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

        this.controls.target.lerp(
            new THREE.Vector3(position.x, position.y + 80, position.z), 0.1
        );
    }

    // Called when MediaPipe landmarks arrive
    updatePose(landmarks) {
        if (this.translationMode !== 'direct') return;
        if (!this.model || !landmarks || landmarks.length < 29) return;
        if (!Object.keys(this.bones).length) return;

        this._landmarkTimestamp = performance.now();

        const alpha = 0.18;
        if (!this._smoothLm) {
            this._smoothLm = landmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z }));
        } else {
            for (let i = 0; i < landmarks.length; i++) {
                this._smoothLm[i].x += alpha * (landmarks[i].x - this._smoothLm[i].x);
                this._smoothLm[i].y += alpha * (landmarks[i].y - this._smoothLm[i].y);
                this._smoothLm[i].z += alpha * (landmarks[i].z - this._smoothLm[i].z);
            }
        }

        const j = this._smoothLm;
        const v = (lm) => new THREE.Vector3(lm.x - 0.5, -lm.y, -lm.z * 0.5);
        // For legs, ignore Z depth entirely — MediaPipe leg depth is too noisy
        // and causes bones to rotate in the sagittal (front-back) plane incorrectly.
        const vLeg = (lm) => new THREE.Vector3(lm.x - 0.5, -lm.y, 0);

        // Arms
        this._aimBone(this.boneMap.leftArm,      v(j[11]), v(j[13]));
        this._aimBone(this.boneMap.leftForeArm,  v(j[13]), v(j[15]));
        this._aimBone(this.boneMap.rightArm,     v(j[12]), v(j[14]));
        this._aimBone(this.boneMap.rightForeArm, v(j[14]), v(j[16]));

        // Legs (heavy smoothing 0.1 to prevent depth jitter when crouching)
        this._aimBone(this.boneMap.leftUpLeg,  vLeg(j[23]), vLeg(j[25]), 0.1, true);
        this._aimBone(this.boneMap.leftLeg,    vLeg(j[25]), vLeg(j[27]), 0.1, true);
        this._aimBone(this.boneMap.rightUpLeg, vLeg(j[24]), vLeg(j[26]), 0.1, true);
        this._aimBone(this.boneMap.rightLeg,   vLeg(j[26]), vLeg(j[28]), 0.1, true);

        // Spine
        const hipMid = v(j[23]).clone().add(v(j[24])).multiplyScalar(0.5);
        const shdMid = v(j[11]).clone().add(v(j[12])).multiplyScalar(0.5);
        this._aimBone(this.boneMap.spine, hipMid, shdMid);
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
                [this.boneMap.leftUpLeg, this.boneMap.rightUpLeg,
                 this.boneMap.leftLeg,   this.boneMap.rightLeg,
                 this.boneMap.leftFoot,  this.boneMap.rightFoot,
                 this.boneMap.spine].forEach(n => returnToRest(n));
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

        const dir = new THREE.Vector3().subVectors(toVec, fromVec).normalize();
        if (dir.length() < 0.001) return;

        const parentQ = new THREE.Quaternion();
        if (bone.parent) bone.parent.getWorldQuaternion(parentQ);
        const localDir = dir.clone().applyQuaternion(parentQ.invert());

        let targetQ;
        if (useRestDir) {
            // Use bone's actual rest direction to avoid antiparallel singularity.
            // Only needed for leg bones whose rest direction is ~(0,-1,0).
            const restDir = new THREE.Vector3(0, 1, 0).applyQuaternion(rest).normalize();
            const deltaQ = new THREE.Quaternion().setFromUnitVectors(restDir, localDir.normalize());
            targetQ = deltaQ.multiply(rest.clone());
        } else {
            targetQ = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), localDir);
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
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
}