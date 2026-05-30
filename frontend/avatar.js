// ── AvatarRenderer: Mixamo GLB + Kalidokit holistic rigging ────────────────
// KEY DIFFERENCE from VRM: Mixamo bones have baked rest-pose rotations.
// Kalidokit outputs euler angles relative to a T-pose/identity rest.
// So we must apply kalidokit rotations ON TOP OF the captured rest pose,
// not replace the bone quaternion directly (which is what VRM can do).

class AvatarRenderer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);

        this.camera = new THREE.PerspectiveCamera(
            35,
            this.container.clientWidth / this.container.clientHeight,
            0.1, 2000
        );
        this.camera.position.set(0, 140, 400);
        this.camera.lookAt(0, 100, 0);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.toneMapping = THREE.NoToneMapping;
        this.container.appendChild(this.renderer.domElement);

        this._setupLighting();
        this._buildEnvironment();

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 100, 0);
        this.controls.update();

        this.model = null;
        this.bones = {};
        this.restPose = {};   // captured AFTER model added to scene
        this._modelYOffset = 0;

        // Camera auto-zoom state
        this._camMode = 'body';
        this._faceOnlyFrames = 0;
        this._bodyFrames = 0;
        this._camThreshold = 20;
        this._camLerping = false;

        // Target values lerped each frame
        this._camPosTarget = new THREE.Vector3(0, 140, 400);
        this._camLookTarget = new THREE.Vector3(0, 100, 0);

        // Mixamo bone name map (humanoid key → mixamorig bone name)
        this.boneMap = {
            hips: 'mixamorigHips',
            spine: 'mixamorigSpine',
            chest: 'mixamorigSpine1',
            neck: 'mixamorigNeck',
            head: 'mixamorigHead',

            leftUpperArm: 'mixamorigLeftArm',
            leftLowerArm: 'mixamorigLeftForeArm',
            leftHand: 'mixamorigLeftHand',
            rightUpperArm: 'mixamorigRightArm',
            rightLowerArm: 'mixamorigRightForeArm',
            rightHand: 'mixamorigRightHand',

            leftUpperLeg: 'mixamorigLeftUpLeg',
            leftLowerLeg: 'mixamorigLeftLeg',
            leftFoot: 'mixamorigLeftFoot',
            rightUpperLeg: 'mixamorigRightUpLeg',
            rightLowerLeg: 'mixamorigRightLeg',
            rightFoot: 'mixamorigRightFoot',

            // Left fingers
            leftThumbProximal: 'mixamorigLeftHandThumb1',
            leftThumbIntermediate: 'mixamorigLeftHandThumb2',
            leftThumbDistal: 'mixamorigLeftHandThumb3',
            leftIndexProximal: 'mixamorigLeftHandIndex1',
            leftIndexIntermediate: 'mixamorigLeftHandIndex2',
            leftIndexDistal: 'mixamorigLeftHandIndex3',
            leftMiddleProximal: 'mixamorigLeftHandMiddle1',
            leftMiddleIntermediate: 'mixamorigLeftHandMiddle2',
            leftMiddleDistal: 'mixamorigLeftHandMiddle3',
            leftRingProximal: 'mixamorigLeftHandRing1',
            leftRingIntermediate: 'mixamorigLeftHandRing2',
            leftRingDistal: 'mixamorigLeftHandRing3',
            leftLittleProximal: 'mixamorigLeftHandPinky1',
            leftLittleIntermediate: 'mixamorigLeftHandPinky2',
            leftLittleDistal: 'mixamorigLeftHandPinky3',

            // Right fingers
            rightThumbProximal: 'mixamorigRightHandThumb1',
            rightThumbIntermediate: 'mixamorigRightHandThumb2',
            rightThumbDistal: 'mixamorigRightHandThumb3',
            rightIndexProximal: 'mixamorigRightHandIndex1',
            rightIndexIntermediate: 'mixamorigRightHandIndex2',
            rightIndexDistal: 'mixamorigRightHandIndex3',
            rightMiddleProximal: 'mixamorigRightHandMiddle1',
            rightMiddleIntermediate: 'mixamorigRightHandMiddle2',
            rightMiddleDistal: 'mixamorigRightHandMiddle3',
            rightRingProximal: 'mixamorigRightHandRing1',
            rightRingIntermediate: 'mixamorigRightHandRing2',
            rightRingDistal: 'mixamorigRightHandRing3',
            rightLittleProximal: 'mixamorigRightHandPinky1',
            rightLittleIntermediate: 'mixamorigRightHandPinky2',
            rightLittleDistal: 'mixamorigRightHandPinky3',
        };

        this._clock = new THREE.Clock();
        this._loadModel();
        this._animate();
        window.addEventListener('resize', () => this._onWindowResize());
    }

    // ── Lighting ─────────────────────────────────────────────────────────────
    _setupLighting() {
        this.scene.add(new THREE.AmbientLight(0xffffff, 4.0));

        const key = new THREE.DirectionalLight(0xffffff, 5.0);
        key.position.set(3, 10, 6);
        key.castShadow = true;
        key.shadow.mapSize.setScalar(1024);
        key.shadow.camera.near = 1;
        key.shadow.camera.far = 600;
        key.shadow.camera.left = key.shadow.camera.bottom = -200;
        key.shadow.camera.right = key.shadow.camera.top = 200;
        this.scene.add(key);

        const fill = new THREE.DirectionalLight(0xffeedd, 3.0);
        fill.position.set(-4, 6, -4);
        this.scene.add(fill);

        const top = new THREE.DirectionalLight(0xffffff, 2.0);
        top.position.set(0, 20, 0);
        this.scene.add(top);
    }

    _buildEnvironment() {
        this.scene.add(new THREE.GridHelper(1000, 40, 0x334477, 0x223366));

        const floor = new THREE.Mesh(
            new THREE.PlaneGeometry(1000, 1000),
            new THREE.MeshLambertMaterial({ color: 0x111133 })
        );
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        this.scene.add(floor);
    }

    // ── Model loading ─────────────────────────────────────────────────────────
    _loadModel() {
        const loader = new THREE.GLTFLoader();
        loader.load('/models/avatar.glb', (gltf) => {
            this.model = gltf.scene;
            this.model.rotation.y = Math.PI;

            const box1 = new THREE.Box3().setFromObject(this.model);
            const size = box1.getSize(new THREE.Vector3());
            const scale = 150 / size.y;
            this.model.scale.setScalar(scale);

            const box2 = new THREE.Box3().setFromObject(this.model);
            const centre = box2.getCenter(new THREE.Vector3());
            this.model.position.x = -centre.x;
            this.model.position.z = -centre.z;
            this._modelYOffset = -box2.min.y;
            this.model.position.y = this._modelYOffset;

            this.model.traverse((child) => {
                if (child.isMesh || child.isSkinnedMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
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

            // CRITICAL: capture rest pose AFTER model is in scene and transforms settled
            // These are the Mixamo baked-in rotations we must preserve as base
            Object.keys(this.bones).forEach(n => {
                this.restPose[n] = this.bones[n].quaternion.clone();
            });

            // Model facing quaternion (root rotation.y = PI). Kalidokit rotations are
            // expressed relative to a model-facing VRM rest, so we map them through this.
            this._faceQ = this.model.getWorldQuaternion(new THREE.Quaternion());
            this._faceQInv = this._faceQ.clone().invert();

            console.log('Avatar loaded. Bones:', Object.keys(this.bones).length);
            console.log('Sample bone names:', Object.keys(this.bones).slice(0, 8));

            const overlay = document.getElementById('loadingOverlay');
            if (overlay) overlay.style.display = 'none';
        },
            (p) => {
                if (p.total > 0) {
                    const el = document.getElementById('loadingPct');
                    if (el) el.textContent = Math.round(p.loaded / p.total * 100) + '%';
                }
            },
            (e) => console.error('Avatar load error:', e));
    }

    // ── Core rig helper (VRM → Mixamo retarget) ────────────────────────────────
    // Kalidokit emits rotations in VRM convention: each bone's rest local frame is
    // world/model-aligned, so the VRM sample just does bone.quaternion = euler(Rk).
    // Mixamo bones have baked rest rotations AND limb-local axes rotated ~90° from
    // VRM's, so applying Rk in the bone's own local frame (rest * Rk) twists the arms.
    //
    // Correct retarget: treat Rk as a deviation in the model's facing frame, re-express
    // it in this bone's PARENT-local frame, then layer it on top of the Mixamo rest:
    //
    //   R'     = faceQ · Rk · faceQ⁻¹                     (Rk in model-facing world space)
    //   target = (parentWorld⁻¹ · R' · parentWorld) · restLocal
    //
    // For top-of-chain bones (upper arms, hips) parentWorld ≈ faceQ, so this reduces to
    // target = Rk · restLocal — i.e. PRE-multiply, the opposite of the broken rest · Rk.
    // For deeper bones (forearms, hands, fingers) it self-corrects from the live parent
    // pose. No bone needs a hand-tuned axis flip; the captured rest carries the basis.
    _rigRotation(humanoidKey, rotation = { x: 0, y: 0, z: 0 }, dampener = 1, lerpAmt = 0.3) {
        const boneName = this.boneMap[humanoidKey];
        if (!boneName) return;
        const bone = this.bones[boneName];
        const rest = this.restPose[boneName];
        if (!bone || !rest || !this._faceQ) return;

        // Kalidokit euler (VRM convention) → quaternion
        const Rk = new THREE.Quaternion().setFromEuler(new THREE.Euler(
            rotation.x * dampener,
            rotation.y * dampener,
            rotation.z * dampener,
            'XYZ'
        ));

        // Map into the model's facing frame: R' = faceQ · Rk · faceQ⁻¹
        const Rworld = this._faceQ.clone().multiply(Rk).multiply(this._faceQInv);

        // Re-express in the bone's parent-local frame and apply on top of rest
        const P = new THREE.Quaternion();
        bone.parent.getWorldQuaternion(P);
        const target = P.clone().invert().multiply(Rworld).multiply(P).multiply(rest);

        bone.quaternion.slerp(target, lerpAmt);
    }

    // ── Apply rigged data from Kalidokit ──────────────────────────────────────
    applyHolisticResults(riggedPose, riggedFace, riggedLeftHand, riggedRightHand) {
        if (!this.model || !Object.keys(this.restPose).length) return;

        // ── Pose ──────────────────────────────────────────────────────────────
        if (riggedPose) {
            this._rigRotation('hips', riggedPose.Hips.rotation, 0.7, 0.07);

            // Keep model grounded — X/Z sway only from hips, no Y
            if (riggedPose.Hips.position) {
                const hx = -riggedPose.Hips.position.x * 60;
                const hz = -riggedPose.Hips.position.z * 60;
                this.model.position.x += (hx - this.model.position.x) * 0.1;
                this.model.position.y = this._modelYOffset;
                this.model.position.z += (hz - this.model.position.z) * 0.1;
            }

            this._rigRotation('chest', riggedPose.Spine, 0.25, 0.3);
            this._rigRotation('spine', riggedPose.Spine, 0.45, 0.3);

            this._rigRotation('rightUpperArm', riggedPose.RightUpperArm, 1, 0.3);
            this._rigRotation('rightLowerArm', riggedPose.RightLowerArm, 1, 0.3);
            this._rigRotation('leftUpperArm', riggedPose.LeftUpperArm, 1, 0.3);
            this._rigRotation('leftLowerArm', riggedPose.LeftLowerArm, 1, 0.3);

            this._rigRotation('leftUpperLeg', riggedPose.LeftUpperLeg, 1, 0.3);
            this._rigRotation('leftLowerLeg', riggedPose.LeftLowerLeg, 1, 0.3);
            this._rigRotation('rightUpperLeg', riggedPose.RightUpperLeg, 1, 0.3);
            this._rigRotation('rightLowerLeg', riggedPose.RightLowerLeg, 1, 0.3);
        }

        // ── Face / Head / Neck ────────────────────────────────────────────────
        if (riggedFace) {
            this._rigRotation('neck', riggedFace.head, 0.7, 0.3);
            this._rigRotation('head', riggedFace.head, 0.3, 0.3);
        }

        // ── Hands ─────────────────────────────────────────────────────────────
        if (riggedLeftHand && riggedPose) {
            this._rigRotation('leftHand', {
                z: riggedPose.LeftHand ? riggedPose.LeftHand.z : 0,
                y: riggedLeftHand.LeftWrist.y,
                x: riggedLeftHand.LeftWrist.x,
            }, 1, 0.3);
            this._applyFingers(riggedLeftHand, 'left');
        }
        if (riggedRightHand && riggedPose) {
            this._rigRotation('rightHand', {
                z: riggedPose.RightHand ? riggedPose.RightHand.z : 0,
                y: riggedRightHand.RightWrist.y,
                x: riggedRightHand.RightWrist.x,
            }, 1, 0.3);
            this._applyFingers(riggedRightHand, 'right');
        }
    }

    _applyFingers(riggedHand, side) {
        const capSide = side === 'left' ? 'Left' : 'Right';
        const fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'];
        const joints = ['Proximal', 'Intermediate', 'Distal'];
        for (const finger of fingers) {
            for (const joint of joints) {
                const kKey = `${capSide}${finger}${joint}`;
                const bKey = `${side}${finger}${joint}`;
                if (riggedHand[kKey]) {
                    this._rigRotation(bKey, riggedHand[kKey], 1, 0.3);
                }
            }
        }
    }

    // ── Smart camera zoom ─────────────────────────────────────────────────────
    // OrbitControls stores camera as spherical coords and recomputes camera.position
    // from them on every controls.update() call — overwriting any lerp we do.
    // Fix: disable controls on mode change, drive camera directly, re-enable + update
    // once at end so controls re-syncs its internal state from the new position.
    updateCameraMode(faceOnly) {
        if (faceOnly) {
            this._faceOnlyFrames++;
            this._bodyFrames = 0;
        } else {
            this._bodyFrames++;
            this._faceOnlyFrames = 0;
        }

        const prevMode = this._camMode;
        if (this._camMode === 'body' && this._faceOnlyFrames >= this._camThreshold) {
            this._camMode = 'face';
        } else if (this._camMode === 'face' && this._bodyFrames >= this._camThreshold) {
            this._camMode = 'body';
        }

        if (this._camMode !== prevMode) {
            this.controls.enabled = false;
            this._camLerping = true;

            let headY = 140;
            const headBone = this.bones[this.boneMap.head];
            if (headBone) {
                const wp = new THREE.Vector3();
                headBone.getWorldPosition(wp);
                if (wp.y > 10) headY = wp.y;
            }

            if (this._camMode === 'face') {
                this._camPosTarget.set(0, headY, 90);
                this._camLookTarget.set(0, headY - 8, 0);
            } else {
                this._camPosTarget.set(0, 140, 400);
                this._camLookTarget.set(0, 100, 0);
            }
        }
    }

    // ── Render loop ───────────────────────────────────────────────────────────
    _animate = () => {
        requestAnimationFrame(this._animate);

        if (this._camLerping) {
            const speed = 0.07;
            this.camera.position.lerp(this._camPosTarget, speed);
            this.controls.target.lerp(this._camLookTarget, speed);
            this.camera.lookAt(this.controls.target);

            if (this.camera.position.distanceTo(this._camPosTarget) < 2 &&
                this.controls.target.distanceTo(this._camLookTarget) < 2) {
                this.camera.position.copy(this._camPosTarget);
                this.controls.target.copy(this._camLookTarget);
                this.camera.lookAt(this.controls.target);
                this.controls.enabled = true;
                this.controls.update(); // re-syncs OrbitControls internal sphericals
                this._camLerping = false;
            }
        } else {
            this.controls.update();
        }

        this.renderer.render(this.scene, this.camera);
    }

    _onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
}