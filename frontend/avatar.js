class AvatarRenderer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 200, 250);
        this.camera.lookAt(0, 100, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        // Lighting
        this.setupLighting();

        // Grid floor
        const grid = new THREE.GridHelper(500, 20, 0xcccccc, 0xdddddd);
        this.scene.add(grid);

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 100, 0);
        this.controls.update();

        // Avatar state
        this.model = null;
        this.bones = {};
        this.lastValidPose = null;

        // MediaPipe landmark index -> Mixamo bone name (mixamorig prefix)
        this.boneMap = {
            leftArm:      'mixamorigLeftArm',
            rightArm:     'mixamorigRightArm',
            leftForeArm:  'mixamorigLeftForeArm',
            rightForeArm: 'mixamorigRightForeArm',
            leftHand:     'mixamorigLeftHand',
            rightHand:    'mixamorigRightHand',
            leftUpLeg:    'mixamorigLeftUpLeg',
            rightUpLeg:   'mixamorigRightUpLeg',
            leftLeg:      'mixamorigLeftLeg',
            rightLeg:     'mixamorigRightLeg',
            leftFoot:     'mixamorigLeftFoot',
            rightFoot:    'mixamorigRightFoot',
            spine:        'mixamorigSpine',
            hips:         'mixamorigHips',
        };

        this.loadModel();
        this.animate();
        window.addEventListener('resize', () => this.onWindowResize());
    }

    setupLighting() {
        const ambient = new THREE.AmbientLight(0xffffff, 0.7);
        this.scene.add(ambient);

        const dirLight = new THREE.DirectionalLight(0x00d4ff, 1.0);
        dirLight.position.set(5, 10, 7);
        dirLight.castShadow = true;
        this.scene.add(dirLight);

        const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
        fillLight.position.set(-5, 5, -5);
        this.scene.add(fillLight);
    }

    loadModel() {
        const loader = new THREE.GLTFLoader();
        loader.load('/models/avatar.glb', (gltf) => {
            this.model = gltf.scene;

            // Fix orientation — some converters export lying down
            this.model.rotation.x = 0;
            this.model.rotation.y = Math.PI;  // face toward camera
            this.model.rotation.z = 0;

            // Auto scale and center
            const box = new THREE.Box3().setFromObject(this.model);
            const size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());

            const scale = 150 / size.y;
            this.model.scale.setScalar(scale);

            // Recompute box after scale
            const box2 = new THREE.Box3().setFromObject(this.model);
            this.model.position.x = -center.x * scale;
            this.model.position.y = -box2.min.y;
            this.model.position.z = -center.z * scale;

            // Enable skinning on existing materials and collect bones
            this.model.traverse((child) => {
                if (child.isMesh || child.isSkinnedMesh) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(m => { m.skinning = true; });
                    } else {
                        child.material.skinning = true;
                    }
                    child.castShadow = true;
                }
                // Collect bones
                if (child.isBone) {
                    this.bones[child.name] = child;
                }
            });

            this.scene.add(this.model);
            this.captureInitialPose();

            console.log('Avatar loaded. Bones found:', Object.keys(this.bones).length);
            console.log('Available bones:', Object.keys(this.bones));
        }, (progress) => {
            if (progress.total > 0) {
                console.log('Loading:', Math.round(progress.loaded / progress.total * 100) + '%');
            }
        }, (error) => {
            console.error('Error loading avatar:', error);
        });
    }

    captureInitialPose() {
        this.lastValidPose = {};
        Object.keys(this.bones).forEach(name => {
            this.lastValidPose[name] = {
                quaternion: this.bones[name].quaternion.clone()
            };
        });
    }

    updatePose(landmarks) {
        if (!this.model || !landmarks || landmarks.length < 29) return;
        if (Object.keys(this.bones).length === 0) return;

        const j = landmarks;

        // Helper — proper 3D conversion
        // X: mirrored, Y: flipped, Z: depth kept positive
        const v = (lm, yScale = 1.0) => new THREE.Vector3(
            1 - lm.x,
            -lm.y * yScale,
            lm.z * 0.5         // don't negate Z
        );

        // Arms
        this.aimBone(this.boneMap.leftArm,      v(j[11]), v(j[13]));
        this.aimBone(this.boneMap.leftForeArm,  v(j[13]), v(j[15]));
        this.aimBone(this.boneMap.rightArm,     v(j[12]), v(j[14]));
        this.aimBone(this.boneMap.rightForeArm, v(j[14]), v(j[16]));

        // Legs — higher Y scale
        this.aimBone(this.boneMap.leftUpLeg,  v(j[23], 1.5), v(j[25], 1.5));
        this.aimBone(this.boneMap.leftLeg,    v(j[25], 1.5), v(j[27], 1.5));
        this.aimBone(this.boneMap.rightUpLeg, v(j[24], 1.5), v(j[26], 1.5));
        this.aimBone(this.boneMap.rightLeg,   v(j[26], 1.5), v(j[28], 1.5));
        // Spine
        const hipV = v(j[23]).add(v(j[24])).multiplyScalar(0.5);
        const shdV = v(j[11]).add(v(j[12])).multiplyScalar(0.5);
        this.aimBone(this.boneMap.spine, hipV, shdV);

        // Hips rotation — rotate avatar body as you rotate
        const leftHip  = v(j[23]);
        const rightHip = v(j[24]);
        const hipDir   = new THREE.Vector3().subVectors(rightHip, leftHip).normalize();
        const targetHipAngle = Math.atan2(hipDir.z, hipDir.x);
        if (this.bones[this.boneMap.hips]) {
            const hipQuat = new THREE.Quaternion();
            hipQuat.setFromAxisAngle(new THREE.Vector3(0, 1, 0), targetHipAngle);
            this.bones[this.boneMap.hips].quaternion.slerp(hipQuat, 0.2);
        }

        this.lastValidPose = this.capturePose();
    }

    aimBone(boneName, fromVec, toVec) {
        const bone = this.bones[boneName];
        if (!bone) return;

        const targetDir = new THREE.Vector3()
            .subVectors(toVec, fromVec)
            .normalize();

        if (targetDir.length() < 0.001) return;

        const parentWorldQuat = new THREE.Quaternion();
        if (bone.parent) {
            bone.parent.getWorldQuaternion(parentWorldQuat);
        }

        const parentWorldQuatInv = parentWorldQuat.clone().invert();
        const localTarget = targetDir.clone().applyQuaternion(parentWorldQuatInv);
        const restDir = new THREE.Vector3(0, 1, 0);
        const localQuat = new THREE.Quaternion().setFromUnitVectors(restDir, localTarget);

        bone.quaternion.slerp(localQuat, 0.25);
    }

    capturePose() {
        const pose = {};
        Object.keys(this.bones).forEach(name => {
            pose[name] = { quaternion: this.bones[name].quaternion.clone() };
        });
        return pose;
    }

    freezePose() {
        if (!this.lastValidPose) return;
        Object.keys(this.lastValidPose).forEach(name => {
            if (this.bones[name]) {
                this.bones[name].quaternion.copy(this.lastValidPose[name].quaternion);
            }
        });
    }

    animate = () => {
        requestAnimationFrame(this.animate);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }
}