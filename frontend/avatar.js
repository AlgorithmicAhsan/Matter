class AvatarRenderer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0f);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 100, 250);
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
        const grid = new THREE.GridHelper(500, 20, 0x222233, 0x111122);
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
            this.model.rotation.y = 0;
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

            // Override materials to solid color
            this.model.traverse((child) => {
                if (child.isMesh || child.isSkinnedMesh) {
                    child.material = new THREE.MeshLambertMaterial({
                        color: 0x00d4ff,
                        emissive: 0x002233,
                    });
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

        // Arms
        this.rotateBone(this.boneMap.leftArm,      j[11], j[13]);
        this.rotateBone(this.boneMap.rightArm,     j[12], j[14]);
        this.rotateBone(this.boneMap.leftForeArm,  j[13], j[15]);
        this.rotateBone(this.boneMap.rightForeArm, j[14], j[16]);

        // Legs
        this.rotateBone(this.boneMap.leftUpLeg,  j[23], j[25]);
        this.rotateBone(this.boneMap.rightUpLeg, j[24], j[26]);
        this.rotateBone(this.boneMap.leftLeg,    j[25], j[27]);
        this.rotateBone(this.boneMap.rightLeg,   j[26], j[28]);

        // Spine — from hip center to shoulder center
        const hipCenter    = { x: (j[23].x + j[24].x) / 2, y: (j[23].y + j[24].y) / 2, z: (j[23].z + j[24].z) / 2 };
        const shoulderCenter = { x: (j[11].x + j[12].x) / 2, y: (j[11].y + j[12].y) / 2, z: (j[11].z + j[12].z) / 2 };
        this.rotateBone(this.boneMap.spine, hipCenter, shoulderCenter);

        this.lastValidPose = this.capturePose();
    }

    rotateBone(boneName, parentJoint, childJoint) {
        const bone = this.bones[boneName];
        if (!bone) return;

        // Direction vector — flip Y because MediaPipe Y goes down, Three.js Y goes up
        const dir = new THREE.Vector3(
            childJoint.x - parentJoint.x,
            -(childJoint.y - parentJoint.y),
            childJoint.z - parentJoint.z
        );

        if (dir.length() < 0.001) return;
        dir.normalize();

        // Rotate from default up direction to target direction
        const up = new THREE.Vector3(0, 1, 0);
        const quat = new THREE.Quaternion();
        quat.setFromUnitVectors(up, dir);

        // Smooth interpolation — 0.2 for responsive but not jittery
        bone.quaternion.slerp(quat, 0.2);
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