/**
 * ObstacleCourseEnvironment
 * --------------------------
 * A self-contained Three.js environment that builds the obstacle course.
 * Each obstacle requires a distinct player action: walk, crouch, jump, climb ramp.
 *
 * Layout (Z+ = forward through the course, avatar starts at Z=0):
 *
 *   Z=0      SPAWN
 *   Z=200    Low Barrier  ← must CROUCH
 *   Z=500    Stagger Walls ← must navigate left/right
 *   Z=750    GAP          ← must JUMP
 *   Z=1000   Ramp up
 *   Z=1200   Elevated Platform
 *
 * All distances are in Three.js world units (avatar ≈ 150 units tall).
 */
class ObstacleCourseEnvironment {
    constructor(scene) {
        this.scene   = scene;
        this._meshes = [];

        this.colliders   = [];   // AABB boxes {cx,cy,cz,hw,hh,hd}
        this.gaps        = [];   // {minZ,maxZ}
        this.lowBarriers = [];   // {minZ,maxZ,clearanceY}
        this.ramps       = [];   // {minZ,maxZ,rampHeight,minX,maxX,flat?}

        this._build();
    }

    // ─── Build ────────────────────────────────────────────────────────────────

    _build() {
        this._addFloor();
        this._addCrouchBarrier();
        this._addStaggerWalls();
        this._addGap();
        this._addRamp();
        this._addLabels();
    }

    /** Single solid floor spanning the entire course (no grid — avoids Z-fighting flicker) */
    _addFloor() {
        // One big solid floor covering everything: Z -400 → 1800, with a gap cut-out
        // Near section: Z -400 → gap start (720)
        this._box(
            { x: 0, y: -10, z: 160 },
            { w: 800, h: 20, d: 1120 },
            this._mat(0x14182a),
            false
        );

        // Far section: past gap → end
        this._box(
            { x: 0, y: -10, z: 1100 },
            { w: 800, h: 20, d: 660 },
            this._mat(0x14182a),
            false
        );
    }

    // ── OBSTACLE 1: Low Barrier ───────────────────────────────────────────────
    /** Glowing orange gate the player must crouch through. Wide enough for 1 person. */
    _addCrouchBarrier() {
        const centreZ   = 200;
        const depth     = 60;     // Z thickness
        const gateW     = 200;    // width of the passage
        const beamH     = 85;     // height of bottom of top beam (player must duck under)
        const beamThick = 18;
        const postH     = beamH + beamThick;
        const totalW    = 600;    // full wall width inc. solid side walls
        const sideW     = (totalW - gateW) / 2;

        // ── Glowing top beam (NOT solid — lowBarrier handles clearance)
        this._emissiveBox(
            { x: 0, y: beamH + beamThick / 2, z: centreZ },
            { w: gateW, h: beamThick, d: depth },
            0xff6020, 0.8, false
        );

        // ── Left solid wall (fills left of gate)
        this._emissiveBox(
            { x: -(gateW / 2 + sideW / 2), y: postH / 2, z: centreZ },
            { w: sideW, h: postH, d: depth },
            0xcc4010, 0.3
        );

        // ── Right solid wall
        this._emissiveBox(
            { x:  (gateW / 2 + sideW / 2), y: postH / 2, z: centreZ },
            { w: sideW, h: postH, d: depth },
            0xcc4010, 0.3
        );

        // Physics: side walls block going around
        this.colliders.push(
            { cx: -(gateW/2+sideW/2), cy: postH/2, cz: centreZ, hw: sideW/2, hh: postH/2, hd: depth/2 },
            { cx:  (gateW/2+sideW/2), cy: postH/2, cz: centreZ, hw: sideW/2, hh: postH/2, hd: depth/2 }
        );

        // Physics: low-barrier zone blocks upright avatar
        this.lowBarriers.push({
            minZ: centreZ - depth / 2,
            maxZ: centreZ + depth / 2,
            clearanceY: beamH,
        });

        // Neon floor strip under gate (decoration, non-solid)
        this._emissiveBox(
            { x: 0, y: 1, z: centreZ },
            { w: gateW, h: 2, d: depth },
            0xff8040, 0.6, false
        );
    }

    // ── OBSTACLE 2: Stagger Walls ─────────────────────────────────────────────
    /** Two offset walls forcing the player to zigzag — requires walking/turning */
    _addStaggerWalls() {
        const wallH = 200;
        const wallD = 50;
        const wallW = 200;

        // Left wall — blocks left side at Z=420, player must pass RIGHT
        // Wall spans x: -300 to -100 → clear passage from x=-100 onward
        this._emissiveBox(
            { x: -200, y: wallH / 2, z: 420 },
            { w: wallW, h: wallH, d: wallD },
            0x1166cc, 0.25
        );
        // Right wall — blocks right side at Z=560, player must pass LEFT
        // Wall spans x: 100 to 300 → clear passage from x=100 onward
        this._emissiveBox(
            { x:  200, y: wallH / 2, z: 560 },
            { w: wallW, h: wallH, d: wallD },
            0x1166cc, 0.25
        );

        // Boundary walls (invisible, keep player in lane)
        this.colliders.push(
            { cx: -410, cy: 200, cz: 800, hw: 10, hh: 400, hd: 1600 },
            { cx:  410, cy: 200, cz: 800, hw: 10, hh: 400, hd: 1600 }
        );
    }

    // ── OBSTACLE 3: Gap ───────────────────────────────────────────────────────
    /** Floor disappears — player must jump across */
    _addGap() {
        const gapMinZ = 720;
        const gapMaxZ = 800;   // 80-unit gap — comfortably jumpable
        this.gaps.push({ minZ: gapMinZ, maxZ: gapMaxZ });

        // Glowing near edge (non-solid — just decoration)
        this._emissiveBox({ x: 0, y: 2, z: gapMinZ - 3 }, { w: 700, h: 4, d: 6 }, 0x00d4ff, 1.0, false);
        // Glowing far edge
        this._emissiveBox({ x: 0, y: 2, z: gapMaxZ + 3 }, { w: 700, h: 4, d: 6 }, 0x00d4ff, 1.0, false);

        // Pit bottom (way below, dark red)
        const pitMat = new THREE.MeshLambertMaterial({ color: 0x1a0000 });
        const pit = new THREE.Mesh(new THREE.PlaneGeometry(700, 200), pitMat);
        pit.rotation.x = -Math.PI / 2;
        pit.position.set(0, -400, (gapMinZ + gapMaxZ) / 2);
        this._add(pit);

        // Arrow chevrons pointing "jump here" (non-solid — decoration only)
        for (let z = gapMinZ - 60; z >= gapMinZ - 200; z -= 60) {
            this._emissiveBox({ x: 0, y: 3, z }, { w: 80, h: 3, d: 8 }, 0x00aaff, 0.5, false);
        }
    }

    // ── OBSTACLE 4: Ramp ──────────────────────────────────────────────────────
    /** Inclined ramp rising to an elevated platform */
    _addRamp() {
        const rampMinZ   = 950;
        const rampMaxZ   = 1150;
        const rampHeight = 130;
        const rampWidth  = 320;

        // Compute ramp geometry
        const rampLen   = Math.sqrt(
            Math.pow(rampMaxZ - rampMinZ, 2) + Math.pow(rampHeight, 2)
        );
        const rampAngle = -Math.atan2(rampHeight, rampMaxZ - rampMinZ);

        // Ramp surface — green emissive tint
        const rampMat = new THREE.MeshLambertMaterial({ color: 0x004422, emissive: new THREE.Color(0x00ff88), emissiveIntensity: 0.12 });
        const rampMesh = new THREE.Mesh(new THREE.BoxGeometry(rampWidth, 14, rampLen), rampMat);
        rampMesh.rotation.x = rampAngle;
        rampMesh.position.set(0, rampHeight / 2, (rampMinZ + rampMaxZ) / 2);
        rampMesh.castShadow = rampMesh.receiveShadow = true;
        this._add(rampMesh);

        // Ramp side rails (decoration)
        for (const sx of [-rampWidth / 2 + 8, rampWidth / 2 - 8]) {
            const rail = new THREE.Mesh(new THREE.BoxGeometry(6, 20, rampLen), new THREE.MeshLambertMaterial({ color: 0x00ff88, emissive: new THREE.Color(0x00ff88), emissiveIntensity: 0.4 }));
            rail.rotation.x = rampAngle;
            rail.position.set(sx, rampHeight / 2 + 12, (rampMinZ + rampMaxZ) / 2);
            this._add(rail);
        }

        // Elevated platform on top
        this._emissiveBox(
            { x: 0, y: rampHeight + 8, z: 1350 },
            { w: rampWidth, h: 16, d: 400 },
            0x002233, 0.05
        );

        // Platform edge glow (non-solid)
        this._emissiveBox({ x: 0, y: rampHeight + 18, z: 1150 }, { w: rampWidth, h: 4, d: 8 }, 0x00ff88, 0.8, false);

        // Physics: ramp slope
        this.ramps.push({
            minZ: rampMinZ, maxZ: rampMaxZ,
            rampHeight,
            minX: -rampWidth / 2, maxX: rampWidth / 2,
        });
        // Physics: flat platform
        this.ramps.push({
            minZ: rampMaxZ, maxZ: 1600,
            rampHeight,
            minX: -rampWidth / 2, maxX: rampWidth / 2,
            flat: true,
        });

        // Finish line glow at end of platform (non-solid)
        this._emissiveBox({ x: 0, y: rampHeight + 19, z: 1560 }, { w: rampWidth, h: 4, d: 14 }, 0xffdd00, 1.0, false);
        // "FINISH" label handled by _addLabels
    }

    // ── Labels ────────────────────────────────────────────────────────────────
    _addLabels() {
        const labels = [
            { text: '↓ CROUCH',   z: 160,  y: 140, color: '#ff7040' },
            { text: '↔ NAVIGATE', z: 480,  y: 300, color: '#4488ff' },
            { text: '↑ JUMP',     z: 760,  y: 100, color: '#00d4ff' },
            { text: '↗ WALK UP',  z: 1050, y: 230, color: '#00ff88' },
            { text: '★ FINISH',   z: 1560, y: 210, color: '#ffdd00' },
        ];

        labels.forEach(({ text, z, y, color }) => {
            const canvas  = document.createElement('canvas');
            canvas.width  = 512; canvas.height = 96;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, 512, 96);
            ctx.font         = 'bold 36px Inter, Arial, sans-serif';
            ctx.fillStyle    = color;
            ctx.textAlign    = 'center';
            ctx.textBaseline = 'middle';
            // Subtle dark backdrop
            ctx.shadowColor  = 'rgba(0,0,0,0.8)';
            ctx.shadowBlur   = 12;
            ctx.fillText(text, 256, 48);

            const tex    = new THREE.CanvasTexture(canvas);
            const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
            sprite.scale.set(200, 38, 1);
            sprite.position.set(0, y, z);
            this._add(sprite);
        });
    }

    // ─── Collision API (called by ActionController each frame) ────────────────

    /** Returns the Y surface height at world (x, z). 0 = flat ground. */
    getGroundY(x, z) {
        for (const r of this.ramps) {
            if (z < r.minZ || z > r.maxZ) continue;
            if (x < r.minX || x > r.maxX) continue;
            if (r.flat) return r.rampHeight;
            const t = (z - r.minZ) / (r.maxZ - r.minZ);
            return t * r.rampHeight;
        }
        return 0;
    }

    /** True if the avatar is over a gap (should fall). */
    isOverGap(x, z) {
        return this.gaps.some(g => z >= g.minZ && z <= g.maxZ);
    }

    /**
     * Max allowed avatar height at (x, z) based on low barriers.
     * Returns Infinity when no barrier present.
     */
    getMaxAvatarHeight(x, z) {
        for (const b of this.lowBarriers) {
            if (z >= b.minZ && z <= b.maxZ) return b.clearanceY;
        }
        return Infinity;
    }

    /**
     * Push avatar out of any solid AABB collider (XZ plane only).
     * @param {THREE.Vector3} pos
     * @param {number} hw - avatar half-width
     * @param {number} h  - avatar height
     */
    resolveCollisions(pos, hw = 22, h = 150) {
        for (const c of this.colliders) {
            const minX = c.cx - c.hw,  maxX = c.cx + c.hw;
            const minY = c.cy - c.hh,  maxY = c.cy + c.hh;
            const minZ = c.cz - c.hd,  maxZ = c.cz + c.hd;

            const aMinX = pos.x - hw, aMaxX = pos.x + hw;
            const aMinY = pos.y,      aMaxY = pos.y + h;
            const aMinZ = pos.z - hw, aMaxZ = pos.z + hw;

            if (aMaxX < minX || aMinX > maxX) continue;
            if (aMaxY < minY || aMinY > maxY) continue;
            if (aMaxZ < minZ || aMinZ > maxZ) continue;

            const ox1 = aMaxX - minX, ox2 = maxX - aMinX;
            const oz1 = aMaxZ - minZ, oz2 = maxZ - aMinZ;
            const min = Math.min(ox1, ox2, oz1, oz2);

            if      (min === ox1) pos.x -= ox1;
            else if (min === ox2) pos.x += ox2;
            else if (min === oz1) pos.z -= oz1;
            else                  pos.z += oz2;
        }
    }

    // ─── Scene cleanup ────────────────────────────────────────────────────────
    removeFromScene() {
        this._meshes.forEach(m => this.scene.remove(m));
        this._meshes = [];
        this.colliders   = [];
        this.gaps        = [];
        this.lowBarriers = [];
        this.ramps       = [];
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    /** Box mesh + optional collider */
    _box(centre, size, mat, solid = true) {
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(size.w, size.h, size.d), mat);
        mesh.position.set(centre.x, centre.y, centre.z);
        mesh.castShadow = mesh.receiveShadow = true;
        this._add(mesh);
        if (solid) this.colliders.push({
            cx: centre.x, cy: centre.y, cz: centre.z,
            hw: size.w/2,  hh: size.h/2,  hd: size.d/2,
        });
    }

    /** Box with emissive glow — automatically adds a collider */
    _emissiveBox(centre, size, color, emissiveIntensity = 0.5, solid = true) {
        const mat = new THREE.MeshLambertMaterial({
            color,
            emissive: new THREE.Color(color),
            emissiveIntensity,
        });
        this._box(centre, size, mat, solid);
    }

    _add(obj) { this.scene.add(obj); this._meshes.push(obj); }

    _mat(color, opacity = 1) {
        return new THREE.MeshLambertMaterial({ color, transparent: opacity < 1, opacity });
    }
}

// ─── EnvironmentManager ───────────────────────────────────────────────────────
class EnvironmentManager {
    static REGISTRY = {
        'obstacle_course': ObstacleCourseEnvironment,
        // Register more here: 'flat': FlatEnvironment
    };

    constructor(scene) {
        this.scene   = scene;
        this.current = null;
    }

    /** Load environment by key, disposing the current one first. */
    load(key) {
        if (this.current) this.current.removeFromScene();
        const Cls = EnvironmentManager.REGISTRY[key];
        if (!Cls) throw new Error(`Unknown environment: "${key}"`);
        this.current = new Cls(this.scene);
        return this.current;
    }
}
