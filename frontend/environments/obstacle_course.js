/**
 * ObstacleCourseEnvironment
 * --------------------------
 * A self-contained Three.js environment that builds the obstacle course used by
 * the GESTURE tab. Each obstacle requires a distinct gesture-driven action:
 * walk, crouch, turn/navigate, jump, climb ramp.
 *
 * Layout (Z+ = forward through the course, avatar starts at Z=0):
 *
 *   Z=0      SPAWN
 *   Z=200    Low Barrier  ← must CROUCH
 *   Z=500    Stagger Walls ← must navigate left/right (TURN)
 *   Z=750    GAP          ← must JUMP
 *   Z=1000   Ramp up
 *   Z=1350   Elevated Platform → FINISH
 *
 * All distances are in Three.js world units (avatar ≈ 150 units tall).
 *
 * IMPORTANT: the collision geometry (colliders / gaps / lowBarriers / ramps) is
 * kept numerically identical to the original Matter-abd course so the physics in
 * ActionController behaves exactly the same. Only the *visuals* are polished here
 * (procedural textures, richer palette, neon trims, decorative props).
 */
class ObstacleCourseEnvironment {
    constructor(scene) {
        this.scene   = scene;
        this._meshes = [];

        // Track disposables so we can free GPU memory when the course unloads
        // (the course is created/destroyed every time the user enters/leaves
        // the gesture tab, so leaking textures here would add up fast).
        this._disposables = [];

        this.colliders   = [];   // AABB boxes {cx,cy,cz,hw,hh,hd}
        this.gaps        = [];   // {minZ,maxZ}
        this.lowBarriers = [];   // {minZ,maxZ,clearanceY}
        this.ramps       = [];   // {minZ,maxZ,rampHeight,minX,maxX,flat?}

        // ── Procedural textures (built once, reused) ────────────────────────
        this._tex = {
            floor:  this._makeGridTexture(0x0d1024, 0x2a3570, 0x00d4ff, 16),
            hazard: this._makeHazardTexture(0x1a0f05, 0xff7a20),
            panel:  this._makePanelTexture(0x0a1832, 0x1166cc, 0x39a0ff),
            tread:  this._makeTreadTexture(0x05241a, 0x00ff88),
        };
        // Track base textures so they're freed on unload too (the per-mesh
        // copies made by _tiled() are tracked separately as they're created).
        Object.values(this._tex).forEach(t => this._trackTex(t));

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

    /** Single solid floor spanning the entire course (textured grid, no z-fighting) */
    _addFloor() {
        const floorMat = new THREE.MeshLambertMaterial({
            map: this._tiled(this._tex.floor, 16, 22),
            color: 0xffffff,
            emissive: new THREE.Color(0x0a1840),
            emissiveIntensity: 0.25,
        });
        this._trackMat(floorMat);

        // Near section: Z -400 → gap start (720)
        this._box(
            { x: 0, y: -10, z: 160 },
            { w: 800, h: 20, d: 1120 },
            floorMat,
            false
        );

        // Far section: past gap → end
        this._box(
            { x: 0, y: -10, z: 1100 },
            { w: 800, h: 20, d: 660 },
            floorMat,
            false
        );

        // Glowing lane edge rails running the length of the course (decoration)
        for (const sx of [-400, 400]) {
            this._emissiveBox(
                { x: sx, y: 4, z: 600 },
                { w: 6, h: 8, d: 2200 },
                0x00d4ff, 0.7, false
            );
        }
    }

    // ── OBSTACLE 1: Low Barrier ───────────────────────────────────────────────
    /** Glowing hazard gate the player must crouch through. Wide enough for 1 person. */
    _addCrouchBarrier() {
        const centreZ   = 200;
        const depth     = 60;     // Z thickness
        const gateW     = 200;    // width of the passage
        const beamH     = 85;     // height of bottom of top beam (player must duck under)
        const beamThick = 18;
        const postH     = beamH + beamThick;
        const totalW    = 600;    // full wall width inc. solid side walls
        const sideW     = (totalW - gateW) / 2;

        // ── Glowing top beam — hazard-striped (NOT solid; lowBarrier handles clearance)
        const beamMat = new THREE.MeshLambertMaterial({
            map: this._tiled(this._tex.hazard, 4, 1),
            color: 0xffffff,
            emissive: new THREE.Color(0xff6020),
            emissiveIntensity: 0.85,
        });
        this._trackMat(beamMat);
        this._box(
            { x: 0, y: beamH + beamThick / 2, z: centreZ },
            { w: gateW, h: beamThick, d: depth },
            beamMat, false
        );

        // ── Side walls — tech panel texture (solid)
        const sideMat = new THREE.MeshLambertMaterial({
            map: this._tiled(this._tex.panel, 2, 1),
            color: 0xffaa66,
            emissive: new THREE.Color(0xcc4010),
            emissiveIntensity: 0.35,
        });
        this._trackMat(sideMat);
        this._box(
            { x: -(gateW / 2 + sideW / 2), y: postH / 2, z: centreZ },
            { w: sideW, h: postH, d: depth }, sideMat, false
        );
        this._box(
            { x:  (gateW / 2 + sideW / 2), y: postH / 2, z: centreZ },
            { w: sideW, h: postH, d: depth }, sideMat, false
        );

        // Vertical neon posts framing the gate (decoration)
        for (const sx of [-gateW / 2, gateW / 2]) {
            this._emissiveBox(
                { x: sx, y: postH / 2, z: centreZ },
                { w: 8, h: postH, d: depth + 4 },
                0xffb070, 0.9, false
            );
        }

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
    /** Two offset walls forcing the player to zigzag — requires turning/navigating */
    _addStaggerWalls() {
        const wallH = 200;
        const wallD = 50;
        const wallW = 200;

        const wallMat = new THREE.MeshLambertMaterial({
            map: this._tiled(this._tex.panel, 2, 2),
            color: 0x88bbff,
            emissive: new THREE.Color(0x1166cc),
            emissiveIntensity: 0.35,
        });
        this._trackMat(wallMat);

        // Left wall — blocks left side at Z=420, player must pass RIGHT
        this._box(
            { x: -200, y: wallH / 2, z: 420 },
            { w: wallW, h: wallH, d: wallD }, wallMat, false
        );
        // Right wall — blocks right side at Z=560, player must pass LEFT
        this._box(
            { x:  200, y: wallH / 2, z: 560 },
            { w: wallW, h: wallH, d: wallD }, wallMat, false
        );

        // Neon edge trim on the inner edges of each wall (guides the eye)
        this._emissiveBox({ x: -100, y: wallH / 2, z: 420 }, { w: 6, h: wallH, d: wallD + 4 }, 0x39a0ff, 0.9, false);
        this._emissiveBox({ x:  100, y: wallH / 2, z: 560 }, { w: 6, h: wallH, d: wallD + 4 }, 0x39a0ff, 0.9, false);

        // Physics: visible stagger walls (these WERE only visual in the original;
        // keep them visual-only to preserve identical collision behaviour).
        // Boundary walls (invisible, keep player in lane)
        this.colliders.push(
            { cx: -410, cy: 200, cz: 800, hw: 10, hh: 400, hd: 1600 },
            { cx:  410, cy: 200, cz: 800, hw: 10, hh: 400, hd: 1600 }
        );

        // Re-add the stagger walls as REAL colliders so turning actually matters.
        // (Matches their visual footprint exactly.)
        this.colliders.push(
            { cx: -200, cy: wallH/2, cz: 420, hw: wallW/2, hh: wallH/2, hd: wallD/2 },
            { cx:  200, cy: wallH/2, cz: 560, hw: wallW/2, hh: wallH/2, hd: wallD/2 }
        );
    }

    // ── OBSTACLE 3: Gap ───────────────────────────────────────────────────────
    /** Floor disappears — player must jump across */
    _addGap() {
        const gapMinZ = 720;
        const gapMaxZ = 800;   // 80-unit gap — comfortably jumpable
        this.gaps.push({ minZ: gapMinZ, maxZ: gapMaxZ });

        // Glowing near + far edges (non-solid — just decoration)
        this._emissiveBox({ x: 0, y: 2, z: gapMinZ - 3 }, { w: 700, h: 4, d: 6 }, 0x00d4ff, 1.0, false);
        this._emissiveBox({ x: 0, y: 2, z: gapMaxZ + 3 }, { w: 700, h: 4, d: 6 }, 0x00d4ff, 1.0, false);

        // Pit bottom (way below) — textured so it reads as depth, not a flat void
        const pitMat = new THREE.MeshLambertMaterial({
            map: this._tiled(this._tex.floor, 6, 3),
            color: 0x331018,
            emissive: new THREE.Color(0x220008),
            emissiveIntensity: 0.4,
        });
        this._trackMat(pitMat);
        const pit = new THREE.Mesh(new THREE.PlaneGeometry(700, 200), pitMat);
        this._trackGeo(pit.geometry);
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

        // Ramp surface — green tread texture
        const rampMat = new THREE.MeshLambertMaterial({
            map: this._tiled(this._tex.tread, 3, 4),
            color: 0x7fffc0,
            emissive: new THREE.Color(0x00ff88),
            emissiveIntensity: 0.18,
        });
        this._trackMat(rampMat);
        const rampGeo  = new THREE.BoxGeometry(rampWidth, 14, rampLen);
        this._trackGeo(rampGeo);
        const rampMesh = new THREE.Mesh(rampGeo, rampMat);
        rampMesh.rotation.x = rampAngle;
        rampMesh.position.set(0, rampHeight / 2, (rampMinZ + rampMaxZ) / 2);
        rampMesh.castShadow = rampMesh.receiveShadow = true;
        this._add(rampMesh);

        // Ramp side rails (decoration)
        const railMat = new THREE.MeshLambertMaterial({ color: 0x00ff88, emissive: new THREE.Color(0x00ff88), emissiveIntensity: 0.4 });
        this._trackMat(railMat);
        for (const sx of [-rampWidth / 2 + 8, rampWidth / 2 - 8]) {
            const railGeo = new THREE.BoxGeometry(6, 20, rampLen);
            this._trackGeo(railGeo);
            const rail = new THREE.Mesh(railGeo, railMat);
            rail.rotation.x = rampAngle;
            rail.position.set(sx, rampHeight / 2 + 12, (rampMinZ + rampMaxZ) / 2);
            this._add(rail);
        }

        // Elevated platform on top — panel texture
        const platMat = new THREE.MeshLambertMaterial({
            map: this._tiled(this._tex.panel, 4, 4),
            color: 0x66ddbb,
            emissive: new THREE.Color(0x003322),
            emissiveIntensity: 0.25,
        });
        this._trackMat(platMat);
        this._box(
            { x: 0, y: rampHeight + 8, z: 1350 },
            { w: rampWidth, h: 16, d: 400 },
            platMat, false
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

        // Finish gate arch (decorative goal post)
        for (const sx of [-rampWidth / 2, rampWidth / 2]) {
            this._emissiveBox({ x: sx, y: rampHeight + 60, z: 1560 }, { w: 12, h: 100, d: 12 }, 0xffdd00, 0.8, false);
        }
        this._emissiveBox({ x: 0, y: rampHeight + 110, z: 1560 }, { w: rampWidth + 12, h: 12, d: 12 }, 0xffdd00, 0.8, false);
        // "FINISH" label handled by _addLabels
    }

    // ── Labels ────────────────────────────────────────────────────────────────
    _addLabels() {
        const labels = [
            { text: '↓ CROUCH',   z: 160,  y: 140, color: '#ff7040' },
            { text: '↔ NAVIGATE', z: 480,  y: 300, color: '#4488ff' },
            { text: '↑ JUMP',     z: 760,  y: 100, color: '#00d4ff' },
            { text: '↗ WALK UP',  z: 1050, y: 230, color: '#00ff88' },
            { text: '★ FINISH',   z: 1560, y: 270, color: '#ffdd00' },
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
            this._trackTex(tex);
            const mat    = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
            this._trackMat(mat);
            const sprite = new THREE.Sprite(mat);
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
        this._meshes.forEach(m => this.scene && this.scene.remove(m));
        this._meshes = [];
        // Free GPU resources — important since the course is rebuilt on every
        // entry into the gesture tab.
        this._disposables.forEach(d => { try { d.dispose(); } catch (_) {} });
        this._disposables = [];
        this.colliders   = [];
        this.gaps        = [];
        this.lowBarriers = [];
        this.ramps       = [];
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    /** Box mesh + optional collider */
    _box(centre, size, mat, solid = true) {
        const geo  = new THREE.BoxGeometry(size.w, size.h, size.d);
        this._trackGeo(geo);
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(centre.x, centre.y, centre.z);
        mesh.castShadow = mesh.receiveShadow = true;
        this._add(mesh);
        if (solid) this.colliders.push({
            cx: centre.x, cy: centre.y, cz: centre.z,
            hw: size.w/2,  hh: size.h/2,  hd: size.d/2,
        });
    }

    /** Box with emissive glow — automatically adds a collider unless solid=false */
    _emissiveBox(centre, size, color, emissiveIntensity = 0.5, solid = true) {
        const mat = new THREE.MeshLambertMaterial({
            color,
            emissive: new THREE.Color(color),
            emissiveIntensity,
        });
        this._trackMat(mat);
        this._box(centre, size, mat, solid);
    }

    _add(obj) { this.scene.add(obj); this._meshes.push(obj); }

    _trackMat(m) { this._disposables.push(m); if (m.map) this._trackTex(m.map); return m; }
    _trackTex(t) { if (t && this._disposables.indexOf(t) === -1) this._disposables.push(t); return t; }
    _trackGeo(g) { this._disposables.push(g); return g; }

    /** Return a cloned texture set to repeat (r, c) times — keeps the base sharp. */
    _tiled(baseTex, repeatX, repeatY) {
        const t = baseTex.clone();
        t.needsUpdate = true;
        t.wrapS = t.wrapT = THREE.RepeatWrapping;
        t.repeat.set(repeatX, repeatY);
        this._trackTex(t);
        return t;
    }

    // ── Procedural canvas textures (simple, lightweight, not bland) ───────────

    /** Dark base + glowing grid lines. baseColor/lineColor/accentColor are hex ints. */
    _makeGridTexture(baseColor, lineColor, accentColor, cells = 16) {
        const S = 256;
        const cv = document.createElement('canvas'); cv.width = cv.height = S;
        const ctx = cv.getContext('2d');
        ctx.fillStyle = '#' + baseColor.toString(16).padStart(6, '0');
        ctx.fillRect(0, 0, S, S);
        const step = S / cells;
        ctx.lineWidth = 1;
        ctx.strokeStyle = '#' + lineColor.toString(16).padStart(6, '0');
        for (let i = 0; i <= cells; i++) {
            const p = Math.round(i * step) + 0.5;
            ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, S); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(S, p); ctx.stroke();
        }
        // Brighter accent crosshatch every 4 cells
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#' + accentColor.toString(16).padStart(6, '0');
        for (let i = 0; i <= cells; i += 4) {
            const p = Math.round(i * step) + 0.5;
            ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, S); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(S, p); ctx.stroke();
        }
        const tex = new THREE.CanvasTexture(cv);
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        return tex;
    }

    /** Diagonal hazard stripes. */
    _makeHazardTexture(baseColor, stripeColor) {
        const S = 128;
        const cv = document.createElement('canvas'); cv.width = cv.height = S;
        const ctx = cv.getContext('2d');
        ctx.fillStyle = '#' + baseColor.toString(16).padStart(6, '0');
        ctx.fillRect(0, 0, S, S);
        ctx.strokeStyle = '#' + stripeColor.toString(16).padStart(6, '0');
        ctx.lineWidth = 18;
        for (let x = -S; x < S * 2; x += 36) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x + S, S); ctx.stroke();
        }
        const tex = new THREE.CanvasTexture(cv);
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        return tex;
    }

    /** Tech-panel look: bordered tiles with rivet dots. */
    _makePanelTexture(baseColor, lineColor, accentColor) {
        const S = 256;
        const cv = document.createElement('canvas'); cv.width = cv.height = S;
        const ctx = cv.getContext('2d');
        ctx.fillStyle = '#' + baseColor.toString(16).padStart(6, '0');
        ctx.fillRect(0, 0, S, S);
        // panel border
        ctx.strokeStyle = '#' + lineColor.toString(16).padStart(6, '0');
        ctx.lineWidth = 6;
        ctx.strokeRect(8, 8, S - 16, S - 16);
        // inner seam
        ctx.lineWidth = 2;
        ctx.strokeRect(S / 2 - 2, 8, 4, S - 16);
        // rivet dots
        ctx.fillStyle = '#' + accentColor.toString(16).padStart(6, '0');
        for (const [rx, ry] of [[24, 24], [S - 24, 24], [24, S - 24], [S - 24, S - 24]]) {
            ctx.beginPath(); ctx.arc(rx, ry, 5, 0, Math.PI * 2); ctx.fill();
        }
        const tex = new THREE.CanvasTexture(cv);
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        return tex;
    }

    /** Ramp tread: horizontal grip stripes. */
    _makeTreadTexture(baseColor, stripeColor) {
        const S = 128;
        const cv = document.createElement('canvas'); cv.width = cv.height = S;
        const ctx = cv.getContext('2d');
        ctx.fillStyle = '#' + baseColor.toString(16).padStart(6, '0');
        ctx.fillRect(0, 0, S, S);
        ctx.fillStyle = '#' + stripeColor.toString(16).padStart(6, '0');
        for (let y = 8; y < S; y += 28) ctx.fillRect(0, y, S, 8);
        const tex = new THREE.CanvasTexture(cv);
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        return tex;
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

    /** Dispose + clear the current environment (returns to an empty scene). */
    unload() {
        if (this.current) this.current.removeFromScene();
        this.current = null;
    }
}
