class UncertaintyUI {
    constructor() {
        this.historyLimit = 100;
        this.uncertaintyHistory = [];
        this.graphCanvas = document.getElementById('uncertaintyGraph');
        this.graphCtx    = this.graphCanvas.getContext('2d');

        this.graphCanvas.width  = this.graphCanvas.offsetWidth;
        this.graphCanvas.height = this.graphCanvas.offsetHeight;

        new ResizeObserver(() => {
            this.graphCanvas.width  = this.graphCanvas.offsetWidth;
            this.graphCanvas.height = this.graphCanvas.offsetHeight;
            this.drawGraph();
        }).observe(this.graphCanvas);
    }

    update(uncertainty) {
        document.getElementById('aleatoricVal').textContent = uncertainty.aleatoric.toFixed(3);
        document.getElementById('epistemicVal').textContent = uncertainty.epistemic.toFixed(3);
        document.getElementById('totalVal').textContent     = uncertainty.total.toFixed(3);

        this._updateBar('aleatoric', uncertainty.aleatoric);
        this._updateBar('epistemic', uncertainty.epistemic);
        this._updateBar('total',     uncertainty.total);
        this._updateTrustBadge(uncertainty.trusted);

        this.uncertaintyHistory.push(uncertainty.total);
        if (this.uncertaintyHistory.length > this.historyLimit) this.uncertaintyHistory.shift();
        this.drawGraph();
    }

    _updateBar(type, value) {
        const barEl = document.getElementById(type + 'Bar');
        if (!barEl) return;
        barEl.style.width      = Math.min(value * 100, 100) + '%';
        barEl.style.background = this._colorForValue(value);
    }

    _colorForValue(value) {
        if (value < 0.4)  return 'linear-gradient(90deg, #00ff88, #00dd77)';
        if (value < 0.65) return 'linear-gradient(90deg, #ffcc00, #ffaa00)';
        return 'linear-gradient(90deg, #ff5555, #ff3232)';
    }

    _updateTrustBadge(trusted) {
        const badge = document.getElementById('trustBadge');
        if (!badge) return;
        if (trusted) {
            badge.textContent = '✓ TRUSTED';
            badge.className   = 'trusted';
        } else {
            badge.textContent = '✗ UNTRUSTED';
            badge.className   = 'untrusted';
        }
    }

    drawGraph() {
        const canvas = this.graphCanvas;
        const ctx    = this.graphCtx;
        const w = canvas.width, h = canvas.height;
        const data = this.uncertaintyHistory;

        ctx.fillStyle = 'rgba(15,15,20,.5)';
        ctx.fillRect(0, 0, w, h);
        this._drawGrid(ctx, w, h);

        if (data.length < 1) return;

        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth   = 2;
        ctx.lineJoin    = 'round';
        ctx.beginPath();
        for (let i = 0; i < data.length; i++) {
            const x = (i / Math.max(data.length - 1, 1)) * w;
            const y = h - data[i] * h;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();

        ctx.lineTo(w, h);
        ctx.lineTo(0, h);
        ctx.closePath();
        ctx.fillStyle = 'rgba(0,212,255,.1)';
        ctx.fill();

        const last  = data[data.length - 1];
        const lastY = h - last * h;
        ctx.fillStyle = '#00d4ff';
        ctx.beginPath();
        ctx.arc(w - 4, lastY, 4, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle     = '#00d4ff';
        ctx.font          = '11px monospace';
        ctx.textAlign     = 'right';
        ctx.textBaseline  = 'bottom';
        ctx.fillText(last.toFixed(2), w - 8, lastY - 8);

        // Threshold line
        ctx.strokeStyle = 'rgba(255,50,50,.3)';
        ctx.lineWidth   = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        const ty = h - 0.7 * h;
        ctx.moveTo(0, ty);
        ctx.lineTo(w, ty);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle    = 'rgba(255,50,50,.5)';
        ctx.font         = '10px monospace';
        ctx.textAlign    = 'right';
        ctx.textBaseline = 'bottom';
        ctx.fillText('threshold (0.7)', w - 4, ty - 2);
    }

    _drawGrid(ctx, w, h) {
        ctx.strokeStyle = 'rgba(0,212,255,.1)';
        ctx.lineWidth   = 1;
        for (let i = 0; i <= 5; i++) {
            const y = (i / 5) * h;
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
            ctx.fillStyle    = 'rgba(0,212,255,.3)';
            ctx.font         = '9px monospace';
            ctx.textAlign    = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText((1 - i / 5).toFixed(1), w - 4, y);
        }
        for (let i = 0; i <= 4; i++) {
            const x = (i / 4) * w;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }
    }
}
