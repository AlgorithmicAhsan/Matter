class UncertaintyUI {
    constructor() {
        this.historyLimit = 100;
        this.uncertaintyHistory = [];
        this.graphCanvas = document.getElementById('uncertainty-graph');
        this.graphCtx = this.graphCanvas.getContext('2d');

        // Set canvas size
        this.graphCanvas.width = this.graphCanvas.offsetWidth;
        this.graphCanvas.height = this.graphCanvas.offsetHeight;

        // Resize observer for canvas
        new ResizeObserver(() => {
            this.graphCanvas.width = this.graphCanvas.offsetWidth;
            this.graphCanvas.height = this.graphCanvas.offsetHeight;
            this.drawGraph();
        }).observe(this.graphCanvas);
    }

    update(uncertainty) {
        // Update uncertainty values
        document.getElementById('aleatoric-val').textContent = uncertainty.aleatoric.toFixed(3);
        document.getElementById('epistemic-val').textContent = uncertainty.epistemic.toFixed(3);
        document.getElementById('total-val').textContent = uncertainty.total.toFixed(3);

        // Update bars
        this.updateBar('aleatoric', uncertainty.aleatoric);
        this.updateBar('epistemic', uncertainty.epistemic);
        this.updateBar('total', uncertainty.total);

        // Update trust badge
        this.updateTrustBadge(uncertainty.trusted);

        // Add to history
        this.uncertaintyHistory.push(uncertainty.total);
        if (this.uncertaintyHistory.length > this.historyLimit) {
            this.uncertaintyHistory.shift();
        }

        // Draw graph
        this.drawGraph();
    }

    updateBar(type, value) {
        const barElement = document.getElementById(`${type}-bar`);
        const width = Math.min(value * 100, 100);
        const color = this.getColorForValue(value);

        barElement.style.width = width + '%';
        barElement.style.background = color;
    }

    getColorForValue(value) {
        if (value < 0.4) {
            return 'linear-gradient(90deg, #00ff88, #00dd77)';
        } else if (value < 0.65) {
            return 'linear-gradient(90deg, #ffcc00, #ffaa00)';
        } else {
            return 'linear-gradient(90deg, #ff5555, #ff3232)';
        }
    }

    updateTrustBadge(trusted) {
        const badge = document.getElementById('trust-badge');
        
        if (trusted) {
            badge.textContent = '✓ TRUSTED';
            badge.classList.remove('untrusted');
            badge.classList.add('trusted');
        } else {
            badge.textContent = '✗ UNTRUSTED';
            badge.classList.remove('trusted');
            badge.classList.add('untrusted');
        }
    }

    drawGraph() {
        const canvas = this.graphCanvas;
        const ctx = this.graphCtx;
        const width = canvas.width;
        const height = canvas.height;
        const data = this.uncertaintyHistory;

        // Clear canvas
        ctx.fillStyle = 'rgba(15, 15, 20, 0.5)';
        ctx.fillRect(0, 0, width, height);

        // Draw grid
        this.drawGrid(ctx, width, height);

        // Draw line graph
        if (data.length > 0) {
            ctx.strokeStyle = '#00d4ff';
            ctx.lineWidth = 2;
            ctx.lineJoin = 'round';
            ctx.beginPath();

            for (let i = 0; i < data.length; i++) {
                const x = (i / Math.max(data.length - 1, 1)) * width;
                const y = height - (data[i] * height);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }

            ctx.stroke();

            // Draw filled area under curve
            ctx.lineTo(width, height);
            ctx.lineTo(0, height);
            ctx.closePath();
            ctx.fillStyle = 'rgba(0, 212, 255, 0.1)';
            ctx.fill();

            // Draw current value indicator
            if (data.length > 0) {
                const lastValue = data[data.length - 1];
                const lastX = width;
                const lastY = height - (lastValue * height);

                ctx.fillStyle = '#00d4ff';
                ctx.beginPath();
                ctx.arc(lastX - 4, lastY, 4, 0, Math.PI * 2);
                ctx.fill();

                // Draw value label
                ctx.fillStyle = '#00d4ff';
                ctx.font = '11px monospace';
                ctx.textAlign = 'right';
                ctx.textBaseline = 'bottom';
                ctx.fillText(lastValue.toFixed(2), width - 8, lastY - 8);
            }
        }

        // Draw threshold line
        ctx.strokeStyle = 'rgba(255, 50, 50, 0.3)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        const thresholdY = height - (0.7 * height);
        ctx.moveTo(0, thresholdY);
        ctx.lineTo(width, thresholdY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw threshold label
        ctx.fillStyle = 'rgba(255, 50, 50, 0.5)';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'bottom';
        ctx.fillText('threshold (0.7)', width - 4, thresholdY - 2);
    }

    drawGrid(ctx, width, height) {
        ctx.strokeStyle = 'rgba(0, 212, 255, 0.1)';
        ctx.lineWidth = 1;

        // Horizontal grid lines
        const gridSteps = 5;
        for (let i = 0; i <= gridSteps; i++) {
            const y = (i / gridSteps) * height;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();

            // Grid labels
            const value = (1 - i / gridSteps).toFixed(1);
            ctx.fillStyle = 'rgba(0, 212, 255, 0.3)';
            ctx.font = '9px monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(value, width - 4, y);
        }

        // Vertical grid lines
        const verticalSteps = 4;
        for (let i = 0; i <= verticalSteps; i++) {
            const x = (i / verticalSteps) * width;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
    }
}
