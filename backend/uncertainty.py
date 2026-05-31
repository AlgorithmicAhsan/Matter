import numpy as np

class UncertaintyScorer:
    """
    Reproduces the uncertainty mechanism from Paper 2:
    'Uncertainty-Aware Mapping from 3D Keypoints to Anatomical Landmarks'

    Two types of uncertainty:
    - Aleatoric: irreducible noise from the observation (uses MediaPipe visibility scores)
    - Epistemic: model uncertainty from not seeing enough of a pose (simulated via Monte Carlo)

    Total uncertainty = aleatoric + epistemic
    If total uncertainty > threshold -> frame is untrusted -> avatar holds last pose
    """

    def __init__(self, mc_passes=20, noise_std=0.01, threshold=0.7):
        """
        mc_passes  : number of Monte Carlo passes to simulate epistemic uncertainty
        noise_std  : std of Gaussian noise injected per MC pass (in normalized coords)
        threshold  : total uncertainty above this = untrusted frame
        """
        self.mc_passes = mc_passes
        self.noise_std = noise_std
        self.threshold = threshold

        # History of uncertainty scores for visualization
        self.history = []

    def aleatoric(self, landmarks):
        """
        Aleatoric uncertainty = how noisy/ambiguous is this observation.
        MediaPipe gives a visibility score (0 to 1) per landmark.
        Low visibility = high aleatoric uncertainty.
        We invert and average across all landmarks.
        """
        if landmarks is None:
            return 1.0  # maximum uncertainty if no pose detected

        visibilities = [lm["visibility"] for lm in landmarks]
        avg_visibility = np.mean(visibilities)

        # Invert: high visibility = low uncertainty
        aleatoric_score = 1.0 - avg_visibility
        return float(aleatoric_score)

    def epistemic(self, landmarks):
        """
        Epistemic uncertainty = how uncertain is the model about this pose.
        Simulated via Monte Carlo: inject small Gaussian noise into landmarks
        multiple times and measure variance across passes.
        High variance = model is uncertain about this configuration.
        """
        if landmarks is None:
            return 1.0

        # Stack landmarks into array shape (33, 3) for x, y, z
        coords = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks])

        # Run mc_passes with small noise injected each time
        mc_predictions = []
        for _ in range(self.mc_passes):
            noise = np.random.normal(0, self.noise_std, coords.shape)
            perturbed = coords + noise
            mc_predictions.append(perturbed)

        # Stack into (mc_passes, 33, 3)
        mc_predictions = np.stack(mc_predictions, axis=0)

        # Variance across passes per joint per axis, then mean across all
        variance = np.var(mc_predictions, axis=0)  # shape (33, 3)
        epistemic_score = float(np.mean(variance))

        # Normalize to roughly 0-1 range (variance is tiny in normalized coords)
        epistemic_score = min(epistemic_score * 1000, 1.0)

        return epistemic_score

    def score(self, landmarks):
        """
        Main method. Call once per frame with the landmarks from PoseExtractor.
        Returns a dict with:
          - aleatoric  : float 0-1
          - epistemic  : float 0-1
          - total      : float 0-1
          - trusted    : bool (True = use this frame, False = skip/hold last pose)
        """
        a = self.aleatoric(landmarks)
        e = self.epistemic(landmarks)
        total = a + e

        # Clamp to 0-1
        total = min(total, 1.0)

        trusted = total < self.threshold

        result = {
            "aleatoric": round(a, 4),
            "epistemic": round(e, 4),
            "total": round(total, 4),
            "trusted": trusted
        }

        self.history.append(result)
        return result

    def coverage_stats(self):
        """
        Risk-coverage analysis from Paper 2.
        Shows what % of frames are trusted at current threshold.
        """
        if not self.history:
            return {}

        total_frames = len(self.history)
        trusted_frames = sum(1 for r in self.history if r["trusted"])
        avg_uncertainty = np.mean([r["total"] for r in self.history])

        return {
            "total_frames": total_frames,
            "trusted_frames": trusted_frames,
            "coverage_pct": round(trusted_frames / total_frames * 100, 2),
            "avg_uncertainty": round(float(avg_uncertainty), 4)
        }


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import cv2
    from pose import PoseExtractor

    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open source: {source}")
        sys.exit(1)

    extractor = PoseExtractor()
    scorer = UncertaintyScorer()

    print("Running uncertainty scoring. Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, annotated, _ = extractor.process_frame(frame)
        result = scorer.score(landmarks)

        # Display uncertainty info on frame
        color = (0, 255, 0) if result["trusted"] else (0, 0, 255)
        label = "TRUSTED" if result["trusted"] else "UNTRUSTED"

        cv2.putText(annotated, f"{label}  uncertainty={result['total']:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(annotated, f"aleatoric={result['aleatoric']:.3f}  epistemic={result['epistemic']:.3f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Print to terminal
        print(f"Frame {len(scorer.history):04d} | "
              f"total={result['total']:.3f} | "
              f"trusted={result['trusted']} | "
              f"aleatoric={result['aleatoric']:.3f} | "
              f"epistemic={result['epistemic']:.3f}")

        cv2.imshow("Uncertainty Scoring", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print final coverage stats (Paper 2 risk-coverage analysis)
    stats = scorer.coverage_stats()
    print("\n── Coverage Stats (Paper 2) ──")
    print(f"Total frames    : {stats['total_frames']}")
    print(f"Trusted frames  : {stats['trusted_frames']}")
    print(f"Coverage        : {stats['coverage_pct']}%")
    print(f"Avg uncertainty : {stats['avg_uncertainty']}")