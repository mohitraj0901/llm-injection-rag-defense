"""
Post-Aware Calibration (PAC) from LATPC
Reference: Yi et al. (Expert Systems With Applications 296, 129101, 2026).
"""
import numpy as np

class PostAwareCalibratorNumpy:
    """
    Numpy implementation of Post-Aware Calibration.
    Shifts benign query latent representations away from the over-refusal boundary.
    """
    def __init__(self, hidden_dim=128, p0=0.5, delta=0.15):
        self.hidden_dim = hidden_dim
        self.p0 = p0
        self.delta = delta
        # Probe direction vector E (unit normalized)
        np.random.seed(42)
        w = np.random.randn(hidden_dim)
        self.e_dir = w / (np.linalg.norm(w) + 1e-8)

    def calibrate(self, h_q, is_safe_intent=True):
        """
        h_q: Latent vector of shape (hidden_dim,)
        """
        # Refusal score
        refusal_score = 1.0 / (1.0 + np.exp(-np.dot(self.e_dir, h_q)))
        applied = False
        calibrated_h = h_q.copy()
        
        # If model exhibits latent over-refusal bias on safe intent, calibrate
        if is_safe_intent and refusal_score > self.p0:
            calibrated_h = calibrated_h + self.delta * self.e_dir
            applied = True
            
        return calibrated_h, applied, float(refusal_score)
