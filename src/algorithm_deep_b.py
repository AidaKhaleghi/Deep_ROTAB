import numpy as np
import torch

from algorithm import soft_threshold
from b_network import BNet


def torch_soft_threshold(z, thresh):
    # Differentiable shrinkage operator (same as algorithm.soft_threshold)
    return torch.sign(z) * torch.relu(torch.abs(z) - thresh)


class ROTABDeepB:
    """ROTAB variant where the closed-form solve for b[t] is replaced by an
    online-trained ConvLSTM network.

    Per frame:
      1. S[t] = soft_threshold(D - X diag(b[t-1]) Y^T, lam') using the
         previous b.
      2. Forward pass with (D, D - S[t]) -> b[t]; this prediction is used by
         the algorithm. Loss = 0.5*||D - L(b) - S[t]||_F^2 + lam'*||S[t]||_1
         with S[t] held fixed, backward, one (or more) optimizer step(s) on
         this frame only (test-then-train: the weight update only affects
         future frames).
      3. RLS updates for X, Y run unchanged, using b[t] and S[t].

    The ConvLSTM hidden state carries across frames but is detached each frame
    (truncated backpropagation through time of length 1).
    """

    def __init__(self, init_frames, rank=5, mu=0.1, alpha=0.95, lam_prime=0.4,
                 lr=1e-3, train_steps=1, device=None):
        init_frames = np.asarray(init_frames)
        K, M, N = init_frames.shape
        self.R = rank
        self.mu = mu
        self.alpha = alpha
        self.lam_prime = lam_prime
        self.train_steps = train_steps

        # Same initialization as the baseline: truncated SVD of the mean frame
        D_init = np.mean(init_frames, axis=0)
        U, s, Vt = np.linalg.svd(D_init, full_matrices=False)
        self.X = U[:, :self.R].copy()  # M x R
        self.Y = Vt[:self.R, :].T.copy()  # N x R
        self.b = s[:self.R].copy()  # R

        self.RX = self.mu * np.eye(self.R)
        self.RY = self.mu * np.eye(self.R)
        self.mu_diff = mu * (1 - alpha)

        # Network that replaces Step 1, warm-started at b0 = s[:R]
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.net = BNet(rank=self.R, b0=self.b).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.state = None  # ConvLSTM (h, c), detached between frames
        self.last_loss = None

    def _predict_and_train(self, D, S):
        D_t = torch.as_tensor(D, dtype=torch.float32, device=self.device)
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        # Input channels: current frame and its background-only residual
        inp = torch.stack([D_t, D_t - S_t]).unsqueeze(0)  # (1, 2, H, W)

        # X, Y are constants for the network update (no gradient through them)
        X_t = torch.as_tensor(self.X, dtype=torch.float32, device=self.device)
        Y_t = torch.as_tensor(self.Y, dtype=torch.float32, device=self.device)

        # Sparsity term of the (fixed) S[t]: constant w.r.t. the network,
        # included so the reported loss matches the model objective.
        sparsity = self.lam_prime * S_t.abs().mean()

        b_used = None
        state_used = None
        for step in range(self.train_steps):
            b_pred, state_new = self.net(inp, self.state)
            if b_used is None:
                # The first prediction (before any weight update on this frame)
                # is the one the algorithm uses: honest online evaluation.
                b_used = b_pred.detach()
                state_used = state_new

            L = (X_t * b_pred) @ Y_t.T  # X diag(b) Y^T
            recon = D_t - L - S_t
            loss = 0.5 * recon.pow(2).mean() + sparsity

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.last_loss = loss.item()
        self.state = tuple(t.detach() for t in state_used)
        return b_used.cpu().numpy().astype(np.float64)

    def process_frame(self, D):
        X, Y = self.X, self.Y
        R = self.R

        # Step 1: solve for S_t (soft thresholding) using the previous b
        L_estimate = X @ np.diag(self.b) @ Y.T
        Z = D - L_estimate
        S_new = soft_threshold(Z, self.lam_prime)

        # Step 2: b[t] predicted by the ConvLSTM (trained online on this
        # frame), using the freshly computed S_new
        b_new = self._predict_and_train(D, S_new)

        # Step 3: update X, Y via RLS — unchanged
        F = D

        # --- Update X
        A = Y @ np.diag(b_new)  # N x R
        self.RX = self.alpha * self.RX + A.T @ A + self.mu_diff * np.eye(R)
        RX_inv = np.linalg.inv(self.RX)

        residual_X = (F - S_new) - X @ A.T
        X_new = X - self.mu_diff * (X @ RX_inv) + residual_X @ A @ RX_inv

        # --- Update Y
        Bmat = X_new @ np.diag(b_new)  # M x R
        self.RY = self.alpha * self.RY + Bmat.T @ Bmat + self.mu_diff * np.eye(R)
        RY_inv = np.linalg.inv(self.RY)

        residual_Y = (F - S_new).T - Y @ Bmat.T  # N x M
        Y_new = Y - self.mu_diff * (Y @ RY_inv) + residual_Y @ Bmat @ RY_inv

        # Store state for next frame
        self.X, self.Y, self.b = X_new, Y_new, b_new

        L_new = X_new @ np.diag(b_new) @ Y_new.T
        return L_new, S_new
