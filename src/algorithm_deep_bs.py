import itertools

import numpy as np
import torch

from b_network import BNet
from s_network import SNet


class ROTABDeepBS:
    """ROTAB variant where both Step 1 (closed-form b[t]) and Step 2
    (soft-thresholding for S[t]) are replaced by online-trained networks:

      - b[t]: ConvLSTM network (BNet), input (D, D - S_prev)
      - S[t]: sequence of conv layers (SNet, RPCANet++ style):
        S = x - epsilon * convs(x) with x = D - L(b[t])
        (S_prev is NOT fed into the S-net — that feedback loop is unstable
        with online training; it enters only as a BNet input channel)

    Per frame (test-then-train):
      1. BNet forward -> b[t]; L = X diag(b[t]) Y^T with X = X[t-1], Y = Y[t-1].
      2. SNet forward -> S[t].
      3. Joint loss = 0.5*||D - L - S||_F^2 + lam'*||S||_1, backward, one
         (or more) Adam step(s) over both networks on this frame only. The
         predictions used by the algorithm are the ones from before the
         weight update.
      4. Step 3 of the algorithm (RLS updates for X, Y) runs unchanged.

    The ConvLSTM hidden state carries across frames but is detached each frame
    (truncated backpropagation through time of length 1).
    """

    def __init__(self, init_frames, rank=5, mu=0.1, alpha=0.95, lam_prime=0.4,
                 lr=1e-3, train_steps=1, s_channel=32, s_layers=6, device=None):
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
        self.S_prev = np.zeros((M, N))
        self.mu_diff = mu * (1 - alpha)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.b_net = BNet(rank=self.R, b0=self.b).to(self.device)
        self.s_net = SNet(channel=s_channel, layers=s_layers).to(self.device)
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.b_net.parameters(), self.s_net.parameters()),
            lr=lr,
        )
        self.state = None  # ConvLSTM (h, c), detached between frames
        self.last_loss = None

    def _predict_and_train(self, D):
        D_t = torch.as_tensor(D, dtype=torch.float32, device=self.device)
        S_prev_t = torch.as_tensor(self.S_prev, dtype=torch.float32, device=self.device)
        b_inp = torch.stack([D_t, D_t - S_prev_t]).unsqueeze(0)  # (1, 2, H, W)
        D_img = D_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # X, Y are constants for the network updates (no gradient through them)
        X_t = torch.as_tensor(self.X, dtype=torch.float32, device=self.device)
        Y_t = torch.as_tensor(self.Y, dtype=torch.float32, device=self.device)

        b_used, S_used, state_used = None, None, None
        for step in range(self.train_steps):
            b_pred, state_new = self.b_net(b_inp, self.state)
            L = ((X_t * b_pred) @ Y_t.T).unsqueeze(0).unsqueeze(0)  # X diag(b) Y^T
            S_pred = self.s_net(D_img, L)

            if b_used is None:
                # The first predictions (before any weight update on this
                # frame) are the ones the algorithm uses: honest online
                # evaluation.
                b_used = b_pred.detach()
                S_used = S_pred.detach()
                state_used = state_new

            recon = D_img - L - S_pred
            loss = 0.5 * recon.pow(2).mean() + self.lam_prime * S_pred.abs().mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(self.b_net.parameters(), self.s_net.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

        self.last_loss = loss.item()
        self.state = tuple(t.detach() for t in state_used)
        b_new = b_used.cpu().numpy().astype(np.float64)
        S_new = S_used.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)
        return b_new, S_new

    def process_frame(self, D):
        X, Y = self.X, self.Y
        R = self.R

        # Steps 1 and 2: b[t] and S[t] predicted by the networks (trained
        # online on this frame)
        b_new, S_new = self._predict_and_train(D)

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
        self.S_prev = S_new

        L_new = X_new @ np.diag(b_new) @ Y_new.T
        return L_new, S_new
