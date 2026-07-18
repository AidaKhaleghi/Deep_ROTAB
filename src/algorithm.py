import numpy as np


def soft_threshold(z, thresh):
    # Shrinkage operator
    return np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)


class ROTAB:
    def __init__(self, init_frames, rank=5, mu=0.1, alpha=0.95, lam_prime=0.4):
        init_frames = np.asarray(init_frames)
        K, M, N = init_frames.shape
        self.R = rank
        self.mu = mu
        self.alpha = alpha
        self.lam_prime = lam_prime

        # Initialize X, Y, b via via truncated SVD of the first frame
        D_init = np.mean(init_frames, axis=0)  # Average over the first K frames
        U, s, Vt = np.linalg.svd(D_init, full_matrices=False)
        self.X = U[:, :self.R].copy() # M x R
        self.Y = Vt[:self.R, :].T.copy() # N x R
        self.b = s[:self.R].copy() # R

        # RLS correlation matrices, seeded with mu * I (standard)
        self.RX = self.mu * np.eye(self.R)
        self.RY = self.mu * np.eye(self.R)

        # Foreground from previous step initialized to 0
        self.S_prev = np.zeros((M, N))

        # Constant term (mu[t] - alpha*mu[t-1])
        self.mu_diff = mu * (1 - alpha)
    
    def process_frame(self, D):
        X, Y = self.X, self.Y
        R = self.R

        # Step 1: solve for b[t]
        G = (X.T @ X) * (Y.T @ Y) # R x R
        rhs = np.diag(X.T @ (D - self.S_prev) @ Y) # R
        b_new = np.linalg.solve(self.mu * np.eye(R) + G, rhs)

        # Step 2: solve for S_t (soft thresholding)
        L_estimate = X @ np.diag(b_new) @ Y.T
        Z = D - L_estimate
        S_new = soft_threshold(Z, self.lam_prime)

        # Step 3: update X, Y via RLS
        F = D

        # --- Update X
        A = Y @ np.diag(b_new) # N x R
        self.RX = self.alpha * self.RX + A.T @ A + self.mu_diff * np.eye(R)
        RX_inv = np.linalg.inv(self.RX)

        residual_X = (F - S_new) - X @ A.T
        X_new = X - self.mu_diff * (X @ RX_inv) + residual_X @ A @ RX_inv

        # --- Update Y
        Bmat = X_new @ np.diag(b_new) # M x R
        self.RY = self.alpha * self.RY + Bmat.T @ Bmat + self.mu_diff * np.eye(R)
        RY_inv = np.linalg.inv(self.RY)

        residual_Y = (F - S_new).T - Y @ Bmat.T # N x M
        Y_new = Y - self.mu_diff * (Y @ RY_inv) + residual_Y @ Bmat @ RY_inv

        # Store state for next frame 
        self.X, self.Y, self.b = X_new, Y_new, b_new
        self.S_prev = S_new

        L_new = X_new @ np.diag(b_new) @ Y_new.T
        return L_new, S_new
