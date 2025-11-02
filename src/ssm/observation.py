# src/ssm/observation.py

# src/ssm/observation.py

import torch
from torch.linalg import cholesky, solve_triangular, eigvalsh
from typing import Optional

def _compute_kernel_matrix(
    X_r: torch.Tensor,
    Y_r: torch.Tensor,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute the cross‐kernel matrix K between X and Y:
      - X: Tensor[M, r]
      - Y: Tensor[N, r]
    Returns K of shape (M, N).
    Supports 'linear' and 'rbf' kernels.
    """
    X = X_r.to(dtype)
    Y = Y_r.to(dtype)
    if kernel == "linear":
        return X @ Y.T
    elif kernel == "rbf":
        if gamma is None:
            gamma = 1.0 / X.size(1)
        X_norm = (X**2).sum(dim=1, keepdim=True)  # (M, 1)
        Y_norm = (Y**2).sum(dim=1, keepdim=True)  # (N, 1)
        sq_dists = X_norm - 2 * (X @ Y.T) + Y_norm.T  # (M, N)
        return torch.exp(-gamma * sq_dists)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel}")


class RealizationError(Exception):
    """Raised when CME observation fails (e.g. ill-conditioned)."""
    pass


class CMEObservation:
    """
    Conditional Mean Embedding (CME) decoder with one‐step prediction:
      ŷ_t = Y_feats[1:] · (Φ_f (K_pp + nλI)^{-1} Φ_p^T φ(x_{t-1}))
    """

    def __init__(
        self,
        past_horizon: int,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        reg_lambda: float = 1e-3,
    ):
        self.h = past_horizon
        self.kernel = kernel
        self.gamma = gamma
        self.reg_lambda = reg_lambda

        # --- 元の placeholder ---
        self.L: Optional[torch.Tensor] = None         # (will hold Cholesky of K_pp_reg)
        self.X: Optional[torch.Tensor] = None   # 全状態系列 (unused in one‐step)
        self.Y_mat: Optional[torch.Tensor] = None     # will hold Y_feats_future.T
        self.n_states: Optional[int] = None                  # = n_past

        # --- １ステップ用に追加 ---
        # self.X_past: Optional[torch.Tensor] = None    # 過去の状態集合
        # self.M: Optional[torch.Tensor] = None         # = K_pf @ (K_pp + nλI)^{-1}
        self.gram_X : Optional[torch.Tensor] = None

        # debug
        self.K_past = None
        self.eigvals = None


    def fit(self, X_states: torch.Tensor, Y_feats: torch.Tensor):
        """
        Precompute factors for one‐step decoding.

        Args:
          X_states: Tensor[N, r] – state sequence from realization (time : h, ...,h+N-1)
          Y_feats:  Tensor[N, p] – encoder feature sequence (time : 0, ..., T)
        """
        self.X = X_states
        self.n_states = int(X_states.size(0))
        n_feats = int(Y_feats.size(0))

        self.Y_mat = Y_feats[self.h : n_feats-self.h+1].T
        K_X = _compute_kernel_matrix(
            X_states, X_states,
            kernel=self.kernel, gamma=self.gamma, dtype = torch.float64
        )
        K_X = 0.5 * (K_X + K_X.T)  # 対称化
        self.gram_X = K_X

        return self

    def decode(self):
        # compute k_x's matrix
        K_past = self.gram_X[:-1, :-1]
        # K_past = 0.5 * (K_past + K_past.T)   # K_Xでやっている
        I_past = torch.eye(self.n_states-1, device=K_past.device).to(K_past.dtype)
        K_past_reg = K_past + self.reg_lambda * (self.n_states - 1) * I_past
        
        # for debug
        # self.K_past = K_past_reg
        self.eigvals = torch.linalg.eigvalsh(K_past_reg)
        
        C_past = cholesky(K_past_reg)
        inv_K_past = torch.cholesky_inverse(C_past)
        self.gram_X = self.gram_X.float(); inv_K_past = inv_K_past.float(); K_past = K_past.float()

        # #LU分解使う計算に変更
        # P, L, U = torch.linalg.lu(K_past_reg)
        # _A = torch.linalg.solve_triangular(L,P.T, upper=False)
        # inv_K_past = torch.linalg.solve_triangular(U, _A, upper=True)
        
        
        
        k_x_pred_mat = self.gram_X[:-1, 1:] @ inv_K_past @ K_past  #計算量のため、K_pastに合わせて各行列をスライス

        Y_pred = self.Y_mat @ inv_K_past @ k_x_pred_mat

        return Y_pred.T  #shape:(\tilde{T}, p)

    def decode_pred(self, forget_flag=True):
        # compute k_x's matrix
        K_past = self.gram_X[:-1, :-1]
        I_past = torch.eye(self.n_states-1, device=K_past.device).to(K_past.dtype)
        K_past_reg = K_past + self.reg_lambda * (self.n_states - 1) * I_past
        
        C_past = cholesky(K_past_reg)
        inv_K_past = torch.cholesky_inverse(C_past)
        inv_K_past = inv_K_past.float()

        #LU分解使う計算に変更
        # P, L, U = torch.linalg.lu(K_past_reg)
        # _A = torch.linalg.solve_triangular(L,P.T, upper=False)
        # inv_K_past = torch.linalg.solve_triangular(U, _A, upper=True)

        self.gram_X = self.gram_X.float(); inv_K_past = inv_K_past.float(); K_past = K_past.float()
        
        k_x_pred = self.gram_X[:-1, 1:] @ inv_K_past @ self.gram_X[:-1, -1] #1D tensor

        Y_pred = self.Y_mat @ inv_K_past @ k_x_pred #1D tensor

        #update gram matrix
        _last_sc_kx = self.gram_X[-1, 1:] @ inv_K_past @ self.gram_X[:-1, -1]
        _last_kx = _last_sc_kx.unsqueeze(0)
        _kx = torch.cat([k_x_pred, _last_kx], dim=0)

        _sc_kernel_new = self.gram_X[:-1, -1] @ inv_K_past @ self.gram_X[1:, 1:] @ inv_K_past @ self.gram_X[1:, -1]

        #construct gram matrix and Y_mat
        n_gram = self.gram_X.size(0)
        _new_gram = self.gram_X.new_empty((n_gram+1, n_gram+1))
        _new_gram[:n_gram, :n_gram] = self.gram_X
        _new_gram[:n_gram, -1] = _kx
        _new_gram[-1, :n_gram] = _kx
        _new_gram[-1, -1] = _sc_kernel_new

        n_row, n_col = self.Y_mat.size()
        _new_Y = self.Y_mat.new_empty((n_row, n_col+1))
        _new_Y[:, :n_col] = self.Y_mat
        _new_Y[:, -1] = Y_pred

        if not forget_flag:
            self.gram_X = _new_gram
            self.Y_mat = _new_Y
        else:
            self.gram_X = _new_gram[1:, 1:]
            self.Y_mat = _new_Y[:, 1:]
            # _new_gram = self.gram_X
            # _new_gram[:-1, :-1] = self.gram_X[1:, 1:]
            # _new_gram[:-1, -1] = _kx
            # _new_gram[-1, :-1] = _kx
            # _new_gram[-1, -1] = _sc_kernel_new
            
            # _new_Y = self.Y_mat
            # _new_Y[:, :-1] = self.Y_mat[:, 1:]
            # _new_Y[:, -1] = Y_pred

            # self.gram_X = _new_gram
            # self.Y_mat = _new_Y

        return Y_pred  #1D Tensor

