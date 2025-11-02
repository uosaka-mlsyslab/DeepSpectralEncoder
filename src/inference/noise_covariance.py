# src/inference/noise_covariance.py

import torch
import warnings
from typing import Union, Tuple, Dict, Any


class ObservationNoiseCovarianceEstimator:
    """
    多変量観測ノイズ共分散推定器

    残差 ρ_t := ψ_ω(m_t) - V_B φ_θ(x_t) ∈ R^{d_B} から
    サンプル共分散 R ∈ R^{d_B × d_B} を推定し、数値安定性のため正則化。
    """

    def __init__(
        self,
        d_B: int,
        regularization: float = 1e-3,
        min_eigenvalue: float = 1e-6,
        max_condition_number: float = 1e8
    ):
        """
        Args:
            d_B: 観測特徴次元
            regularization: 正則化パラメータ γ_R
            min_eigenvalue: 最小固有値
            max_condition_number: 最大条件数
        """
        self.d_B = d_B
        self.gamma_R = regularization
        self.min_eigenvalue = min_eigenvalue
        self.max_condition_number = max_condition_number

    def estimate_covariance(
        self,
        residuals: torch.Tensor,
        return_stats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        残差からサンプル共分散を推定

        Args:
            residuals: 残差系列 (T, d_B)
            return_stats: 統計情報も返すか

        Returns:
            R: 正則化済み観測ノイズ共分散 (d_B, d_B)
            stats: 統計情報（オプション）
        """
        if residuals.dim() != 2:
            raise ValueError(f"residuals must be 2D tensor (T, d_B), got shape: {residuals.shape}")

        T, d_B_actual = residuals.shape
        if d_B_actual != self.d_B:
            raise ValueError(f"Feature dimension mismatch: expected {self.d_B}, got {d_B_actual}")

        if T < 2:
            raise ValueError(f"Too few samples for covariance estimation: T={T}")

        # サンプル共分散推定
        residuals_centered = residuals - residuals.mean(dim=0, keepdim=True)  # (T, d_B)
        R_sample = (residuals_centered.T @ residuals_centered) / (T - 1)  # (d_B, d_B)

        # 正則化適用
        R_regularized = self.regularize_covariance(R_sample)

        if return_stats:
            stats = self._compute_stats(R_sample, R_regularized, T)
            return R_regularized, stats
        return R_regularized

    def regularize_covariance(self, R_sample: torch.Tensor) -> torch.Tensor:
        """
        共分散行列の正則化

        Args:
            R_sample: サンプル共分散 (d_B, d_B)

        Returns:
            R_regularized: 正則化済み共分散 (d_B, d_B)
        """
        # 対称性確保
        R_sample = (R_sample + R_sample.T) / 2

        # 対角正則化: R = R_sample + γ_R * I_{d_B}
        R_regularized = R_sample + self.gamma_R * torch.eye(
            self.d_B, device=R_sample.device, dtype=R_sample.dtype
        )

        # 固有値分解による追加正則化
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(R_regularized)

            # 負・小さい固有値をクリップ
            eigenvalues_clipped = torch.clamp(eigenvalues, min=self.min_eigenvalue)

            # 条件数制御
            max_eigenvalue = eigenvalues_clipped.max()
            min_allowed = max_eigenvalue / self.max_condition_number
            eigenvalues_final = torch.clamp(eigenvalues_clipped, min=min_allowed)

            # 再構成
            R_final = eigenvectors @ torch.diag(eigenvalues_final) @ eigenvectors.T

        except torch.linalg.LinAlgError:
            # フォールバック: より強い対角正則化
            warnings.warn("Eigenvalue decomposition failed, using stronger regularization")
            stronger_reg = max(self.gamma_R * 10, 1e-2)
            R_final = R_sample + stronger_reg * torch.eye(
                self.d_B, device=R_sample.device, dtype=R_sample.dtype
            )

        return R_final

    def _compute_stats(
        self,
        R_sample: torch.Tensor,
        R_regularized: torch.Tensor,
        T: int
    ) -> Dict[str, float]:
        """統計情報計算"""
        try:
            # サンプル共分散の統計
            eigenvals_sample = torch.linalg.eigvals(R_sample).real
            cond_sample = eigenvals_sample.max() / eigenvals_sample.min()
            det_sample = torch.det(R_sample)

            # 正則化後の統計
            eigenvals_reg = torch.linalg.eigvals(R_regularized).real
            cond_reg = eigenvals_reg.max() / eigenvals_reg.min()
            det_reg = torch.det(R_regularized)

            return {
                'sample_size': T,
                'condition_number_sample': cond_sample.item(),
                'condition_number_regularized': cond_reg.item(),
                'determinant_sample': det_sample.item(),
                'determinant_regularized': det_reg.item(),
                'min_eigenvalue_sample': eigenvals_sample.min().item(),
                'min_eigenvalue_regularized': eigenvals_reg.min().item(),
                'regularization_applied': self.gamma_R,
                'numerical_stable': cond_reg < self.max_condition_number
            }
        except Exception as e:
            return {'error': str(e), 'sample_size': T}

    def estimate_from_sequences(
        self,
        psi_obs: torch.Tensor,
        psi_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        観測特徴量と予測から直接推定

        Args:
            psi_obs: 観測特徴量系列 (T, d_B)
            psi_pred: 予測特徴量系列 (T, d_B)

        Returns:
            R: 観測ノイズ共分散 (d_B, d_B)
        """
        if psi_obs.shape != psi_pred.shape:
            raise ValueError(f"Shape mismatch: psi_obs {psi_obs.shape} vs psi_pred {psi_pred.shape}")

        residuals = psi_obs - psi_pred  # (T, d_B)
        return self.estimate_covariance(residuals)

    def adaptive_estimation(
        self,
        residuals: torch.Tensor,
        window_size: int = 50,
        overlap: float = 0.5
    ) -> torch.Tensor:
        """
        適応的共分散推定（滑動窓）

        Args:
            residuals: 残差系列 (T, d_B)
            window_size: 窓サイズ
            overlap: 重複率

        Returns:
            R_adaptive: 適応的観測ノイズ共分散 (d_B, d_B)
        """
        T = residuals.size(0)
        if T < window_size:
            return self.estimate_covariance(residuals)

        step_size = max(1, int(window_size * (1 - overlap)))
        covariance_estimates = []

        for start in range(0, T - window_size + 1, step_size):
            end = start + window_size
            window_residuals = residuals[start:end]
            R_window = self.estimate_covariance(window_residuals)
            covariance_estimates.append(R_window)

        # 推定値の平均（より安定した推定）
        R_adaptive = torch.stack(covariance_estimates).mean(dim=0)
        return self.regularize_covariance(R_adaptive)