# src/inference/kalman_filter.py
"""
Algorithm 1: Operator-based Kalman filtering with Galerkin-projected conditional covariance operators

資料Section 2.7.2の完全実装。学習済み転送作用素VA, VBを用いた逐次状態推定。

Input:
- 学習済み演算子: VA ∈ R^(dA×dA), VB ∈ R^(dB×dA), UB ∈ R^(dB×m), UA^T ∈ R^(r×dA)
- ノイズ共分散: Q ∈ R^(dA×dA), R ∈ R^(m×m)
- エンコーダ: uη : R^n→R^m
- 初期値: (µ0, Σ0)

Output:
- フィルタリング状態: x̂t ∈ R^r
- 共分散: Σt^(x) ∈ R^(r×r) at each t
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings


class OperatorBasedKalmanFilter:
    """
    Algorithm 1の完全実装
    
    作用素ベースKalman更新による逐次状態推定。
    特徴空間でのKalman更新後、元状態空間へ復元。
    """
    
    def __init__(
        self,
        V_A: torch.Tensor,           # 状態転送作用素 (dA, dA)
        V_B: torch.Tensor,           # 観測転送作用素 (dB, dA)
        U_A: torch.Tensor,           # 状態読み出し行列 (dA, r)
        U_B: torch.Tensor,           # 観測読み出し行列 (dB, m)
        Q: torch.Tensor,             # 状態ノイズ共分散 (dA, dA)
        R: Union[torch.Tensor, float], # 観測ノイズ分散（スカラーまたは行列）
        encoder: nn.Module,          # エンコーダネットワーク（frozen）
        df_obs_layer: nn.Module = None,  # DF-B層（学習済みψ_ω含む）
        device: str = 'cpu'
    ):
        """
        Algorithm 1の初期化
        
        Args:
            V_A: 学習済み状態転送作用素 (dA, dA)
            V_B: 学習済み観測転送作用素 (dB, dA)  
            U_A: 学習済み状態読み出し行列 (dA, r)
            U_B: 学習済み観測読み出し行列 (dB, m)
            Q: 状態ノイズ共分散 (dA, dA)
            R: 観測ノイズ分散（スカラーまたは行列）
            encoder: エンコーダネットワーク（frozen）
            device: 計算デバイス
        """
        self.device = torch.device(device)
        
        # 学習済みパラメータ
        self.V_A = V_A.to(self.device)
        self.V_B = V_B.to(self.device)
        self.U_A = U_A.to(self.device)
        self.U_B = U_B.to(self.device)
        
        # ノイズ共分散
        self.Q = Q.to(self.device)
        if isinstance(R, (int, float)):
            self.R = torch.tensor(R, dtype=torch.float32, device=self.device)
        else:
            self.R = R.to(self.device)
            # R can be scalar (univariate) or matrix (multivariate) observation noise
            
        # エンコーダ（推論時は勾配なし）
        self.encoder = encoder
        self.encoder.eval()

        # DF-B層（学習済みψ_ω観測特徴ネットワーク含む）
        self.df_obs_layer = df_obs_layer
        if self.df_obs_layer is not None:
            self.df_obs_layer.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # 次元
        self.dA = int(V_A.size(0))  # 特徴空間状態次元
        self.dB = int(V_B.size(0))  # 特徴空間観測次元
        self.r = int(U_A.size(1))   # 元状態次元
        
        # 定式化では V_B を直接使用する（式51-53）
        # H = V_B.T @ u_B は旧実装で、新定式化では不要
        
        # 内部状態（逐次処理用）
        self.mu: Optional[torch.Tensor] = None      # µ ∈ R^dA
        self.Sigma: Optional[torch.Tensor] = None   # Σ ∈ R^(dA×dA)
        self.is_initialized = False
        
        # 数値安定性パラメータ
        self.min_eigenvalue = 1e-6
        self.condition_threshold = 1e12
        self.jitter = 1e-5

    def initialize_state(
        self,
        initial_observations: torch.Tensor,
        method: str = "data_driven"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        µ₀, Σ₀の初期化 (式47-48)
        
        Args:
            initial_observations: 初期観測サンプル (N0, n)
            method: 初期化方法 ("data_driven" | "zero")
            
        Returns:
            µ₀: 初期状態平均 (dA,)
            Σ₀: 初期状態共分散 (dA, dA)
        """
        N0 = initial_observations.size(0)
        
        if method == "data_driven":
            # 式47-48: データ駆動初期化
            with torch.no_grad():
                # エンコード
                if initial_observations.dim() == 2:
                    # (N0, n) → (N0,) スカラー特徴量
                    m_initial = self.encoder(initial_observations.unsqueeze(0)).squeeze()
                else:
                    m_initial = self.encoder(initial_observations)
                    
                # ψ_ω による特徴変換（簡易版：線形変換で近似）
                # 注意: 実際の実装では DF-B の ψ_ω を使用
                phi_samples = self._approximate_feature_mapping(m_initial)  # (N0, dA)
                
                # µ₀ = (1/N₀) Σ_{i=1}^{N₀} φ_θ(x₀^{(i)})
                mu_0 = torch.mean(phi_samples, dim=0)  # (dA,)
                
                # Σ₀ = (1/(N₀-1)) Σ_{i=1}^{N₀} (φ_θ(x₀^{(i)}) - µ₀)(φ_θ(x₀^{(i)}) - µ₀)^T
                centered = phi_samples - mu_0.unsqueeze(0)  # (N0, dA)
                Sigma_0 = (centered.T @ centered) / (N0 - 1)  # (dA, dA)
                
        elif method == "zero":
            # ゼロ初期化
            mu_0 = torch.zeros(self.dA, device=self.device)
            Sigma_0 = torch.eye(self.dA, device=self.device)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
            
        # 正定値性確保
        Sigma_0 = self._regularize_covariance(Sigma_0)
        
        # 内部状態設定
        self.mu = mu_0.clone()
        self.Sigma = Sigma_0.clone()
        self.is_initialized = True
        
        return mu_0, Sigma_0

    def _approximate_feature_mapping(self, m_series: torch.Tensor) -> torch.Tensor:
        """
        多変量特徴量からdA次元特徴への近似写像

        注意: この関数は簡易版。実際の実装では学習済みDF-A/DF-Bの
        特徴写像 φ_θ を使用する必要がある。

        Args:
            m_series: 多変量特徴系列 (T, m) または (T,) for backward compatibility

        Returns:
            torch.Tensor: 近似特徴 (T, dA)
        """
        T = m_series.size(0)

        # 多変量対応: 入力が(T,)の場合は(T,1)に変換
        if m_series.dim() == 1:
            m_series = m_series.unsqueeze(1)  # (T, 1)

        m = m_series.size(1)  # 特徴量次元

        # 簡易的な特徴拡張（実際はφ_θを使用）
        # 例: 遅延埋め込み + ランダムフーリエ特徴
        features = []

        # 遅延埋め込み（多変量対応）
        max_delay = min(5, T)
        for t in range(T):
            delayed = []
            for d in range(max_delay):
                if t - d >= 0:
                    delayed.append(m_series[t - d])  # (m,)
                else:
                    delayed.append(torch.zeros_like(m_series[0]))  # (m,)
            delayed_features = torch.cat(delayed, dim=0)  # (max_delay * m,)
            features.append(delayed_features)

        feature_matrix = torch.stack(features)  # (T, max_delay * m)
        
        # 線形変換でdA次元に拡張
        if feature_matrix.size(1) < self.dA:
            # パディング
            padding = torch.randn(T, self.dA - feature_matrix.size(1), device=self.device) * 0.1
            feature_matrix = torch.cat([feature_matrix, padding], dim=1)
        elif feature_matrix.size(1) > self.dA:
            # 切り取り
            feature_matrix = feature_matrix[:, :self.dA]
            
        return feature_matrix

    def predict_step(
        self, 
        mu_prev: torch.Tensor, 
        Sigma_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        時間更新 (Time update)
        
        µ⁻ₜ = V_A µ⁺ₜ₋₁
        Σ⁻ₜ = V_A Σ⁺ₜ₋₁ V_A^T + Q
        
        Args:
            mu_prev: 前時刻状態平均 µ⁺ₜ₋₁ (dA,)
            Sigma_prev: 前時刻状態共分散 Σ⁺ₜ₋₁ (dA, dA)
            
        Returns:
            mu_minus: 予測状態平均 µ⁻ₜ (dA,)
            Sigma_minus: 予測状態共分散 Σ⁻ₜ (dA, dA)
        """
        # µ⁻ₜ = V_A µ⁺ₜ₋₁
        mu_minus = self.V_A @ mu_prev  # (dA,)
        
        # Σ⁻ₜ = V_A Σ⁺ₜ₋₁ V_A^T + Q
        Sigma_minus = self.V_A @ Sigma_prev @ self.V_A.T + self.Q  # (dA, dA)
        
        # 正定値性確保
        Sigma_minus = self._regularize_covariance(Sigma_minus)
        
        return mu_minus, Sigma_minus

    def update_step(
        self,
        mu_minus: torch.Tensor,
        Sigma_minus: torch.Tensor,
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        観測更新 (Innovation update)
        
        Algorithm 1のKalman innovation update実装
        
        Args:
            mu_minus: 予測状態平均 µ⁻ₜ (dA,)
            Sigma_minus: 予測状態共分散 Σ⁻ₜ (dA, dA)
            observation: 現在観測 y_t (n,) または (1, n)
            
        Returns:
            mu_plus: 更新状態平均 µ⁺ₜ (dA,)
            Sigma_plus: 更新状態共分散 Σ⁺ₜ (dA, dA)
            likelihood: ダミー値（正規分布非仮定のため廃止）
        """
        # 1. 現在観測のエンコード: m_t ← u_η(y_t) ∈ R^m
        with torch.no_grad():
            # TCNエンコーダ用の形状調整: [d] → [1, 1, d]
            if observation.dim() == 1:
                observation = observation.unsqueeze(0).unsqueeze(0)  # (1, 1, d)
            elif observation.dim() == 2:
                observation = observation.unsqueeze(1)  # (B, 1, d)
            m_t = self.encoder(observation).squeeze()  # (m,) multivariate features

            # 1.5. 観測特徴の生成: ψ_ω(m_t) ∈ ℝ^{d_B} (式54で使用)
            # 多変量特徴量から観測特徴量への変換
            psi_omega_m_t = self._generate_obs_features_from_multivariate(m_t)  # (dB,)

        # 2. 観測予測 (Observation prediction): 式51
        # ハット{ψ}^{-}_{t} = V_B μ^-_{t} ∈ ℝ^{d_B}
        psi_pred = self.V_B @ mu_minus  # (dB,)

        # 3. イノベーション共分散 (Innovation covariance): 式52
        # S_t = V_B Σ^-_t V_B^T + R ∈ ℝ^{d_B × d_B}
        S = self.V_B @ Sigma_minus @ self.V_B.T  # (dB, dB)

        # 観測ノイズ共分散の追加（多変量対応）
        if isinstance(self.R, torch.Tensor) and self.R.dim() == 2:
            # 行列 R ∈ R^{d_B × d_B}
            S = S + self.R
        elif isinstance(self.R, (int, float)) or (isinstance(self.R, torch.Tensor) and self.R.dim() == 0):
            # スカラー R → 対角行列 R * I_{d_B}
            R_scalar = float(self.R) if isinstance(self.R, torch.Tensor) else self.R
            S = S + R_scalar * torch.eye(self.dB, device=S.device, dtype=S.dtype)
        else:
            raise ValueError(f"Unsupported observation noise covariance type: {type(self.R)}, shape: {self.R.shape if hasattr(self.R, 'shape') else 'N/A'}")

        # 数値安定性チェック (Check positive definiteness)
        try:
            eigenvalues = torch.linalg.eigvals(S).real
            if torch.any(eigenvalues <= 0):
                warnings.warn(f"Non-positive definite innovation covariance matrix detected")
                # Add jitter to diagonal for regularization
                S = S + self.jitter * torch.eye(S.size(0), device=self.device)
        except Exception as e:
            warnings.warn(f"Failed to check positive definiteness: {e}")
            S = S + self.jitter * torch.eye(S.size(0), device=self.device)

        # 4. Kalmanゲイン (Kalman gain): 式53
        # K_t = Σ^-_t V_B^T S_t^{-1} ∈ ℝ^{d_A × d_B}
        try:
            S_inv = torch.linalg.inv(S)  # (dB, dB)
            K = Sigma_minus @ self.V_B.T @ S_inv  # (dA, dB)
        except Exception as e:
            warnings.warn(f"Failed to invert innovation covariance: {e}")
            # Use pseudo-inverse as fallback
            S_pinv = torch.linalg.pinv(S)
            K = Sigma_minus @ self.V_B.T @ S_pinv  # (dA, dB)
        
        # 5. イノベーション更新 (Innovation update): 式54
        # μ^+_t = μ^-_t + K_t(ψ_ω(m_t) - ハット{ψ}^{-}_t) ∈ ℝ^{d_A}
        innovation = psi_omega_m_t - psi_pred  # (dB,)
        mu_plus = mu_minus + K @ innovation  # (dA,) = (dA, dB) @ (dB,)
        
        # 6. 共分散更新 (Covariance update): 式55
        # Σ^+_t = Σ^-_t - K_t V_B Σ^-_t ∈ ℝ^{d_A × d_A}
        Sigma_plus = Sigma_minus - K @ self.V_B @ Sigma_minus  # (dA, dA)
        
        # 正定値性確保
        Sigma_plus = self._regularize_covariance(Sigma_plus)

        # 6. 観測尤度計算（廃止: 正規分布非仮定のため）
        # likelihood = self._compute_likelihood(innovation, S)
        likelihood = 0.0  # ダミー値

        return mu_plus, Sigma_plus, likelihood

    def filter_step(
        self,
        observation: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        1ステップフィルタリング（逐次処理用）
        
        内部状態を保持してストリーミング対応
        
        Args:
            observation: 現在観測 y_t (n,)
            state: 外部状態 (mu, Sigma) or None（内部状態使用）
            
        Returns:
            x_hat_t: 推定状態 x̂_t (r,)
            Sigma_x_t: 状態共分散 Σ_t^(x) (r, r)
            likelihood: ダミー値（正規分布非仮定のため廃止）
        """
        if state is not None:
            mu_prev, Sigma_prev = state
        else:
            if not self.is_initialized:
                raise RuntimeError("Filter not initialized. Call initialize_state() first.")
            mu_prev, Sigma_prev = self.mu, self.Sigma
            
        # 時間更新
        mu_minus, Sigma_minus = self.predict_step(mu_prev, Sigma_prev)
        
        # 観測更新
        mu_plus, Sigma_plus, likelihood = self.update_step(
            mu_minus, Sigma_minus, observation
        )
        
        # 元状態空間での復元
        x_hat_t, Sigma_x_t = self._recover_original_state(mu_plus, Sigma_plus)
        
        # 内部状態更新
        if state is None:
            self.mu = mu_plus
            self.Sigma = Sigma_plus
            
        return x_hat_t, Sigma_x_t, likelihood

    def filter_sequence(
        self,
        observations: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_likelihood: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], 
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        バッチフィルタリング（従来型）
        
        観測系列全体を一度に処理
        
        Args:
            observations: 観測系列 (T, n)
            initial_state: 初期状態 (µ₀, Σ₀) or None
            return_likelihood: 尤度も返すかどうか
            
        Returns:
            X_means: 状態平均系列 (T, r)
            X_covariances: 状態共分散系列 (T, r, r)
            likelihoods: ダミー値系列 (T,) [正規分布非仮定のため廃止]
        """
        T = observations.size(0)
        
        # 初期化
        if initial_state is not None:
            self.mu, self.Sigma = initial_state
            self.is_initialized = True
        elif not self.is_initialized:
            # データ駆動初期化（最初の数サンプル使用）
            n_init = min(10, T)
            self.initialize_state(observations[:n_init])
            
        # 結果格納用
        X_means = torch.zeros(T, self.r, device=self.device)
        X_covariances = torch.zeros(T, self.r, self.r, device=self.device)
        if return_likelihood:
            likelihoods = torch.zeros(T, device=self.device)
        
        # 逐次フィルタリング
        for t in range(T):
            x_hat_t, Sigma_x_t, likelihood = self.filter_step(observations[t])
            
            X_means[t] = x_hat_t
            X_covariances[t] = Sigma_x_t
            if return_likelihood:
                likelihoods[t] = likelihood
                
        if return_likelihood:
            return X_means, X_covariances, likelihoods
        return X_means, X_covariances

    def _recover_original_state(
        self, 
        mu: torch.Tensor, 
        Sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        元状態空間での復元
        
        x̂_t ← U_A^T µ ∈ R^r
        Σ_t^(x) ← U_A^T Σ U_A ∈ R^(r×r)
        
        Args:
            mu: 特徴空間状態平均 (dA,)
            Sigma: 特徴空間状態共分散 (dA, dA)
            
        Returns:
            x_hat: 元状態空間での状態推定 (r,)
            Sigma_x: 元状態空間での共分散 (r, r)
        """
        # x̂_t ← U_A^T µ
        x_hat = self.U_A.T @ mu  # (r,)
        
        # Σ_t^(x) ← U_A^T Σ U_A  
        Sigma_x = self.U_A.T @ Sigma @ self.U_A  # (r, r)
        
        # 正定値性確保
        Sigma_x = self._regularize_covariance(Sigma_x)
        
        return x_hat, Sigma_x

    def _regularize_covariance(self, cov_matrix: torch.Tensor) -> torch.Tensor:
        """
        共分散行列の正則化
        
        正定値性確保と数値安定性向上
        
        Args:
            cov_matrix: 共分散行列 (d, d)
            
        Returns:
            torch.Tensor: 正則化済み共分散行列 (d, d)
        """
        # 対称性確保
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        # 固有値分解
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        except:
            # 失敗時はjitter追加で対応
            cov_matrix += self.jitter * torch.eye(cov_matrix.size(0), device=self.device)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
            
        # 負の固有値をクリップ
        eigenvalues = torch.clamp(eigenvalues, min=self.min_eigenvalue)
        
        # 再構成
        cov_matrix = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        
        return cov_matrix

    def _compute_likelihood(self, innovation: torch.Tensor, S: torch.Tensor) -> float:
        """
        観測尤度計算（廃止）

        注意: 定式化では正規分布を仮定していないため、尤度計算は理論的に不適切。
        平均と共分散のみを特徴空間から元の状態空間へ射影することで復元する。

        Args:
            innovation: イノベーション (多変量) - 使用されない
            S: イノベーション共分散 (行列) - 使用されない

        Returns:
            float: ダミー値（0.0を返す）
        """
        # 引数は後方互換性のためのみ保持
        _ = innovation  # 未使用警告回避
        _ = S          # 未使用警告回避

        # TODO: 正規分布仮定なしの定式化のため、尤度計算を削除
        # 元の実装（正規分布仮定）:
        # log_likelihood = -0.5 * (
        #     torch.log(2 * torch.pi * S) +
        #     (innovation ** 2) / S
        # )

        # 正規分布を仮定しない定式化では尤度計算は行わない
        return 0.0  # ダミー値

    def get_current_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        現在の状態推定値と信頼区間を取得
        
        Returns:
            x_hat: 現在状態推定 (r,)
            Sigma_x: 現在状態共分散 (r, r)
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized")
            
        return self._recover_original_state(self.mu, self.Sigma)

    def reset_state(self):
        """
        内部状態リセット（新しい系列開始時）
        """
        self.mu = None
        self.Sigma = None
        self.is_initialized = False

    def check_numerical_stability(self) -> Dict[str, Any]:
        """
        数値安定性診断
        
        Returns:
            Dict: 診断結果
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}
            
        # 条件数チェック
        try:
            cond_Q = torch.linalg.cond(self.Q).item()
            cond_Sigma = torch.linalg.cond(self.Sigma).item()
            
            # 固有値チェック
            eigenvals_Q = torch.linalg.eigvals(self.Q).real
            eigenvals_Sigma = torch.linalg.eigvals(self.Sigma).real
            
            return {
                "status": "ok",
                "condition_numbers": {
                    "Q": cond_Q,
                    "Sigma": cond_Sigma
                },
                "min_eigenvalues": {
                    "Q": eigenvals_Q.min().item(),
                    "Sigma": eigenvals_Sigma.min().item()
                },
                "numerical_stable": (
                    cond_Q < self.condition_threshold and 
                    cond_Sigma < self.condition_threshold and
                    eigenvals_Q.min() > 0 and 
                    eigenvals_Sigma.min() > 0
                )
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _generate_obs_features_from_multivariate(self, m_t: torch.Tensor) -> torch.Tensor:
        """
        多変量特徴から観測特徴を生成: ψ_ω(m_t) ∈ ℝ^{d_B}

        定式化の式54で必要な観測特徴ベクトルを生成。
        学習済みDF-B層のψ_ωネットワークが必須。

        Args:
            m_t: 多変量特徴 (エンコーダ出力) ∈ R^m - (m,) または (1, m)

        Returns:
            psi_omega_m_t: 観測特徴ベクトル (dB,)

        Raises:
            RuntimeError: DF-B層またはψ_ωネットワークが存在しない場合
        """
        with torch.no_grad():
            # DF-B層とψ_ωネットワークの存在確認（必須）
            if self.df_obs_layer is None or not hasattr(self.df_obs_layer, 'psi_omega'):
                raise RuntimeError(
                    "DF-B layer with psi_omega network is required for observation feature generation. "
                    "Cannot proceed without learned ψ_ω(m_t) transformation."
                )

            # m_t の形状を ψ_ω の入力形式に調整
            # 多変量特徴: (m,) または (1, m) → (m,) for single sample
            if m_t.dim() == 1:
                # (m,) - 単一サンプル
                m_t_input = m_t  # そのまま
            elif m_t.dim() == 2 and m_t.size(0) == 1:
                # (1, m) - バッチサイズ1
                m_t_input = m_t.squeeze(0)  # (m,)
            else:
                raise ValueError(f"m_t must be multivariate feature tensor with shape (m,) or (1, m), got shape: {m_t.shape}")

            # 学習済みψ_ωネットワークで観測特徴生成
            psi_omega_m_t = self.df_obs_layer.psi_omega(m_t_input)  # (dB,) or (1, dB)

            # 出力を(dB,)形状に正規化
            while psi_omega_m_t.dim() > 1 and psi_omega_m_t.size(0) == 1:
                psi_omega_m_t = psi_omega_m_t.squeeze(0)

            # 最終確認: 1次元テンソルであることを保証
            if psi_omega_m_t.dim() != 1:
                raise RuntimeError(
                    f"psi_omega output has unexpected shape: {psi_omega_m_t.shape}. "
                    f"Expected 1D tensor (dB,) for observation features."
                )

            return psi_omega_m_t