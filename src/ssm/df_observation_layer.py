# src/ssm/df_observation_layer.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import warnings

from .cross_fitting import CrossFittingManager, TwoStageCrossFitter, CrossFittingError


class ObservationFeatureNet(nn.Module):
    """
    観測特徴写像 ψ_ω: R^m → R^{d_B}

    多変量特徴量（エンコーダ出力）を高次元特徴空間に写像する。
    """
    
    def __init__(
        self,
        input_dim: int = 8,  # 多変量特徴量次元 m
        output_dim: int = 16,
        hidden_sizes: list[int] = [32, 32],
        activation: str = "ReLU",
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: 入力次元 m（多変量特徴量）
            output_dim: 出力特徴次元 d_B
            hidden_sizes: 中間層のユニット数リスト
            activation: 活性化関数名
            dropout: ドロップアウト率
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 中間層
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(nn, activation)())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 出力層
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
        # 初期化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m: 多変量特徴量 (batch_size, m) または (m,)

        Returns:
            torch.Tensor: 観測特徴量 (batch_size, d_B) または (d_B,)
        """
        # 多変量特徴量の形状チェック
        if m.dim() == 1:  # (m,)
            m = m.unsqueeze(0)  # (1, m)
            return self.net(m).squeeze(0)  # (d_B,)
        elif m.dim() == 2:  # (batch_size, m)
            return self.net(m)  # (batch_size, d_B)
        else:
            raise ValueError(f"Unsupported input shape: {m.shape}. Expected (m,) or (batch_size, m)")


class DFObservationLayer(nn.Module):
    """
    DF-B: Deep Feature Instrumental Variable for Observation Process
    
    資料のSection 1.4.2に対応。DF-Aで得られた状態予測を操作変数として、
    スカラー特徴量の1ステップ予測を2SLSで実現。
    
    **修正点**: 勾配制御の明確化
    
    計算フロー:
    1. DF-Aから状態予測 x̂_{t|t-1} と共有特徴写像 φ_θ を受け取る
    2. φ_θ(x̂_{t|t-1}) を操作変数として使用
    3. ψ_ω(m_t) で観測特徴量を計算
    4. Stage-1: V_B推定 + φ_θ更新（ψ_ω完全固定）
    5. Stage-2: u_B推定 + ψ_ω更新（φ_θ完全固定）
    6. 予測: m̂_{t|t-1} = u_B^T V_B φ_θ(x̂_{t|t-1})
    """
    
    def __init__(
        self,
        df_state_layer,  # DFStateLayerインスタンス（必須）
        obs_feature_dim: int = 16,
        multivariate_feature_dim: int = 8,  # 多変量特徴量次元 m
        lambda_B: float = 1e-3,
        lambda_dB: float = 1e-3,
        obs_net_config: Optional[Dict[str, Any]] = None,
        cross_fitting_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            df_state_layer: **必須** 学習済みDFStateLayerインスタンス
            obs_feature_dim: 観測特徴次元 d_B
            multivariate_feature_dim: 多変量特徴量次元 m
            lambda_B: Stage-1正則化パラメータ λ_B
            lambda_dB: Stage-2正則化パラメータ λ_{dB}
            obs_net_config: ObservationFeatureNetの設定
            cross_fitting_config: CrossFittingManagerの設定
        """
        if df_state_layer is None:
            raise ValueError("df_state_layerは必須です。DFStateLayerインスタンスを渡してください。")
        
        super().__init__()
        self.df_state = df_state_layer
        self.obs_feature_dim = obs_feature_dim
        self.multivariate_feature_dim = multivariate_feature_dim
        self.lambda_B = float(lambda_B)  # 文字列対応
        self.lambda_dB = float(lambda_dB)  # 文字列対応
        
        # **重要**: 共有特徴ネットワーク（DF-Aから直接参照）
        self.phi_theta = df_state_layer.phi_theta
        
        # 観測特徴ネットワーク ψ_ω
        obs_config = obs_net_config or {}
        self.psi_omega = ObservationFeatureNet(
            input_dim=multivariate_feature_dim,
            output_dim=obs_feature_dim,
            **obs_config
        )
        
        # クロスフィッティング設定
        self.cf_config = cross_fitting_config or {'n_blocks': 5, 'min_block_size': 10}
        
        # 学習済みパラメータ
        self.V_B: Optional[torch.Tensor] = None  # 観測転送作用素 (d_B, d_A)
        self.U_B: Optional[torch.Tensor] = None  # 観測読み出し行列 (d_B, m)
        self._is_fitted = False
        
        # **新機能**: Phase-1学習用の内部状態管理
        self._stage1_cache = {}  # V_B計算結果をキャッシュ
        self._stage2_cache = {}  # U_B計算結果をキャッシュ
    
    def _ridge_stage1_vb(
        self, 
        Phi_instrument: torch.Tensor, 
        Psi_treatment: torch.Tensor, 
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-1 Ridge回帰: 観測転送作用素推定
        
        V_B = (Ψ^T Φ)(Φ^T Φ + λI)^{-1}
        
        Args:
            Phi_instrument: 操作変数特徴量 (N, d_A)
            Psi_treatment: 観測特徴量 (N, d_B)
            reg_lambda: 正則化パラメータ λ_B
            
        Returns:
            torch.Tensor: 観測転送作用素 V_B (d_B, d_A)
        """
        N, d_A = Phi_instrument.shape
        N_t, d_B = Psi_treatment.shape

        if N != N_t:
            raise ValueError(f"操作変数と観測の サンプル数不一致: {N} vs {N_t}")
        
        if N < max(d_A, d_B):
            warnings.warn(f"サンプル数 {N} < 特徴次元 max({d_A}, {d_B})。数値不安定の可能性")
        
        # グラム行列 + 正則化
        PhiTPhi = Phi_instrument.T @ Phi_instrument  # (d_A, d_A)
        PhiTPhi_reg = PhiTPhi + reg_lambda * torch.eye(d_A, device=Phi_instrument.device, dtype=Phi_instrument.dtype)
        
        # クロス共分散（デバイス整合性を確保）
        Psi_treatment = Psi_treatment.to(Phi_instrument.device)
        PsiTPhi = Psi_treatment.T @ Phi_instrument  # (d_B, d_A)
        
        # 逆行列計算（数値安定化）
        try:
            PhiTPhi_inv = torch.linalg.inv(PhiTPhi_reg)
            V_B = PsiTPhi @ PhiTPhi_inv
        except torch.linalg.LinAlgError:
            # Cholesky分解 fallback
            try:
                L = torch.linalg.cholesky(PhiTPhi_reg)
                PhiTPhi_inv = torch.cholesky_inverse(L)
                V_B = PsiTPhi @ PhiTPhi_inv
            except torch.linalg.LinAlgError:
                # SVD fallback
                U, S, Vh = torch.linalg.svd(PhiTPhi_reg)
                S_inv = torch.where(S > 1e-10, 1.0 / S, 0.0)
                PhiTPhi_inv = (Vh.T * S_inv) @ Vh
                V_B = PsiTPhi @ PhiTPhi_inv
        
        return V_B

    def _ridge_stage1_vb_with_grad(
        self,
        Phi_instrument: torch.Tensor,
        Psi_treatment: torch.Tensor,
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-1 Ridge回帰（勾配計算あり）: φ_θ更新用

        V_B = (Ψ^T Φ)(Φ^T Φ + λI)^{-1}
        """
        N, d_A = Phi_instrument.shape
        N_t, d_B = Psi_treatment.shape

        if N != N_t:
            raise ValueError(f"特徴量とターゲットのサンプル数不一致: {N} vs {N_t}")

        # グラム行列 + 正則化（勾配計算あり）
        PhiPhi = Phi_instrument.T @ Phi_instrument  # (d_A, d_A)
        PhiPhi_reg = PhiPhi + reg_lambda * torch.eye(d_A, device=Phi_instrument.device, dtype=Phi_instrument.dtype)

        # クロス共分散（勾配計算あり・デバイス整合性を確保）
        Psi_treatment = Psi_treatment.to(Phi_instrument.device)
        PsiPhi = Psi_treatment.T @ Phi_instrument  # (d_B, d_A)

        # 逆行列計算（勾配計算あり）
        try:
            PhiPhi_inv = torch.linalg.inv(PhiPhi_reg)
            V_B = PsiPhi @ PhiPhi_inv
        except torch.linalg.LinAlgError:
            # フォールバック：疑似逆行列
            PhiPhi_inv = torch.linalg.pinv(PhiPhi_reg)
            V_B = PsiPhi @ PhiPhi_inv

        return V_B

    def _compute_cross_fitting_prediction_vb(
        self,
        phi_prev: torch.Tensor,
        psi_curr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        **理論準拠**: V_B用クロスフィッティングでout-of-fold予測計算

        Args:
            phi_prev: 操作変数特徴量 (T-1, d_A)
            psi_curr: 観測特徴量 (T-1, d_B)

        Returns:
            tuple: (psi_pred_cf, V_B_final)
                - psi_pred_cf: out-of-fold予測 (T-1, d_B)
                - V_B_final: 最終V_B行列 (d_B, d_A)
        """
        T_eff = phi_prev.size(0)

        # クロスフィッティング設定取得
        n_blocks = getattr(self, 'cf_config', {}).get('n_blocks', 6)
        min_block_size = getattr(self, 'cf_config', {}).get('min_block_size', 20)

        # データ量チェック：クロスフィッティング実行可能性
        if T_eff < max(n_blocks * min_block_size, 100):
            # データ不足時：従来の全データRidge回帰（勾配あり）
            V_B = self._ridge_stage1_vb_with_grad(phi_prev, psi_curr, self.lambda_B)
            psi_pred = (V_B @ phi_prev.T).T
            return psi_pred, V_B

        # クロスフィッティング実行
        try:
            from .cross_fitting import CrossFittingManager, TwoStageCrossFitter

            # クロスフィッティング管理
            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)
            cf_fitter = TwoStageCrossFitter(cf_manager)

            # V_B推定（クロスフィッティング、勾配あり）
            # 注: psi_currはno_grad()で計算済み（ω固定）
            V_B_list = cf_fitter.cross_fit_stage1(
                phi_prev, psi_curr,
                stage1_estimator=lambda X, Y: self._ridge_stage1_vb_with_grad(X, Y, self.lambda_B)
            )

            # **理論準拠**: out-of-fold予測計算（勾配あり）
            psi_pred_cf = cf_fitter.compute_out_of_fold_features(phi_prev, V_B_list)

            # 最終V_B：全データでの推定（勾配あり、正則化用）
            V_B_final = self._ridge_stage1_vb_with_grad(phi_prev, psi_curr, self.lambda_B)

            # 情報をキャッシュ
            if not hasattr(self, '_cross_fitting_cache'):
                self._cross_fitting_cache = {}
            self._cross_fitting_cache.update({
                'V_B_list': V_B_list,
                'cf_manager': cf_manager
            })

            return psi_pred_cf, V_B_final

        except ImportError:
            # フォールバック（勾配あり）
            V_B = self._ridge_stage1_vb_with_grad(phi_prev, psi_curr, self.lambda_B)
            psi_pred = (V_B @ phi_prev.T).T
            return psi_pred, V_B
        except Exception as e:
            print(f"V_Bクロスフィッティング失敗、従来方式を使用: {e}")
            V_B = self._ridge_stage1_vb_with_grad(phi_prev, psi_curr, self.lambda_B)
            psi_pred = (V_B @ phi_prev.T).T
            return psi_pred, V_B

    def _ridge_stage2_ub_matrix(
        self,
        H_instrument: torch.Tensor,
        M_target: torch.Tensor,
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-2 Ridge回帰: 観測読み出し行列推定（多変量版）

        U_B = (H H^T + λI)^{-1} H M^T

        Args:
            H_instrument: クロスフィット操作変数特徴量 (N, d_B)
            M_target: 目標多変量特徴量 (N, m)
            reg_lambda: 正則化パラメータ λ_{dB}

        Returns:
            torch.Tensor: 観測読み出し行列 U_B (d_B, m)
        """
        N, d_B = H_instrument.shape
        N_t, m = M_target.shape

        if N != N_t:
            raise ValueError(f"操作変数とターゲットのサンプル数不一致: {N} vs {N_t}")

        # グラム行列 + 正則化
        HHt = H_instrument.T @ H_instrument  # (d_B, d_B)
        HHt_reg = HHt + reg_lambda * torch.eye(d_B, device=H_instrument.device, dtype=H_instrument.dtype)

        # クロス項（デバイス整合性を確保）
        M_target = M_target.to(H_instrument.device)
        HM = H_instrument.T @ M_target  # (d_B, m)

        # 逆行列計算（数値安定化）
        try:
            HHt_inv = torch.linalg.inv(HHt_reg)
            U_B = HHt_inv @ HM
        except torch.linalg.LinAlgError:
            # Cholesky分解 fallback
            try:
                L = torch.linalg.cholesky(HHt_reg)
                HHt_inv = torch.cholesky_inverse(L)
                U_B = HHt_inv @ HM
            except torch.linalg.LinAlgError:
                # SVD fallback
                U, S, Vh = torch.linalg.svd(HHt_reg)
                S_inv = torch.where(S > 1e-10, 1.0 / S, 0.0)
                HHt_inv = (Vh.T * S_inv) @ Vh
                U_B = HHt_inv @ HM

        return U_B

    def _ridge_stage2_ub_matrix_with_grad(
        self,
        H_instrument: torch.Tensor,
        M_target: torch.Tensor,
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-2 Ridge回帰（勾配計算あり）: ψ_ω更新用（多変量版）

        U_B = (H H^T + λI)^{-1} H M^T
        """
        N, d_B = H_instrument.shape
        N_t, m = M_target.shape

        if N != N_t:
            raise ValueError(f"特徴量とターゲットのサンプル数不一致: {N} vs {N_t}")

        # グラム行列 + 正則化（勾配計算あり）
        HHt = H_instrument.T @ H_instrument  # (d_B, d_B)
        HHt_reg = HHt + reg_lambda * torch.eye(d_B, device=H_instrument.device, dtype=H_instrument.dtype)

        # クロス項（勾配計算あり・デバイス整合性を確保）
        M_target = M_target.to(H_instrument.device)
        HM = H_instrument.T @ M_target  # (d_B, m)

        # 逆行列計算（勾配計算あり）
        try:
            HHt_inv = torch.linalg.inv(HHt_reg)
            U_B = HHt_inv @ HM
        except torch.linalg.LinAlgError:
            # フォールバック：疑似逆行列
            HHt_inv = torch.linalg.pinv(HHt_reg)
            U_B = HHt_inv @ HM

        return U_B

    def _compute_cross_fitting_prediction_ub_matrix(
        self,
        H_features: torch.Tensor,
        M_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        **理論準拠**: U_B用クロスフィッティングでout-of-fold予測計算（多変量版）

        Args:
            H_features: 操作変数特徴量 (T-1, d_B)
            M_target: 観測目標 (T-1, m)

        Returns:
            tuple: (M_pred_cf, U_B_final)
                - M_pred_cf: out-of-fold予測 (T-1, m)
                - U_B_final: 最終U_B行列 (d_B, m)
        """
        T_eff = H_features.size(0)

        # クロスフィッティング設定取得
        n_blocks = getattr(self, 'cf_config', {}).get('n_blocks', 6)
        min_block_size = getattr(self, 'cf_config', {}).get('min_block_size', 20)

        # データ量チェック：クロスフィッティング実行可能性
        if T_eff < max(n_blocks * min_block_size, 100):
            # データ不足時：従来の全データRidge回帰（勾配あり）
            U_B = self._ridge_stage2_ub_matrix_with_grad(H_features, M_target, self.lambda_dB)
            M_pred = (H_features @ U_B)  # (T-1, d_B) @ (d_B, m) = (T-1, m)
            return M_pred, U_B

        # クロスフィッティング実行
        try:
            from .cross_fitting import CrossFittingManager, TwoStageCrossFitter

            # クロスフィッティング管理
            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)
            cf_fitter = TwoStageCrossFitter(cf_manager)

            # U_B推定（クロスフィッティング、勾配あり）
            # 注: H_featuresは勾配あり（ω依存）
            U_B_list = cf_fitter.cross_fit_stage2_matrix(
                H_features, M_target,
                stage2_estimator=lambda H, M: self._ridge_stage2_ub_matrix_with_grad(H, M, self.lambda_dB)
            )

            # **理論準拠**: out-of-fold予測計算（勾配あり）
            M_pred_cf = torch.zeros_like(M_target)
            for k in range(cf_manager.n_blocks):
                block_indices = cf_manager.get_block_indices(k)
                H_block = H_features[block_indices]
                M_pred_cf[block_indices] = H_block @ U_B_list[k]

            # 最終U_B：全データでの推定（勾配あり、正則化用）
            U_B_final = self._ridge_stage2_ub_matrix_with_grad(H_features, M_target, self.lambda_dB)

            return M_pred_cf, U_B_final

        except ImportError:
            # フォールバック（勾配あり）
            U_B = self._ridge_stage2_ub_matrix_with_grad(H_features, M_target, self.lambda_dB)
            M_pred = H_features @ U_B  # (T-1, d_B) @ (d_B, m) = (T-1, m)
            return M_pred, U_B
        except Exception as e:
            print(f"U_Bクロスフィッティング失敗、従来方式を使用: {e}")
            U_B = self._ridge_stage2_ub_matrix_with_grad(H_features, M_target, self.lambda_dB)
            M_pred = H_features @ U_B  # (T-1, d_B) @ (d_B, m) = (T-1, m)
            return M_pred, U_B

    def _freeze_parameters(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        """パラメータを一時的に固定し、元の状態を保存"""
        original_states = {}
        for name, param in module.named_parameters():
            original_states[name] = param.requires_grad
            param.requires_grad = False
        return original_states

    def _restore_parameters(self, module: nn.Module, original_states: Dict[str, torch.Tensor]):
        """パラメータの requires_grad 状態を復元"""
        for name, param in module.named_parameters():
            if name in original_states:
                param.requires_grad = original_states[name]
    
    def train_stage1_with_gradients(
        self,
        X_hat_states: torch.Tensor,
        m_features: torch.Tensor,
        optimizer_phi: torch.optim.Optimizer,
        fix_psi_omega: bool = True,
        epoch: int = 0
    ) -> Dict[str, float]:
        """
        Stage-1学習（エポックごとに1ブロック処理）

        理論:
        L_Stage-1(θ) = Σ_{t∈B_k} ||ψ_ω(m_t) - V_B^{(-k)} φ_θ(x̂_{t|t-1})||² + λ_B ||V_B^{(-k)}||²_F

        注: ψ_ωは固定（fix_psi_omega=True）

        Args:
            X_hat_states: 状態予測系列 (T, r)
            m_features: 多変量特徴量 (T, m)
            optimizer_phi: φ_θ用オプティマイザ
            fix_psi_omega: ψ_ωを固定するか
            epoch: エポック番号

        Returns:
            Dict[str, float]: 損失メトリクス
        """
        T_x = X_hat_states.size(0)

        # 多変量特徴量の次元チェック
        if m_features.dim() != 2:
            raise ValueError(f"多変量特徴量は2次元 (T, m): got {m_features.shape}")

        if m_features.size(1) != self.multivariate_feature_dim:
            raise ValueError(f"特徴量次元不一致: expected {self.multivariate_feature_dim}, got {m_features.size(1)}")

        if T_x != m_features.size(0):
            error_details = {
                "X_hat_states_shape": tuple(X_hat_states.shape),
                "m_features_shape": tuple(m_features.shape),
                "size_diff": T_x - m_features.size(0)
            }
            raise ValueError(
                f"系列長不一致: X_hat_states={T_x} vs m_features={m_features.size(0)}. "
                f"詳細: {error_details}. "
                f"ヒント: 時間インデックス調整が必要です。"
            )

        if T_x < 2:
            raise ValueError(f"系列が短すぎます: T={T_x}")

        # ψ_ω パラメータの固定処理
        psi_original_states = {}
        if fix_psi_omega:
            psi_original_states = self._freeze_parameters(self.psi_omega)

        try:
            optimizer_phi.zero_grad()

            # 操作変数特徴量（φ_θ使用、勾配あり）
            phi_instrument = self.phi_theta(X_hat_states)  # (T, d_A)

            # # 診断: φ(X_hat)統計量
            # if not hasattr(self, '_phi_xhat_diag_logged'):
            #     with torch.no_grad():
            #         phi_mean = phi_instrument.abs().mean().item()
            #         phi_std = phi_instrument.std().item()
            #         phi_min = phi_instrument.min().item()
            #         phi_max = phi_instrument.max().item()
            #         print(f"[診断A-φ(X_hat)-epoch{epoch+1}] 範囲: [{phi_min:.3f}, {phi_max:.3f}], "
            #               f"平均絶対値: {phi_mean:.3f}, 標準偏差: {phi_std:.3f}")
            #     self._phi_xhat_diag_logged = True

            # 観測特徴量（ψ_ω固定、勾配なし）
            with torch.no_grad():
                psi_obs = self.psi_omega(m_features)  # (T, d_B)

            # 時間合わせ
            phi_prev = phi_instrument[:-1]  # (T-1, d_A)
            psi_curr = psi_obs[1:]          # (T-1, d_B)

            T_eff = phi_prev.size(0)

            # クロスフィッティング設定
            n_blocks = self.cf_config.get('n_blocks', 5)
            min_block_size = self.cf_config.get('min_block_size', 20)

            # データ量チェック
            if T_eff < max(n_blocks * min_block_size, 100):
                # 小データ時：クロスフィッティングなし
                V_B = self._ridge_stage1_vb_with_grad(phi_prev, psi_curr, self.lambda_B)
                psi_pred = (V_B @ phi_prev.T).T

                prediction_loss = torch.norm(psi_pred - psi_curr, p='fro') ** 2 / psi_curr.size(0)
                regularization_loss = self.lambda_B * torch.norm(V_B, p='fro') ** 2
                loss_stage1 = prediction_loss + regularization_loss

                loss_stage1.backward()
                optimizer_phi.step()

                # キャッシュ更新（通常パスと同じパターン）
                with torch.no_grad():
                    self._stage1_cache = {
                        'V_B': V_B,
                        'phi_prev': phi_prev,
                        'psi_curr': psi_curr,
                        'X_hat': X_hat_states
                    }

                return {
                    'stage1_loss': loss_stage1.item(),
                    'n_blocks': 0,
                    'mode': 'no_crossfitting'
                }

            # クロスフィッティング実行
            from .cross_fitting import CrossFittingManager

            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)

            # エポックに対応するブロックを選択
            k = epoch % cf_manager.n_blocks

            optimizer_phi.zero_grad()

            # 現在のφ_θで操作変数特徴量を計算
            phi_instrument = self.phi_theta(X_hat_states)
            phi_prev = phi_instrument[:-1]

            # 観測特徴量（ψ_ω固定）
            with torch.no_grad():
                psi_obs = self.psi_omega(m_features)
                psi_curr = psi_obs[1:]

            # out-of-foldインデックス
            oof_indices = cf_manager.get_out_of_fold_indices(k)
            phi_prev_oof = phi_prev[oof_indices]
            psi_curr_oof = psi_curr[oof_indices]

            # V_B^{(-k)}計算
            V_B_k = self._ridge_stage1_vb_with_grad(phi_prev_oof, psi_curr_oof, self.lambda_B)

            # in-foldインデックス
            block_indices = cf_manager.get_block_indices(k)
            phi_prev_block = phi_prev[block_indices]
            psi_curr_block = psi_curr[block_indices]

            # 予測
            psi_pred_block = (V_B_k @ phi_prev_block.T).T

            # ブロックkの損失
            prediction_loss_k = torch.norm(psi_pred_block - psi_curr_block, p='fro') ** 2 / psi_curr_block.size(0)
            regularization_loss_k = self.lambda_B * torch.norm(V_B_k, p='fro') ** 2
            loss_k = prediction_loss_k + regularization_loss_k

            # # 診断: DF-B損失詳細・V_Bノルム
            # if not hasattr(self, '_df_b_loss_logged'):
            #     print(f"[診断-DF-B-S1-epoch{epoch+1}-Block{k}] 損失 - 予測: {prediction_loss_k.item():.6e}, "
            #           f"正則化: {regularization_loss_k.item():.6e}")
            #     print(f"[診断-DF-B-S1-epoch{epoch+1}-Block{k}] ψ_pred範囲: [{psi_pred_block.min().item():.3f}, "
            #           f"{psi_pred_block.max().item():.3f}], ψ_curr範囲: [{psi_curr_block.min().item():.3f}, "
            #           f"{psi_curr_block.max().item():.3f}]")
            #     print(f"[診断-DF-B-S1-epoch{epoch+1}-Block{k}] ブロックサイズ: {psi_curr_block.size(0)}, "
            #           f"特徴次元: {psi_curr_block.size(1)}")
            #     self._df_b_loss_logged = True

            loss_k.backward()
            optimizer_phi.step()

            # # 診断: V_Bノルム追跡
            # with torch.no_grad():
            #     V_B_norm = torch.norm(V_B_k, p='fro').item()
            #     print(f"[診断D-V_B-epoch{epoch+1}] V_Bノルム: {V_B_norm:.2f}")

            # キャッシュ更新
            with torch.no_grad():
                phi_final = self.phi_theta(X_hat_states)
                psi_final = self.psi_omega(m_features)

                phi_prev_final = phi_final[:-1]
                psi_curr_final = psi_final[1:]
                V_B_final = self._ridge_stage1_vb(phi_prev_final, psi_curr_final, self.lambda_B)

                self._stage1_cache = {
                    'V_B': V_B_final,
                    'phi_prev': phi_prev_final,
                    'psi_curr': psi_curr_final,
                    'X_hat': X_hat_states
                }

            return {
                'stage1_loss': loss_k.item(),
                'current_block': k,
                'n_blocks': cf_manager.n_blocks
            }

        finally:
            # ψ_ω パラメータの復元
            if fix_psi_omega and psi_original_states:
                self._restore_parameters(self.psi_omega, psi_original_states)

    
    def train_stage2_with_gradients(
            self,
            M_features: torch.Tensor,
            optimizer_psi: torch.optim.Optimizer,
            fix_phi_theta: bool = True,
            epoch: int = 0
        ) -> Dict[str, float]:
        """
        Stage-2学習（エポックごとに1ブロック処理）

        理論:
        L_Stage-2(ω) = Σ_{t∈B_k} ||M_t - U_B^{(-k)}^T H_k||² + λ_{dB} ||U_B^{(-k)}||²_F

        注: φ_θは固定（fix_phi_theta=True）

        Args:
            M_features: 多変量特徴量 (T, m)
            optimizer_psi: ψ_ω用オプティマイザ
            fix_phi_theta: φ_θを固定するか
            epoch: エポック番号

        Returns:
            Dict[str, float]: 損失メトリクス
        """
        if 'V_B' not in self._stage1_cache:
            raise RuntimeError("Stage-1が先に実行されている必要があります")

        # φ_θ パラメータの固定処理
        phi_original_states = {}
        if fix_phi_theta:
            phi_original_states = self._freeze_parameters(self.phi_theta)

        try:
            # Stage-1からの結果取得
            phi_prev = self._stage1_cache['phi_prev']
            T_eff = phi_prev.size(0)

            # 境界チェック
            if M_features.size(0) < T_eff + 1:
                raise RuntimeError(f"M_features長不足: required {T_eff+1}, got {M_features.size(0)}")
            M_curr = M_features[1:T_eff+1]  # (T-1, m)

            optimizer_psi.zero_grad()

            # ψ_ω依存のV_B動的計算
            psi_obs_current = self.psi_omega(M_features)
            psi_curr_grad = psi_obs_current[1:T_eff+1]  # (T-1, d_B) 勾配あり

            # V_B動的計算（ψ_ω勾配あり、φ_θ固定）
            V_B_current = self._ridge_stage1_vb_with_grad(phi_prev, psi_curr_grad, self.lambda_B)

            # H計算（ψ_ω勾配あり）
            H = (V_B_current @ phi_prev.T).T  # (T-1, d_B)

            # クロスフィッティング設定
            n_blocks = self.cf_config.get('n_blocks', 5)
            min_block_size = self.cf_config.get('min_block_size', 20)

            # データ量チェック
            if T_eff < max(n_blocks * min_block_size, 100):
                # 小データ時：クロスフィッティングなし
                U_B = self._ridge_stage2_ub_matrix_with_grad(H, M_curr, self.lambda_dB)
                M_pred = H @ U_B  # (T-1, m)

                prediction_loss = torch.norm(M_pred - M_curr, p='fro') ** 2 / M_curr.size(0)
                regularization_loss = self.lambda_dB * torch.norm(U_B, p='fro') ** 2
                loss_stage2 = prediction_loss + regularization_loss

                loss_stage2.backward()
                optimizer_psi.step()

                self._stage2_cache['U_B'] = U_B.detach()
                return {'stage2_loss': loss_stage2.item(), 'n_blocks': 0, 'mode': 'no_crossfitting'}

            # クロスフィッティング実行
            from .cross_fitting import CrossFittingManager
            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)

            # エポックに対応するブロックを選択
            k = epoch % cf_manager.n_blocks

            optimizer_psi.zero_grad()

            # 現在のψ_ωでHを計算
            psi_obs_current = self.psi_omega(M_features)
            psi_curr_grad = psi_obs_current[1:T_eff+1]
            V_B_current = self._ridge_stage1_vb_with_grad(phi_prev, psi_curr_grad, self.lambda_B)
            H = (V_B_current @ phi_prev.T).T

            # out-of-foldでU_B^{(-k)}計算
            oof_indices = cf_manager.get_out_of_fold_indices(k)
            U_B_k = self._ridge_stage2_ub_matrix_with_grad(H[oof_indices], M_curr[oof_indices], self.lambda_dB)

            # in-foldブロックで予測
            block_indices = cf_manager.get_block_indices(k)
            M_pred_k = H[block_indices] @ U_B_k

            # ブロックkの損失
            pred_loss_k = torch.norm(M_pred_k - M_curr[block_indices], p='fro') ** 2 / M_curr[block_indices].size(0)
            reg_loss_k = self.lambda_dB * torch.norm(U_B_k, p='fro') ** 2
            loss_k = pred_loss_k + reg_loss_k

            # ブロックkでbackward
            loss_k.backward()

            # ブロックkで更新
            optimizer_psi.step()

            # 推論用キャッシュ更新（毎回全データで再計算、φ_θ・ψ_ω更新を反映）
            with torch.no_grad():
                X_hat_states = self._stage1_cache['X_hat']
                phi_final = self.phi_theta(X_hat_states)
                psi_final = self.psi_omega(M_features)
                phi_prev_final = phi_final[:-1]
                psi_curr_final = psi_final[1:T_eff+1]
                V_B_final = self._ridge_stage1_vb(phi_prev_final, psi_curr_final, self.lambda_B)
                H_final = (V_B_final @ phi_prev_final.T).T
                M_curr_final = M_features[1:T_eff+1]
                U_B_final = self._ridge_stage2_ub_matrix(H_final, M_curr_final, self.lambda_dB)
                self._stage2_cache['U_B'] = U_B_final

            return {
                'stage2_loss': loss_k.item(),
                'current_block': k,
                'n_blocks': cf_manager.n_blocks
            }

        finally:
            # φ_θ パラメータの復元
            if fix_phi_theta and phi_original_states:
                self._restore_parameters(self.phi_theta, phi_original_states)
    
    def fit_two_stage(
        self,
        X_hat_states: torch.Tensor,  # DF-Aからの状態予測
        M_features: torch.Tensor,    # エンコーダからの多変量特徴量
        use_cross_fitting: bool = True,
        verbose: bool = False
    ) -> 'DFObservationLayer':
        """
        従来の2段階クロスフィッティング学習（多変量対応）

        **注意**: これは既存の学習メソッドです。
        新しいPhase-1学習では train_stage1_with_gradients と
        train_stage2_with_gradients を使用してください。
        """
        T_x, r = X_hat_states.shape

        # 多変量特徴量の次元チェック
        if M_features.dim() != 2:
            raise ValueError(f"多変量特徴量は2次元であるべき: got shape {M_features.shape}")

        if M_features.size(1) != self.multivariate_feature_dim:
            raise ValueError(f"特徴量次元不一致: expected {self.multivariate_feature_dim}, got {M_features.size(1)}")

        T_m = M_features.size(0)
        
        if T_x != T_m:
            raise ValueError(f"状態予測と特徴量の時系列長不一致: {T_x} vs {T_m}")
        
        if T_x < 2:
            raise ValueError(f"時系列が短すぎます: T={T_x}")
        
        # 操作変数特徴量（状態予測から）
        with torch.no_grad():
            Phi_instrument = self.phi_theta(X_hat_states)  # (T, d_A)
        
        # 観測特徴量
        with torch.no_grad():
            Psi_obs = self.psi_omega(M_features)  # (T, d_B)
        
        # 時間合わせ: 現時刻の観測を次時刻で予測
        # 操作変数: t-1 時刻の状態予測特徴量
        # 目標: t 時刻の観測特徴量
        Phi_prev = Phi_instrument[:-1]  # (T-1, d_A)
        Psi_curr = Psi_obs[1:]          # (T-1, d_B)
        M_curr = M_features[1:]         # (T-1, m)
        
        if use_cross_fitting and T_x >= 20:
            self._fit_with_cross_fitting(Phi_prev, Psi_curr, M_curr, verbose)
        else:
            self._fit_without_cross_fitting(Phi_prev, Psi_curr, M_curr, verbose)
        
        self._is_fitted = True
        return self
    
    def _fit_with_cross_fitting(
        self,
        Phi_prev: torch.Tensor,
        Psi_curr: torch.Tensor,
        M_curr: torch.Tensor,
        verbose: bool
    ):
        """クロスフィッティング付き学習（多変量対応）"""
        T_eff = Phi_prev.size(0)  # T-1
        
        # クロスフィッティング管理
        cf_manager = CrossFittingManager(T_eff, **self.cf_config)
        cf_fitter = TwoStageCrossFitter(cf_manager)
        
        if verbose:
            print(f"DF-B クロスフィッティング: T={T_eff}, n_blocks={cf_manager.n_blocks}")
        
        # Stage-1: 観測転送作用素推定（クロスフィッティング）
        VB_list = cf_fitter.cross_fit_stage1(
            Phi_prev, Psi_curr,
            self._ridge_stage1_vb,
            reg_lambda=self.lambda_B
        )
        
        # 平均観測転送作用素（最終的な V_B）
        self.V_B = torch.stack(VB_list).mean(dim=0)
        
        # Out-of-fold操作変数特徴量計算
        H_cf = cf_fitter.compute_out_of_fold_features(Phi_prev, VB_list)
        
        # Stage-2: 観測読み出し行列推定（多変量版）
        U_B_list = cf_fitter.cross_fit_stage2_matrix(
            H_cf, M_curr,
            self._ridge_stage2_ub_matrix,
            detach_features=True,
            reg_lambda=self.lambda_dB
        )

        # 平均U_B行列（最終的な U_B）
        self.U_B = torch.stack(U_B_list).mean(dim=0)

        if verbose:
            print(f"V_B shape: {self.V_B.shape}, U_B shape: {self.U_B.shape}")
    
    def _fit_without_cross_fitting(
        self,
        Phi_prev: torch.Tensor,
        Psi_curr: torch.Tensor,
        M_curr: torch.Tensor,
        verbose: bool
    ):
        """クロスフィッティングなし学習（小データ用、多変量対応）"""
        if verbose:
            print("DF-B クロスフィッティングなしで学習（多変量版）")

        # Stage-1: 直接推定
        self.V_B = self._ridge_stage1_vb(Phi_prev, Psi_curr, self.lambda_B)

        # 中間特徴量
        H = (self.V_B @ Phi_prev.T).T  # (T-1, d_B)

        # Stage-2: 読み出し行列推定（多変量版）
        self.U_B = self._ridge_stage2_ub_matrix(H, M_curr, self.lambda_dB)

        if verbose:
            print(f"V_B shape: {self.V_B.shape}, U_B shape: {self.U_B.shape}")
    
    def predict_one_step(self, x_hat_prev: torch.Tensor) -> torch.Tensor:
        """
        1ステップ特徴量予測: M̂_{t|t-1} = U_B^T V_B φ_θ(x̂_{t|t-1})

        Args:
            x_hat_prev: 前時刻の状態予測 (r,) または (batch, r)

        Returns:
            torch.Tensor: 予測多変量特徴量 (m,) または (batch, m)
        """
        # 次元チェックを追加
        expected_state_dim = self.df_state.phi_theta.net[0].in_features
        if x_hat_prev.dim() == 1:
            actual_dim = x_hat_prev.shape[0]
        else:
            actual_dim = x_hat_prev.shape[-1]

        if actual_dim != expected_state_dim:
            raise ValueError(
                f"predict_one_step expects state input with dimension {expected_state_dim}, "
                f"but got {actual_dim}. "
                f"Input shape: {x_hat_prev.shape}. "
                f"Note: This method expects x_hat_prev (state), not M_features (multivariate features)."
            )

        if not self._is_fitted:
            # **修正**: キャッシュされた結果も使用可能に
            if 'V_B' in self._stage1_cache and 'U_B' in self._stage2_cache:
                V_B = self._stage1_cache['V_B']
                U_B = self._stage2_cache['U_B']
            else:
                raise RuntimeError("fit_two_stage() または train_stage1/2_with_gradients() を先に実行してください")
        else:
            V_B = self.V_B
            U_B = self.U_B
        
        # 状態特徴写像（共有φ_θ）
        phi_prev = self.phi_theta(x_hat_prev)

        # 観測転送作用素適用 + 読み出し（GPUデバイス整合性を確保）
        V_B = V_B.to(phi_prev.device)
        U_B = U_B.to(phi_prev.device)
        if phi_prev.dim() == 1:
            h_pred = V_B @ phi_prev  # (d_B,)
            return U_B.T @ h_pred    # (m,) - スカラー入力: U_B^T @ h
        else:
            h_pred = (V_B @ phi_prev.T).T  # (batch, d_B)
            return h_pred @ U_B
    
    def predict_sequence(
        self,
        X_hat_states: torch.Tensor
    ) -> torch.Tensor:
        """
        系列予測: 各時刻でのone-step-ahead特徴量予測

        Args:
            X_hat_states: 状態予測系列 (T, r)

        Returns:
            torch.Tensor: 予測特徴量系列 (T-1, m)
        """
        if not self._is_fitted and 'V_B' not in self._stage1_cache:
            raise RuntimeError("学習が完了していません")

        T = X_hat_states.size(0)
        predictions = []

        for t in range(T - 1):
            M_pred = self.predict_one_step(X_hat_states[t])
            predictions.append(M_pred)

        return torch.stack(predictions)
    
    def get_observation_operator(self) -> torch.Tensor:
        """観測転送作用素 V_B を取得"""
        if self._is_fitted:
            return self.V_B.clone()
        elif 'V_B' in self._stage1_cache:
            return self._stage1_cache['V_B'].clone()
        else:
            raise RuntimeError("学習が完了していません")
    
    def get_readout_matrix(self) -> torch.Tensor:
        """観測読み出し行列 U_B を取得"""
        if self._is_fitted:
            return self.U_B.clone()
        elif 'U_B' in self._stage2_cache:
            return self._stage2_cache['U_B'].clone()
        else:
            raise RuntimeError("学習が完了していません")
    
    def get_state_dict(self) -> Dict[str, Any]:
        """学習済みパラメータを辞書で取得"""
        state_dict = {
            'psi_omega': self.psi_omega.state_dict(),
            'config': {
                'obs_feature_dim': self.obs_feature_dim,
                'lambda_B': self.lambda_B,
                'lambda_dB': self.lambda_dB
            }
        }
        
        if self._is_fitted:
            state_dict.update({
                'V_B': self.V_B,
                'U_B': self.U_B,
            })
        
        if self._stage1_cache:
            state_dict['stage1_cache'] = self._stage1_cache.copy()
        if self._stage2_cache:
            state_dict['stage2_cache'] = self._stage2_cache.copy()
            
        return state_dict
    
    def get_inference_state_dict(self) -> Dict[str, Any]:
        """推論用のstate_dictを取得（filtering評価用にV_B/u_Bを含める）"""
        state_dict = {
            'psi_omega': self.psi_omega.state_dict(),
        }

        # filtering評価では学習済みV_B, U_Bが必要なので含める
        if hasattr(self, 'V_B') and self.V_B is not None:
            state_dict['V_B'] = self.V_B
        if hasattr(self, 'U_B') and self.U_B is not None:
            state_dict['U_B'] = self.U_B

        # phi_theta は推論時にdf_state_layerから共有参照されるため保存不要
        # configは除外（推論時には設定ファイルから読み込むため不要）
        # キャッシュも推論には不要なので除外

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """カスタムload_state_dict: V_B/U_Bも適切に設定"""
        # V_B, U_Bを別途処理
        v_b = state_dict.pop('V_B', None)
        u_b = state_dict.pop('U_B', None)

        # 通常のパラメータを読み込み
        super().load_state_dict(state_dict, strict=strict)

        # V_B, U_Bを設定
        if v_b is not None:
            self.V_B = v_b.to(self.device) if hasattr(self, 'device') else v_b
        if u_b is not None:
            self.U_B = u_b.to(self.device) if hasattr(self, 'device') else u_b