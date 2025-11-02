# src/ssm/df_state_layer.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import warnings

from .cross_fitting import CrossFittingManager, TwoStageCrossFitter, CrossFittingError


class StateFeatureNet(nn.Module):
    """
    状態特徴写像 ϕ_θ: R^r → R^{d_A}
    
    状態変数を高次元特徴空間に写像するニューラルネットワーク。
    DF-AとDF-Bで共有される。
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_sizes: list[int] = [64, 64],
        activation: str = "ReLU",
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: 状態次元 r
            output_dim: 特徴次元 d_A
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 状態 (batch_size, r) または (r,)

        Returns:
            torch.Tensor: 特徴 (batch_size, d_A) または (d_A,)
        """
        # GPUデバイス整合性を確保
        if hasattr(self.net, 'to'):
            self.net = self.net.to(x.device)

        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.net(x).squeeze(0)
        return self.net(x)


class DFStateLayer(nn.Module):
    """
    DF-A: Deep Feature Instrumental Variable for State Process
    
    資料のSection 1.4.1に対応。状態系列からの1ステップ予測を
    2段階回帰（2SLS）とクロスフィッティングで実現。
    
    **修正点**: クロスフィッティングの一貫性向上
    
    計算フロー:
    1. ϕ_θ(x_t) で状態を特徴空間に写像
    2. Stage-1: V_A^{(-k)}推定（クロスフィッティング）+ ϕ_θ勾配更新
    3. Stage-2: U_A推定（閉形式解のみ）
    4. 予測: x̂_{t|t-1} = U_A^T V_A ϕ_θ(x_{t-1})
    """
    
    def __init__(
        self,
        state_dim: int,
        feature_dim: int,
        lambda_A: float = 1e-3,
        lambda_B: float = 1e-3,
        feature_net_config: Optional[Dict[str, Any]] = None,
        cross_fitting_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            state_dim: 状態次元 r
            feature_dim: 特徴次元 d_A
            lambda_A: Stage-1正則化パラメータ λ_A
            lambda_B: Stage-2正則化パラメータ λ_B
            feature_net_config: StateFeatureNetの設定
            cross_fitting_config: CrossFittingManagerの設定
        """
        super().__init__()
        self.state_dim = int(state_dim)
        self.feature_dim = int(feature_dim)
        self.lambda_A = float(lambda_A)  # 文字列対応
        self.lambda_B = float(lambda_B)  # 文字列対応
        
        # 特徴ネットワーク
        feature_config = feature_net_config or {}
        self.phi_theta = StateFeatureNet(
            input_dim=state_dim,
            output_dim=feature_dim,
            **feature_config
        )
        
        # クロスフィッティング設定
        self.cf_config = cross_fitting_config or {'n_blocks': 5, 'min_block_size': 10}
        
        # 学習済みパラメータ
        self.V_A: Optional[torch.Tensor] = None  # 転送作用素 (d_A, d_A)
        self.U_A: Optional[torch.Tensor] = None  # 読み出し行列 (d_A, r)
        self._is_fitted = False

        self._stage1_cache = {}  # V_A計算結果キャッシュ
        self._stage2_cache = {}  # U_A計算結果キャッシュ
        self._cf_manager: Optional[CrossFittingManager] = None
        # # self._diagnostic_call_count = 0  # 診断用カウンター
    

    
    def _ridge_stage1(
        self, 
        X_features: torch.Tensor, 
        Y_targets: torch.Tensor, 
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-1 Ridge回帰: 転送作用素推定
        
        V = (Y^T X)(X^T X + λI)^{-1}
        """

        N, d_A = X_features.shape
        N_t, d_A_t = Y_targets.shape
        
        # 明示的な型変換を追加
        N = int(N.item() if hasattr(N, 'item') else N)
        d_A = int(d_A.item() if hasattr(d_A, 'item') else d_A)
        N_t = int(N_t.item() if hasattr(N_t, 'item') else N_t)
        d_A_t = int(d_A_t.item() if hasattr(d_A_t, 'item') else d_A_t)
        
        if N != N_t:
            raise ValueError(f"特徴量とターゲットのサンプル数不一致: {N} vs {N_t}")
        
        if d_A != d_A_t:
            raise ValueError(f"特徴量とターゲットの次元不一致: {d_A} vs {d_A_t}")
        
        if N < d_A:
            warnings.warn(f"サンプル数 {N} < 特徴次元 {d_A}。数値不安定の可能性")
        
        # デバッグプリント追加 debug - コメントアウト
        
        # グラム行列 + 正則化（d_Aは確実にint）
        XtX = X_features.T @ X_features  # (d_A, d_A)
        XtX_reg = XtX + reg_lambda * torch.eye(d_A).type_as(XtX).to(XtX.device)
        
        # クロス共分散
        YtX = Y_targets.T @ X_features  # (d_A, d_A)
        
        # 逆行列計算（数値安定化 + デバイス維持）
        original_device = X_features.device
        try:
            XtX_inv = torch.linalg.inv(XtX_reg)
            V = YtX @ XtX_inv
        except torch.linalg.LinAlgError:
            # Cholesky分解 fallback
            try:
                L = torch.linalg.cholesky(XtX_reg)
                XtX_inv = torch.cholesky_inverse(L)
                V = YtX @ XtX_inv
            except torch.linalg.LinAlgError:
                # SVD fallback
                U, S, Vh = torch.linalg.svd(XtX_reg)
                S_inv = torch.where(S > 1e-10, 1.0 / S, 0.0)
                XtX_inv = (Vh.T * S_inv) @ Vh
                V = YtX @ XtX_inv

        # 強制的に元のデバイスに保持（linalg操作でのCPU移動を防止）
        V = V.to(original_device)
        return V

    def _ridge_stage1_with_grad(
        self,
        X_features: torch.Tensor,
        Y_targets: torch.Tensor,
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-1 Ridge回帰（勾配計算あり）: φ_θ更新用

        V = (Y^T X)(X^T X + λI)^{-1}
        """
        N, d_A = X_features.shape
        N_t, d_A_t = Y_targets.shape

        if N != N_t:
            raise ValueError(f"特徴量とターゲットのサンプル数不一致: {N} vs {N_t}")

        if d_A != d_A_t:
            raise ValueError(f"特徴量とターゲットの次元不一致: {d_A} vs {d_A_t}")

        # グラム行列 + 正則化（勾配計算あり）
        XtX = X_features.T @ X_features  # (d_A, d_A)
        XtX_reg = XtX + reg_lambda * torch.eye(d_A, device=X_features.device, dtype=X_features.dtype)

        # クロス共分散（勾配計算あり）
        YtX = Y_targets.T @ X_features  # (d_A, d_A)

        # 逆行列計算（勾配計算あり + デバイス維持）
        original_device = X_features.device
        try:
            XtX_inv = torch.linalg.inv(XtX_reg)
            V = YtX @ XtX_inv
        except torch.linalg.LinAlgError:
            # フォールバック：疑似逆行列
            XtX_inv = torch.linalg.pinv(XtX_reg)
            V = YtX @ XtX_inv

        # 強制的に元のデバイスに保持（勾配グラフも維持）
        V = V.to(original_device)
        return V

    def _ridge_stage2(
        self, 
        H_features: torch.Tensor, 
        X_targets: torch.Tensor, 
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-2 Ridge回帰: 読み出し行列推定
        
        U = (H H^T + λI)^{-1} H X^T
        
        Args:
            H_features: クロスフィット特徴量 H^{(cf)}_A (N, d_A)
            X_targets: 目標状態 X^+ (N, r)
            reg_lambda: 正則化パラメータ λ_B
            
        Returns:
            torch.Tensor: 読み出し行列 U (d_A, r)
        """
        N, d_A = H_features.shape
        N_t, r = X_targets.shape
        
        if N != N_t:
            raise ValueError(f"特徴量とターゲットのサンプル数不一致: {N} vs {N_t}")
        
        # グラム行列 + 正則化
        HHt = H_features.T @ H_features  # (d_A, d_A)
        HHt_reg = HHt + reg_lambda * torch.eye(d_A, device=H_features.device, dtype=H_features.dtype)
        
        # クロス項
        HXt = H_features.T @ X_targets  # (d_A, r)
        
        # 逆行列計算（数値安定化 + デバイス維持）
        original_device = H_features.device
        try:
            HHt_inv = torch.linalg.inv(HHt_reg)
            U = HHt_inv @ HXt
        except torch.linalg.LinAlgError:
            # Cholesky分解 fallback
            try:
                L = torch.linalg.cholesky(HHt_reg)
                HHt_inv = torch.cholesky_inverse(L)
                U = HHt_inv @ HXt
            except torch.linalg.LinAlgError:
                # SVD fallback
                U_svd, S, Vh = torch.linalg.svd(HHt_reg)
                S_inv = torch.where(S > 1e-10, 1.0 / S, 0.0)
                HHt_inv = (Vh.T * S_inv) @ Vh
                U = HHt_inv @ HXt

        # 強制的に元のデバイスに保持（linalg操作でのCPU移動を防止）
        U = U.to(original_device)
        return U
    
    def _initialize_cross_fitting(self, T_eff: int) -> CrossFittingManager:
        """
        **新機能**: クロスフィッティング管理の初期化
        
        Args:
            T_eff: 有効時系列長
            
        Returns:
            CrossFittingManager: 初期化されたクロスフィッティング管理
        """
        # データサイズに応じてクロスフィッティング設定を調整
        cf_config = self.cf_config.copy()
        
        # 最小ブロックサイズの確保
        min_block_size = cf_config.get('min_block_size', 10)
        max_blocks = T_eff // min_block_size
        n_blocks = min(cf_config.get('n_blocks', 5), max_blocks)
        
        if n_blocks < 2:
            # データが小さすぎる場合は、非クロスフィッティング
            warnings.warn(f"データサイズ {T_eff} が小さすぎるため、クロスフィッティングを無効化")
            return None
        
        cf_config['n_blocks'] = n_blocks
        
        return CrossFittingManager(T_eff, **cf_config)
    
    def _compute_crossfit_stage1_loss(
        self, 
        X_states: torch.Tensor, 
        use_simple_fallback: bool = False
    ) -> torch.Tensor:
        """
        **修正版**: クロスフィッティング対応のStage-1損失計算
        
        Args:
            X_states: 状態系列 (T, r)
            use_simple_fallback: 簡易版を使用するかどうか
            
        Returns:
            torch.Tensor: Stage-1損失（スカラー）
        """
        T, r = X_states.shape
        
        # 特徴量計算
        phi_seq = self.phi_theta(X_states)  # (T, d_A)
        
        # 過去/未来特徴量分割
        phi_minus = phi_seq[:-1]  # (T-1, d_A)
        phi_plus = phi_seq[1:]    # (T-1, d_A)
        
        T_eff = phi_minus.size(0)
        
        # **修正**: クロスフィッティングの実装
        if use_simple_fallback or T_eff < 20:
            # 小データまたは簡易版：全データで推定（従来の実装）
            V_A = self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)
            phi_pred = (V_A @ phi_minus.T).T
            loss = torch.norm(phi_pred - phi_plus, p='fro') ** 2 / phi_plus.numel()
            
            # **追加**: 非クロスフィッティング用キャッシュクリア
            self._stage1_cache.pop('V_A_list', None)
            self._stage1_cache.pop('cf_manager', None)
        else:
            # **修正**: 真のクロスフィッティング実装
            cf_manager = self._initialize_cross_fitting(T_eff)
            
            if cf_manager is None:
                # フォールバック：全データ使用
                V_A = self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)
                phi_pred = (V_A @ phi_minus.T).T
                loss = torch.norm(phi_pred - phi_plus, p='fro') ** 2 / phi_plus.numel()
                
                # **追加**: 非クロスフィッティング用キャッシュクリア
                self._stage1_cache.pop('V_A_list', None)
                self._stage1_cache.pop('cf_manager', None)
            else:
                # クロスフィッティングによる損失計算
                cf_fitter = TwoStageCrossFitter(cf_manager)
                
                # Stage-1: V_A^{(-k)} 推定（勾配なし）
                with torch.no_grad():
                    V_list = cf_fitter.cross_fit_stage1(
                        phi_minus, phi_plus,  # **修正**: detach()削除
                        self._ridge_stage1,
                        reg_lambda=self.lambda_A
                    )
                
                # Out-of-fold予測誤差の計算（勾配あり）
                total_loss = 0.0
                for k in range(cf_manager.n_blocks):
                    # ブロックkのインデックス
                    block_indices = cf_manager.get_block_indices(k)
                    
                    # ブロックkでの予測（勾配あり）
                    phi_minus_k = phi_minus[block_indices]  # 勾配あり
                    phi_plus_k = phi_plus[block_indices]    # 勾配あり
                    
                    # V_A^{(-k)}による予測
                    V_k = V_list[k]  # 勾配なし（detach済み）
                    phi_pred_k = (V_k @ phi_minus_k.T).T
                    
                    # ブロックkの損失
                    loss_k = torch.norm(phi_pred_k - phi_plus_k, p='fro') ** 2 / phi_plus_k.numel()
                    total_loss += loss_k
                
                loss = total_loss / cf_manager.n_blocks
                
                # キャッシュ更新（平均転送作用素）
                self._stage1_cache['V_A_list'] = V_list
                self._stage1_cache['cf_manager'] = cf_manager
        
        return loss
    
    def train_stage1_with_gradients(
        self,
        X_states: torch.Tensor,
        optimizer_phi: torch.optim.Optimizer,
        epoch: int = 0
    ) -> Dict[str, float]:
        """
        Stage-1学習（エポックごとに1ブロック処理）

        理論（式42a）:
        L_Stage-1(θ) = Σ_{t∈B_k} ||φ_θ(x_t) - V_A^{(-k)} φ_θ(x_{t-1})||² + λ_A ||V_A^{(-k)}||²_F

        ポイント:
        - エポックeではブロック k = (e mod K) を処理
        - 1エポック = 1回パラメータ更新
        - K個のエポックで全ブロックを1巡

        Args:
            X_states: 状態系列 (T, r)
            optimizer_phi: φ_θ用オプティマイザ
            epoch: エポック番号

        Returns:
            Dict[str, float]: 損失メトリクス
        """
        if X_states.size(0) < 2:
            raise ValueError(f"状態系列が短すぎます: T={X_states.size(0)}")

        optimizer_phi.zero_grad()

        # 特徴量計算（1回のみ）
        phi_seq = self.phi_theta(X_states)  # (T, d_A)

        phi_minus = phi_seq[:-1]  # (T-1, d_A)
        phi_plus = phi_seq[1:]    # (T-1, d_A)

        T_eff = phi_minus.size(0)

        # クロスフィッティング設定
        n_blocks = self.cf_config.get('n_blocks', 5)
        min_block_size = self.cf_config.get('min_block_size', 20)

        # データ量チェック
        if T_eff < max(n_blocks * min_block_size, 100):
            # 小データ時：クロスフィッティングなし（全データRidge）
            V_A = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)
            phi_pred = (V_A @ phi_minus.T).T

            prediction_loss = torch.norm(phi_pred - phi_plus, p='fro') ** 2 / phi_plus.size(0)
            regularization_loss = self.lambda_A * torch.norm(V_A, p='fro') ** 2
            loss_stage1 = prediction_loss + regularization_loss

            loss_stage1.backward()

            # # 診断: φ勾配測定・損失バランス・φ/V_A変化量
            # if not hasattr(self, '_diagnostic_call_count'):
            #     self._diagnostic_call_count = 0
            # self._diagnostic_call_count += 1
            # if self._diagnostic_call_count <= 5:
            #     total_grad_norm = 0.0
            #     max_grad = 0.0
            #     min_grad = float('inf')
            #     n_params_with_grad = 0
            #     for p in self.phi_theta.parameters():
            #         if p.grad is not None:
            #             param_grad_norm = p.grad.norm().item()
            #             total_grad_norm += param_grad_norm ** 2
            #             max_grad = max(max_grad, p.grad.abs().max().item())
            #             min_grad = min(min_grad, p.grad.abs().min().item())
            #             n_params_with_grad += 1
            #     total_grad_norm = total_grad_norm ** 0.5
            #     print(f"[診断-DF-A-S1-通常-{self._diagnostic_call_count}] φ勾配ノルム: {total_grad_norm:.6e}, "
            #           f"最大: {max_grad:.6e}, 最小: {min_grad:.6e}, パラメータ数: {n_params_with_grad}")
            #     if prediction_loss.item() > 0:
            #         loss_ratio = regularization_loss.item() / prediction_loss.item()
            #     else:
            #         loss_ratio = float('inf')
            #     print(f"[診断-DF-A-S1-通常-{self._diagnostic_call_count}] 損失 - 予測: {prediction_loss.item():.6e}, "
            #           f"正則化: {regularization_loss.item():.6e}, 比率: {loss_ratio:.2f}")

            optimizer_phi.step()

            # # 診断: φ/V_A変化量測定
            # if self._diagnostic_call_count <= 5:
            #     with torch.no_grad():
            #         phi_seq_after = self.phi_theta(X_states)
            #         phi_change = torch.norm(phi_seq_after - phi_seq_before)
            #         print(f"[診断-DF-A-S1-通常-{self._diagnostic_call_count}] φ変化量: {phi_change.item():.6e}")
            #         if hasattr(self, '_prev_V_A'):
            #             V_A_change = torch.norm(V_A - self._prev_V_A)
            #             print(f"[診断-DF-A-S1-通常-{self._diagnostic_call_count}] V_A変化量: {V_A_change.item():.6e}")
            #         self._prev_V_A = V_A.detach().clone()

            # キャッシュ更新
            with torch.no_grad():
                self._stage1_cache = {
                    'V_A': V_A,
                    'phi_minus': phi_minus,
                    'phi_plus': phi_plus,
                    'X_plus': X_states[1:]
                }

            return {
                'stage1_loss': loss_stage1.item(),
                'n_blocks': 0,
                'mode': 'no_crossfitting'
            }

        # クロスフィッティング実行
        from .cross_fitting import CrossFittingManager

        cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)

        # # 診断: カウンター初期化
        # if not hasattr(self, '_diagnostic_call_count'):
        #     self._diagnostic_call_count = 0
        # self._diagnostic_call_count += 1

        k = epoch % cf_manager.n_blocks
        optimizer_phi.zero_grad()

        # # 診断: φ学習前出力保存
        # with torch.no_grad():
        #     phi_seq_before = self.phi_theta(X_states).detach().clone()

        # 現在のφ_θで特徴量を計算
        phi_seq = self.phi_theta(X_states)
        phi_minus = phi_seq[:-1]
        phi_plus = phi_seq[1:]

        # out-of-foldインデックス取得
        oof_indices = cf_manager.get_out_of_fold_indices(k)
        phi_minus_oof = phi_minus[oof_indices]
        phi_plus_oof = phi_plus[oof_indices]

        # V_A^{(-k)}計算
        V_A_k = self._ridge_stage1_with_grad(phi_minus_oof, phi_plus_oof, self.lambda_A)

        # in-foldインデックス取得
        block_indices = cf_manager.get_block_indices(k)
        phi_minus_block = phi_minus[block_indices]
        phi_plus_block = phi_plus[block_indices]

        # 予測
        phi_pred_block = (V_A_k @ phi_minus_block.T).T

        # ブロックkの損失
        prediction_loss_k = torch.norm(phi_pred_block - phi_plus_block, p='fro') ** 2 / phi_plus_block.size(0)
        regularization_loss_k = self.lambda_A * torch.norm(V_A_k, p='fro') ** 2
        loss_k = prediction_loss_k + regularization_loss_k

        # # 診断: ブロック損失測定
        # if self._diagnostic_call_count <= 1:
        #     if prediction_loss_k.item() > 0:
        #         loss_ratio_k = regularization_loss_k.item() / prediction_loss_k.item()
        #     else:
        #         loss_ratio_k = float('inf')
        #     print(f"[診断-DF-A-S1-CF-Block{k}] 損失 - 予測: {prediction_loss_k.item():.6e}, "
        #           f"正則化: {regularization_loss_k.item():.6e}, 比率: {loss_ratio_k:.2f}")

        loss_k.backward()
        optimizer_phi.step()

        # # 診断: φ出力変化量・V_Aノルム追跡
        # with torch.no_grad():
        #     phi_seq_after = self.phi_theta(X_states)
        #     phi_output_change = torch.norm(phi_seq_after - phi_seq_before).item()
        #     print(f"[診断-DF-A-S1-epoch{epoch+1}] φ出力変化: {phi_output_change:.2e}")
        #     V_A_norm = torch.norm(V_A_k, p='fro').item()
        #     print(f"[診断C-V_A-epoch{epoch+1}] V_Aノルム: {V_A_norm:.2f}")

        # キャッシュ更新
        with torch.no_grad():
            phi_seq_final = self.phi_theta(X_states)
            phi_minus_final = phi_seq_final[:-1]
            phi_plus_final = phi_seq_final[1:]
            V_A_final = self._ridge_stage1(phi_minus_final, phi_plus_final, self.lambda_A)

            self._stage1_cache = {
                'V_A': V_A_final,
                'phi_minus': phi_minus_final,
                'phi_plus': phi_plus_final,
                'X_plus': X_states[1:]
            }

        return {
            'stage1_loss': loss_k.item(),
            'current_block': k,
            'n_blocks': cf_manager.n_blocks
        }
    
    def _compute_cross_fitting_prediction_ua_matrix(
        self,
        H_features: torch.Tensor,
        X_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        **DF-B参照**: U_A用クロスフィッティングでout-of-fold予測計算

        Args:
            H_features: 操作変数特徴量 (T-1, d_A)
            X_targets: 状態目標 (T-1, r)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (X_pred_cf, U_A_final)
                - X_pred_cf: out-of-fold予測 (T-1, r)
                - U_A_final: 最終U_A行列 (d_A, r)
        """
        T_eff = H_features.size(0)

        # クロスフィッティング設定取得
        n_blocks = self.cf_config.get('n_blocks', 6)
        min_block_size = self.cf_config.get('min_block_size', 20)

        # データ量チェック：クロスフィッティング実行可能性
        if T_eff < max(n_blocks * min_block_size, 100):
            # データ不足時：従来の全データRidge回帰（勾配あり）
            U_A = self._ridge_stage2(H_features, X_targets, self.lambda_B)
            X_pred = (U_A.T @ H_features.T).T
            return X_pred, U_A

        # クロスフィッティング実行
        try:
            from .cross_fitting import CrossFittingManager, TwoStageCrossFitter

            # クロスフィッティング管理
            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)
            cf_fitter = TwoStageCrossFitter(cf_manager)

            # U_A推定（クロスフィッティング、勾配あり）
            # U_A^{(-k)}リスト構築（out-of-fold）
            U_A_list = []
            for k in range(cf_manager.n_blocks):
                # out-of-foldインデックス I_{-k}
                oof_indices = cf_manager.get_out_of_fold_indices(k)
                H_oof = H_features[oof_indices]
                X_oof = X_targets[oof_indices]

                # U_A^{(-k)} 推定（勾配計算あり）
                U_A_k = self._ridge_stage2(H_oof, X_oof, self.lambda_B)
                U_A_list.append(U_A_k)

            # **理論準拠**: out-of-fold予測計算（勾配あり）
            X_pred_cf = torch.zeros_like(X_targets)
            for k in range(cf_manager.n_blocks):
                block_indices = cf_manager.get_block_indices(k)
                H_block = H_features[block_indices]  # 勾配あり

                # U_A^{(-k)}による予測: X_pred = (U_A^T @ H^T)^T = H @ U_A
                X_pred_cf[block_indices] = (U_A_list[k].T @ H_block.T).T

            # 最終U_A：全データでの推定（勾配あり、正則化用）
            U_A_final = self._ridge_stage2(H_features, X_targets, self.lambda_B)

            return X_pred_cf, U_A_final

        except Exception as e:
            print(f"クロスフィッティング失敗、従来方式を使用: {e}")
            U_A = self._ridge_stage2(H_features, X_targets, self.lambda_B)
            X_pred = (U_A.T @ H_features.T).T
            return X_pred, U_A

    def train_stage2_with_gradients(
        self,
        X_states: torch.Tensor,
        optimizer_phi: torch.optim.Optimizer,
        epoch: int = 0
    ) -> Dict[str, float]:
        """
        Stage-2学習（エポックごとに1ブロック処理）

        理論（式42b）:
        L_Stage-2(θ) = Σ_{t∈B_k} ||x_t - U_A^{(-k)}^T H_k||² + λ_B ||U_A^{(-k)}||²_F

        ポイント:
        - エポックeではブロック k = (e mod K) を処理
        - V_Aを動的再計算（φ_θ勾配を通す）
        - 1エポック = 1回パラメータ更新

        Args:
            X_states: 状態系列 (T, r)
            optimizer_phi: φ_θ用オプティマイザ
            epoch: エポック番号

        Returns:
            Dict[str, float]: 損失メトリクス
        """
        if 'X_plus' not in self._stage1_cache:
            raise RuntimeError("Stage-1が先に実行されている必要があります")

        X_plus = self._stage1_cache['X_plus']  # (T-1, r)

        optimizer_phi.zero_grad()

        # φ_θで特徴量を動的計算（勾配あり）
        phi_seq = self.phi_theta(X_states)  # (T, d_A)
        phi_minus = phi_seq[:-1]  # (T-1, d_A)
        phi_plus = phi_seq[1:]    # (T-1, d_A)

        # V_A動的再計算（φ_θ勾配を通す）
        V_A_current = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)

        # H計算（φ_θ勾配あり）
        H = (V_A_current @ phi_minus.T).T  # (T-1, d_A)
        T_eff = H.size(0)

        # クロスフィッティング設定
        n_blocks = self.cf_config.get('n_blocks', 5)
        min_block_size = self.cf_config.get('min_block_size', 20)

        # データ量チェック
        if T_eff < max(n_blocks * min_block_size, 100):
            # 小データ時：クロスフィッティングなし
            U_A = self._ridge_stage2(H, X_plus, self.lambda_B)
            X_pred = (U_A.T @ H.T).T

            prediction_loss = torch.norm(X_pred - X_plus, p='fro') ** 2 / X_plus.size(0)
            regularization_loss = self.lambda_B * torch.norm(U_A, p='fro') ** 2
            loss_stage2 = prediction_loss + regularization_loss

            loss_stage2.backward()
            optimizer_phi.step()

            self._stage2_cache['U_A'] = U_A.detach()
            return {'stage2_loss': loss_stage2.item(), 'n_blocks': 0, 'mode': 'no_crossfitting'}

        # クロスフィッティング実行
        from .cross_fitting import CrossFittingManager
        cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)

        # エポックに対応するブロックを選択
        k = epoch % cf_manager.n_blocks

        optimizer_phi.zero_grad()

        # 現在のφ_θでHを計算
        phi_seq = self.phi_theta(X_states)
        phi_minus = phi_seq[:-1]
        phi_plus = phi_seq[1:]
        V_A_current = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)
        H = (V_A_current @ phi_minus.T).T

        # out-of-foldでU_A^{(-k)}計算
        oof_indices = cf_manager.get_out_of_fold_indices(k)
        U_A_k = self._ridge_stage2(H[oof_indices], X_plus[oof_indices], self.lambda_B)

        # in-foldブロックで予測
        block_indices = cf_manager.get_block_indices(k)
        X_pred_k = (U_A_k.T @ H[block_indices].T).T

        # ブロックkの損失
        pred_loss_k = torch.norm(X_pred_k - X_plus[block_indices], p='fro') ** 2 / X_plus[block_indices].size(0)
        reg_loss_k = self.lambda_B * torch.norm(U_A_k, p='fro') ** 2
        loss_k = pred_loss_k + reg_loss_k

        # ブロックkでbackward
        loss_k.backward()

        # ブロックkで更新
        optimizer_phi.step()

        # 推論用キャッシュ更新（毎回全データで再計算、φ_θ更新を反映）
        with torch.no_grad():
            phi_seq_final = self.phi_theta(X_states)
            phi_minus_final = phi_seq_final[:-1]
            phi_plus_final = phi_seq_final[1:]
            V_A_final = self._ridge_stage1(phi_minus_final, phi_plus_final, self.lambda_A)
            H_final = (V_A_final @ phi_minus_final.T).T
            U_A_final = self._ridge_stage2(H_final, X_plus, self.lambda_B)
            self._stage2_cache['U_A'] = U_A_final

        return {
            'stage2_loss': loss_k.item(),
            'current_block': k,
            'n_blocks': cf_manager.n_blocks
        }
    
    def fit_two_stage(
        self, 
        X_states: torch.Tensor, 
        use_cross_fitting: bool = True,
        verbose: bool = False
    ) -> 'DFStateLayer':
        """
        従来の2段階クロスフィッティング学習（変更なし）
        
        **注意**: これは既存の学習メソッドです。
        新しいPhase-1学習では train_stage1_with_gradients と 
        train_stage2_closed_form を使用してください。
        """
        T, r = X_states.shape
        
        if r != self.state_dim:
            raise ValueError(f"状態次元不一致: expected {self.state_dim}, got {r}")
        
        if T < 2:
            raise ValueError(f"時系列が短すぎます: T={T}")
        
        # 特徴量計算
        with torch.no_grad():
            phi_seq = self.phi_theta(X_states)  # (T, d_A)
        
        # 過去/未来特徴量
        Phi_minus = phi_seq[:-1]  # (T-1, d_A)
        Phi_plus = phi_seq[1:]    # (T-1, d_A)
        X_plus = X_states[1:]     # (T-1, r)
        
        if use_cross_fitting and T >= 20:  # 最小限のサンプルサイズ
            self._fit_with_cross_fitting(Phi_minus, Phi_plus, X_plus, verbose)
        else:
            self._fit_without_cross_fitting(Phi_minus, Phi_plus, X_plus, verbose)
        
        self._is_fitted = True
        return self
    
    def _fit_with_cross_fitting(
        self, 
        Phi_minus: torch.Tensor, 
        Phi_plus: torch.Tensor, 
        X_plus: torch.Tensor,
        verbose: bool
    ):
        """クロスフィッティング付き学習"""
        T_eff = int(Phi_minus.size(0))  # T-1)
        
        # クロスフィッティング管理
        cf_manager = CrossFittingManager(T_eff, **self.cf_config)
        cf_fitter = TwoStageCrossFitter(cf_manager)
        
        if verbose:
            print(f"クロスフィッティング: T={T_eff}, n_blocks={cf_manager.n_blocks}")
        
        # Stage-1: 転送作用素推定（クロスフィッティング）
        V_list = cf_fitter.cross_fit_stage1(
            Phi_minus, Phi_plus,
            self._ridge_stage1,
            reg_lambda=self.lambda_A
        )
        
        # 平均転送作用素（最終的な V_A）
        self.V_A = torch.stack(V_list).mean(dim=0)
        
        # Out-of-fold特徴量計算
        H_cf = cf_fitter.compute_out_of_fold_features(Phi_minus, V_list)
        
        # Stage-2: 読み出し行列推定
        self.U_A = cf_fitter.cross_fit_stage2(
            H_cf, X_plus,
            self._ridge_stage2,
            detach_features=True,
            reg_lambda=self.lambda_B
        )
        
        if verbose:
            print(f"V_A shape: {self.V_A.shape}, U_A shape: {self.U_A.shape}")
    
    def _fit_without_cross_fitting(
        self, 
        Phi_minus: torch.Tensor, 
        Phi_plus: torch.Tensor, 
        X_plus: torch.Tensor,
        verbose: bool
    ):
        """クロスフィッティングなし学習（小データ用）"""
        if verbose:
            print("クロスフィッティングなしで学習")
        
        # Stage-1: 直接推定
        self.V_A = self._ridge_stage1(Phi_minus, Phi_plus, self.lambda_A)
        
        # 中間特徴量
        H = (self.V_A @ Phi_minus.T).T  # (T-1, d_A)
        
        # Stage-2: 読み出し推定
        self.U_A = self._ridge_stage2(H, X_plus, self.lambda_B)
        
        if verbose:
            print(f"V_A shape: {self.V_A.shape}, U_A shape: {self.U_A.shape}")
    
    def _compute_cross_fitting_prediction(
        self,
        phi_minus: torch.Tensor,
        phi_plus: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        **理論準拠**: クロスフィッティングでout-of-fold予測計算

        Args:
            phi_minus: 過去特徴量 (T-1, d_A)
            phi_plus: 未来特徴量 (T-1, d_A)

        Returns:
            tuple: (phi_pred_cf, V_A_final)
                - phi_pred_cf: out-of-fold予測 (T-1, d_A)
                - V_A_final: 最終V_A行列 (d_A, d_A)
        """
        T_eff = phi_minus.size(0)

        # クロスフィッティング設定取得
        n_blocks = self.cf_config.get('n_blocks', 6)
        min_block_size = self.cf_config.get('min_block_size', 20)

        # データ量チェック：クロスフィッティング実行可能性
        if T_eff < max(n_blocks * min_block_size, 100):
            # データ不足時：従来の全データRidge回帰（勾配あり）
            V_A = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)
            phi_pred = (V_A @ phi_minus.T).T
            return phi_pred, V_A

        # クロスフィッティング実行
        try:
            from .cross_fitting import CrossFittingManager, TwoStageCrossFitter

            # クロスフィッティング管理
            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)
            cf_fitter = TwoStageCrossFitter(cf_manager)

            # V_A推定（クロスフィッティング）- 勾配計算あり（θ更新用）
            # 修正: no_grad()を削除し、_ridge_stage1_with_gradを使用してθへの勾配を有効化
            V_A_list = cf_fitter.cross_fit_stage1(
                phi_minus, phi_plus,
                stage1_estimator=lambda X, Y: self._ridge_stage1_with_grad(X, Y, self.lambda_A)
            )

            # **理論準拠**: out-of-fold予測計算（勾配あり）
            phi_pred_cf = cf_fitter.compute_out_of_fold_features(phi_minus, V_A_list)

            # 最終V_A：全データでの推定（勾配あり、正則化用）
            V_A_final = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)

            # 情報をキャッシュ
            if not hasattr(self, '_cross_fitting_cache'):
                self._cross_fitting_cache = {}
            self._cross_fitting_cache.update({
                'V_A_list': V_A_list,
                'cf_manager': cf_manager
            })

            return phi_pred_cf, V_A_final

        except ImportError:
            # フォールバック（勾配あり）
            V_A = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)
            phi_pred = (V_A @ phi_minus.T).T
            return phi_pred, V_A
        except Exception as e:
            print(f"クロスフィッティング失敗、従来方式を使用: {e}")
            V_A = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)
            phi_pred = (V_A @ phi_minus.T).T
            return phi_pred, V_A

    def _compute_V_A_with_cross_fitting(
        self,
        phi_minus: torch.Tensor,
        phi_plus: torch.Tensor
    ) -> torch.Tensor:
        """
        **理論準拠**: クロスフィッティングを用いたV_A計算（閉形式解）

        Args:
            phi_minus: 過去特徴量 (T-1, d_A)
            phi_plus: 未来特徴量 (T-1, d_A)

        Returns:
            torch.Tensor: V_A行列 (d_A, d_A)
        """
        T_eff = phi_minus.size(0)

        # クロスフィッティング設定取得
        n_blocks = self.cf_config.get('n_blocks', 6)
        min_block_size = self.cf_config.get('min_block_size', 20)

        # データ量チェック：クロスフィッティング実行可能性
        if T_eff < max(n_blocks * min_block_size, 100):
            # データ不足時：従来の全データRidge回帰
            return self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)

        # クロスフィッティング実行
        try:
            from .cross_fitting import CrossFittingManager, TwoStageCrossFitter

            # クロスフィッティング管理
            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)
            cf_fitter = TwoStageCrossFitter(cf_manager)

            # V_A推定（クロスフィッティング）
            V_A_list = cf_fitter.cross_fit_stage1(
                phi_minus, phi_plus,
                stage1_estimator=lambda X, Y: self._ridge_stage1(X, Y, self.lambda_A)
            )

            # 最終V_A：全データでの推定
            V_A = self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)

            # V_Aリストをキャッシュ（Stage-2で使用）
            if not hasattr(self, '_cross_fitting_cache'):
                self._cross_fitting_cache = {}
            self._cross_fitting_cache.update({
                'V_A_list': V_A_list,
                'cf_manager': cf_manager
            })

            return V_A

        except ImportError:
            # クロスフィッティングモジュール未使用時：従来方式
            return self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)
        except Exception as e:
            # エラー時：従来方式にフォールバック
            print(f"クロスフィッティング失敗、従来方式を使用: {e}")
            return self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)

    def apply_transfer_operator(self, phi_prev: torch.Tensor) -> torch.Tensor:
        """
        転送作用素の適用: φ̂_{t|t-1} = V_A φ_{t-1}
        
        Args:
            phi_prev: 前時刻の特徴量 (d_A,) または (batch, d_A)
            
        Returns:
            torch.Tensor: 予測特徴量 (d_A,) または (batch, d_A)
        """
        if not self._is_fitted:
            # **修正**: キャッシュされた結果も使用可能に
            if 'V_A' in self._stage1_cache:
                V_A = self._stage1_cache['V_A']
            else:
                raise RuntimeError("fit_two_stage() または train_stage1_with_gradients() を先に実行してください")
        else:
            V_A = self.V_A
        
        # GPUデバイス整合性を確保
        V_A = V_A.to(phi_prev.device)

        if phi_prev.dim() == 1:
            return V_A @ phi_prev
        else:
            return (V_A @ phi_prev.T).T
    
    def predict_one_step(self, x_prev: torch.Tensor) -> torch.Tensor:
        """
        1ステップ状態予測: x̂_{t|t-1} = U_A^T V_A ϕ_θ(x_{t-1})
        
        Args:
            x_prev: 前時刻の状態 (r,) または (batch, r)
            
        Returns:
            torch.Tensor: 予測状態 (r,) または (batch, r)
        """
        V_A = None
        U_A = None
        
        if self._is_fitted:
            # 完全学習済み
            V_A = self.V_A
            U_A = self.U_A
        elif 'V_A' in self._stage1_cache and 'U_A' in self._stage2_cache:
            # Phase-1学習済み
            V_A = self._stage1_cache['V_A']
            U_A = self._stage2_cache['U_A']
        elif 'V_A' in self._stage1_cache:
            # **新規**: Stage-1のみ完了の場合の対応
            V_A = self._stage1_cache['V_A']
            # 簡易的なU_A推定
            if 'phi_minus' in self._stage1_cache and 'X_plus' in self._stage1_cache:
                with torch.no_grad():
                    phi_minus = self._stage1_cache['phi_minus']
                    X_plus = self._stage1_cache['X_plus']
                    H_simple = (V_A @ phi_minus.T).T
                    U_A = self._ridge_stage2(H_simple, X_plus, self.lambda_B)
                    # キャッシュに保存
                    self._stage2_cache['U_A'] = U_A.detach()
            else:
                raise RuntimeError("Stage-1は完了していますが、Stage-2実行に必要なデータが不足しています")
        else:
            raise RuntimeError("学習が完了していません。fit_two_stage() または train_stage1_with_gradients() を先に実行してください")
        
        # 特徴写像
        phi_prev = self.phi_theta(x_prev)
        
        # 転送作用素適用
        phi_pred = self.apply_transfer_operator(phi_prev)
        
        # 状態空間に戻す（GPUデバイス整合性を確保）
        U_A = U_A.to(phi_pred.device)
        if phi_pred.dim() == 1:
            return U_A.T @ phi_pred
        else:
            return (U_A.T @ phi_pred.T).T
    
    def predict_sequence(
        self, 
        X_states: torch.Tensor, 
        return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        系列予測: 各時刻でのone-step-ahead予測（推論専用モード）
        
        Args:
            X_states: 状態系列 (T, r)
            return_features: 特徴量も返すかどうか
            
        Returns:
            torch.Tensor: 予測系列 (T-1, r)
            Optional[torch.Tensor]: 特徴量系列 (T-1, d_A)
        """
        if not self._is_fitted and 'V_A' not in self._stage1_cache:
            raise RuntimeError("学習が完了していません")
        
        T = X_states.size(0)
        predictions = []
        features = []
        
        # 推論専用モードで実行（勾配グラフを切断）
        with torch.no_grad():
            for t in range(T - 1):
                x_pred = self.predict_one_step(X_states[t])
                predictions.append(x_pred)
                
                if return_features:
                    phi_prev = self.phi_theta(X_states[t])
                    phi_pred = self.apply_transfer_operator(phi_prev)
                    features.append(phi_pred)
        
        pred_tensor = torch.stack(predictions)
        
        if return_features:
            feat_tensor = torch.stack(features)
            return pred_tensor, feat_tensor
        
        return pred_tensor
    
    def get_transfer_operator(self) -> torch.Tensor:
        """転送作用素 V_A を取得"""
        if self._is_fitted:
            return self.V_A.clone()
        elif 'V_A' in self._stage1_cache:
            return self._stage1_cache['V_A'].clone()
        else:
            raise RuntimeError("学習が完了していません")
    
    def get_readout_matrix(self) -> torch.Tensor:
        """読み出し行列 U_A を取得"""
        if self._is_fitted:
            return self.U_A.clone()
        elif 'U_A' in self._stage2_cache:
            return self._stage2_cache['U_A'].clone()
        else:
            raise RuntimeError("学習が完了していません")
    
    def get_state_dict(self) -> Dict[str, Any]:
        """学習済みパラメータを辞書で取得"""
        state_dict = {
            'phi_theta': self.phi_theta.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'feature_dim': self.feature_dim,
                'lambda_A': self.lambda_A,
                'lambda_B': self.lambda_B
            }
        }
        
        if self._is_fitted:
            state_dict.update({
                'V_A': self.V_A,
                'U_A': self.U_A,
            })
        
        if self._stage1_cache:
            state_dict['stage1_cache'] = self._stage1_cache.copy()
        if self._stage2_cache:
            state_dict['stage2_cache'] = self._stage2_cache.copy()
            
        return state_dict
    
    def get_inference_state_dict(self) -> Dict[str, Any]:
        """推論用のstate_dictを取得（filtering評価用にV_A/U_Aを含める）"""
        state_dict = {
            'phi_theta': self.phi_theta.state_dict(),
        }

        # filtering評価では学習済みV_A, U_Aが必要なので含める
        if hasattr(self, 'V_A') and self.V_A is not None:
            state_dict['V_A'] = self.V_A
        if hasattr(self, 'U_A') and self.U_A is not None:
            state_dict['U_A'] = self.U_A

        # configは除外（推論時には設定ファイルから読み込むため不要）
        # キャッシュも推論には不要なので除外

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """カスタムload_state_dict: V_A/U_Aも適切に設定"""
        # V_A, U_Aを別途処理
        v_a = state_dict.pop('V_A', None)
        u_a = state_dict.pop('U_A', None)

        # 通常のパラメータを読み込み
        super().load_state_dict(state_dict, strict=strict)

        # V_A, U_Aを設定
        if v_a is not None:
            self.V_A = v_a.to(self.device) if hasattr(self, 'device') else v_a
        if u_a is not None:
            self.U_A = u_a.to(self.device) if hasattr(self, 'device') else u_a