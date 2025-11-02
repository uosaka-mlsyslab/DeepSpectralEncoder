import torch
import torch.nn as nn
from torch.linalg import cholesky, solve_triangular, eigvalsh, svd
import warnings
from typing import Optional, Dict, Any, Tuple, Union, List
import math

class RealizationError(Exception):
    """
    Exception raised when stochastic realization (SSM identification)
    fails due to numerical issues (ill-conditioning, non-positive-definite, etc.).
    """
    pass


class StochasticRealizationWithEncoder(nn.Module):
    """
    定式化準拠: Hilbert空間での関数的CCAに基づく確率的実現

    定式化対応:
    - Hilbert空間特徴量プロセス: u_t = u_η(y_t) ∈ R^m (Section 3.1)
    - 過去・未来部分空間: H^- := span{u_{t-1}, u_{t-2}, ...}, H^+ := span{u_t, u_{t+1}, ...}
    - 関数的CCA制約最適化: max Cov(v^-, v^+) / (√Var(v^-) √Var(v^+))
    - Galerkin投影による有限次元近似: ℓ長ウィンドウでの実装

    実装特徴:
    1. 時不変エンコーダー u_η との完全統合
    2. 弱定常性に基づく共分散演算子の正確な計算
    3. Ridge正則化とSVDによる数値安定化
    4. 正準方向から状態変数への理論準拠変換

    ==== 設定ファイル対応: 必要なパラメータ設定 ====
    # configs/realization.yaml 例:
    # ssm:
    #   realization:
    #     # 基本パラメータ
    #     encoder_output_dim: 16          # エンコーダー出力次元 m
    #     window_length: 10               # Galerkin近似ウィンドウ長 ℓ
    #     num_components: 8               # 状態次元 r (上位正準方向数)
    #     ridge_param: 1e-3               # Ridge正則化パラメータ λ
    #
    #     # 数値安定化設定
    #     min_eigenvalue: 1e-8            # 最小固有値クリッピング
    #     condition_threshold: 1e12       # 条件数閾値
    #
    #     # 特徴量マッピング設定
    #     feature_mapping:
    #       method: "component_averaging"  # "component_averaging" | "component_mlp"
    #       mlp_hidden_dims: [32, 16]     # MLP使用時の隠れ層次元
    #
    #     # 状態変数計算設定
    #     state_computation:
    #       use_past_block: true          # 過去ブロック使用 (既存実装準拠)
    #       apply_sqrt_weights: true      # Λ^{1/2} 重み付け適用
    #       correlation_threshold: 0.1    # 正準相関選択閾値
    #
    #     # デバイス設定
    #     device: "cpu"                   # "cpu" | "cuda"
    ===================================================
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_output_dim: int,  # 設定ファイルから明示指定
        past_horizon: int = 10,   # 旧: window_length → 互換性のため変更
        rank: int = 8,            # 旧: num_components → 互換性のため変更
        ridge_param: float = 1e-3,
        jitter: float = 1e-8,     # 旧: min_eigenvalue → 互換性のため変更
        m: int = 500,             # ラグ共分散推定サンプル数
        device: str = 'cpu',
        # ======== 新規追加: 特徴写像設定 ========
        feature_mapping_type: str = "averaging",  # "averaging" | "linear" | "mlp"
        feature_mapping_hidden_dims: Optional[List[int]] = None,  # MLPの隠れ層 (例: [32] or [64, 32])
        feature_mapping_activation: str = "relu"  # "relu" | "tanh" | "gelu"
    ):
        """
        Args:
            encoder: 時不変エンコーダー u_η: R^n → R^m
            past_horizon: Galerkin近似のウィンドウ長 ℓ (旧名: window_length)
            rank: 状態次元 r（上位正準方向数）(旧名: num_components)
            ridge_param: Ridge正則化パラメータ λ
            jitter: 最小固有値クリッピング (旧名: min_eigenvalue)
            m: ラグ共分散推定サンプル数
            device: 計算デバイス
        """
        super().__init__()

        self.encoder = encoder
        self.window_length = int(past_horizon)  # ℓ (内部では元の変数名を使用)
        self.num_components = int(rank)  # r (内部では元の変数名を使用)
        self.ridge_param = float(ridge_param)  # λ
        self.min_eigenvalue = float(jitter)  # 数値安定化 (内部では元の変数名を使用)
        self.m = int(m)  # ラグ共分散推定サンプル数
        self.device = device

        # 定式化対応: エンコーダー出力次元 m の取得
        # encoder_output_dimを優先し、フォールバックで自動検出
        if encoder_output_dim is not None:
            self.feature_dim = encoder_output_dim
        else:
            self.feature_dim = self._get_encoder_output_dim()

        # 学習済みパラメータ（fit後に設定）
        self.canonical_directions_past: Optional[torch.Tensor] = None  # a_i ∈ R^m
        self.canonical_directions_future: Optional[torch.Tensor] = None  # b_i ∈ R^m
        self.canonical_correlations: Optional[torch.Tensor] = None  # ρ_i
        self.is_fitted = False

        # 互換性のため: 古いRealizationクラスとの互換性
        self.h = self.window_length  # past_horizon相当
        self.rank = self.num_components  # rank相当（低ランク近似の次数）

        # 計算フロー最適化: 中間結果のキャッシュ
        self._cached_feature_matrices: Optional[Dict[str, torch.Tensor]] = None  # Φ_X, Φ_Y
        self._cached_covariance_blocks: Optional[Dict[str, torch.Tensor]] = None  # G, H, A
        self._cached_whitening_matrices: Optional[Dict[str, torch.Tensor]] = None  # G_inv_sqrt, H_inv_sqrt
        self._last_input_shape: Optional[Tuple[int, int]] = None  # (T, n) for cache validation

        # デバッグ・統計用
        self._feature_statistics: Optional[Dict[str, torch.Tensor]] = None

        # ======== 追加: 特徴写像変換の初期化 ========
        self.feature_mapping_type = feature_mapping_type

        if feature_mapping_type == "averaging":
            # 従来の時間平均（パラメータなし）
            self.component_transforms = None

        elif feature_mapping_type in ["linear", "mlp"]:
            # 成分別変換: m個の独立したネットワーク
            # 各ネットワークは ℓ → 1 の変換

            if feature_mapping_type == "linear":
                # 線形層のみ（隠れ層なし）
                hidden_dims = []
            else:
                # MLP（隠れ層あり）
                if feature_mapping_hidden_dims is None:
                    hidden_dims = [32]
                else:
                    hidden_dims = feature_mapping_hidden_dims

            # m個の成分別変換を構築
            self.component_transforms = nn.ModuleList([
                self._build_component_transform(
                    input_dim=self.window_length,  # ℓ
                    hidden_dims=hidden_dims,
                    activation=feature_mapping_activation
                )
                for _ in range(self.feature_dim)  # m個の成分
            ])
        else:
            raise ValueError(
                f"Unknown feature_mapping_type: {feature_mapping_type}. "
                f"Choose from: 'averaging', 'linear', 'mlp'"
            )

    def _build_component_transform(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str
    ) -> nn.Sequential:
        """
        成分別変換ネットワーク構築: ℝˡ → ℝ

        Theory Section 4.4.1, equation 333:
        φ̃_m^(i)(p(t)) = MLP(φ_u^(i)(p(t))) ∈ ℝ

        実装選択肢:
        - hidden_dims = [] → 線形層のみ: Linear(ℓ, 1)
        - hidden_dims = [h] → 浅いMLP: Linear(ℓ, h) → Act → Linear(h, 1)
        - hidden_dims = [h1, h2] → 2層MLP: Linear(ℓ, h1) → Act → Linear(h1, h2) → Act → Linear(h2, 1)

        Args:
            input_dim: 入力次元 ℓ (window_length)
            hidden_dims: 隠れ層次元リスト（空リスト=線形層のみ）
            activation: 活性化関数 ("relu", "tanh", "gelu")

        Returns:
            nn.Sequential: 変換ネットワーク
        """
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU
        }

        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}")

        act_fn = activation_map[activation]

        layers = []
        in_dim = input_dim

        # 隠れ層の構築
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn())
            in_dim = h_dim

        # 最終層: → ℝ (スカラー出力)
        layers.append(nn.Linear(in_dim, 1))

        return nn.Sequential(*layers)

    def _get_encoder_output_dim(self) -> int:
        """
        定式化対応: エンコーダー出力次元 m の取得（設定ファイル対応）

        実環境対応: ダミー入力を避け、設定パラメータで明示指定
        """
        # エンコーダーの属性から直接取得を試行
        if hasattr(self.encoder, 'output_dim'):
            return self.encoder.output_dim

        # 実環境では設定ファイルで明示指定することを推奨
        raise ValueError(
            "Cannot determine encoder output dimension. "
            "Please specify 'encoder_output_dim' in config or "
            "ensure encoder has 'output_dim' attribute. "
            "Example config: ssm.realization.encoder_output_dim: 16"
        )

    def _build_feature_matrices(
        self,
        Y: torch.Tensor,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        定式化対応: 時不変エンコーダーを用いた特徴量行列構築（キャッシュ対応）

        計算フロー最適化: 同一データに対する再計算を避ける

        定式化 Section 3.2-3.3:
        1. エンコーダー適用: Y → M = [u_η(y_1), ..., u_η(y_T)]
        2. 過去・未来ブロック定義（修正版）:
           p(t) = (y(t-1)^T, y(t-2)^T, ..., y(t-ℓ)^T)^T ∈ R^{d·ℓ}
           f(t) = (y(t)^T, y(t+1)^T, ..., y(t+ℓ-1)^T)^T ∈ R^{d·ℓ}
        3. 特徴量マッピング φ_m 構築:
           過去特徴量: Φ_X[i] = φ_m(p(t))
           未来特徴量: Φ_Y[i] = φ_m(f(t))

        Args:
            Y: 観測時系列 (T, n)
            use_cache: キャッシュを使用するか

        Returns:
            Φ_X: 過去特徴量行列 (m, N)
            Φ_Y: 未来特徴量行列 (m, N)
        """
        # キャッシュ機能: 一時的に無効化（安全性のため）
        # 将来的にデータハッシュベースのキャッシュに変更予定
        use_cache = False

        # 入力形状の記録（キャッシュ機構維持のため）
        input_shape = Y.shape

        # キャッシュチェック（現在は無効化されているため常にスキップ）
        if (use_cache and
            self._cached_feature_matrices is not None and
            self._last_input_shape == input_shape):
            return self._cached_feature_matrices['Φ_X'], self._cached_feature_matrices['Φ_Y']

        # Step 1: 全時点でエンコーダー適用（バッチ処理対応）
        # u_η(y_t) → m_t ∈ R^m (定式化 Section 3.1)

        # バッチ処理でエンコーダーを呼び出し
        # 訓練時は勾配保持、推論時は効率化のため勾配無効化
        if self.encoder.training:
            # 訓練時: 勾配を保持してエンコーダーパラメータ更新を可能にする
            # Phase-2でのCCA損失勾配フロー確保のため重要
            M = self.encoder(Y)  # (T, H, W, C) or (T, n) → (T, m)
        else:
            # 推論時: 勾配無効化で効率化
            self.encoder.eval()  # BatchNorm無効化
            with torch.no_grad():
                M = self.encoder(Y)  # (T, H, W, C) or (T, n) → (T, m)

        # エンコーダー出力から統一的に形状を取得
        T, m = M.shape  # エンコーダー後は常に(T, m)

        L = self.window_length
        N = T - 2 * L + 1  # 有効サンプル数

        if N <= 0:
            raise ValueError(f"Time series too short for window length {L}")

            # 次元確認と調整
            if M.dim() == 1:  # (m,) の場合
                M = M.unsqueeze(0)  # (1, m)
            elif M.dim() == 3:  # (T, 1, m) の場合
                M = M.squeeze(1)  # (T, m)

        # Step 2: 特徴量マッピング φ_m の構築
        # 定式化 Section 3.3: 同一成分のグループ化 → MLP変換
        Φ_X_list = []
        Φ_Y_list = []

        for i in range(N):
            t = i + L  # 実際の時点（Lから開始）

            # 過去ブロック: p(t) = (y(t-1)^T, y(t-2)^T, ..., y(t-L)^T)^T
            # 時点t-1からt-Lまで逆順で取得
            past_indices = list(range(t-1, t-L-1, -1))  # [t-1, t-2, ..., t-L]
            past_features = M[past_indices]  # (L, m)

            # 未来ブロック: f(t) = (y(t)^T, y(t+1)^T, ..., y(t+L-1)^T)^T
            future_features = M[t:t+L]  # (L, m)

            # φ_m 変換: 成分別グループ化 → スカラー出力
            φ_past = self._apply_feature_mapping(past_features)  # (m,)
            φ_future = self._apply_feature_mapping(future_features)  # (m,)

            Φ_X_list.append(φ_past)
            Φ_Y_list.append(φ_future)

        Φ_X = torch.stack(Φ_X_list, dim=1)  # (m, N)
        Φ_Y = torch.stack(Φ_Y_list, dim=1)  # (m, N)

        # キャッシュ保存
        if use_cache:
            self._cached_feature_matrices = {'Φ_X': Φ_X.clone(), 'Φ_Y': Φ_Y.clone()}
            self._last_input_shape = input_shape

        return Φ_X, Φ_Y

    def _apply_feature_mapping(self, features: torch.Tensor) -> torch.Tensor:
        """
        定式化対応: 特徴量マッピング φ_m の実装（成分別変換）

        Theory Section 4.4.1:
        1. 同一成分のグループ化: φ_u^(i)(p(t)) := [φ_u^(i)(y(t-1)), ..., φ_u^(i)(y(t-ℓ))]^T ∈ ℝˡ
        2. 成分別スカラー変換: φ̃_m^(i)(p(t)) = Transform_i(φ_u^(i)(p(t))) ∈ ℝ
        3. 最終ベクトル構成: φ_m(p(t)) = [φ̃_m^(1)(p(t)), ..., φ̃_m^(m)(p(t))]^T ∈ ℝᵐ

        実装選択肢（feature_mapping_type設定で切り替え）:
        - "averaging": 時間平均 (1/ℓ) Σ φ_u^(i)(y(t-j)) （Theory equation 345）
        - "linear":    線形変換 w_i^T φ_u^(i)(p(t)) （パラメータ学習可能）
        - "mlp":       浅いMLP MLP_i(φ_u^(i)(p(t))) （Theory equation 333）

        Args:
            features: (ℓ, m) 特徴量ブロック
                ℓ: window_length
                m: feature_dim (エンコーダー出力次元)

        Returns:
            φ_output: (m,) マッピング結果
        """
        ℓ, m = features.shape

        if self.feature_mapping_type == "averaging":
            # 従来の時間平均実装 (Theory equation 345)
            # φ̃_m^(i) ≈ (1/ℓ) Σ_{j=1}^ℓ φ_u^(i)(y(t-j))
            φ_output = torch.mean(features, dim=0)  # (m,)

        elif self.feature_mapping_type in ["linear", "mlp"]:
            # 成分別変換実装 (Theory equation 333)
            φ_components = []

            for i in range(m):
                # Step 1: 成分iのグループ化
                # φ_u^(i)(p(t)) = [φ_u^(i)(y(t-1)), ..., φ_u^(i)(y(t-ℓ))]^T ∈ ℝˡ
                φ_u_i = features[:, i]  # (ℓ,)

                # Step 2: 成分別変換適用
                # φ̃_m^(i)(p(t)) = Transform_i(φ_u^(i)(p(t))) ∈ ℝ
                φ_tilde_i = self.component_transforms[i](φ_u_i)  # (1,)
                φ_components.append(φ_tilde_i.squeeze())

            # Step 3: 最終ベクトル構成
            φ_output = torch.stack(φ_components)  # (m,)

        else:
            raise ValueError(f"Unknown feature_mapping_type: {self.feature_mapping_type}")

        return φ_output

    def _compute_covariance_blocks(
        self,
        Φ_X: torch.Tensor,
        Φ_Y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        定式化対応: サンプル共分散ブロック計算

        定式化 Section 3.4:
        - 中心化特徴量行列（サンプル平均減算）
        - サンプル共分散ブロック:
          G = (1/N) Φ_X Φ_X^T ∈ R^{m×m} (過去の分散)
          H = (1/N) Φ_Y Φ_Y^T ∈ R^{m×m} (未来の分散)
          A = (1/N) Φ_Y Φ_X^T ∈ R^{m×m} (過去-未来の共分散)

        Args:
            Φ_X: 過去特徴量行列 (m, N)
            Φ_Y: 未来特徴量行列 (m, N)

        Returns:
            G: 過去分散行列 (m, m)
            H: 未来分散行列 (m, m)
            A: 交差共分散行列 (m, m)
        """
        m, N = Φ_X.shape

        # 中心化（サンプル平均減算）
        Φ_X_mean = torch.mean(Φ_X, dim=1, keepdim=True)  # (m, 1)
        Φ_Y_mean = torch.mean(Φ_Y, dim=1, keepdim=True)  # (m, 1)

        Φ_X_centered = Φ_X - Φ_X_mean  # (m, N)
        Φ_Y_centered = Φ_Y - Φ_Y_mean  # (m, N)

        # サンプル共分散ブロック
        G = (Φ_X_centered @ Φ_X_centered.T) / N  # (m, m)
        H = (Φ_Y_centered @ Φ_Y_centered.T) / N  # (m, m)
        A = (Φ_Y_centered @ Φ_X_centered.T) / N  # (m, m)

        # 統計情報保存（デバッグ用）
        self._feature_statistics = {
            'past_mean': Φ_X_mean.squeeze(),
            'future_mean': Φ_Y_mean.squeeze(),
            'num_samples': N
        }

        return G, H, A

    def _apply_ridge_regularization(
        self,
        G: torch.Tensor,
        H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        定式化対応: Ridge正則化による数値安定化

        定式化 Section 3.4:
        Ridge正則化: G_λ = G + λI_m, H_λ = H + λI_m, λ > 0

        Args:
            G: 過去分散行列 (m, m)
            H: 未来分散行列 (m, m)

        Returns:
            G_λ: 正則化過去分散行列 (m, m)
            H_λ: 正則化未来分散行列 (m, m)
        """
        m = G.size(0)
        I_m = torch.eye(m, device=G.device, dtype=G.dtype)

        G_λ = G + self.ridge_param * I_m
        H_λ = H + self.ridge_param * I_m

        return G_λ, H_λ

    def _compute_whitening_matrices(
        self,
        G_λ: torch.Tensor,
        H_λ: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        定式化対応: 白色化行列の安定計算

        定式化 Section 3.4:
        白色化行列計算: G_λ^{-1/2}, H_λ^{-1/2} の安定的計算（固有値分解使用）

        数値安定化:
        - 最小固有値クリッピング
        - 条件数管理

        Args:
            G_λ: 正則化過去分散行列 (m, m)
            H_λ: 正則化未来分散行列 (m, m)

        Returns:
            G_inv_sqrt: G_λ^{-1/2} (m, m)
            H_inv_sqrt: H_λ^{-1/2} (m, m)
        """
        def stable_matrix_inv_sqrt(A: torch.Tensor, min_eigval: float = 1e-8) -> torch.Tensor:
            """数値安定的な A^{-1/2} 計算"""
            # 対称化
            A_sym = 0.5 * (A + A.T)

            # 固有値分解
            eigvals, eigvecs = torch.linalg.eigh(A_sym)

            # 最小固有値クリッピング（数値安定化）
            eigvals_clipped = torch.clamp(eigvals, min=min_eigval)

            # A^{-1/2} = V diag(λ^{-1/2}) V^T
            inv_sqrt_eigvals = eigvals_clipped.rsqrt()
            A_inv_sqrt = eigvecs @ torch.diag(inv_sqrt_eigvals) @ eigvecs.T

            return A_inv_sqrt

        G_inv_sqrt = stable_matrix_inv_sqrt(G_λ)
        H_inv_sqrt = stable_matrix_inv_sqrt(H_λ)

        # 白色化行列保存（デバッグ用）
        self._whitening_matrices = {
            'G_inv_sqrt': G_inv_sqrt,
            'H_inv_sqrt': H_inv_sqrt
        }

        return G_inv_sqrt, H_inv_sqrt

    def _solve_canonical_correlation(
        self,
        A: torch.Tensor,
        G_inv_sqrt: torch.Tensor,
        H_inv_sqrt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        定式化対応: SVDによる正準相関分析

        定式化 Section 3.5:
        1. 白色化された交差ブロック: Ã := H_λ^{-1/2} A G_λ^{-1/2}
        2. SVD分解: Ã = U Σ V^T
        3. 正準方向の逆変換:
           a_i = G_λ^{-1/2} v_i ∈ R^m
           b_i = H_λ^{-1/2} u_i ∈ R^m

        Args:
            A: 交差共分散行列 (m, m)
            G_inv_sqrt: G_λ^{-1/2} (m, m)
            H_inv_sqrt: H_λ^{-1/2} (m, m)

        Returns:
            U: 左特異ベクトル (m, m)
            Σ: 特異値 (正準相関係数) (m,)
            V^T: 右特異ベクトル転置 (m, m)
        """
        # 白色化された交差ブロック
        Ã = H_inv_sqrt @ A @ G_inv_sqrt  # (m, m)

        # SVD分解
        U, Σ, Vt = torch.linalg.svd(Ã, full_matrices=False)

        return U, Σ, Vt

    def _extract_canonical_directions(
        self,
        U: torch.Tensor,
        Vt: torch.Tensor,
        G_inv_sqrt: torch.Tensor,
        H_inv_sqrt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        定式化対応: 正準方向の抽出と逆変換

        定式化 Section 3.5:
        正準方向の逆変換:
        a_i = G_λ^{-1/2} v_i ∈ R^m (過去空間での正準方向)
        b_i = H_λ^{-1/2} u_i ∈ R^m (未来空間での正準方向)

        Args:
            U: 左特異ベクトル (m, m)
            Vt: 右特異ベクトル転置 (m, m)
            G_inv_sqrt: G_λ^{-1/2} (m, m)
            H_inv_sqrt: H_λ^{-1/2} (m, m)

        Returns:
            canonical_dirs_past: 過去正準方向 a_i (m, r)
            canonical_dirs_future: 未来正準方向 b_i (m, r)
        """
        # 上位 r 個の正準方向選択
        r = min(self.num_components, U.size(1))

        # 正準方向の逆変換
        canonical_dirs_past = G_inv_sqrt @ Vt[:r, :].T  # (m, r)
        canonical_dirs_future = H_inv_sqrt @ U[:, :r]   # (m, r)

        return canonical_dirs_past, canonical_dirs_future

    def fit(self, Y: torch.Tensor, encoder: Optional[nn.Module] = None) -> 'StochasticRealizationWithEncoder':
        """
        定式化対応: 関数的CCAに基づく確率的実現の学習

        定式化 Section 3 全体の実装:
        1. 時不変エンコーダーとの統合
        2. Hilbert空間での特徴量プロセス構築
        3. 関数的CCA制約最適化問題の解決
        4. 正準方向から状態変数への変換

        Args:
            Y: 観測時系列 (T, n)
            encoder: エンコーダー（Noneの場合は初期化時のものを使用）

        Returns:
            self: 学習済みインスタンス
        """
        if encoder is not None:
            self.encoder = encoder

        Y = Y.to(self.device)
        self.encoder = self.encoder.to(self.device)

        # Step 1: 時不変エンコーダーを用いた特徴量行列構築
        Φ_X, Φ_Y = self._build_feature_matrices(Y)

        # Step 2: サンプル共分散ブロック計算
        G, H, A = self._compute_covariance_blocks(Φ_X, Φ_Y)

        # Step 3: Ridge正則化
        G_λ, H_λ = self._apply_ridge_regularization(G, H)

        # Step 4: 白色化行列計算
        G_inv_sqrt, H_inv_sqrt = self._compute_whitening_matrices(G_λ, H_λ)

        # Step 5: 正準相関分析（SVD）
        U, Σ, Vt = self._solve_canonical_correlation(A, G_inv_sqrt, H_inv_sqrt)

        # Step 6: 正準方向抽出
        canonical_dirs_past, canonical_dirs_future = self._extract_canonical_directions(
            U, Vt, G_inv_sqrt, H_inv_sqrt
        )

        # 学習結果保存
        self.canonical_directions_past = canonical_dirs_past
        self.canonical_directions_future = canonical_dirs_future
        self.canonical_correlations = Σ[:min(self.num_components, len(Σ))]

        # 既存実装との整合性: B行列の構築 (B = Σ^{1/2} a^T 形式)
        # 既存filter()メソッドとの互換性確保のため
        sqrt_correlations = torch.sqrt(self.canonical_correlations)  # Σ^{1/2}
        self.B_matrix = torch.diag(sqrt_correlations) @ canonical_dirs_past.T  # (r, m)

        self.is_fitted = True

        return self

    def estimate_states(self, Y: torch.Tensor) -> torch.Tensor:
        """
        定式化準拠: 正準変量から状態変数への変換

        定式化対応:
        過去プロセス正準変量: z_p(t) = [a_1^T φ_m(p(t)), ..., a_r^T φ_m(p(t))]^T ∈ R^r
        状態変数: x(t) = Σ^{1/2} z_p(t)

        計算フロー最適化: fit()で計算済みの情報を再利用

        Args:
            Y: 観測時系列 (T, n)

        Returns:
            X: 状態系列 (T_eff, r)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        Y = Y.to(self.device)

        # キャッシュされた特徴量行列を再利用（fit()で計算済み）
        if self._cached_feature_matrices is not None:
            Φ_X = self._cached_feature_matrices['Φ_X']  # 過去ブロック特徴量
            Φ_Y = self._cached_feature_matrices['Φ_Y']  # 未来ブロック特徴量 (未使用)
        else:
            # キャッシュがない場合のみ再計算
            Φ_X, Φ_Y = self._build_feature_matrices(Y)

        # 定式化準拠の状態変数計算
        # Step 1: 過去プロセス正準変量 z_p(t) = [a_1^T φ_m(p(t)), ..., a_r^T φ_m(p(t))]^T
        z_p = self.canonical_directions_past.T @ Φ_X  # (r, N)

        # Step 2: 正準相関係数による重み付け x(t) = Σ^{1/2} z_p(t)
        sqrt_canonical_corrs = torch.sqrt(self.canonical_correlations)  # Σ^{1/2} ∈ R^r

        # 重み付け適用: 各正準変量に対応する相関係数の平方根を乗算
        X_weighted = sqrt_canonical_corrs.unsqueeze(1) * z_p  # (r, N)
        X = X_weighted.T  # (N, r)

        return X

    def filter_compatible(self, Y: torch.Tensor) -> torch.Tensor:
        """
        既存filter()メソッドとの互換性確保

        定式化準拠: estimate_states()の結果と数値的に一致するよう修正
        短期解決として内部でestimate_states()を呼び出して数値精度を統一

        Args:
            Y: 観測時系列 (T, n)

        Returns:
            X: 状態系列 (T_eff, r) - estimate_states()と数値的に同一結果
        """
        # 定式化準拠のestimate_states()を呼び出して数値精度を統一
        return self.estimate_states(Y)

    def get_canonical_analysis_results(self) -> Dict[str, Any]:
        """
        正準相関分析の結果取得

        Returns:
            Dict: 分析結果
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}

        return {
            "canonical_correlations": self.canonical_correlations.cpu().numpy().tolist(),
            "num_components": self.num_components,
            "feature_dim": self.feature_dim,
            "window_length": self.window_length,
            "ridge_param": self.ridge_param,
            "condition_numbers": {
                "correlations_range": {
                    "min": self.canonical_correlations.min().item(),
                    "max": self.canonical_correlations.max().item()
                }
            }
        }

class Realization:
    """
    Exact subspace identification via canonical correlation analysis.
    
    Steps in fit(Y):
      1) Build past/future block Hankel matrices H_p, H_f
      2) Compute empirical covariances S_pp, S_ff, S_fp
      3) Add diagonal jitter ε to S_pp, S_ff for stability
      4) Check condition numbers; reject if too large
      5) Compute Cholesky factors L_pp, L_ff (S = L @ L.T)
      6) Solve triangular systems to get whitening matrices W_pp = L_pp⁻¹, W_ff = L_ff⁻¹
      7) Form T = W_ff @ S_fp @ W_pp and take its SVD → U, Lambda, Vᵀ
      8) Build state map B = Lambda^{1/2} Vᵀ W_pp
      9) Save singular values Lambda for downstream loss or analysis
    """
    def __init__(
        self,
        past_horizon: int,
        jitter: float = 1e-6,
        cond_thresh: float = 1e12,
        rank: int | None = None,
        reg_type : str = "sum",
        m: int | None = None,
    ):
        """
        Args:
        past_horizon:  number of time-lags in past/future Hankel blocks
        jitter:        small epsilon to add to diagonals of covariances
        cond_thresh:   max allowed condition number before rejection
        """
        # **修正**: 数値パラメータの型変換を明示的に実行
        self.h = int(past_horizon)
        self.jitter = float(jitter)  # **追加**: 明示的なfloat変換
        self.cond_thresh = float(cond_thresh)  # **追加**: 明示的なfloat変換
        
        # rank処理
        if rank is not None:
            self.rank = int(rank)  # **修正**: 明示的なint変換
        else:
            self.rank = rank
            
        self.reg_type = str(reg_type)  # **追加**: 明示的なstr変換

        # mパラメータ（ラグ共分散推定用サンプル数）
        self.m = int(m) if m is not None else None

        # 初期化
        self.B = None
        self._L_vals = None

        # for debug
        self._Spp_eigvals = None
        self.H = None

    def fit(self, Y: torch.Tensor):
        T, p = Y.shape
        h = self.h
        N = T - 2*h + 1
        if N <= 0:
            # raise RealizationError("Time series too short for horizon h")
            return

        # 0) 中心化
        mu = Y.mean(dim=0, keepdim=True)
        Y_c = Y - mu 
        
        # # 1) ブロック‐ハンケル行列
        # H_p_float = torch.stack([Y_c[i : i+h].flip(dims=(0,)).reshape(-1) for i in range(N)], dim=1)
        # H_f_float = torch.stack([Y_c[i+1 : i+h+1].reshape(-1) for i in range(N)], dim=1)
        # H_p = H_p_float.double()
        # H_f = H_f_float.double()

        device = Y_c.device
        # mを自動決定（設定可能、適応的デフォルト）
        max_available = T - 2 * h
        if max_available <= 0:
            # past_horizonを自動調整
            h_new = max(1, (T - 1) // 2)
            print(f"realization調整: past_horizon {h} → {h_new} (データ長: {T})")
            h = h_new
            self.h = h_new  # 属性も更新
            max_available = T - 2 * h

        m_default = min(500, max(1, max_available))
        m = getattr(self, 'm', None)
        if m is None:
            m = m_default
        else:
            m = min(m, max_available)  # 利用可能範囲に制限
        rank   = self.rank      # 低ランク近似の次数
        # eps_chol   = 1e-7
        # eps_jitter = 1e-10
        eps_chol   = float(self.jitter)
        eps_jitter = float(self.jitter)
        q_over     = 5

        # 1. ── ラグ共分散 (バッチ外積, float32) -----------------
        idx = torch.randint(0, max_available, (m,), device=device)
        Y0  = Y_c[idx]                          # (m, p)
        Lambda    = {}
        
        for l in range(0, 2 * h):
            Yl = Y_c[idx + l]                  # (m, p)
            cov = (Yl.T @ Y0) / m              # (p, p)
            Lambda[ l]  = cov
            if h + 1 > l > 0:
                Lambda[-l] = cov.T

        # 2. ── ブロック行列 (k×k ブロック) ----------------------
        dim_H   = h * p
        H32  = torch.zeros(dim_H, dim_H, dtype=torch.float32, device=device)
        Tp32 = torch.zeros_like(H32)
        
        z = torch.zeros(p, p, dtype=torch.float32, device=device)
        for i in range(h):
            for j in range(h):
                H32 [i*p:(i+1)*p, j*p:(j+1)*p] = Lambda.get(i + j + 1, z)
                Tp32[i*p:(i+1)*p, j*p:(j+1)*p] = Lambda.get(i - j, z)
        
        Tp32.diagonal().add_(eps_jitter)        # jitter for SPD

        # 3. ── 逆平方根 (float64 で) -----------------------------
        Tp64 = Tp32.to(torch.float64)
        Tp64 = 0.5 * (Tp64 + Tp64.T)
        try:
            L64  = torch.linalg.cholesky(
                      Tp64 + eps_chol*torch.eye(dim_H, device=device, dtype=torch.float64))
            W64  = torch.linalg.solve_triangular(
                      L64, torch.eye(dim_H, device=device, dtype=torch.float64), upper=False)
        except RuntimeError as e:
            print(f"real.fit failed cholesky decom: {e}")
            
            # ========================================
            # 数値安定性失敗時の対応方針メモ:
            # 
            # 選択肢1: 微分可能な変換で数値修正
            #   - torch.nn.functional.softplus(eigvals, beta=10) + eps_chol
            #   - 勾配は保持されるが、手法の数学的意味が変わる可能性
            # 
            # 選択肢2: 例外エポックスキップ（現在の方針）
            #   - RealizationErrorを投げてトレーナー側でエポック継続
            #   - 手法の学習戦略に破綻しない
            #
            # 選択肢3: 正則化による事前防止
            #   - 学習時に固有値負値化を防ぐ正則化項
            #   - 根本的解決だが実装が複雑
            # ========================================
            
            # NaN/Inf チェック - これ自体が数値破綻の兆候
            if not torch.isfinite(Tp64).all():
                print("Critical: Tp64 contains non-finite values.")
                raise RealizationError("Matrix contains non-finite values - numerical breakdown")
            
            # 対称性と固有値分解の最後の試行
            try:
                # 対称性チェック
                is_symmetric = torch.allclose(Tp64, Tp64.T, atol=1e-8)
                
                if is_symmetric:
                    eigvals, eigvecs = torch.linalg.eigh(Tp64)
                else:
                    print("Warning: Matrix is not symmetric")
                    eigvals_complex, eigvecs_complex = torch.linalg.eig(Tp64)
                    eigvals = eigvals_complex.real
                    eigvecs = eigvecs_complex.real
                
                # 固有値の数値安定性チェック
                if torch.any(eigvals <= 0) or not torch.isfinite(eigvals).all():
                    print(f"Unstable eigenvalues: min={eigvals.min().item()}, has_inf={not torch.isfinite(eigvals).all()}")
                    raise RealizationError("Eigenvalues are numerically unstable")
                
                # TODO: 方針変更時はここで eigvals のクリップ等を実装
                # eigvals = torch.clamp(eigvals, min=eps_chol)  # 勾配途切れ注意
                
                inv_sqrt = eigvals.rsqrt()
                D = torch.diag(inv_sqrt)
                W64 = eigvecs @ D @ eigvecs.T
                
                if not torch.isfinite(W64).all():
                    raise RealizationError("Final W64 matrix contains non-finite values")
                    
            except (RuntimeError, torch.linalg.LinAlgError) as inner_e:
                print(f"All numerical recovery attempts failed: {inner_e}")
                raise RealizationError(f"Complete numerical breakdown in realization: {inner_e}")
        
        # 4. ── 正規化 Hankel (float64) --------------------------
        T64 = W64.T @ (H32.to(torch.float64) @ W64)
        
        # 5. ── ランダム化 rank‑r SVD  or SVD----------------------------
        if rank is None or rank >= dim_H:
            U64, S64, Vh64 = torch.linalg.svd(T64, full_matrices=False)
            U, S, Vh = U64.to(torch.float32), S64.to(torch.float32), Vh64.to(torch.float32)
            # B = Lambda^{1/2} Vᵀ W_pp
            self.B = torch.diag(S.pow(0.5)) @ Vh @ W64.to(torch.float32)
            self._L_vals = S
        else:                                  # ランク r の切り出し
            U64, S64, Vh64 = torch.linalg.svd(T64, full_matrices=False)
            U, S, Vh = U64.to(torch.float32), S64.to(torch.float32), Vh64.to(torch.float32)
            U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank,:]
            self.B = torch.diag(S_r.pow(0.5)) @ Vh_r @ W64.to(torch.float32)
            self._L_vals = S_r
            
            
        # r = min(rank, dim_H)
        # G   = torch.randn(dim_H, r + q_over, dtype=torch.float64, device=device)
        # Y2  = T64 @ G
        # Q, _= torch.linalg.qr(Y2, mode='reduced')
        # B   = Q.T @ T64
        # Ub, S64, Vh64 = torch.linalg.svd(B, full_matrices=False)
        # U64 = Q @ Ub
        # U_r = U64[:, :r]
        # S_r = S64[:r]
        # V_r = Vh64[:r, :].T       
        
        

        

        # #for debug
        # self.H =H_p
        # # print(f'first Y: {Y.detach().cpu()[:4, :4]}')
        
        # # 2) 経験共分散
        # # print(f'print N before computing S_pp: {N}') #debug
        # S_pp = (H_p @ H_p.T) / (N-1)
        # S_ff = (H_f @ H_f.T) / (N-1)
        # S_fp = (H_f @ H_p.T) / (N-1)


        # # 3) ジッター
        # I_pp = torch.eye(S_pp.size(0), device=S_pp.device).to(S_pp.dtype)
        # I_ff = torch.eye(S_ff.size(0), device=S_ff.device).to(S_pp.dtype)
        # S_pp = S_pp + self.jitter * I_pp
        # S_ff = S_ff + self.jitter * I_ff

        # # 3.5)対称化
        # S_pp = 0.5 * (S_pp + S_pp.T)
        # S_ff = 0.5 * (S_ff + S_ff.T)

        # #debug
        # self._Spp_eigvals = eigvalsh(S_pp)
        
        # # 4) 条件数チェック
        # def compute_eigvals(A):
        #     try:
        #         return eigvalsh(A)
        #     except RuntimeError:
        #         # eigh がダメなら SVD で特異値（≒固有値）取得
        #         return svd(A, compute_uv=False)
        '''
        #一旦除去
        # cond_pp = compute_eigvals(S_pp).max() / compute_eigvals(S_pp).min()
        # cond_ff = compute_eigvals(S_ff).max() / compute_eigvals(S_ff).min()
        # if cond_pp > self.cond_thresh or cond_ff > self.cond_thresh:
        #     raise RealizationError("block Covariance : ill-conditioned")
        # print(f'(real) debag: eigvalsh(S_pp):{eigvalsh(S_pp)}')
        '''
        '''
        # # -- debug dump --
        # # 1) そもそも finite か？
        # if not torch.isfinite(S_pp).all():
        #     print("S_pp contains non-finite entries!")
        # # 2) 先頭要素が何か
        # # print(f"S_pp[0,0]={S_pp[0,0].item()}, diag min={torch.min(torch.diag(S_pp)).item()}")
        # # 3) 固有値最小値／最大値
        # try:
        #     eigs = torch.linalg.eigvalsh(S_pp)
        #     # print(f"eigvals (min, max) = ({eigs[0].item()}, {eigs[-1].item()})")
        # except Exception as e:
        #     print("eigvalsh failed:", e)
        # # 4) 完全な行列のサンプル
        # # print("S_pp[0:3,0:3] =\n", S_pp[:3,:3].cpu().numpy())
        # # -- end debug dump --
        '''

        # # 5) Cholesky
        # L_pp = cholesky(S_pp)  # lower
        # L_ff = cholesky(S_ff)
        
        # # L_pp = L_pp.float(); L_ff = L_ff.float()
        # # I_pp = L_pp.float(); I_ff = L_ff.float()


        # # 6) Calculate root-inverse
        # W_pp = solve_triangular(L_pp, I_pp, upper=False)
        # W_ff = solve_triangular(L_ff, I_ff, upper=False)

        # W_pp = W_pp.float(); W_ff = W_ff.float(); S_fp = S_fp.float()

        '''
        # # 5.6) compute square-root inverse
        # eigval_p, Eigvecs_p = torch.linalg.eigh(S_pp)
        # inv_sqrt_p = eigval_p.rsqrt()
        # W_pp = Eigvecs_p @ torch.diag(inv_sqrt_p) @ Eigvecs_p.T
        # eigval_f, Eigvecs_f = torch.linalg.eigh(S_ff) 
        # inv_sqrt_f = eigval_f.rsqrt()
        # W_ff = Eigvecs_f @ torch.diag(inv_sqrt_f) @ Eigvecs_f.T
        '''

        # # 7) SVD
        # T_mat = W_ff @ S_fp @ W_pp.T
        # U, L_vals, Vt = svd(T_mat)

        # # 8) 低ランク切り出し
        # if 0 < self.rank < L_vals.numel():
        #     U_r   = U[:, :self.rank]            # (ph, rank)
        #     L_r   = L_vals[: self.rank]         # (rank,)
        #     Vt_r  = Vt[: self.rank, :]          # (rank, ph)
        #     L_vals = L_r
        #     Vt     = Vt_r
        #     # B = Lambda^{1/2} Vᵀ W_pp
        #     self.B = torch.diag(L_r.pow(0.5)) @ Vt_r @ W_pp
        # else:
        #     # フルランク版
        #     self.B = torch.diag(L_vals.pow(0.5)) @ Vt @ W_pp

        # # 9) 特異値を保存
        # self._L_vals = L_vals

    def filter(self, Y: torch.Tensor) -> torch.Tensor:
        h = self.h
        N = Y.shape[0] - 2*h + 1
        # Yf = torch.stack([Y[i+1 : i+h+1].reshape(-1) for i in range(N)], dim=1)
        Yp = torch.stack([Y[i : i + h].flip(dims=(0, )).reshape(-1) for i in range(N)], dim=1)
        X_state = (self.B @ Yp).T  # shape (N, r), time ; t = h,...,h+N-1

        self.X_state_torch = X_state

        return X_state

    def singular_value_reg(self, sv_weight : float) -> torch.Tensor:
        """
        特異値正則化を返す。
        reg_type=="sum"     -> sum(σ_i)
        reg_type=="squared" -> sum((1 - σ_i)^2)
        """
        if self.reg_type == "sum":
            _tr = self._L_vals.sum()
            _reg = -_tr
        elif self.reg_type == "squared":
            _reg = ((1 - self._L_vals) ** 2).sum()
        elif self.reg_type == "abs":
            _reg = (1 - self._L_vals).abs().sum()
            
        elif self.reg_type == "bounded":
            _reg = (1 - self._L_vals ** 2).sum()
        else:
            raise ValueError(f"Unknown reg_type: {self.reg_type}")

        return sv_weight * _reg

    # =====================================
    # 既存機能の改善と追加ユーティリティ
    # =====================================

    def get_state_statistics(self) -> Dict[str, Any]:
        """
        状態推定の統計情報取得
        
        Returns:
            Dict: 統計情報
        """
        if not hasattr(self, 'X_state_torch') or self.X_state_torch is None:
            return {"status": "not_fitted"}
        
        X = self.X_state_torch
        
        return {
            "state_shape": X.shape,
            "state_dimension": X.size(1),
            "sequence_length": X.size(0),
            "state_statistics": {
                "mean": torch.mean(X, dim=0).tolist(),
                "std": torch.std(X, dim=0).tolist(),
                "min": torch.min(X, dim=0)[0].tolist(),
                "max": torch.max(X, dim=0)[0].tolist()
            },
            "singular_values": {
                "values": self._L_vals.tolist() if hasattr(self, '_L_vals') and self._L_vals is not None else None,
                "condition_number": (self._L_vals.max() / self._L_vals.min()).item() if hasattr(self, '_L_vals') and self._L_vals is not None else None
            }
        }

    def predict_states(
        self,
        n_steps: int = 1,
        method: str = "linear"
    ) -> torch.Tensor:
        """
        状態の将来予測
        
        Args:
            n_steps: 予測ステップ数
            method: 予測手法 ("linear" | "last_value")
            
        Returns:
            torch.Tensor: 予測状態 (n_steps, r)
        """
        if not hasattr(self, 'X_state_torch') or self.X_state_torch is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = self.X_state_torch
        T, r = X.shape
        
        if method == "linear":
            # 線形外挿
            if T >= 2:
                trend = X[-1] - X[-2]  # 最新のトレンド
                predictions = []
                for step in range(1, n_steps + 1):
                    pred = X[-1] + step * trend
                    predictions.append(pred)
                return torch.stack(predictions)
            else:
                method = "last_value"  # フォールバック
        
        if method == "last_value":
            # 最後の値を繰り返し
            last_state = X[-1]
            return last_state.unsqueeze(0).expand(n_steps, r).clone()
        
        else:
            raise ValueError(f"Unknown prediction method: {method}")

    def validate_kalman_compatibility(
        self,
        df_state_layer,
        df_obs_layer
    ) -> Dict[str, Any]:
        """
        Kalman更新との互換性検証
        
        Args:
            df_state_layer: DF-A層
            df_obs_layer: DF-B層
            
        Returns:
            Dict: 検証結果
        """
        validation = {
            "compatible": True,
            "issues": [],
            "requirements_met": {},
            "recommendations": []
        }
        
        # 必須コンポーネントの存在確認
        requirements = [
            ("df_state_layer.V_A", hasattr(df_state_layer, 'V_A') and df_state_layer.V_A is not None),
            ("df_state_layer.U_A", hasattr(df_state_layer, 'U_A') and df_state_layer.U_A is not None),
            ("df_obs_layer.V_B", hasattr(df_obs_layer, 'V_B') and df_obs_layer.V_B is not None),
            ("df_obs_layer.U_B", hasattr(df_obs_layer, 'U_B') and df_obs_layer.U_B is not None),
            ("realization_fitted", hasattr(self, 'X_state_torch') and self.X_state_torch is not None)
        ]
        
        for req_name, req_met in requirements:
            validation["requirements_met"][req_name] = req_met
            if not req_met:
                validation["compatible"] = False
                validation["issues"].append(f"Missing requirement: {req_name}")
        
        # 次元整合性チェック
        if validation["compatible"]:
            try:
                r_realization = self.rank if self.rank is not None else self.X_state_torch.size(1)
                r_df_state = df_state_layer.state_dim
                
                if r_realization != r_df_state:
                    validation["issues"].append(f"State dimension mismatch: realization={r_realization}, df_state={r_df_state}")
                    validation["compatible"] = False
                    
                # 特徴次元
                dA = df_state_layer.feature_dim
                dB = df_obs_layer.obs_feature_dim
                
                validation["dimensions"] = {
                    "state_dim": r_realization,
                    "feature_dim_A": dA,
                    "feature_dim_B": dB
                }
                
                # 推奨事項
                if dA < 2 * r_realization:
                    validation["recommendations"].append(f"Consider increasing feature_dim_A (current: {dA}, recommended: >={2*r_realization})")
                    
            except Exception as e:
                validation["compatible"] = False
                validation["issues"].append(f"Dimension check failed: {e}")
        
        return validation

    # =====================================
    # 設定ファイル対応
    # =====================================

    def create_kalman_config(
        self,
        **overrides
    ) -> Dict[str, Any]:
        """
        Kalman Filter用設定作成
        
        Args:
            **overrides: 設定上書き
            
        Returns:
            Dict: Kalman設定
        """
        default_config = {
            'noise_estimation': {
                'method': 'residual_based',
                'gamma_Q': 1e-6,
                'gamma_R': 1e-6
            },
            'initialization': {
                'method': 'data_driven',
                'n_init_samples': 10
            },
            'numerical': {
                'condition_threshold': 1e12,
                'min_eigenvalue': 1e-8,
                'jitter': 1e-6
            },
            'device': 'cpu'
        }
        
        # 上書き適用
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        config = default_config.copy()
        deep_update(config, overrides)
        
        return config
    

def build_realization(cfg):
    """
    修正版Factory: エンコーダーの有無によって適切な実装を選択

    設定エラーハンドリング対応:
    - エンコーダーありの場合: encoder_output_dim必須
    - エンコーダーなしの場合: 従来のRealization

    Args:
        cfg: 設定オブジェクト

    Returns:
        Realization または StochasticRealizationWithEncoder

    Raises:
        ValueError: encoder_output_dim未指定など設定エラー
    """
    # エンコーダーの有無で分岐
    use_encoder = getattr(cfg, 'use_encoder', False) and hasattr(cfg, 'encoder')

    if use_encoder:
        # 新型: エンコーダー対応版
        # encoder_output_dimの明示指定を要求（設定エラーハンドリング）
        if not hasattr(cfg, 'encoder_output_dim'):
            raise ValueError(
                "encoder_output_dim must be specified when using encoder. "
                "Example: cfg.encoder_output_dim = 16"
            )

        return StochasticRealizationWithEncoder(
            encoder=cfg.encoder,
            encoder_output_dim=cfg.encoder_output_dim,
            window_length=getattr(cfg, 'window_length', 5),
            num_components=getattr(cfg, 'num_components', 4),
            ridge_param=getattr(cfg, 'ridge_param', 1e-6),
            device=getattr(cfg, 'device', 'cpu')
        )
    else:
        # 従来型: Realizationクラス
        r = getattr(cfg, "rank", None)
        if r is not None and r < 0:
            raise ValueError(f"Invalid rank: {r} (must be >= 0 or None)")

        return Realization(
            past_horizon=cfg.h,
            jitter=cfg.jitter,
            cond_thresh=cfg.cond_thresh,
            rank=r,
            reg_type=getattr(cfg, "svd_reg_type", "sum"),
        )