# src/models/architectures/rkn.py
"""
RKN画像用アーキテクチャ - Factory Pattern準拠
ファイル名: rkn.py → クラス名: rknEncoder, rknDecoder

RKN_ARCHITECTURE.md仕様準拠:
- rknEncoder: 画像(H,W,C) → 潜在表現(100次元)
- rknDecoder: 潜在表現(100次元) → 画像(H,W,C)
- ZToStateMeanDecoder: 潜在表現(100次元) → 状態平均(d_state次元)

形状処理: time_invariant.py準拠パターン
- 単一画像: (H, W, C)
- 時系列画像: (T, H, W, C) ← 基本形状
- バッチ時系列画像: (B, T, H, W, C)
"""

import math
from typing import Optional, List, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class rknEncoder(nn.Module):
    """
    画像エンコーダ: (H,W,C) → 潜在表現(100次元)
    Factory Pattern準拠: type="rkn" で自動登録

    RKN_ARCHITECTURE.md 仕様:
    (H×W×C) → Conv(5×5, 32), stride=1, same → ReLU
            → MaxPool(2×2)
            → Conv(3×3, 64), stride=1, same → ReLU
            → MaxPool(2×2)
            → Flatten → FC(hidden=200) → ReLU
            → FC(100)
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int, int] = (48, 48, 1),
        feature_dim: int = 100,
        hidden: int = 200,
        conv_channels: Tuple[int, int] = (32, 64),
        activation: str = "relu",
        normalize_input: bool = False,  # 画像用は通常False
        normalize_output: bool = False,
        track_running_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
        **kwargs
    ):
        """
        Args:
            input_resolution: 入力画像解像度 (H, W, C)
            feature_dim: 潜在特徴次元数（100次元固定推奨）
            hidden: FC層の隠れ次元数
            conv_channels: CNN層のチャネル数 (conv1_ch, conv2_ch)
            activation: 活性化関数名
            normalize_input: 入力正規化（画像では通常False）
            normalize_output: 出力正規化
            track_running_stats: 統計量追跡（time_invariant準拠）
            momentum: BatchNorm慣性
            eps: 数値安定化パラメータ
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.feature_dim = feature_dim
        self.hidden = hidden
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.eps = eps

        H, W, C = input_resolution

        # 活性化関数（time_invariant準拠）
        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.ReLU()

        # CNN層構築（RKN_ARCHITECTURE.md仕様）
        self.conv1 = nn.Conv2d(C, conv_channels[0], kernel_size=5, padding=2)  # same padding
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)  # same padding
        self.pool2 = nn.MaxPool2d(2)

        # 畳み込み後のサイズ計算: (H/4, W/4)
        conv_h, conv_w = H // 4, W // 4
        conv_output_size = conv_h * conv_w * conv_channels[1]

        # FC層（RKN_ARCHITECTURE.md仕様）
        self.fc1 = nn.Linear(conv_output_size, hidden)
        self.fc2 = nn.Linear(hidden, feature_dim)

        # 入力正規化層（time_invariant準拠、ただし画像用は通常不要）
        if self.normalize_input:
            # 注意: 画像データの場合、チャネル次元でBatchNorm2d使用
            self.input_norm = nn.BatchNorm2d(C, momentum=momentum, eps=eps)

        # 出力正規化層（time_invariant準拠）
        if self.normalize_output:
            self.output_norm = nn.BatchNorm1d(feature_dim, momentum=momentum, eps=eps)

        # 統計量（time_invariant準拠、画像形状対応）
        if track_running_stats:
            self.register_buffer('input_mean', torch.zeros(C, H, W))  # 画像形状
            self.register_buffer('input_var', torch.ones(C, H, W))
            self.register_buffer('output_mean', torch.zeros(feature_dim))
            self.register_buffer('output_var', torch.ones(feature_dim))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # 重み初期化（time_invariant準拠）
        self._initialize_weights()

    def _initialize_weights(self):
        """重み初期化（time_invariant準拠）"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # CNN用：活性化関数に応じた初期化
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        時不変性保証付き前向き計算（time_invariant準拠）

        Args:
            o: 画像データ
               - (H, W, C): 単一画像
               - (T, H, W, C): 時系列画像 ← 基本形状
               - (B, T, H, W, C): バッチ時系列画像

        Returns:
            z: 潜在表現
               - (feature_dim,): 単一潜在
               - (T, feature_dim): 時系列潜在 ← 基本形状
               - (B, T, feature_dim): バッチ時系列潜在
        """
        original_shape = o.shape
        is_single_step = len(original_shape) == 3  # (H, W, C)
        is_no_batch = len(original_shape) == 4  # (T, H, W, C) ← 基本形状

        # 形状統一化: (B, T, H, W, C)（time_invariant準拠パターン）
        if is_single_step:
            o = o.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, C)
        elif is_no_batch:
            o = o.unsqueeze(0)  # (1, T, H, W, C)
        elif len(original_shape) == 4:
            # (B, H, W, C) → (B, 1, H, W, C) の場合
            o = o.unsqueeze(1)

        B, T, H, W, C = o.shape

        # 形状検証
        expected_H, expected_W, expected_C = self.input_resolution
        if H != expected_H or W != expected_W or C != expected_C:
            raise ValueError(f"入力形状不一致: expected {self.input_resolution}, got ({H}, {W}, {C})")

        # 時間軸に沿った独立処理（時不変性保証）
        o_flat = o.view(B * T, H, W, C)  # (B*T, H, W, C)

        # (B*T, H, W, C) → (B*T, C, H, W) for Conv2d
        o_flat = o_flat.permute(0, 3, 1, 2)

        # 入力正規化（time_invariant準拠、画像用）
        if self.normalize_input:
            if self.training:
                o_flat = self.input_norm(o_flat)
            else:
                # 推論時は保存された統計量を使用（time_invariant準拠）
                if self.track_running_stats:
                    input_mean = self.input_mean.to(o_flat.device)
                    input_var = self.input_var.to(o_flat.device)
                    # 画像用正規化
                    o_flat = (o_flat - input_mean.unsqueeze(0)) / torch.sqrt(input_var.unsqueeze(0) + self.eps)
                else:
                    o_flat = self.input_norm(o_flat)

        # CNN処理（RKN_ARCHITECTURE.md仕様）
        x = self.activation(self.conv1(o_flat))      # (B*T, conv_ch[0], H, W)
        x = self.pool1(x)                            # (B*T, conv_ch[0], H/2, W/2)

        x = self.activation(self.conv2(x))           # (B*T, conv_ch[1], H/2, W/2)
        x = self.pool2(x)                            # (B*T, conv_ch[1], H/4, W/4)

        # Flatten + FC（RKN_ARCHITECTURE.md仕様）
        x = x.reshape(x.size(0), -1)                 # (B*T, flatten_size)
        x = self.activation(self.fc1(x))             # (B*T, hidden)
        z_flat = self.fc2(x)                         # (B*T, feature_dim)

        # 出力正規化（time_invariant準拠）
        if self.normalize_output:
            if self.training:
                z_flat = self.output_norm(z_flat)
            else:
                if self.track_running_stats:
                    output_mean = self.output_mean.to(z_flat.device)
                    output_var = self.output_var.to(z_flat.device)
                    z_flat = (z_flat - output_mean) / torch.sqrt(output_var + self.eps)
                else:
                    z_flat = self.output_norm(z_flat)

        # 形状復元
        z = z_flat.view(B, T, self.feature_dim)

        # 統計量更新（time_invariant準拠）
        if self.training and self.track_running_stats:
            self._update_statistics(o_flat, z_flat)

        # 元の形状に復元（time_invariant準拠）
        if is_single_step:
            return z.squeeze(0).squeeze(0)  # (feature_dim,)
        elif is_no_batch:
            return z.squeeze(0)  # (T, feature_dim) ← 基本形状
        elif len(original_shape) == 4:
            return z.squeeze(1)  # (B, feature_dim)
        else:
            return z  # (B, T, feature_dim)

    def _update_statistics(self, o_flat: torch.Tensor, z_flat: torch.Tensor):
        """統計量の指数移動平均更新（time_invariant準拠）"""
        with torch.no_grad():
            # 入力統計量（画像用）
            input_mean_batch = o_flat.mean(dim=0)  # (C, H, W)
            input_var_batch = o_flat.var(dim=0, unbiased=False)  # (C, H, W)

            # 出力統計量
            output_mean_batch = z_flat.mean(dim=0)  # (feature_dim,)
            output_var_batch = z_flat.var(dim=0, unbiased=False)  # (feature_dim,)

            # 指数移動平均更新（time_invariant準拠）
            n = self.num_batches_tracked.item()
            momentum = self.momentum if n > 0 else 1.0

            # GPU/CPUデバイス整合性確保
            input_mean_batch = input_mean_batch.to(self.input_mean.device)
            input_var_batch = input_var_batch.to(self.input_var.device)
            output_mean_batch = output_mean_batch.to(self.output_mean.device)
            output_var_batch = output_var_batch.to(self.output_var.device)

            self.input_mean.mul_(1 - momentum).add_(input_mean_batch, alpha=momentum)
            self.input_var.mul_(1 - momentum).add_(input_var_batch, alpha=momentum)
            self.output_mean.mul_(1 - momentum).add_(output_mean_batch, alpha=momentum)
            self.output_var.mul_(1 - momentum).add_(output_var_batch, alpha=momentum)

            self.num_batches_tracked += 1

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """正規化統計量の取得（time_invariant準拠）"""
        if not self.track_running_stats:
            return {}

        return {
            'input_mean': self.input_mean.clone(),
            'input_var': self.input_var.clone(),
            'output_mean': self.output_mean.clone(),
            'output_var': self.output_var.clone(),
            'num_batches_tracked': self.num_batches_tracked.clone()
        }

    def verify_time_invariance(self, o1: torch.Tensor, o2: torch.Tensor, tol: float = 1e-6) -> bool:
        """
        時不変性の検証（time_invariant準拠）

        Args:
            o1, o2: 同じ値の画像（異なる時刻）
            tol: 許容誤差

        Returns:
            bool: 時不変性が保証されているか
        """
        with torch.no_grad():
            self.eval()
            z1 = self.forward(o1)
            z2 = self.forward(o2)

            max_diff = torch.max(torch.abs(z1 - z2)).item()
            return max_diff < tol


class rknDecoder(nn.Module):
    """
    画像デコーダ: 潜在表現(100次元) → (H,W,C)
    Factory Pattern準拠: type="rkn" で自動登録

    RKN_ARCHITECTURE.md 仕様:
    z ∈ R^{100} → FC(hidden=200) → ReLU
                → FC(S) → ReLU           # S = H'·W'·C'
                → Reshape (H', W', C')   # 既定: (H/4, W/4, 64)
                → Upsample(×2) → Conv(3×3, 64), same → ReLU
                → Upsample(×2) → Conv(5×5, 32), same → ReLU
                → Conv(1×1, C), same → Sigmoid
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int, int] = (48, 48, 1),
        feature_dim: int = 100,
        grid: Optional[Tuple[int, int, int]] = None,
        hidden: int = 200,
        upsample_mode: str = "nearest",
        conv_channels: Tuple[int, int, int] = (64, 32, 1),
        activation: str = "relu",
        output_activation: str = "sigmoid",  # 画像用は sigmoid
        **kwargs
    ):
        """
        Args:
            input_resolution: 出力画像解像度 (H, W, C)
            feature_dim: 入力潜在特徴次元数
            grid: 中間グリッドサイズ (H', W', C') - None時は自動設定
            hidden: FC層の隠れ次元数
            upsample_mode: アップサンプリング方法 ("nearest", "bilinear")
            conv_channels: 逆畳み込み層のチャネル数 (ch1, ch2, output_ch)
            activation: 活性化関数名
            output_activation: 出力活性化関数名
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.feature_dim = feature_dim
        self.upsample_mode = upsample_mode

        H, W, C = input_resolution

        # 活性化関数（time_invariant準拠）
        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.ReLU()

        # 出力活性化関数
        if output_activation.lower() == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation.lower() == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation.lower() == "linear":
            self.output_activation = nn.Identity()
        else:
            self.output_activation = nn.Sigmoid()  # デフォルト

        # グリッドサイズ自動設定（RKN_ARCHITECTURE.md仕様）
        if grid is None:
            grid = (H // 4, W // 4, conv_channels[0])  # (12, 12, 64) for (48,48,1)
        self.grid = grid

        grid_h, grid_w, grid_c = grid
        grid_size = grid_h * grid_w * grid_c

        # FC層: 潜在→グリッド（RKN_ARCHITECTURE.md仕様）
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.fc2 = nn.Linear(hidden, grid_size)

        # Upsample + Conv層（RKN_ARCHITECTURE.md仕様）
        self.upsample1 = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.conv1 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.conv2 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=5, padding=2)

        # 重み初期化（time_invariant準拠）
        self._initialize_weights()

    def _initialize_weights(self):
        """重み初期化（time_invariant準拠）"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        潜在表現→画像: 時不変デコーディング（time_invariant準拠）

        Args:
            z: 潜在表現
               - (feature_dim,): 単一潜在
               - (T, feature_dim): 時系列潜在 ← 基本形状
               - (B, T, feature_dim): バッチ時系列潜在

        Returns:
            o: 再構成画像
               - (H, W, C): 単一画像
               - (T, H, W, C): 時系列画像 ← 基本形状
               - (B, T, H, W, C): バッチ時系列画像
        """
        original_shape = z.shape
        is_single_step = len(original_shape) == 1  # (feature_dim,)
        is_no_batch = len(original_shape) == 2  # (T, feature_dim) ← 基本形状

        # 形状統一化: (B, T, feature_dim)（time_invariant準拠）
        if is_single_step:
            z = z.unsqueeze(0).unsqueeze(0)  # (1, 1, feature_dim)
        elif is_no_batch:
            z = z.unsqueeze(0)  # (1, T, feature_dim)
        elif len(original_shape) == 2:
            # (B, feature_dim) → (B, 1, feature_dim) の場合もあり得る
            z = z.unsqueeze(1)

        B, T, feature_dim = z.shape

        # 形状検証
        if feature_dim != self.feature_dim:
            raise ValueError(f"潜在特徴次元不一致: expected {self.feature_dim}, got {feature_dim}")

        # 時間軸に沿った独立処理（時不変性保証）
        z_flat = z.view(B * T, feature_dim)  # (B*T, feature_dim)

        # FC処理（RKN_ARCHITECTURE.md仕様）
        x = self.activation(self.fc1(z_flat))       # (B*T, hidden)
        x = self.activation(self.fc2(x))            # (B*T, grid_size)

        # Reshape to grid
        grid_h, grid_w, grid_c = self.grid
        x = x.view(-1, grid_c, grid_h, grid_w)      # (B*T, grid_c, grid_h, grid_w)

        # Upsample + Conv処理（RKN_ARCHITECTURE.md仕様）
        x = self.upsample1(x)                       # (B*T, grid_c, 2*grid_h, 2*grid_w)
        x = self.activation(self.conv1(x))          # (B*T, conv_ch[1], 2*grid_h, 2*grid_w)

        x = self.upsample2(x)                       # (B*T, conv_ch[1], 4*grid_h, 4*grid_w)
        x = self.output_activation(self.conv2(x))   # (B*T, conv_ch[2], H, W) ∈ [0,1]

        # (B*T, C, H, W) → (B*T, H, W, C)
        x = x.permute(0, 2, 3, 1)

        # 形状復元
        H, W, C = self.input_resolution
        o = x.view(B, T, H, W, C)

        # 元の形状に復元（time_invariant準拠）
        if is_single_step:
            return o.squeeze(0).squeeze(0)  # (H, W, C)
        elif is_no_batch:
            return o.squeeze(0)  # (T, H, W, C) ← 基本形状
        elif len(original_shape) == 2:
            return o.squeeze(1)  # (B, H, W, C)
        else:
            return o  # (B, T, H, W, C)


class rkn_targetDecoder(nn.Module):
    """
    ターゲット予測デコーダ (rkn_targetDecoder): 潜在表現(d次元) → 状態平均(d_state次元)
    命名規則: <type>_targetDecoder (ファクトリパターン準拠)

    構造（各時刻）:
    z_t ∈ R^{feature_dim} → FC(hidden) → ReLU → FC(d_state) → output_activation

    数式マッピング:
    μ̂_{s_t} = output_activation(W_2 σ(W_1 z_t + b_1) + b_2) ∈ R^{d_state}
    """

    def __init__(
        self,
        feature_dim: int = 100,
        state_dim: int = 8,  # quad制御状態: 8次元
        hidden: int = 50,
        activation: str = "relu",
        output_activation: str = "linear",  # 出力活性化関数
        **kwargs
    ):
        """
        Args:
            feature_dim: 入力潜在特徴次元数
            state_dim: 出力状態次元数（quadコプター制御状態: 8次元）
            hidden: FC層の隠れ次元数
            activation: 活性化関数名
            output_activation: 出力活性化関数名 ("linear", "tanh", "sigmoid")
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.hidden = hidden

        # 活性化関数（time_invariant準拠）
        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.ReLU()

        # 出力活性化関数
        if output_activation.lower() == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation.lower() == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation.lower() == "linear":
            self.output_activation = nn.Identity()
        else:
            self.output_activation = nn.Identity()  # デフォルト

        # FC層（RKN_ARCHITECTURE.md仕様）
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.fc2 = nn.Linear(hidden, state_dim)  # 出力活性化は後で適用

        # 重み初期化（time_invariant準拠）
        self._initialize_weights()

    def _initialize_weights(self):
        """重み初期化（time_invariant準拠）"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        潜在表現→状態平均: 時不変状態推定（time_invariant準拠）

        Args:
            z: 潜在表現
               - (feature_dim,): 単一潜在
               - (T, feature_dim): 時系列潜在 ← 基本形状
               - (B, T, feature_dim): バッチ時系列潜在

        Returns:
            μ̂_s: 状態平均
                - (state_dim,): 単一状態
                - (T, state_dim): 時系列状態 ← 基本形状
                - (B, T, state_dim): バッチ時系列状態
        """
        original_shape = z.shape
        is_single_step = len(original_shape) == 1  # (feature_dim,)
        is_no_batch = len(original_shape) == 2  # (T, feature_dim) ← 基本形状

        # 形状統一化: (B, T, feature_dim)（time_invariant準拠）
        if is_single_step:
            z = z.unsqueeze(0).unsqueeze(0)  # (1, 1, feature_dim)
        elif is_no_batch:
            z = z.unsqueeze(0)  # (1, T, feature_dim)
        elif len(original_shape) == 2:
            z = z.unsqueeze(1)  # (B, 1, feature_dim)

        B, T, feature_dim = z.shape

        # 形状検証
        if feature_dim != self.feature_dim:
            raise ValueError(f"潜在特徴次元不一致: expected {self.feature_dim}, got {feature_dim}")

        # 時間軸に沿った独立処理（時不変性保証）
        z_flat = z.view(B * T, feature_dim)  # (B*T, feature_dim)

        # FC処理（RKN_ARCHITECTURE.md仕様）
        x = self.activation(self.fc1(z_flat))       # (B*T, hidden)
        x = self.fc2(x)                             # (B*T, state_dim)
        mu_s_flat = self.output_activation(x)       # (B*T, state_dim) ← 出力活性化適用

        # 形状復元
        mu_s = mu_s_flat.view(B, T, self.state_dim)

        # 元の形状に復元（time_invariant準拠）
        if is_single_step:
            return mu_s.squeeze(0).squeeze(0)  # (state_dim,)
        elif is_no_batch:
            return mu_s.squeeze(0)  # (T, state_dim) ← 基本形状
        elif len(original_shape) == 2:
            return mu_s.squeeze(1)  # (B, state_dim)
        else:
            return mu_s  # (B, T, state_dim)


# Factory関数（オプション）: 状態デコーダ用（互換性のため保持）
def make_state_decoder(cfg: Dict[str, Any]) -> rkn_targetDecoder:
    """
    状態デコーダのファクトリ関数
    RKN_ARCHITECTURE.md仕様準拠

    Args:
        cfg: 設定辞書
            - feature_dim: 潜在特徴次元数（デフォルト100）
            - state_dim: 状態次元数（必須）
            - hidden: 隠れ層次元数（デフォルト50）

    Returns:
        rkn_targetDecoder: 状態デコーダインスタンス
    """
    # デフォルト値設定（RKN_ARCHITECTURE.md準拠）
    feature_dim = cfg.get('feature_dim', 100)
    state_dim = cfg.get('state_dim')
    hidden = cfg.get('hidden', 50)

    if state_dim is None:
        raise ValueError("state_dim は必須パラメータです")

    return rkn_targetDecoder(
        feature_dim=feature_dim,
        state_dim=state_dim,
        hidden=hidden,
        **{k: v for k, v in cfg.items() if k not in ['feature_dim', 'state_dim', 'hidden']}
    )