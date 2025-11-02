# src/inference/state_estimator.py
"""
統合推論クラス: StateEstimator

学習済みDFIVモデルからKalman Filtering推論エンジンを構築。
DF-A/DF-B コンポーネントとの統合、ノイズ推定、推論実行を管理。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path
import warnings
import yaml

from .kalman_filter import OperatorBasedKalmanFilter
from .utils import (
    estimate_noise_covariances,
    compute_residuals_from_operators,
    validate_kalman_inputs,
    initialize_state_data_driven,
    format_filter_results
)
from ..ssm.realization import Realization


class StateEstimator:
    """
    統合推論クラス
    
    学習済みDFIVモデル（DF-A + DF-B）から転送作用素を抽出し、
    Algorithm 1による逐次状態推定を実行。
    
    機能:
    - 学習済みモデル読み込み
    - ノイズ共分散推定
    - Kalman Filter初期化・実行
    - バッチ・オンライン推論対応
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        設定ファイルから初期化
        
        Args:
            config: 推論設定辞書
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        # コンポーネント
        self.df_state_layer = None      # DF-A
        self.df_obs_layer = None        # DF-B
        self.encoder = None             # エンコーダ
        self.realization = None         # 状態空間実現
        self.kalman_filter = None       # Kalman Filter
        
        # 学習済みパラメータ
        self.V_A: Optional[torch.Tensor] = None
        self.V_B: Optional[torch.Tensor] = None
        self.U_A: Optional[torch.Tensor] = None
        self.U_B: Optional[torch.Tensor] = None
        self.Q: Optional[torch.Tensor] = None
        self.R: Optional[Union[torch.Tensor, float]] = None
        
        # 状態
        self.is_initialized = False
        self.calibration_data: Optional[torch.Tensor] = None

    @classmethod
    def from_trained_model(
        cls,
        model_path: Union[str, Path],
        config_path: Union[str, Path]
    ) -> 'StateEstimator':
        """
        学習済みモデルから初期化
        
        V_A, V_B, φ_θ, ψ_ω, エンコーダを読み込み
        
        Args:
            model_path: 学習済みモデルパス
            config_path: 設定ファイルパス
            
        Returns:
            StateEstimator: 初期化済みインスタンス
        """
        # 設定読み込み
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        estimator = cls(config['inference'])
        estimator.load_components(model_path)
        
        return estimator

    def load_components(self, model_path: Union[str, Path]):
        """
        個別コンポーネントの読み込み

        DF-A, DF-B, エンコーダから演算子を抽出

        Args:
            model_path: 学習済みモデルパス
        """
        model_path = Path(model_path)

        try:
            # モデル状態読み込み
            checkpoint = torch.load(model_path, map_location=self.device)

            # checkpoint['config']から学習時設定を取得
            training_config = checkpoint.get('config', {})

            # エンコーダタイプを取得
            encoder_type = training_config.get('model', {}).get('encoder', {}).get('type')

            # 次元情報を更新
            self._update_config_from_checkpoint(training_config)

            # 各コンポーネントの状態辞書取得（checkpointから直接）
            state_dict = checkpoint

            # DF-A コンポーネント
            if 'df_state' in state_dict:
                self._load_df_state_component(state_dict['df_state'])
            else:
                raise KeyError("DF-A component not found in model")

            # DF-B コンポーネント
            if 'df_obs' in state_dict:
                self._load_df_obs_component(state_dict['df_obs'])
            else:
                raise KeyError("DF-B component not found in model")

            # エンコーダ（タイプ情報付き）
            if 'encoder' in state_dict:
                self._load_encoder_component(state_dict['encoder'], encoder_type=encoder_type)
            else:
                raise KeyError("Encoder component not found in model")

            # 転送作用素抽出
            self._extract_operators()

            print(f"Successfully loaded components from {model_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model components: {e}")

    def _update_config_from_checkpoint(self, training_config: Dict[str, Any]):
        """checkpointから設定を更新"""
        if not training_config:
            return

        # モデル設定を更新
        if 'model' not in self.config:
            self.config['model'] = {}

        # エンコーダ設定
        if 'encoder' in training_config.get('model', {}):
            self.config['model']['encoder'] = training_config['model']['encoder']

        # DF-A設定
        if 'ssm' in training_config and 'df_state' in training_config['ssm']:
            if 'df_state' not in self.config.get('model', {}):
                self.config['model']['df_state'] = {}
            df_state_cfg = training_config['ssm']['df_state']
            self.config['model']['df_state'].update({
                'feature_dim': df_state_cfg.get('feature_dim'),
                'state_dim': training_config['ssm']['realization'].get('rank')
            })

        # DF-B設定
        if 'ssm' in training_config and 'df_observation' in training_config['ssm']:
            if 'df_obs' not in self.config.get('model', {}):
                self.config['model']['df_obs'] = {}
            df_obs_cfg = training_config['ssm']['df_observation']
            self.config['model']['df_obs'].update({
                'obs_feature_dim': df_obs_cfg.get('obs_feature_dim')
            })

    def _flatten_nested_state_dict(self, nested_dict: Dict[str, Any]) -> Dict[str, Any]:
        """ネストしたstate_dictを平坦化"""
        flattened = {}
        for key, value in nested_dict.items():
            if isinstance(value, dict) and hasattr(value, 'keys'):
                # OrderedDict等の辞書形式の場合、キーを結合
                for sub_key, sub_value in value.items():
                    flattened[f"{key}.{sub_key}"] = sub_value
            else:
                # 通常の値はそのまま
                flattened[key] = value
        return flattened

    def _load_df_state_component(self, df_state_dict: Dict[str, Any]):
        """DF-A コンポーネント読み込み"""
        from ..ssm.df_state_layer import DFStateLayer
        
        # 設定から次元取得
        state_config = self.config.get('model', {}).get('df_state', {})
        
        self.df_state_layer = DFStateLayer(
            state_dim=state_config.get('state_dim', 5),
            feature_dim=state_config.get('feature_dim', 16),
            lambda_A=state_config.get('lambda_A', 1e-3),
            lambda_B=state_config.get('lambda_B', 1e-3),
            feature_net_config=state_config.get('feature_net'),
            cross_fitting_config=state_config.get('cross_fitting')
        ).to(self.device)
        
        # ネストしたstate_dictを平坦化
        flattened_state_dict = self._flatten_nested_state_dict(df_state_dict)
        self.df_state_layer.load_state_dict(flattened_state_dict)
        self.df_state_layer.eval()

    def _load_df_obs_component(self, df_obs_dict: Dict[str, Any]):
        """DF-B コンポーネント読み込み"""
        from ..ssm.df_observation_layer import DFObservationLayer
        
        obs_config = self.config.get('model', {}).get('df_obs', {})
        
        self.df_obs_layer = DFObservationLayer(
            df_state_layer=self.df_state_layer,  # DF-A参照
            obs_feature_dim=obs_config.get('obs_feature_dim', 8),
            lambda_B=obs_config.get('lambda_B', 1e-3),
            lambda_dB=obs_config.get('lambda_dB', 1e-3),
            obs_net_config=obs_config.get('obs_net'),
            cross_fitting_config=obs_config.get('cross_fitting')
        ).to(self.device)
        
        # ネストしたstate_dictを平坦化
        flattened_obs_dict = self._flatten_nested_state_dict(df_obs_dict)
        # phi_thetaは既にdf_state_layerから共有参照されるため、strict=Falseで読み込み
        self.df_obs_layer.load_state_dict(flattened_obs_dict, strict=False)
        self.df_obs_layer.eval()

    def _load_encoder_component(self, encoder_dict: Dict[str, Any], encoder_type: str = None):
        """
        動的エンコーダ読み込み

        Args:
            encoder_dict: エンコーダのstate_dict
            encoder_type: エンコーダタイプ（'rkn', 'time_invariant', 'tcn'等）
                          Noneの場合はstate_dictから自動検出
        """
        # エンコーダタイプの決定
        if encoder_type is None:
            encoder_type = self._detect_encoder_type(encoder_dict)

        # Factory Patternを使用
        from ..models.encoder import build_encoder

        # エンコーダ設定の構築
        encoder_config = self._build_encoder_config_from_state_dict(
            encoder_dict, encoder_type
        )
        encoder_config['type'] = encoder_type

        # エンコーダ作成
        self.encoder = build_encoder(encoder_config).to(self.device)

        # state_dict読み込み
        self.encoder.load_state_dict(encoder_dict)
        self.encoder.eval()

        print(f"✅ Loaded {encoder_type}Encoder successfully")

    def _detect_encoder_type(self, encoder_dict: Dict[str, Any]) -> str:
        """state_dictからエンコーダタイプを自動検出"""
        keys = set(encoder_dict.keys())

        # rknEncoder の特徴的なキー
        if any('conv' in k for k in keys):
            return 'rkn'

        # time_invariantEncoder の特徴的なキー
        if any('core_net' in k for k in keys):
            return 'time_invariant'

        # tcnEncoder の特徴的なキー
        if any('temporal' in k for k in keys):
            return 'tcn'

        # デフォルト
        return 'time_invariant'

    def _build_encoder_config_from_state_dict(
        self,
        encoder_dict: Dict[str, Any],
        encoder_type: str
    ) -> Dict[str, Any]:
        """state_dictからエンコーダ設定を復元"""

        if encoder_type == 'rkn':
            # rknEncoder の設定を推定
            # conv1.weight の形状から input_resolution を推定
            conv1_weight = encoder_dict.get('conv1.weight')
            if conv1_weight is not None:
                C = conv1_weight.shape[1]  # 入力チャネル数
                # H, W は実行時データから決定（または config から取得）
                config_from_checkpoint = self.config.get('model', {}).get('encoder', {})
                input_resolution_from_cfg = config_from_checkpoint.get('input_resolution', [48, 48, 1])
                if len(input_resolution_from_cfg) >= 2:
                    H, W = input_resolution_from_cfg[:2]
                else:
                    H, W = 48, 48  # デフォルト
                input_resolution = [H, W, C]
            else:
                input_resolution = [48, 48, 1]  # デフォルト

            # feature_dim を推定（fc_out.weight の形状から）
            fc_out_weight = encoder_dict.get('fc_out.weight')
            if fc_out_weight is not None:
                feature_dim = fc_out_weight.shape[0]
            else:
                feature_dim = 50  # デフォルト

            # conv_channels を推定
            conv_channels = []
            if 'conv1.weight' in encoder_dict:
                conv_channels.append(encoder_dict['conv1.weight'].shape[0])
            if 'conv2.weight' in encoder_dict:
                conv_channels.append(encoder_dict['conv2.weight'].shape[0])

            # hidden を推定（fc1.weight の形状から）
            fc1_weight = encoder_dict.get('fc1.weight')
            if fc1_weight is not None:
                hidden = fc1_weight.shape[0]
            else:
                hidden = 200  # デフォルト

            return {
                'input_resolution': input_resolution,
                'feature_dim': feature_dim,
                'hidden': hidden,
                'conv_channels': conv_channels if conv_channels else [32, 64],
                'activation': 'relu',
                'normalize_input': False,
                'normalize_output': True,
                'track_running_stats': True
            }

        elif encoder_type == 'time_invariant':
            # 既存の _detect_time_invariant_output_dim() 等を使用
            input_dim = self.config.get('model', {}).get('encoder', {}).get('input_dim', 6)
            output_dim = self._detect_time_invariant_output_dim(encoder_dict)

            return {
                'input_dim': input_dim,
                'output_dim': output_dim,
                'architecture': 'mlp',
                'normalize_input': True,
                'normalize_output': True,
                'track_running_stats': True
            }

        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def _detect_time_invariant_output_dim(self, encoder_dict: Dict[str, Any]) -> int:
        """time_invariantエンコーダーの出力次元を検出"""
        if 'output_mean' in encoder_dict:
            return encoder_dict['output_mean'].shape[0]
        # core_netの最終層から推定
        for key in encoder_dict.keys():
            if key.endswith('.bias') and 'core_net' in key:
                return encoder_dict[key].shape[0]
        return 8  # デフォルト

    def _extract_operators(self):
        """
        学習済みコンポーネントから転送作用素を抽出
        
        DF-A: V_A, U_A
        DF-B: V_B, u_B
        """
        if not all([self.df_state_layer, self.df_obs_layer]):
            raise RuntimeError("DF components not loaded")
            
        # DF-A から V_A, U_A 抽出
        if hasattr(self.df_state_layer, 'V_A') and self.df_state_layer.V_A is not None:
            self.V_A = self.df_state_layer.V_A.clone().detach()
        else:
            raise RuntimeError("V_A not found in DF-A component")
            
        if hasattr(self.df_state_layer, 'U_A') and self.df_state_layer.U_A is not None:
            self.U_A = self.df_state_layer.U_A.clone().detach()
        else:
            raise RuntimeError("U_A not found in DF-A component")
            
        # DF-B から V_B, u_B 抽出
        if hasattr(self.df_obs_layer, 'V_B') and self.df_obs_layer.V_B is not None:
            self.V_B = self.df_obs_layer.V_B.clone().detach()
        else:
            raise RuntimeError("V_B not found in DF-B component")
            
        if hasattr(self.df_obs_layer, 'U_B') and self.df_obs_layer.U_B is not None:
            self.U_B = self.df_obs_layer.U_B.clone().detach()
        else:
            raise RuntimeError("u_B not found in DF-B component")
            
        print("Operators extracted successfully:")
        print(f"  V_A: {self.V_A.shape}")
        print(f"  V_B: {self.V_B.shape}")
        print(f"  U_A: {self.U_A.shape}")
        print(f"  U_B: {self.U_B.shape}")

    def estimate_noise_covariances(
        self,
        calibration_data: torch.Tensor,
        method: str = "residual_based"
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, float]]:
        """
        キャリブレーション用データでQ, R推定
        
        式45-46の実装
        
        Args:
            calibration_data: キャリブレーション観測 (T_cal, n)
            method: ノイズ推定法 ("residual_based")
            
        Returns:
            Q: 状態ノイズ共分散 (dA, dA)
            R: 観測ノイズ分散
        """
        if method != "residual_based":
            raise ValueError(f"Unknown noise estimation method: {method}")
            
        self.calibration_data = calibration_data
        T_cal = calibration_data.size(0)
        
        print(f"Estimating noise covariances from {T_cal} calibration samples...")
        
        with torch.no_grad():
            # 1. エンコード: {y_t} → {m_t} (多変量特徴量対応)
            m_series = self.encoder(calibration_data.unsqueeze(0)).squeeze(0)  # (T_cal, output_dim)

            # 2. 状態空間実現: {m_t} → {x_t} (多変量入力対応)
            if self.realization is None:
                # 設定からrealizationパラメータを取得
                realization_config = self.config.get('ssm', {}).get('realization', {})
                past_horizon = realization_config.get('past_horizon', 10)

                # データ長に基づいてpast_horizonを自動調整
                T_cal = m_series.size(0)
                max_horizon = (T_cal - 1) // 2  # N >= 1を保証するための最大horizon
                if past_horizon > max_horizon:
                    past_horizon = max(1, max_horizon)
                    print(f"past_horizon調整: {realization_config.get('past_horizon', 10)} → {past_horizon} (データ長: {T_cal})")

                self.realization = Realization(
                    past_horizon=past_horizon,
                    jitter=realization_config.get('jitter', 1e-6),
                    cond_thresh=realization_config.get('cond_thresh', 1e10),
                    rank=realization_config.get('rank', 4),
                    reg_type=realization_config.get('reg_type', 'sum')
                )
                # 多変量特徴量 (T_cal, output_dim) をそのままfit
                self.realization.fit(m_series)

            # realizationで状態系列を生成: (N, rank) where N = T_cal - 2*h + 1
            x_series = self.realization.filter(m_series)

            # 時間対応を合わせるため、m_seriesも同じ範囲を使用
            # realization.filter は時点 h から h+N-1 に対応
            h = self.realization.h
            m_series_aligned = m_series[h:h + x_series.size(0)]  # (N,)

            # 3. 特徴写像適用: {x_t} → {φ_t}, {ψ_t}
            # 状態系列x_tをφ_θに、対応する時間範囲のスカラー系列m_tをψ_ωに渡す
            phi_sequence = self._apply_state_feature_mapping(x_series)           # (N+1, dA)
            psi_sequence = self._apply_obs_feature_mapping(m_series_aligned)     # (N+1, dB)
            
            # 3. 残差計算
            residuals_state, residuals_obs = compute_residuals_from_operators(
                phi_sequence, psi_sequence, self.V_A, self.V_B
            )
            
            # 4. 共分散推定
            regularization = self.config.get('noise_estimation', {})
            Q, R = estimate_noise_covariances(
                residuals_state, residuals_obs, regularization
            )
            
        self.Q = Q
        self.R = R
        
        print(f"Noise covariances estimated:")
        print(f"  Q condition number: {torch.linalg.cond(Q).item():.2e}")
        if isinstance(R, torch.Tensor):
            print(f"  R condition number: {torch.linalg.cond(R).item():.2e}")
        else:
            print(f"  R (scalar): {R:.6f}")
            
        return Q, R

    def _apply_state_feature_mapping(self, x_series: torch.Tensor) -> torch.Tensor:
        """
        状態特徴写像 φ_θ の適用

        学習済みDF-A層のφ_θネットワークが必須。

        Args:
            x_series: 状態系列 (N, rank) - realization.filter()の出力

        Returns:
            torch.Tensor: 状態特徴系列 (N+1, dA)

        Raises:
            RuntimeError: DF-A層またはφ_θネットワークが存在しない場合
        """
        # DF-A層とφ_θネットワークの存在確認（必須）
        if self.df_state_layer is None or not hasattr(self.df_state_layer, 'phi_theta'):
            raise RuntimeError(
                "DF-A layer with phi_theta network is required for state feature generation. "
                "Cannot proceed without learned φ_θ(m_t) transformation."
            )

        N = x_series.size(0)

        with torch.no_grad():
            phi_sequence = []
            for t in range(N + 1):
                if t < N:
                    # 学習済みφ_θネットワークで状態特徴生成
                    x_t_input = x_series[t]  # (rank,) - 状態ベクトル
                    phi_t = self.df_state_layer.phi_theta(x_t_input)  # (dA,)
                else:
                    # 最後の値を使用（予測時）
                    x_t_input = x_series[-1]  # (rank,) - 状態ベクトル
                    phi_t = self.df_state_layer.phi_theta(x_t_input)  # (dA,)

                # 出力を(dA,)形状に正規化
                while phi_t.dim() > 1 and phi_t.size(0) == 1:
                    phi_t = phi_t.squeeze(0)

                if phi_t.dim() != 1:
                    raise RuntimeError(
                        f"phi_theta output has unexpected shape: {phi_t.shape}. "
                        f"Expected 1D tensor (dA,) for state features."
                    )

                phi_sequence.append(phi_t)

            return torch.stack(phi_sequence)  # (T+1, dA)

    def _apply_obs_feature_mapping(self, m_series: torch.Tensor) -> torch.Tensor:
        """
        観測特徴写像 ψ_ω の適用

        学習済みDF-B層のψ_ωネットワークが必須。

        Args:
            m_series: スカラー特徴系列 (T,)

        Returns:
            torch.Tensor: 観測特徴系列 (T+1, dB)

        Raises:
            RuntimeError: DF-B層またはψ_ωネットワークが存在しない場合
        """
        # DF-B層とψ_ωネットワークの存在確認（必須）
        if self.df_obs_layer is None or not hasattr(self.df_obs_layer, 'psi_omega'):
            raise RuntimeError(
                "DF-B layer with psi_omega network is required for observation feature generation. "
                "Cannot proceed without learned ψ_ω(m_t) transformation."
            )

        T = m_series.size(0)

        with torch.no_grad():
            psi_sequence = []
            for t in range(T + 1):
                if t < T:
                    # 学習済みψ_ωネットワークで観測特徴生成
                    m_t_input = m_series[t]  # (r,) - 正しい入力次元を保持
                    psi_t = self.df_obs_layer.psi_omega(m_t_input)  # (dB,)
                else:
                    # 最後の値を使用（予測時）
                    m_t_input = m_series[-1]  # (r,) - 正しい入力次元を保持
                    psi_t = self.df_obs_layer.psi_omega(m_t_input)  # (dB,)

                # 出力を(dB,)形状に正規化
                while psi_t.dim() > 1 and psi_t.size(0) == 1:
                    psi_t = psi_t.squeeze(0)

                if psi_t.dim() != 1:
                    raise RuntimeError(
                        f"psi_omega output has unexpected shape: {psi_t.shape}. "
                        f"Expected 1D tensor (dB,) for observation features."
                    )

                psi_sequence.append(psi_t)

            return torch.stack(psi_sequence)  # (T+1, dB)

    def initialize_filtering(
        self,
        initial_data: Optional[torch.Tensor] = None,
        method: str = "data_driven"
    ):
        """
        フィルタリング開始前の初期化
        
        KalmanFilterの作成、初期状態設定
        
        Args:
            initial_data: 初期化用データ (N0, n) or None
            method: 初期化方法 ("data_driven" | "zero")
        """
        if not all([self.V_A is not None, self.V_B is not None, self.U_A is not None, self.U_B is not None]):
            raise RuntimeError("Operators not extracted. Call load_components() first.")
            
        if self.Q is None or self.R is None:
            # デフォルトノイズ設定
            warnings.warn("Noise covariances not estimated. Using defaults.")
            dA = int(self.V_A.size(0))
            self.Q = 0.01 * torch.eye(dA, device=self.device)
            self.R = 0.1
            
        # 入力検証
        validation = validate_kalman_inputs(
            self.V_A, self.V_B, self.U_A, self.U_B, self.Q, self.R
        )
        
        if not validation["valid"]:
            raise RuntimeError(f"Invalid Kalman inputs: {validation['errors']}")
            
        if validation["warnings"]:
            for warning in validation["warnings"]:
                warnings.warn(warning)
                
        # Kalman Filter作成（学習済みDF-B層を渡す）
        self.kalman_filter = OperatorBasedKalmanFilter(
            V_A=self.V_A,
            V_B=self.V_B,
            U_A=self.U_A,
            U_B=self.U_B,
            Q=self.Q,
            R=self.R,
            encoder=self.encoder,
            df_obs_layer=self.df_obs_layer,  # 学習済みψ_ω含む
            device=str(self.device)
        )
        
        # 初期状態設定
        if initial_data is not None:
            self.kalman_filter.initialize_state(initial_data, method)
        elif self.calibration_data is not None:
            # キャリブレーションデータから初期化
            n_init = min(10, self.calibration_data.size(0))
            self.kalman_filter.initialize_state(
                self.calibration_data[:n_init], method
            )
        else:
            # ゼロ初期化
            self.kalman_filter.initialize_state(
                torch.zeros(1, self.encoder.input_dim, device=self.device), "zero"
            )
            
        self.is_initialized = True
        print("Kalman Filter initialized successfully")

    def filter_sequence(
        self,
        observations: torch.Tensor,
        return_likelihood: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], 
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        バッチフィルタリング（従来型）
        
        観測系列全体を一度に処理
        
        Args:
            observations: 観測系列 (T, n)
            return_likelihood: 尤度も返すかどうか
            
        Returns:
            X_means: 状態平均系列 (T, r)
            X_covariances: 状態共分散系列 (T, r, r)
            likelihoods: 観測尤度系列 (T,) [optional]
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize_filtering() first.")
            
        return self.kalman_filter.filter_sequence(observations, None, return_likelihood)

    def filter_online(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        オンラインフィルタリング（逐次型）
        
        1観測ずつ処理、内部状態保持
        
        Args:
            observation: 現在観測 (n,)
            
        Returns:
            x_hat: 推定状態 (r,)
            Sigma_x: 状態共分散 (r, r)  
            likelihood: 観測尤度
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize_filtering() first.")
            
        return self.kalman_filter.filter_step(observation)

    def reset_state(self):
        """
        内部状態リセット（新しい系列開始時）
        """
        if self.kalman_filter is not None:
            self.kalman_filter.reset_state()

    def get_current_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        現在の状態推定値と信頼区間を取得
        
        Returns:
            x_hat: 現在状態推定 (r,)
            Sigma_x: 現在状態共分散 (r, r)
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized")
            
        return self.kalman_filter.get_current_state()

    def predict_ahead(
        self,
        n_steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        n期先予測
        
        現在状態から将来予測
        
        Args:
            n_steps: 予測ステップ数
            
        Returns:
            x_pred: 予測状態 (n_steps, r)
            Sigma_pred: 予測共分散 (n_steps, r, r)
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized")
            
        # 現在状態取得
        mu_current, Sigma_current = self.kalman_filter.mu, self.kalman_filter.Sigma
        
        # 逐次予測
        predictions = []
        covariances = []
        
        mu, Sigma = mu_current.clone(), Sigma_current.clone()
        
        for step in range(n_steps):
            # 時間更新のみ（観測なし）
            mu, Sigma = self.kalman_filter.predict_step(mu, Sigma)
            
            # 元状態空間での復元
            x_pred, Sigma_x_pred = self.kalman_filter._recover_original_state(mu, Sigma)
            
            predictions.append(x_pred)
            covariances.append(Sigma_x_pred)
            
        return torch.stack(predictions), torch.stack(covariances)

    def get_filter_diagnostics(self) -> Dict[str, Any]:
        """
        フィルタ診断情報取得
        
        Returns:
            Dict: 診断結果
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}
            
        diagnostics = {
            "initialization_status": self.is_initialized,
            "operator_shapes": {
                "V_A": self.V_A.shape,
                "V_B": self.V_B.shape,
                "U_A": self.U_A.shape,
                "U_B": self.U_B.shape
            },
            "numerical_stability": self.kalman_filter.check_numerical_stability()
        }
        
        if self.Q is not None:
            diagnostics["noise_covariances"] = {
                "Q_condition": torch.linalg.cond(self.Q).item(),
                "Q_trace": torch.trace(self.Q).item()
            }
            
        if isinstance(self.R, torch.Tensor):
            diagnostics["noise_covariances"]["R_condition"] = torch.linalg.cond(self.R).item()
        else:
            diagnostics["noise_covariances"]["R_scalar"] = self.R
            
        return diagnostics

    def export_for_deployment(self, export_path: Union[str, Path]):
        """
        デプロイ用ファイル出力
        
        推論のみ必要なコンポーネント抽出
        
        Args:
            export_path: 出力パス
        """
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Cannot export.")
            
        # 推論用パラメータ
        inference_params = {
            "operators": {
                "V_A": self.V_A.cpu(),
                "V_B": self.V_B.cpu(),
                "U_A": self.U_A.cpu(),
                "U_B": self.U_B.cpu()
            },
            "noise_covariances": {
                "Q": self.Q.cpu(),
                "R": self.R if isinstance(self.R, (int, float)) else self.R.cpu()
            },
            "config": self.config
        }
        
        # エンコーダ状態
        encoder_state = self.encoder.state_dict()
        
        # 保存
        torch.save(inference_params, export_path / "inference_params.pth")
        torch.save(encoder_state, export_path / "encoder.pth")
        
        with open(export_path / "inference_config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        print(f"Inference components exported to {export_path}")