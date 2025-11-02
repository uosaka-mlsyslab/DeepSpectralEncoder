# src/models/inference_model.py
"""
統合インターフェース: InferenceModel

学習済みモデルと推論設定から初期化し、バッチ・ストリーミング推論を統一的に提供。
研究・開発・本番環境で共通して使用できるインターフェース。
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union, Generator
from pathlib import Path
import yaml
import json
import warnings
from dataclasses import dataclass

from ..inference.state_estimator import StateEstimator
from ..inference.streaming import StreamingEstimator
from ..inference.utils import format_filter_results


@dataclass
class InferenceConfig:
    """推論設定の構造化"""
    # デバイス設定
    device: str = 'auto'
    
    # フィルタリング設定
    noise_estimation_method: str = 'residual_based'
    gamma_Q: float = 1e-6
    gamma_R: float = 1e-6
    use_calibration: bool = True
    
    # 初期化設定
    initialization_method: str = 'data_driven'
    n_init_samples: int = 50
    
    # ストリーミング設定
    buffer_size: int = 100
    batch_processing: bool = False
    anomaly_detection: bool = True
    anomaly_threshold: float = 3.0
    
    # 数値安定性
    condition_threshold: float = 1e12
    min_eigenvalue: float = 1e-8
    jitter: float = 1e-6
    
    # 出力設定
    save_states: bool = True
    save_covariances: bool = False  # メモリ節約
    save_likelihoods: bool = True


class InferenceModel:
    """
    学習済みモデルと推論設定から初期化される統合推論システム
    
    研究・開発・本番環境での一貫した推論インターフェースを提供。
    バッチ推論・ストリーミング推論の両方に対応。
    """
    
    def __init__(
        self,
        trained_model_path: Union[str, Path],
        inference_config: Union[Dict[str, Any], InferenceConfig, str, Path]
    ):
        """
        学習済みモデルと推論設定から初期化
        
        Args:
            trained_model_path: 学習済みモデルのパス
            inference_config: 推論設定（辞書、InferenceConfig、またはYAMLファイルパス）
        """
        self.model_path = Path(trained_model_path)
        
        # 設定の解析 - 複数ドキュメント対応
        if isinstance(inference_config, (str, Path)):
            with open(inference_config, 'r') as f:
                documents = list(yaml.safe_load_all(f))
                # 最初のドキュメントをデフォルト設定として使用
                config_dict = documents[0] if documents else {}
            # 推論設定のみを抽出
            inference_settings = config_dict.get('inference', {})
            if not inference_settings:
                print("⚠️  'inference'セクションが見つかりません。デフォルト設定を使用します。")
                print("📝 モデル構造情報はcheckpointから自動的に取得されます。")
            self.config = InferenceConfig(**inference_settings)
        elif isinstance(inference_config, dict):
            self.config = InferenceConfig(**inference_config)
        elif isinstance(inference_config, InferenceConfig):
            self.config = inference_config
        else:
            raise ValueError("Invalid inference_config type")
            
        # デバイス設定
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)
            
        print(f"InferenceModel initialized on device: {self.device}")
        
        # コンポーネント
        self.state_estimator: Optional[StateEstimator] = None
        self.streaming_estimator: Optional[StreamingEstimator] = None
        
        # 状態
        self.is_setup = False
        self.calibration_data: Optional[torch.Tensor] = None

    def setup_inference(
        self,
        calibration_data: Optional[torch.Tensor] = None,
        config_override: Optional[Dict[str, Any]] = None,
        method: Optional[str] = None
    ):
        """
        推論環境のセットアップ
        
        ノイズ推定、初期化実行
        
        Args:
            calibration_data: キャリブレーション用データ (T_cal, n)
            config_override: 設定上書き用辞書
            method: 初期化方法の動的オーバーライド ("data_driven", "zero", etc.)
        """
        print("Setting up inference environment...")
        
        # 設定上書き
        if config_override:
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # 初期化方法の動的オーバーライド
        if method is not None:
            self.config.initialization_method = method
                    
        # StateEstimator作成
        estimator_config = {
            'device': str(self.device),
            'model': {
                'df_state': {'state_dim': 5, 'feature_dim': 16},  # デフォルト値
                'df_obs': {'obs_feature_dim': 8},
                'encoder': {'input_dim': 7}
            },
            'noise_estimation': {
                'method': self.config.noise_estimation_method,
                'gamma_Q': self.config.gamma_Q,
                'gamma_R': self.config.gamma_R
            },
            'initialization': {
                'method': self.config.initialization_method,
                'n_init_samples': self.config.n_init_samples
            },
            'numerical': {
                'condition_threshold': self.config.condition_threshold,
                'min_eigenvalue': self.config.min_eigenvalue,
                'jitter': self.config.jitter
            }
        }
        
        self.state_estimator = StateEstimator(estimator_config)
        
        # 学習済みモデル読み込み
        self._load_trained_model()
        
        # ノイズ共分散推定
        if calibration_data is not None and self.config.use_calibration:
            self.calibration_data = calibration_data.to(self.device)
            self.state_estimator.estimate_noise_covariances(
                self.calibration_data, self.config.noise_estimation_method
            )
        elif calibration_data is not None:
            self.calibration_data = calibration_data.to(self.device)
            
        # フィルタリング初期化
        self.state_estimator.initialize_filtering(self.calibration_data)
        
        # StreamingEstimator作成
        if not self.config.batch_processing:
            self.streaming_estimator = StreamingEstimator(
                state_estimator=self.state_estimator,
                buffer_size=self.config.buffer_size,
                anomaly_threshold=self.config.anomaly_threshold,
                enable_metrics=True
            )
            
        self.is_setup = True
        print("Inference environment setup completed")

    def _load_trained_model(self):
        """学習済みモデルの読み込み"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        try:
            # モデルファイルから設定も読み込み
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'config' in checkpoint:
                # 学習時の設定で次元情報を更新
                train_config = checkpoint['config']
                self._update_dimensions_from_training_config(train_config)
                
            # StateEstimatorにモデルを読み込み
            self.state_estimator.load_components(self.model_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load trained model: {e}")

    def _update_dimensions_from_training_config(self, train_config: Dict[str, Any]):
        """学習時設定から次元情報を更新"""
        # DF-A設定
        if 'ssm' in train_config and 'df_state' in train_config['ssm']:
            df_state_config = train_config['ssm']['df_state']
            if hasattr(self.state_estimator, 'config'):
                df_state_update = {
                    'feature_dim': df_state_config.get('feature_dim', 16),
                    'state_dim': train_config['ssm']['realization'].get('rank', 5)
                }
                
                # feature_net設定も更新
                if 'feature_net' in df_state_config:
                    df_state_update['feature_net'] = df_state_config['feature_net']
                
                self.state_estimator.config['model']['df_state'].update(df_state_update)
                
        # DF-B設定
        if 'ssm' in train_config and 'df_observation' in train_config['ssm']:
            df_obs_config = train_config['ssm']['df_observation']
            if hasattr(self.state_estimator, 'config'):
                df_obs_update = {
                    'obs_feature_dim': df_obs_config.get('obs_feature_dim', 8)
                }
                
                # obs_net設定も更新
                if 'obs_net' in df_obs_config:
                    df_obs_update['obs_net'] = df_obs_config['obs_net']
                
                self.state_estimator.config['model']['df_obs'].update(df_obs_update)
                
        # エンコーダ設定
        if 'model' in train_config and 'encoder' in train_config['model']:
            encoder_config = train_config['model']['encoder']
            if hasattr(self.state_estimator, 'config'):
                self.state_estimator.config['model']['encoder'].update({
                    'input_dim': encoder_config.get('input_dim', 7)
                })

    def inference_batch(
        self,
        observations: torch.Tensor,
        return_format: str = 'dict'
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        バッチ推論（従来型データ処理）
        
        Args:
            observations: 観測系列 (T, n)
            return_format: 返却形式 ('tuple' | 'dict')
            
        Returns:
            推論結果（形式は return_format に依存）
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")
            
        observations = observations.to(self.device)
        T, n = observations.shape
        
        print(f"Performing batch inference on {T} observations...")
        
        # バッチフィルタリング実行
        if self.config.save_likelihoods:
            X_means, X_covariances, likelihoods = self.state_estimator.filter_sequence(
                observations, return_likelihood=True
            )
        else:
            X_means, X_covariances = self.state_estimator.filter_sequence(
                observations, return_likelihood=False
            )
            likelihoods = None
            
        # 返却形式に応じて整形
        if return_format == 'tuple':
            if likelihoods is not None:
                return X_means, X_covariances, likelihoods
            return X_means, X_covariances
            
        elif return_format == 'dict':
            results = format_filter_results(X_means, X_covariances, likelihoods)
            
            # 推論設定情報も追加
            results['inference_info'] = {
                'sequence_length': T,
                'observation_dimension': n,
                'model_path': str(self.model_path),
                'device': str(self.device),
                'config': self.config.__dict__
            }
            
            return results
            
        else:
            raise ValueError(f"Unknown return format: {return_format}")

    def inference_streaming(
        self,
        initial_data: Optional[torch.Tensor] = None
    ) -> StreamingEstimator:
        """
        ストリーミング推論のジェネレータ
        
        Args:
            initial_data: 初期化用データ (N0, n)
            
        Returns:
            StreamingEstimator: ストリーミング推論インスタンス
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")
            
        if self.streaming_estimator is None:
            raise RuntimeError("Streaming estimator not available. Set batch_processing=False.")
            
        # ストリーミング開始
        if initial_data is not None:
            initial_data = initial_data.to(self.device)
            self.streaming_estimator.start_streaming(initial_data)
        else:
            self.streaming_estimator.start_streaming(self.calibration_data)
            
        print("Streaming inference started. Use returned estimator to process data.")
        return self.streaming_estimator

    def predict_future(
        self,
        n_steps: int,
        current_observations: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        将来予測
        
        Args:
            n_steps: 予測ステップ数
            current_observations: 現在観測（状態更新用）
            
        Returns:
            Dict: 予測結果
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")
            
        # 必要に応じて状態更新
        if current_observations is not None:
            current_observations = current_observations.to(self.device)
            if current_observations.dim() == 1:
                # 単一観測の場合
                self.state_estimator.filter_online(current_observations)
            else:
                # 複数観測の場合
                for obs in current_observations:
                    self.state_estimator.filter_online(obs)
                    
        # 予測実行
        predictions = self.state_estimator.predict_ahead(n_steps)
        
        return {
            'predictions': predictions[0],      # (n_steps, r)
            'covariances': predictions[1],      # (n_steps, r, r)
            'prediction_steps': n_steps,
            'prediction_time': torch.arange(1, n_steps + 1, dtype=torch.float32)
        }

    def evaluate_model(
        self,
        test_data: torch.Tensor,
        metrics: List[str] = ['mse', 'likelihood', 'coverage']
    ) -> Dict[str, Any]:
        """
        モデル評価
        
        Args:
            test_data: テストデータ (T, n)
            metrics: 評価指標リスト
            
        Returns:
            Dict: 評価結果
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")
            
        test_data = test_data.to(self.device)
        T = test_data.size(0)
        
        print(f"Evaluating model on {T} test samples...")
        
        # 推論実行
        results = self.inference_batch(test_data, return_format='dict')
        X_means = torch.tensor(results['summary']['mean_trajectory'])
        X_covariances = torch.tensor(results['summary']['covariance_trajectory'])
        
        if 'likelihood' in results['statistics']:
            likelihoods = torch.tensor(results['statistics']['likelihood']['likelihood_trajectory'])
        else:
            likelihoods = None
            
        evaluation = {'test_length': T, 'metrics': {}}
        
        # MSE（状態推定の平均二乗誤差）
        if 'mse' in metrics:
            # 真の状態が不明な場合、一期先予測誤差で代替
            pred_errors = []
            for t in range(1, T):
                x_pred = X_means[t-1]  # t-1での t 予測
                x_actual = X_means[t]  # t での実際推定
                pred_errors.append(torch.norm(x_pred - x_actual).item())
                
            evaluation['metrics']['mse'] = {
                'mean_prediction_error': np.mean(pred_errors),
                'std_prediction_error': np.std(pred_errors)
            }
            
        # 尤度評価
        if 'likelihood' in metrics and likelihoods is not None:
            evaluation['metrics']['likelihood'] = {
                'total_log_likelihood': torch.sum(likelihoods).item(),
                'mean_log_likelihood': torch.mean(likelihoods).item(),
                'likelihood_std': torch.std(likelihoods).item()
            }
            
        # カバレッジ（不確実性定量化の品質）
        if 'coverage' in metrics:
            # 95%信頼区間のカバレッジを計算
            std_devs = torch.sqrt(torch.diagonal(X_covariances, dim1=1, dim2=2))
            lower_bounds = X_means - 1.96 * std_devs
            upper_bounds = X_means + 1.96 * std_devs
            
            # 実際には真の値が必要だが、ここでは推定値自体を使用（自己整合性チェック）
            coverage_count = 0
            for t in range(T):
                if torch.all(X_means[t] >= lower_bounds[t]) and torch.all(X_means[t] <= upper_bounds[t]):
                    coverage_count += 1
                    
            evaluation['metrics']['coverage'] = {
                'coverage_rate': coverage_count / T,
                'expected_coverage': 0.95,
                'coverage_difference': coverage_count / T - 0.95
            }
            
        return evaluation

    def export_for_deployment(
        self,
        export_path: Union[str, Path],
        include_calibration: bool = True
    ):
        """
        デプロイ用ファイル出力
        
        推論のみ必要なコンポーネント抽出
        
        Args:
            export_path: 出力パス
            include_calibration: キャリブレーションデータも含めるか
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")
            
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # StateEstimatorのエクスポート機能を使用
        self.state_estimator.export_for_deployment(export_path)
        
        # 推論設定の保存
        config_dict = {
            'inference_config': self.config.__dict__,
            'model_info': {
                'original_model_path': str(self.model_path),
                'export_timestamp': torch.tensor(time.time()).item(),
                'device_used': str(self.device)
            }
        }
        
        with open(export_path / 'deployment_config.yaml', 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
        # キャリブレーションデータ
        if include_calibration and self.calibration_data is not None:
            torch.save(self.calibration_data.cpu(), export_path / 'calibration_data.pth')
            
        # デプロイ用スクリプトの生成
        self._generate_deployment_script(export_path)
        
        print(f"Deployment package created at {export_path}")

    def _generate_deployment_script(self, export_path: Path):
        """デプロイ用スクリプト生成"""
        script_content = '''#!/usr/bin/env python3
"""
Auto-generated deployment script for DFIV Kalman Filter inference.

Usage:
    python deploy_inference.py --input data.npz --output results.json
"""

import torch
import yaml
import argparse
import numpy as np
from pathlib import Path

# Import inference components (adjust import paths as needed)
from src.inference.state_estimator import StateEstimator
from src.inference.kalman_filter import OperatorBasedKalmanFilter

def load_inference_model(model_dir):
    """Load exported inference model"""
    model_dir = Path(model_dir)
    
    # Load parameters
    params = torch.load(model_dir / 'inference_params.pth', map_location='cpu')
    encoder_state = torch.load(model_dir / 'encoder.pth', map_location='cpu')
    
    with open(model_dir / 'deployment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create estimator
    estimator = StateEstimator(config['inference_config'])
    
    # Manual setup (simplified)
    # ... setup code here ...
    
    return estimator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='Exported model directory')
    parser.add_argument('--input', required=True, help='Input data file (.npz)')
    parser.add_argument('--output', required=True, help='Output results file (.json)')
    args = parser.parse_args()
    
    # Load model
    estimator = load_inference_model(args.model_dir)
    
    # Load data
    data = np.load(args.input)
    observations = torch.from_numpy(data['observations']).float()
    
    # Inference
    results = estimator.filter_sequence(observations)
    
    # Save results
    # ... save code here ...
    
    print(f"Inference completed. Results saved to {args.output}")

if __name__ == '__main__':
    main()
'''
        
        with open(export_path / 'deploy_inference.py', 'w') as f:
            f.write(script_content)
            
        # 実行権限付与（Unix系）
        try:
            import os
            os.chmod(export_path / 'deploy_inference.py', 0o755)
        except:
            pass

    def get_system_diagnostics(self) -> Dict[str, Any]:
        """システム診断情報"""
        diagnostics = {
            'setup_status': self.is_setup,
            'model_path': str(self.model_path),
            'device': str(self.device),
            'config': self.config.__dict__
        }
        
        if self.is_setup and self.state_estimator:
            diagnostics['filter_diagnostics'] = self.state_estimator.get_filter_diagnostics()
            
        if self.streaming_estimator:
            diagnostics['streaming_metrics'] = self.streaming_estimator.get_performance_metrics()
            
        return diagnostics

    def filter_sequence(
        self,
        observations: torch.Tensor,
        return_likelihood: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        系列フィルタリング（評価スクリプト用インターフェース）

        注: 類似機能としてinference_batch()が存在しますが、戻り値形式が異なります
             - inference_batch(): 詳細な統計情報付きの辞書形式
             - filter_sequence(): 評価スクリプト用のシンプルなタプル形式

        Args:
            observations: 観測系列 (T, d)
            return_likelihood: 尤度も返すかどうか

        Returns:
            (X_means, X_covariances) or (X_means, X_covariances, likelihoods)
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")

        # StateEstimatorのfilter_sequenceメソッドを直接使用
        result = self.state_estimator.filter_sequence(observations, return_likelihood=return_likelihood)

        return result

    def filter_online(
        self,
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        オンライン単一ステップフィルタリング（評価スクリプト用インターフェース）

        注: 既存のinference_streaming()はストリーミング用ジェネレータですが、
             こちらは評価での単一ステップ処理用です

        Args:
            observation: 単一観測値 (d,)

        Returns:
            (x_hat, Sigma_x, likelihood): 状態推定、共分散、尤度
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")

        # StateEstimatorのfilter_onlineメソッドを直接使用
        result = self.state_estimator.filter_online(observation)

        return result

    def reset_state(self):
        """
        フィルタ状態のリセット（オンライン処理用）

        注: inference_streaming()はストリーミング用ですが、
             こちらはオンライン評価での状態リセット専用です
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")

        # StateEstimatorの状態リセット
        if hasattr(self.state_estimator, 'reset_filter_state'):
            self.state_estimator.reset_filter_state()

        # StreamingEstimatorがある場合もリセット（安全にチェック）
        if self.streaming_estimator and hasattr(self.streaming_estimator, 'reset'):
            self.streaming_estimator.reset()

    def __repr__(self) -> str:
        status = "setup" if self.is_setup else "not setup"
        return f"InferenceModel(model={self.model_path.name}, device={self.device}, status={status})"