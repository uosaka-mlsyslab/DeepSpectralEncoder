"""
評価指標計算モジュール

DFIV Kalman Filterの状態推定性能を包括的に評価するための指標を提供。
- 推定精度（MSE, MAE, RMSE）
- 不確実性定量化品質（カバレッジ率、区間幅）
- 予測性能（対数尤度、キャリブレーション）
- 計算効率（時間、メモリ）
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from sklearn.metrics import r2_score
import warnings


class StateEstimationMetrics:
    """状態推定性能評価クラス"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        
    def compute_all_metrics(
        self,
        X_estimated: torch.Tensor,
        X_true: Optional[torch.Tensor] = None,
        X_covariances: Optional[torch.Tensor] = None,
        observations: Optional[torch.Tensor] = None,
        likelihoods: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> Dict[str, Union[float, Dict]]:
        """
        包括的評価指標の計算
        
        Args:
            X_estimated: 推定状態 (T, r)
            X_true: 真の状態 (T, r) [optional]
            X_covariances: 状態共分散 (T, r, r) [optional]
            observations: 観測データ (T, n) [optional]
            likelihoods: 観測尤度 (T,) [optional]
            verbose: ターミナル出力するかどうか
            
        Returns:
            Dict: 全評価指標
        """
        metrics = {}
        
        # 基本統計
        metrics['basic_stats'] = self._compute_basic_stats(X_estimated)
        
        # 精度評価（真値がある場合）
        if X_true is not None:
            metrics['accuracy'] = self._compute_accuracy_metrics(X_estimated, X_true)
            
        # 不確実性評価（共分散がある場合）
        if X_covariances is not None:
            metrics['uncertainty'] = self._compute_uncertainty_metrics(
                X_estimated, X_covariances, X_true
            )
            
        # 尤度評価
        if likelihoods is not None:
            metrics['likelihood'] = self._compute_likelihood_metrics(likelihoods)
            
        # 予測性能（観測データがある場合）
        if observations is not None:
            metrics['prediction'] = self._compute_prediction_metrics(
                X_estimated, observations
            )
            
        # ターミナル出力
        if verbose:
            self._print_metrics_summary(metrics)
            
        return metrics
    
    def _compute_basic_stats(self, X_estimated: torch.Tensor) -> Dict[str, float]:
        """基本統計情報"""
        with torch.no_grad():
            return {
                'sequence_length': X_estimated.size(0),
                'state_dimension': X_estimated.size(1),
                'mean_state_norm': torch.norm(X_estimated, dim=1).mean().item(),
                'std_state_norm': torch.norm(X_estimated, dim=1).std().item(),
                'max_state_value': X_estimated.max().item(),
                'min_state_value': X_estimated.min().item()
            }
    
    def _compute_accuracy_metrics(
        self, 
        X_estimated: torch.Tensor, 
        X_true: torch.Tensor
    ) -> Dict[str, float]:
        """推定精度指標"""
        with torch.no_grad():
            # エラー計算
            errors = X_estimated - X_true
            squared_errors = errors ** 2
            abs_errors = torch.abs(errors)
            
            # 次元ごとの指標
            mse_per_dim = squared_errors.mean(dim=0)
            mae_per_dim = abs_errors.mean(dim=0)
            
            return {
                'mse': squared_errors.mean().item(),
                'mae': abs_errors.mean().item(),
                'rmse': torch.sqrt(squared_errors.mean()).item(),
                'mse_per_dimension': mse_per_dim.tolist(),
                'mae_per_dimension': mae_per_dim.tolist(),
                'relative_error': (torch.norm(errors, dim=1) / torch.norm(X_true, dim=1)).mean().item(),
                'correlation': self._compute_correlation(X_estimated, X_true).item()
            }
    
    def _compute_uncertainty_metrics(
        self,
        X_estimated: torch.Tensor,
        X_covariances: torch.Tensor,
        X_true: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[float, List]]:
        """不確実性定量化品質"""
        with torch.no_grad():
            # 標準偏差抽出
            std_devs = torch.sqrt(torch.diagonal(X_covariances, dim1=1, dim2=2))
            
            metrics = {
                'mean_uncertainty': std_devs.mean().item(),
                'std_uncertainty': std_devs.std().item(),
                'uncertainty_per_dimension': std_devs.mean(dim=0).tolist(),
                'determinant_mean': torch.det(X_covariances).mean().item(),
                'trace_mean': torch.trace(X_covariances.view(-1, X_covariances.size(-1), X_covariances.size(-1))).mean().item()
            }
            
            # カバレッジ率計算（真値がある場合）
            if X_true is not None:
                coverage_results = self._compute_coverage_rates(
                    X_estimated, std_devs, X_true
                )
                metrics.update(coverage_results)
                
            return metrics
    
    def _compute_coverage_rates(
        self,
        X_estimated: torch.Tensor,
        std_devs: torch.Tensor,
        X_true: torch.Tensor,
        confidence_levels: List[float] = [0.68, 0.95, 0.99]
    ) -> Dict[str, float]:
        """信頼区間カバレッジ率"""
        coverage_results = {}
        
        for conf_level in confidence_levels:
            # Z値計算
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            
            # 信頼区間
            lower = X_estimated - z_score * std_devs
            upper = X_estimated + z_score * std_devs
            
            # カバレッジ判定
            covered = (X_true >= lower) & (X_true <= upper)
            coverage_rate = covered.all(dim=1).float().mean().item()
            
            coverage_results[f'coverage_{int(conf_level*100)}'] = coverage_rate
            coverage_results[f'coverage_error_{int(conf_level*100)}'] = abs(coverage_rate - conf_level)
            
        return coverage_results
    
    def _compute_likelihood_metrics(self, likelihoods: torch.Tensor) -> Dict[str, float]:
        """尤度関連指標"""
        with torch.no_grad():
            return {
                'total_log_likelihood': likelihoods.sum().item(),
                'mean_log_likelihood': likelihoods.mean().item(),
                'std_log_likelihood': likelihoods.std().item(),
                'perplexity': torch.exp(-likelihoods.mean()).item(),
                'likelihood_trend': self._compute_likelihood_trend(likelihoods)
            }
    
    def _compute_prediction_metrics(
        self, 
        X_estimated: torch.Tensor, 
        observations: torch.Tensor
    ) -> Dict[str, float]:
        """予測性能指標"""
        # 一期先予測誤差（簡略版）
        if X_estimated.size(0) > 1:
            pred_errors = []
            for t in range(1, X_estimated.size(0)):
                # 簡単な線形予測
                if t >= 2:
                    predicted = X_estimated[t-1] + (X_estimated[t-1] - X_estimated[t-2])
                else:
                    predicted = X_estimated[t-1]
                actual = X_estimated[t]
                pred_errors.append(torch.norm(actual - predicted).item())
                
            return {
                'one_step_prediction_error': np.mean(pred_errors),
                'prediction_error_std': np.std(pred_errors),
                'prediction_stability': 1.0 / (1.0 + np.std(pred_errors))
            }
        else:
            return {'prediction_error': 0.0}
    
    def _compute_correlation(self, X_estimated: torch.Tensor, X_true: torch.Tensor) -> torch.Tensor:
        """状態推定の相関係数"""
        # 各次元での相関計算
        correlations = []
        for dim in range(X_estimated.size(1)):
            est_dim = X_estimated[:, dim]
            true_dim = X_true[:, dim]
            
            # 標準化
            est_norm = (est_dim - est_dim.mean()) / est_dim.std()
            true_norm = (true_dim - true_dim.mean()) / true_dim.std()
            
            # 相関計算
            corr = (est_norm * true_norm).mean()
            correlations.append(corr)
            
        return torch.stack(correlations).mean()
    
    def _compute_likelihood_trend(self, likelihoods: torch.Tensor) -> float:
        """尤度の傾向分析"""
        if len(likelihoods) < 3:
            return 0.0
            
        # 線形回帰による傾向
        x = torch.arange(len(likelihoods), dtype=torch.float32)
        y = likelihoods
        
        # 傾きを計算
        x_mean = x.mean()
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        
        return slope.item()
    
    def _print_metrics_summary(self, metrics: Dict) -> None:
        """メトリクス結果のターミナル出力"""
        print("\n" + "="*50)
        print("フィルタリング性能評価結果")
        print("="*50)
        
        # 基本統計
        if 'basic_stats' in metrics:
            stats = metrics['basic_stats']
            print(f"\n📈 基本統計:")
            print(f"  系列長: {stats['sequence_length']}")
            print(f"  状態次元: {stats['state_dimension']}")
            print(f"  平均状態ノルム: {stats['mean_state_norm']:.4f}")
            print(f"  状態ノルム標準偏差: {stats['std_state_norm']:.4f}")
        
        # 精度指標
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            print(f"\n推定精度:")
            print(f"  MSE: {acc['mse']:.6f}")
            print(f"  MAE: {acc['mae']:.6f}")
            print(f"  RMSE: {acc['rmse']:.6f}")
            print(f"  相関係数: {acc['correlation']:.4f}")
            print(f"  相対誤差: {acc['relative_error']:.4f}")
        
        # 不確実性指標
        if 'uncertainty' in metrics:
            unc = metrics['uncertainty']
            print(f"\n🎲 不確実性定量化:")
            print(f"  平均不確実性: {unc['mean_uncertainty']:.6f}")
            
            # カバレッジ率
            for key, value in unc.items():
                if key.startswith('coverage_') and not key.endswith('_error'):
                    conf_level = key.split('_')[1]
                    error_key = f'coverage_error_{conf_level}'
                    error = unc.get(error_key, 0.0)
                    print(f"  {conf_level}%信頼区間カバレッジ: {value:.4f} (誤差: {error:.4f})")
        
        # 尤度指標
        if 'likelihood' in metrics:
            like = metrics['likelihood']
            print(f"\n📈 尤度評価:")
            print(f"  総対数尤度: {like['total_log_likelihood']:.2f}")
            print(f"  平均対数尤度: {like['mean_log_likelihood']:.4f}")
            print(f"  パープレキシティ: {like['perplexity']:.4f}")
            
        print("\n" + "="*50)


class ComputationalMetrics:
    """計算効率評価クラス"""
    
    def __init__(self):
        self.timing_results = {}
        self.memory_results = {}
        
    def measure_inference_time(
        self, 
        inference_func, 
        *args, 
        n_trials: int = 5,
        warmup: int = 2
    ) -> Dict[str, float]:
        """推論時間測定"""
        times = []
        
        # ウォームアップ
        for _ in range(warmup):
            _ = inference_func(*args)
            
        # 測定
        for trial in range(n_trials):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            result = inference_func(*args)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'trials': n_trials
        }
    
    def measure_memory_usage(self, function_to_measure, *args) -> Dict[str, float]:
        """メモリ使用量測定"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            initial_memory = torch.cuda.memory_allocated()
            
            result = function_to_measure(*args)
            
            peak_memory = torch.cuda.max_memory_allocated()
            final_memory = torch.cuda.memory_allocated()
            
            return {
                'initial_memory_mb': initial_memory / (1024**2),
                'peak_memory_mb': peak_memory / (1024**2),
                'final_memory_mb': final_memory / (1024**2),
                'memory_increase_mb': (final_memory - initial_memory) / (1024**2),
                'peak_increase_mb': (peak_memory - initial_memory) / (1024**2)
            }
        else:
            return {'message': 'CUDA not available, memory measurement skipped'}


class CalibrationMetrics:
    """キャリブレーション評価クラス"""
    
    @staticmethod
    def compute_calibration_error(
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        true_values: torch.Tensor,
        n_bins: int = 10
    ) -> float:
        """キャリブレーション誤差の計算"""
        # 不確実性を確率に変換（正規分布仮定）
        probabilities = torch.sigmoid(uncertainties)
        
        # ビンごとの分析
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        calibration_error = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # ビン内のサンプル
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
            if in_bin.sum() == 0:
                continue
                
            # 期待信頼度と実際の精度
            expected_confidence = probabilities[in_bin].mean()
            actual_accuracy = ((predictions[in_bin] - true_values[in_bin]).abs() < uncertainties[in_bin]).float().mean()
            
            # ビンの重み
            bin_weight = in_bin.sum().float() / len(probabilities)
            
            # 誤差累積
            calibration_error += bin_weight * abs(expected_confidence - actual_accuracy)
            
        return calibration_error.item()


def create_metrics_evaluator(device: str = 'cpu') -> StateEstimationMetrics:
    """メトリクス評価器の作成"""
    return StateEstimationMetrics(device=device)


def print_comparison_summary(
    method1_metrics: Dict,
    method2_metrics: Dict,
    method1_name: str = "Method 1",
    method2_name: str = "Method 2"
) -> None:
    """2手法の比較結果を出力"""
    print(f"\n🔍 手法比較: {method1_name} vs {method2_name}")
    print("="*60)

    # 精度比較
    if 'accuracy' in method1_metrics and 'accuracy' in method2_metrics:
        acc1 = method1_metrics['accuracy']
        acc2 = method2_metrics['accuracy']

        print(f"\n精度比較:")
        print(f"  MSE:  {method1_name}: {acc1['mse']:.6f}  |  {method2_name}: {acc2['mse']:.6f}")
        print(f"  MAE:  {method1_name}: {acc1['mae']:.6f}  |  {method2_name}: {acc2['mae']:.6f}")
        print(f"  RMSE: {method1_name}: {acc1['rmse']:.6f}  |  {method2_name}: {acc2['rmse']:.6f}")

        # 改善率計算
        mse_improvement = (acc1['mse'] - acc2['mse']) / acc1['mse'] * 100
        mae_improvement = (acc1['mae'] - acc2['mae']) / acc1['mae'] * 100

        print(f"\n改善率 ({method2_name} vs {method1_name}):")
        print(f"  MSE改善: {mse_improvement:+.2f}%")
        print(f"  MAE改善: {mae_improvement:+.2f}%")

    print("="*60)


# =======================================
# ターゲット予測評価機能（Step 6追加）
# =======================================

class TargetPredictionMetrics:
    """
    ターゲット予測評価クラス（既存StateEstimationMetricsと統一設計）

    RKN ターゲット予測実験用の評価指標計算・可視化を提供。
    既存の StateEstimationMetrics クラスと統一したインターフェースを採用。

    機能:
    - 選択可能な評価指標計算 (RMSE, MAE, R², 次元別R²)
    - 指標別独立可視化 (英語ラベル)
    - 研究発表用高品質プロット生成
    - 既存メトリクスと統一されたターミナル出力
    """

    def __init__(self, device: str = 'cpu'):
        """
        初期化

        Args:
            device: 計算デバイス ('cpu' or 'cuda')
        """
        self.device = torch.device(device)

    def compute_target_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        metrics: List[str] = ['rmse'],
        verbose: bool = True
    ) -> Dict[str, Union[float, List[float]]]:
        """
        ターゲット予測評価指標計算（既存メトリクスと統一インターフェース）

        Args:
            y_true: 真値テンソル (T, d)
            y_pred: 予測値テンソル (T, d)
            metrics: 計算する指標のリスト ['rmse', 'mae', 'r2', 'r2_per_dim']
            verbose: ターミナル出力するかどうか

        Returns:
            Dict: 評価指標結果

        例:
            target_evaluator = TargetPredictionMetrics()
            metrics = target_evaluator.compute_target_metrics(
                y_true, y_pred, metrics=['rmse', 'mae'], verbose=True
            )
            # {'rmse': 0.1234, 'mae': 0.0987}
        """
        # 入力検証
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        # デバイス統一
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)

        # 利用可能な評価指標定義
        available_metrics = {
            'rmse': lambda: torch.sqrt(F.mse_loss(y_pred, y_true)).item(),
            'mae': lambda: F.l1_loss(y_pred, y_true).item(),
            'r2': lambda: self._compute_r2_score(y_true, y_pred),
            'r2_per_dim': lambda: self._compute_r2_per_dimension(y_true, y_pred)
        }

        # 指定された指標のみ計算
        results = {}
        for metric in metrics:
            if metric in available_metrics:
                try:
                    results[metric] = available_metrics[metric]()
                except Exception as e:
                    print(f"指標 '{metric}' の計算でエラー: {e}")
                    results[metric] = None
            else:
                print(f"未知の指標: '{metric}'. 利用可能: {list(available_metrics.keys())}")

        # ターミナル出力（既存パターンと統一）
        if verbose:
            self._print_target_metrics_summary(results)

        return results

    def save_target_metrics_results(
        self,
        results: Dict[str, Union[float, List[float]]],
        output_dir: str,
        experiment_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        ターゲット予測評価結果をJSONファイルに保存

        Args:
            results: compute_target_metrics()の結果
            output_dir: 出力ディレクトリ
            experiment_info: 追加の実験情報（オプション）

        Returns:
            保存されたファイルパス
        """
        import json
        from datetime import datetime
        from pathlib import Path

        # 出力ディレクトリ作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存データ構造
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_type': 'target_prediction',
            'metrics': results,
            'experiment_info': experiment_info or {}
        }

        # JSON保存（既存パターンと統一）
        save_file = output_path / 'target_prediction_metrics.json'
        with open(save_file, 'w') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"ターゲット予測評価結果保存: {save_file}")
        return str(save_file)

    def create_target_visualizations(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        metrics: List[str] = ['rmse'],
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        ターゲット予測可視化（既存パターンと統一）

        Args:
            y_true: 真値テンソル
            y_pred: 予測値テンソル
            metrics: 可視化する指標リスト
            output_dir: 出力ディレクトリ（Noneの場合は保存しない）

        Returns:
            List[str]: 生成されたファイルパスのリスト

        生成ファイル例:
            - target_prediction_rmse.png
            - target_prediction_mae.png
            - target_prediction_r2.png
            - target_prediction_r2_per_dim.png
        """
        # TODO: 実装検討中
        # 可視化機能は一旦無効化し、数値出力・保存のみに変更
        print(f"可視化処理はスキップ（数値出力・保存機能で代替）")

        generated_files = []
        # 可視化コードはコメントアウト
        # for metric in metrics:
        #     if output_dir:
        #         save_path = Path(output_dir) / f"target_prediction_{metric}.png"
        #     else:
        #         save_path = None
        #
        #     try:
        #         self._plot_individual_metric(y_true, y_pred, metric, save_path)
        #         if save_path:
        #             generated_files.append(str(save_path))
        #     except Exception as e:
        #         print(f"指標 '{metric}' の可視化でエラー: {e}")
        #
        # if output_dir and generated_files:
        #     print(f"ターゲット予測可視化生成完了: {len(generated_files)}個のファイル")
        #     print(f"📁 保存先: {output_dir}")

        return generated_files

    def _compute_r2_score(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """R²スコア計算（全体）"""
        try:
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            return float(r2_score(y_true_np, y_pred_np, multioutput='uniform_average'))
        except Exception as e:
            print(f"R²計算エラー: {e}")
            return 0.0

    def _compute_r2_per_dimension(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> List[float]:
        """次元別R²スコア計算"""
        try:
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            r2_values = r2_score(y_true_np, y_pred_np, multioutput='raw_values')
            return r2_values.tolist()
        except Exception as e:
            print(f"次元別R²計算エラー: {e}")
            return [0.0]

    def _plot_individual_metric(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        metric: str,
        save_path: Optional[Path] = None,
        max_samples: int = 200
    ) -> None:
        """
        個別指標用可視化（英語ラベル）

        各指標に最適化された可視化:
        - rmse: 時系列比較プロット
        - mae: 絶対誤差分布ヒストグラム
        - r2: 散布図（真値 vs 予測値）
        - r2_per_dim: 次元別R²棒グラフ
        """
        # TODO: 実装検討中
        # 可視化機能は一旦無効化、数値出力・保存機能で代替
        print(f"可視化機能（{metric}）はスキップされました")

        # 全ての可視化コードをコメントアウト
        # plt.style.use('default')  # スタイル初期化

        # if metric == 'rmse':
        #     # RMSE用時系列比較プロット
        #     plt.figure(figsize=(12, 6))
        #     n_samples = min(len(y_true), max_samples)
        #
        #     # 最初の次元のみプロット（複数次元の場合）
        #     if y_true.dim() > 1 and y_true.shape[1] > 1:
        #         true_values = y_true[:n_samples, 0].cpu().numpy()
        #         pred_values = y_pred[:n_samples, 0].cpu().numpy()
        #         plt.title(f'Time Series Comparison (RMSE Analysis) - Dimension 0')
        #     else:
        #         true_values = y_true[:n_samples].cpu().numpy()
        #         pred_values = y_pred[:n_samples].cpu().numpy()
        #         plt.title('Time Series Comparison (RMSE Analysis)')
        #
        #     plt.plot(true_values, label='True Values', alpha=0.8, linewidth=1.5)
        #     plt.plot(pred_values, label='Predicted Values', alpha=0.8, linewidth=1.5)
        #     plt.xlabel('Time Step')
        #     plt.ylabel('Value')
        #     plt.legend()
        #     plt.grid(True, alpha=0.3)
        #
        # elif metric == 'mae':
        #     # MAE用絶対誤差分布プロット
        #     plt.figure(figsize=(10, 6))
        #     absolute_errors = torch.abs(y_pred - y_true).cpu().numpy()
        #
        #     plt.hist(absolute_errors.flatten(), bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
        #     plt.title('Absolute Error Distribution (MAE Analysis)')
        #     plt.xlabel('Absolute Error')
        #     plt.ylabel('Frequency')
        #     plt.grid(True, alpha=0.3)
        #
        #     # 統計情報を追加
        #     mean_error = np.mean(absolute_errors)
        #     plt.axvline(mean_error, color='red', linestyle='--',
        #                label=f'Mean Absolute Error: {mean_error:.4f}')
        #     plt.legend()
        #
        # elif metric == 'r2':
        #     # R²用散布図
        #     plt.figure(figsize=(8, 8))
        #
        #     y_true_flat = y_true.cpu().numpy().flatten()
        #     y_pred_flat = y_pred.cpu().numpy().flatten()
        #
        #     plt.scatter(y_true_flat, y_pred_flat, alpha=0.5, s=10)
        #
        #     # 完全予測線
        #     min_val = min(y_true_flat.min(), y_pred_flat.min())
        #     max_val = max(y_true_flat.max(), y_pred_flat.max())
        #     plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        #
        #     plt.title('Prediction Accuracy Scatter (R² Analysis)')
        #     plt.xlabel('True Values')
        #     plt.ylabel('Predicted Values')
        #     plt.legend()
        #     plt.grid(True, alpha=0.3)
        #     plt.axis('equal')
        #
        # elif metric == 'r2_per_dim':
        #     # 次元別R²棒グラフ
        #     plt.figure(figsize=(10, 6))
        #
        #     try:
        #         r2_values = self._compute_r2_per_dimension(y_true, y_pred)
        #         dimensions = range(len(r2_values))
        #
        #         bars = plt.bar(dimensions, r2_values, alpha=0.7, edgecolor='black', linewidth=0.5)
        #         plt.title('R² Score per Dimension')
        #         plt.xlabel('Dimension')
        #         plt.ylabel('R² Score')
        #         plt.grid(True, alpha=0.3)
        #
        #         # カラーコーディング（R²値によって色分け）
        #         for bar, r2_val in zip(bars, r2_values):
        #             if r2_val < 0.3:
        #                 bar.set_color('red')
        #             elif r2_val < 0.7:
        #                 bar.set_color('orange')
        #             else:
        #                 bar.set_color('green')
        #
        #         # 平均R²ライン
        #         mean_r2 = np.mean(r2_values)
        #         plt.axhline(mean_r2, color='blue', linestyle='--',
        #                    label=f'Mean R²: {mean_r2:.3f}')
        #         plt.legend()
        #
        #     except Exception as e:
        #         plt.text(0.5, 0.5, f'Error computing R² per dimension: {e}',
        #                 ha='center', va='center', transform=plt.gca().transAxes)
        #         plt.title('R² Score per Dimension (Error)')
        #
        # else:
        #     # 未知の指標
        #     plt.figure(figsize=(8, 6))
        #     plt.text(0.5, 0.5, f'Unknown metric: {metric}\nAvailable: rmse, mae, r2, r2_per_dim',
        #             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        #     plt.title(f'Unsupported Metric: {metric}')
        #
        # # 保存処理
        # if save_path:
        #     save_path.parent.mkdir(parents=True, exist_ok=True)
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #     print(f"可視化保存: {save_path}")
        #
        # plt.close()  # メモリリーク防止

    def _print_target_metrics_summary(self, metrics: Dict[str, Union[float, List[float]]]) -> None:
        """ターゲット予測メトリクス結果のターミナル出力（既存パターンと統一）"""
        print("\n" + "="*50)
        print("ターゲット予測評価結果")
        print("="*50)

        for metric_name, value in metrics.items():
            if value is None:
                print(f"  {metric_name.upper()}: 計算エラー")
            elif isinstance(value, list):
                if metric_name == 'r2_per_dim':
                    print(f"  R² PER DIMENSION:")
                    for i, dim_r2 in enumerate(value):
                        print(f"    Dim {i}: {dim_r2:.4f}")
                    print(f"    Average: {np.mean(value):.4f}")
                else:
                    print(f"  {metric_name.upper()}: {value}")
            else:
                print(f"  {metric_name.upper()}: {value:.4f}")

        print("="*50)


# =======================================
# 再構成評価メトリクス（Step 8実装）
# =======================================

class ReconstructionMetrics:
    """
    データ再構成評価クラス（TargetPredictionMetricsと統一設計）

    任意のデータ型（画像・時系列・ベクトル等）の再構成実験用評価指標計算・可視化を提供。
    既存の TargetPredictionMetrics クラスと統一したインターフェースを採用。

    機能:
    - 選択可能な評価指標計算 (reconstruction_rmse, psnr, temporal_correlation)
    - 任意のデータ形状対応 (画像: T,H,W,C / 時系列: T,d / その他)
    - 指標別独立可視化 (英語ラベル)
    - 研究発表用高品質プロット生成
    - 既存メトリクスと統一されたターミナル出力
    """

    def __init__(self, device: str = 'cpu'):
        """
        初期化

        Args:
            device: 計算デバイス ('cpu' or 'cuda')
        """
        self.device = torch.device(device)

    def compute_reconstruction_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        metrics: List[str] = ['reconstruction_rmse'],
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        データ再構成評価指標計算（既存メトリクスと統一インターフェース）

        Args:
            y_true: 真値テンソル (任意形状: T,H,W,C / T,d / etc.)
            y_pred: 予測テンソル (任意形状: y_trueと同じ)
            metrics: 計算する指標のリスト ['reconstruction_rmse', 'psnr', 'temporal_correlation']
            verbose: ターミナル出力するかどうか

        Returns:
            Dict: 再構成評価指標結果

        例:
            reconstruction_evaluator = ReconstructionMetrics()
            metrics = reconstruction_evaluator.compute_reconstruction_metrics(
                y_true, y_pred, metrics=['reconstruction_rmse', 'psnr'], verbose=True
            )
            # {'reconstruction_rmse': 0.1234, 'psnr': 25.67}
        """
        # デバイス移行・形状調整
        y_true = y_true.to(self.device).detach()
        y_pred = y_pred.to(self.device).detach()

        # 形状確認・警告
        if y_true.shape != y_pred.shape:
            print(f"形状不一致: y_true{y_true.shape} vs y_pred{y_pred.shape}")
            return {'error': 1.0}

        results = {}

        for metric in metrics:
            try:
                if metric == 'reconstruction_rmse':
                    results[metric] = self._compute_reconstruction_rmse(y_true, y_pred)
                elif metric == 'psnr':
                    results[metric] = self._compute_psnr(y_true, y_pred)
                elif metric == 'temporal_correlation':
                    results[metric] = self._compute_temporal_correlation(y_true, y_pred)
                else:
                    print(f"未知の指標: '{metric}'")
                    results[metric] = 0.0

            except Exception as e:
                print(f"指標 '{metric}' 計算エラー: {e}")
                results[metric] = 0.0

        # ターミナル出力
        if verbose:
            self._print_reconstruction_metrics_summary(results)

        return results

    def create_reconstruction_visualizations(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        metrics: List[str] = ['reconstruction_rmse'],
        output_dir: str = None
    ) -> List[str]:
        """
        再構成可視化生成（既存パターンと統一）

        Args:
            y_true: 真値テンソル
            y_pred: 予測テンソル
            metrics: 可視化する指標のリスト
            output_dir: 出力ディレクトリ

        Returns:
            List[str]: 生成されたファイルパスのリスト
        """
        # TODO: 可視化機能は段階的実装
        # Step 8では評価指標計算に集中し、可視化は後続で実装
        print(f"可視化処理はスキップ（数値出力・保存機能で代替）")

        generated_files = []
        return generated_files

    def _compute_reconstruction_rmse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """再構成RMSE計算（任意データ型対応）"""
        mse = torch.mean((y_true - y_pred) ** 2).item()
        rmse = mse ** 0.5
        return float(rmse)

    def _compute_psnr(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """PSNR計算（Peak Signal-to-Noise Ratio）任意データ型対応"""
        try:
            # MSE計算
            mse = torch.mean((y_true - y_pred) ** 2).item()
            if mse == 0:
                return float('inf')  # 完全一致

            # データ範囲自動推定
            data_range = torch.max(y_true).item() - torch.min(y_true).item()
            if data_range <= 0:
                return float('inf')

            psnr = 20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10(torch.tensor(mse))
            return float(psnr.item())

        except Exception as e:
            print(f"PSNR計算エラー: {e}")
            return 0.0

    def _compute_temporal_correlation(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """時系列復元相関係数計算（任意データ型対応）"""
        try:
            # 時間軸（第0次元）での相関係数を計算
            if len(y_true.shape) < 2:
                # 1次元データの場合、全体での相関
                if torch.std(y_true) > 1e-8 and torch.std(y_pred) > 1e-8:
                    corr = torch.corrcoef(torch.stack([y_true, y_pred]))[0, 1].item()
                    return float(corr) if not torch.isnan(torch.tensor(corr)) else 0.0
                return 0.0

            # 多次元データ: 各時刻での相関係数を計算し、平均を取る
            correlations = []
            for t in range(y_true.shape[0]):  # 時間軸でループ
                true_t = y_true[t].flatten()
                pred_t = y_pred[t].flatten()

                # ピアソン相関係数
                if torch.std(true_t) > 1e-8 and torch.std(pred_t) > 1e-8:
                    corr = torch.corrcoef(torch.stack([true_t, pred_t]))[0, 1].item()
                    if not torch.isnan(torch.tensor(corr)):
                        correlations.append(corr)

            if correlations:
                return float(sum(correlations) / len(correlations))
            else:
                return 0.0

        except Exception as e:
            print(f"時系列相関計算エラー: {e}")
            return 0.0

    def _print_reconstruction_metrics_summary(self, results: Dict[str, float]):
        """再構成評価結果ターミナル出力（統一フォーマット）"""
        print("\n" + "="*50)
        print("データ再構成評価結果 (Reconstruction Metrics)")
        print("="*50)

        for metric, value in results.items():
            if metric == 'reconstruction_rmse':
                print(f"🔸 Reconstruction RMSE: {value:.6f}")
            elif metric == 'psnr':
                if value == float('inf'):
                    print(f"🔸 PSNR: ∞ dB (Perfect Match)")
                else:
                    print(f"🔸 PSNR: {value:.2f} dB")
            elif metric == 'temporal_correlation':
                print(f"🔸 Temporal Correlation: {value:.6f}")
            else:
                print(f"🔸 {metric}: {value:.6f}")

        print("="*50)

    def save_reconstruction_metrics_results(
        self,
        results: Dict[str, float],
        output_dir: str,
        experiment_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        再構成評価結果をJSONファイルに保存（TargetPredictionMetricsパターン継承）

        Args:
            results: compute_reconstruction_metrics()の結果
            output_dir: 出力ディレクトリ
            experiment_info: 追加の実験情報（オプション）

        Returns:
            保存されたファイルパス
        """
        import json
        from datetime import datetime
        from pathlib import Path

        # 出力ディレクトリ作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存データ構造（TargetPredictionパターン継承）
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_type': 'reconstruction',
            'metrics': results,
        }

        # 実験情報追加
        if experiment_info:
            save_data['experiment_info'] = experiment_info

        # ファイル保存
        save_file = output_path / 'reconstruction_metrics.json'
        with open(save_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"📁 再構成評価結果保存: {save_file}")
        return str(save_file)


def create_target_prediction_evaluator(device: str = 'cpu') -> TargetPredictionMetrics:
    """ターゲット予測評価器の作成（既存パターンと統一）"""
    return TargetPredictionMetrics(device=device)


def create_reconstruction_evaluator(device: str = 'cpu') -> ReconstructionMetrics:
    """再構成評価器の作成（統一インターフェース）"""
    return ReconstructionMetrics(device=device)