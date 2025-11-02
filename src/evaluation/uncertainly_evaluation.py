"""
不確実性評価モジュール

DFIV Kalman Filterの不確実性定量化品質を詳細に評価。
- 信頼区間の品質評価
- キャリブレーション分析
- 不確実性の妥当性検証
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import seaborn as sns
from pathlib import Path


class UncertaintyEvaluator:
    """不確実性定量化評価クラス"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_uncertainty_quality(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        true_values: Optional[torch.Tensor] = None,
        save_plots: bool = True,
        verbose: bool = True
    ) -> Dict[str, Union[float, Dict, List]]:
        """
        不確実性品質の包括的評価
        
        Args:
            predictions: 予測値 (T, d)
            uncertainties: 不確実性（標準偏差） (T, d)
            true_values: 真値 (T, d) [optional]
            save_plots: プロット保存するか
            verbose: 詳細出力するか
            
        Returns:
            Dict: 不確実性評価結果
        """
        if verbose:
            print("\n🎲 不確実性定量化評価開始")
            print("="*50)
            
        evaluation_results = {}
        
        # 1. 基本統計
        evaluation_results['basic_stats'] = self._compute_uncertainty_stats(
            predictions, uncertainties, verbose
        )
        
        # 2. 信頼区間評価（真値がある場合）
        if true_values is not None:
            evaluation_results['confidence_intervals'] = self._evaluate_confidence_intervals(
                predictions, uncertainties, true_values, verbose
            )
            
            # 3. キャリブレーション評価
            evaluation_results['calibration'] = self._evaluate_calibration(
                predictions, uncertainties, true_values, verbose
            )
            
            # 4. 不確実性の有用性評価
            evaluation_results['uncertainty_utility'] = self._evaluate_uncertainty_utility(
                predictions, uncertainties, true_values, verbose
            )
        
        # 5. 不確実性の時系列特性
        evaluation_results['temporal_analysis'] = self._analyze_temporal_uncertainty(
            uncertainties, verbose
        )
        
        # 6. 可視化
        if save_plots and self.output_dir:
            self._create_uncertainty_plots(
                predictions, uncertainties, true_values, evaluation_results
            )
        
        if verbose:
            self._print_uncertainty_summary(evaluation_results)
            
        return evaluation_results
    
    def _compute_uncertainty_stats(
        self, 
        predictions: torch.Tensor, 
        uncertainties: torch.Tensor,
        verbose: bool
    ) -> Dict[str, float]:
        """不確実性の基本統計"""
        with torch.no_grad():
            stats = {
                'mean_uncertainty': uncertainties.mean().item(),
                'std_uncertainty': uncertainties.std().item(),
                'min_uncertainty': uncertainties.min().item(),
                'max_uncertainty': uncertainties.max().item(),
                'median_uncertainty': uncertainties.median().item(),
                'uncertainty_range': (uncertainties.max() - uncertainties.min()).item()
            }
            
            # 次元別統計
            if uncertainties.dim() > 1:
                stats['uncertainty_per_dimension'] = uncertainties.mean(dim=0).tolist()
                stats['uncertainty_std_per_dimension'] = uncertainties.std(dim=0).tolist()
            
            if verbose:
                print(f"\n不確実性基本統計:")
                print(f"  平均不確実性: {stats['mean_uncertainty']:.6f}")
                print(f"  不確実性標準偏差: {stats['std_uncertainty']:.6f}")
                print(f"  不確実性範囲: [{stats['min_uncertainty']:.6f}, {stats['max_uncertainty']:.6f}]")
                
        return stats
    
    def _evaluate_confidence_intervals(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        true_values: torch.Tensor,
        verbose: bool
    ) -> Dict[str, Union[float, List]]:
        """信頼区間の評価"""
        confidence_levels = [0.68, 0.90, 0.95, 0.99]
        interval_results = {}
        
        for conf_level in confidence_levels:
            # Z値計算
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            
            # 信頼区間
            lower_bound = predictions - z_score * uncertainties
            upper_bound = predictions + z_score * uncertainties
            
            # カバレッジ判定
            covered = (true_values >= lower_bound) & (true_values <= upper_bound)
            
            if predictions.dim() > 1:
                # 全次元がカバーされているかチェック
                full_coverage = covered.all(dim=1).float()
                partial_coverage = covered.float().mean(dim=1)
            else:
                full_coverage = covered.float()
                partial_coverage = covered.float()
            
            # カバレッジ率
            coverage_rate = full_coverage.mean().item()
            partial_coverage_rate = partial_coverage.mean().item()
            
            # 区間幅
            interval_width = (upper_bound - lower_bound).mean().item()
            
            # カバレッジ誤差
            coverage_error = abs(coverage_rate - conf_level)
            
            interval_results[f'confidence_{int(conf_level*100)}'] = {
                'expected_coverage': conf_level,
                'actual_coverage': coverage_rate,
                'partial_coverage': partial_coverage_rate,
                'coverage_error': coverage_error,
                'mean_interval_width': interval_width,
                'z_score_used': z_score
            }
        
        if verbose:
            print(f"\n信頼区間評価:")
            for level, result in interval_results.items():
                expected = result['expected_coverage']
                actual = result['actual_coverage']
                error = result['coverage_error']
                width = result['mean_interval_width']
                print(f"  {level}: カバレッジ {actual:.3f} (期待: {expected:.3f}, 誤差: {error:.3f}, 幅: {width:.4f})")
        
        return interval_results
    
    def _evaluate_calibration(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        true_values: torch.Tensor,
        verbose: bool,
        n_bins: int = 10
    ) -> Dict[str, Union[float, List]]:
        """キャリブレーション評価"""
        with torch.no_grad():
            # 正規化された誤差
            normalized_errors = (predictions - true_values) / uncertainties
            
            # Expected Calibration Error (ECE)
            ece = self._compute_ece(predictions, uncertainties, true_values, n_bins)
            
            # 正規性検定
            if normalized_errors.numel() > 0:
                normalized_flat = normalized_errors.flatten().cpu().numpy()
                
                # Shapiro-Wilk検定（サンプル数制限）
                if len(normalized_flat) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(normalized_flat[:5000])
                else:
                    # 大きなサンプルの場合はランダムサンプリング
                    sample_indices = np.random.choice(len(normalized_flat), 5000, replace=False)
                    shapiro_stat, shapiro_p = stats.shapiro(normalized_flat[sample_indices])
                    
                # Kolmogorov-Smirnov検定
                ks_stat, ks_p = stats.kstest(normalized_flat, 'norm')
                
                normality_tests = {
                    'shapiro_stat': float(shapiro_stat),
                    'shapiro_p_value': float(shapiro_p),
                    'ks_stat': float(ks_stat),
                    'ks_p_value': float(ks_p),
                    'is_normal_shapiro': shapiro_p > 0.05,
                    'is_normal_ks': ks_p > 0.05
                }
            else:
                normality_tests = {'error': 'No data for normality test'}
            
            # 残差分析
            residual_stats = {
                'mean_normalized_error': normalized_errors.mean().item(),
                'std_normalized_error': normalized_errors.std().item(),
                'skewness': float(stats.skew(normalized_errors.flatten().cpu().numpy())),
                'kurtosis': float(stats.kurtosis(normalized_errors.flatten().cpu().numpy()))
            }
            
            calibration_result = {
                'ece': ece,
                'normality_tests': normality_tests,
                'residual_statistics': residual_stats
            }
            
            if verbose:
                print(f"\nキャリブレーション評価:")
                print(f"  Expected Calibration Error: {ece:.4f}")
                print(f"  正規化残差 - 平均: {residual_stats['mean_normalized_error']:.4f}")
                print(f"  正規化残差 - 標準偏差: {residual_stats['std_normalized_error']:.4f}")
                print(f"  正規性検定 (Shapiro): p={normality_tests.get('shapiro_p_value', 'N/A'):.4f}")
                print(f"  正規性検定 (KS): p={normality_tests.get('ks_p_value', 'N/A'):.4f}")
            
        return calibration_result
    
    def _evaluate_uncertainty_utility(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        true_values: torch.Tensor,
        verbose: bool
    ) -> Dict[str, float]:
        """不確実性の有用性評価"""
        with torch.no_grad():
            # 絶対誤差
            abs_errors = torch.abs(predictions - true_values)
            
            # 不確実性と誤差の相関
            if abs_errors.dim() > 1:
                # 各サンプルの平均誤差と平均不確実性
                mean_errors = abs_errors.mean(dim=1)
                mean_uncertainties = uncertainties.mean(dim=1)
            else:
                mean_errors = abs_errors
                mean_uncertainties = uncertainties
            
            # 相関計算
            if len(mean_errors) > 1:
                correlation = np.corrcoef(
                    mean_errors.cpu().numpy(), 
                    mean_uncertainties.cpu().numpy()
                )[0, 1]
            else:
                correlation = 0.0
            
            # 不確実性の予測的価値
            # 高不確実性サンプルが実際に高誤差かどうか
            uncertainty_threshold = uncertainties.quantile(0.8)  # 上位20%
            high_uncertainty_mask = uncertainties >= uncertainty_threshold
            
            if high_uncertainty_mask.any():
                high_unc_errors = abs_errors[high_uncertainty_mask].mean().item()
                low_unc_errors = abs_errors[~high_uncertainty_mask].mean().item()
                uncertainty_effectiveness = (high_unc_errors - low_unc_errors) / low_unc_errors
            else:
                uncertainty_effectiveness = 0.0
            
            utility_result = {
                'error_uncertainty_correlation': float(correlation),
                'uncertainty_effectiveness': uncertainty_effectiveness,
                'high_uncertainty_error_ratio': high_unc_errors / low_unc_errors if 'low_unc_errors' in locals() and low_unc_errors > 0 else 1.0
            }
            
            if verbose:
                print(f"\n不確実性有用性:")
                print(f"  誤差-不確実性相関: {correlation:.4f}")
                print(f"  不確実性効果: {uncertainty_effectiveness:.4f}")
            
        return utility_result
    
    def _analyze_temporal_uncertainty(
        self, 
        uncertainties: torch.Tensor,
        verbose: bool
    ) -> Dict[str, float]:
        """時系列不確実性の分析"""
        with torch.no_grad():
            T = uncertainties.size(0)
            
            if T < 2:
                return {'error': 'Insufficient time steps for temporal analysis'}
            
            # 時系列特性
            if uncertainties.dim() > 1:
                # 各時点での平均不確実性
                temporal_uncertainty = uncertainties.mean(dim=1)
            else:
                temporal_uncertainty = uncertainties
            
            # 傾向分析
            time_indices = torch.arange(T, dtype=torch.float32)
            
            # 線形傾向
            time_mean = time_indices.mean()
            unc_mean = temporal_uncertainty.mean()
            
            numerator = ((time_indices - time_mean) * (temporal_uncertainty - unc_mean)).sum()
            denominator = ((time_indices - time_mean) ** 2).sum()
            
            if denominator > 0:
                trend_slope = (numerator / denominator).item()
            else:
                trend_slope = 0.0
            
            # 変動性
            uncertainty_volatility = temporal_uncertainty.std().item()
            
            # 自己相関（lag-1）
            if T > 2:
                autocorr = np.corrcoef(
                    temporal_uncertainty[:-1].cpu().numpy(),
                    temporal_uncertainty[1:].cpu().numpy()
                )[0, 1]
            else:
                autocorr = 0.0
            
            temporal_result = {
                'trend_slope': trend_slope,
                'volatility': uncertainty_volatility,
                'autocorrelation_lag1': float(autocorr),
                'initial_uncertainty': temporal_uncertainty[0].item(),
                'final_uncertainty': temporal_uncertainty[-1].item(),
                'uncertainty_change': (temporal_uncertainty[-1] - temporal_uncertainty[0]).item()
            }
            
            if verbose:
                print(f"\n📈 時系列不確実性:")
                print(f"  不確実性傾向: {trend_slope:+.6f}/ステップ")
                print(f"  不確実性変動: {uncertainty_volatility:.6f}")
                print(f"  自己相関(lag-1): {autocorr:.4f}")
                print(f"  不確実性変化: {temporal_result['uncertainty_change']:+.6f}")
            
        return temporal_result
    
    def _compute_ece(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        true_values: torch.Tensor,
        n_bins: int = 10
    ) -> float:
        """Expected Calibration Error計算"""
        with torch.no_grad():
            # 正規化された不確実性を確信度に変換
            # 簡略版: 不確実性の逆数を使用
            max_uncertainty = uncertainties.max()
            confidences = 1.0 - (uncertainties / max_uncertainty)
            
            # 正確性（閾値ベース）
            threshold = uncertainties.mean()
            accuracies = (torch.abs(predictions - true_values) < threshold).float()
            
            if predictions.dim() > 1:
                confidences = confidences.mean(dim=1)
                accuracies = accuracies.mean(dim=1)
            
            # ビン分割
            bin_boundaries = torch.linspace(0, 1, n_bins + 1)
            ece = 0.0
            
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = in_bin.float().mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = accuracies[in_bin].mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
        return ece.item()
    
    def _create_uncertainty_plots(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        true_values: Optional[torch.Tensor],
        evaluation_results: Dict
    ):
        """不確実性可視化"""
        if not self.output_dir:
            return
        
        plt.style.use('default')
        
        # 1. 不確実性ヒストグラム
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 3, 1)
        plt.hist(uncertainties.flatten().cpu().numpy(), bins=50, alpha=0.7, color='skyblue')
        plt.title('Uncertainty Distribution')
        plt.xlabel('Uncertainty')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 2. 時系列不確実性
        plt.subplot(2, 3, 2)
        if uncertainties.dim() > 1:
            temporal_unc = uncertainties.mean(dim=1).cpu().numpy()
        else:
            temporal_unc = uncertainties.cpu().numpy()
        plt.plot(temporal_unc, color='orange', linewidth=2)
        plt.title('Uncertainty Time Series')
        plt.xlabel('Time')
        plt.ylabel('Mean Uncertainty')
        plt.grid(True, alpha=0.3)
        
        if true_values is not None:
            # 3. 誤差 vs 不確実性
            plt.subplot(2, 3, 3)
            abs_errors = torch.abs(predictions - true_values)
            if abs_errors.dim() > 1:
                errors_flat = abs_errors.mean(dim=1).cpu().numpy()
                uncertainties_flat = uncertainties.mean(dim=1).cpu().numpy()
            else:
                errors_flat = abs_errors.cpu().numpy()
                uncertainties_flat = uncertainties.cpu().numpy()
            
            plt.scatter(uncertainties_flat, errors_flat, alpha=0.6, color='green')
            plt.xlabel('Uncertainty')
            plt.ylabel('Absolute Error')
            plt.title('Uncertainty vs Error')
            plt.grid(True, alpha=0.3)
            
            # 4. 信頼区間プロット（サンプル）
            plt.subplot(2, 3, 4)
            n_samples = min(100, len(predictions))
            indices = torch.randperm(len(predictions))[:n_samples]
            
            sample_pred = predictions[indices]
            sample_true = true_values[indices]
            sample_unc = uncertainties[indices]
            
            if sample_pred.dim() > 1:
                sample_pred = sample_pred.mean(dim=1)
                sample_true = sample_true.mean(dim=1)
                sample_unc = sample_unc.mean(dim=1)
            
            x_pos = torch.arange(n_samples)
            plt.errorbar(
                x_pos.cpu().numpy(),
                sample_pred.cpu().numpy(),
                yerr=1.96 * sample_unc.cpu().numpy(),
                fmt='o',
                color='blue',
                alpha=0.6,
                capsize=3,
                label='Prediction ±95% CI'
            )
            plt.scatter(x_pos.cpu().numpy(), sample_true.cpu().numpy(), 
                       color='red', s=20, label='True Value', zorder=5)
            plt.title('Confidence Interval Sample')
            plt.xlabel('Sample')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        # 5. キャリブレーションプロット
        if true_values is not None and 'calibration' in evaluation_results:
            plt.subplot(2, 3, 5)
            # 簡略版キャリブレーションプロット
            self._plot_calibration_curve(predictions, uncertainties, true_values)
            
        # 6. カバレッジ率プロット
        if 'confidence_intervals' in evaluation_results:
            plt.subplot(2, 3, 6)
            intervals = evaluation_results['confidence_intervals']
            levels = []
            expected = []
            actual = []
            
            for key, value in intervals.items():
                if key.startswith('confidence_'):
                    level = int(key.split('_')[1])
                    levels.append(level)
                    expected.append(value['expected_coverage'])
                    actual.append(value['actual_coverage'])
            
            plt.plot(levels, expected, 'r--', label='Expected Coverage', linewidth=2)
            plt.plot(levels, actual, 'b-o', label='Actual Coverage', linewidth=2)
            plt.xlabel('Confidence Level (%)')
            plt.ylabel('Coverage Rate')
            plt.title('Coverage Rate Evaluation')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'uncertainty_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"不確実性可視化保存: {plot_path}")
    
    def _plot_calibration_curve(
        self, 
        predictions: torch.Tensor, 
        uncertainties: torch.Tensor, 
        true_values: torch.Tensor
    ):
        """キャリブレーション曲線のプロット"""
        # 簡略版実装
        confidence_levels = np.linspace(0.1, 0.9, 9)
        empirical_coverage = []
        
        for conf in confidence_levels:
            z_score = stats.norm.ppf((1 + conf) / 2)
            lower = predictions - z_score * uncertainties
            upper = predictions + z_score * uncertainties
            
            covered = ((true_values >= lower) & (true_values <= upper))
            if covered.dim() > 1:
                coverage = covered.all(dim=1).float().mean().item()
            else:
                coverage = covered.float().mean().item()
            
            empirical_coverage.append(coverage)
        
        plt.plot(confidence_levels, confidence_levels, 'r--', label='Perfect Calibration')
        plt.plot(confidence_levels, empirical_coverage, 'b-o', label='Actual')
        plt.xlabel('Expected Coverage')
        plt.ylabel('Actual Coverage')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _print_uncertainty_summary(self, results: Dict):
        """不確実性評価サマリ出力"""
        print(f"\n不確実性定量化評価完了")
        print("="*50)
        
        if 'confidence_intervals' in results:
            print(f"\n主要カバレッジ結果:")
            ci_results = results['confidence_intervals']
            for level in [68, 95]:
                key = f'confidence_{level}'
                if key in ci_results:
                    result = ci_results[key]
                    print(f"  {level}%信頼区間: {result['actual_coverage']:.3f} (誤差: {result['coverage_error']:.3f})")
        
        if 'calibration' in results:
            cal_results = results['calibration']
            print(f"\nキャリブレーション:")
            print(f"  ECE: {cal_results['ece']:.4f}")
            
        if 'uncertainty_utility' in results:
            utility = results['uncertainty_utility']
            print(f"\n不確実性有用性:")
            print(f"  誤差-不確実性相関: {utility['error_uncertainty_correlation']:.4f}")


def create_uncertainty_evaluator(output_dir: str = None) -> UncertaintyEvaluator:
    """不確実性評価器作成"""
    return UncertaintyEvaluator(output_dir)


def quick_uncertainty_evaluation(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    true_values: torch.Tensor,
    output_dir: str = None
) -> Dict:
    """クイック不確実性評価"""
    evaluator = UncertaintyEvaluator(output_dir)
    return evaluator.evaluate_uncertainty_quality(
        predictions, uncertainties, true_values
    )