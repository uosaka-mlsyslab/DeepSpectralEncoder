"""
フィルタリング分析モジュール

DFIV Kalman Filterのフィルタリング性能を詳細に分析し、
ターミナル出力と数値保存を行う。
"""

import torch
import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import warnings

from .metrics import StateEstimationMetrics, ComputationalMetrics, CalibrationMetrics


class FilteringAnalyzer:
    """フィルタリング性能の包括的分析クラス"""
    
    def __init__(self, output_dir: str, device: str = 'cpu'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        
        # 分析結果保存用
        self.analysis_results = {}
        self.experiment_metadata = {}
        
        # 評価器の初期化
        self.metrics_evaluator = StateEstimationMetrics(device=str(device))
        self.computational_metrics = ComputationalMetrics()
        
    def analyze_filtering_performance(
        self,
        inference_model,
        test_data: torch.Tensor,
        true_states: Optional[torch.Tensor] = None,
        experiment_name: str = "filtering_experiment",
        save_results: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        フィルタリング性能の包括的分析
        
        Args:
            inference_model: 推論モデル
            test_data: テストデータ (T, n)
            true_states: 真の状態 (T, r) [optional]
            experiment_name: 実験名
            save_results: 結果を保存するか
            verbose: 詳細出力するか
            
        Returns:
            Dict: 分析結果
        """
        if verbose:
            print(f"\n🚀 フィルタリング性能分析開始: {experiment_name}")
            print("="*60)
            
        analysis_start_time = datetime.now()
        
        # 1. バッチフィルタリング実行・分析
        batch_results = self._analyze_batch_filtering(
            inference_model, test_data, true_states, verbose
        )
        
        # 2. オンラインフィルタリング実行・分析  
        online_results = self._analyze_online_filtering(
            inference_model, test_data, true_states, verbose
        )
        
        # 3. 計算効率分析
        efficiency_results = self._analyze_computational_efficiency(
            inference_model, test_data, verbose
        )
        
        # 4. 比較分析（バッチ vs オンライン）
        comparison_results = self._compare_filtering_methods(
            batch_results, online_results, verbose
        )
        
        # 5. 結果統合
        complete_analysis = {
            'experiment_info': {
                'name': experiment_name,
                'timestamp': analysis_start_time.isoformat(),
                'data_shape': list(test_data.shape),
                'has_true_states': true_states is not None,
                'device': str(self.device)
            },
            'batch_filtering': batch_results,
            'online_filtering': online_results,
            'computational_efficiency': efficiency_results,
            'method_comparison': comparison_results
        }
        
        # 6. 結果保存
        if save_results:
            self._save_analysis_results(complete_analysis, experiment_name)
            
        if verbose:
            print(f"\n分析完了: {experiment_name}")
            print(f"📁 結果保存先: {self.output_dir}")
            
        return complete_analysis
    
    def _analyze_batch_filtering(
        self,
        inference_model,
        test_data: torch.Tensor,
        true_states: Optional[torch.Tensor],
        verbose: bool
    ) -> Dict[str, Any]:
        """バッチフィルタリングの分析"""
        if verbose:
            print("\nバッチフィルタリング分析...")
            
        # フィルタリング実行
        start_time = datetime.now()
        
        try:
            # 推論実行（防御的チェック付き）
            if hasattr(inference_model, 'filter_sequence'):
                filtering_result = inference_model.filter_sequence(
                    test_data, return_likelihood=True
                )
            else:
                # フォールバック: inference_batchを使用
                batch_result = inference_model.inference_batch(test_data, return_format='dict')
                X_means = torch.tensor(batch_result['summary']['mean_trajectory'])
                X_covariances = torch.tensor(batch_result['summary']['covariance_trajectory'])
                if 'likelihood' in batch_result['statistics']:
                    likelihoods = torch.tensor(batch_result['statistics']['likelihood']['likelihood_trajectory'])
                    filtering_result = (X_means, X_covariances, likelihoods)
                else:
                    filtering_result = (X_means, X_covariances)
            
            if len(filtering_result) == 3:
                X_means, X_covariances, likelihoods = filtering_result
            else:
                X_means, X_covariances = filtering_result
                likelihoods = None
                
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            if verbose:
                print(f"  バッチフィルタリング完了 ({processing_time:.2f}秒)")
                print(f"  📏 推定状態形状: {X_means.shape}")
                print(f"  📏 共分散形状: {X_covariances.shape}")
                
        except Exception as e:
            import traceback
            error_details = {
                'error_message': str(e),
                'error_type': type(e).__name__,
                'full_traceback': traceback.format_exc(),
                'inference_model_type': type(inference_model).__name__,
                'available_methods': [m for m in dir(inference_model) if not m.startswith('_')],
                'filter_methods': [m for m in dir(inference_model) if not m.startswith('_') and 'filter' in m.lower()],
                'test_data_shape': list(test_data.shape),
                'test_data_type': str(test_data.dtype),
                'model_setup_status': getattr(inference_model, 'is_setup', 'unknown')
            }

            if verbose:
                print(f"  ❌ バッチフィルタリングエラー: {e}")
                print(f"  🔍 エラータイプ: {type(e).__name__}")
                print(f"  データ形状: {test_data.shape} (dtype: {test_data.dtype})")
                print(f"  モデルタイプ: {type(inference_model).__name__}")
                print(f"  🔧 利用可能なfilterメソッド: {error_details['filter_methods']}")
                print(f"  ⚙️  モデルセットアップ状況: {error_details['model_setup_status']}")
                print(f"  📝 詳細トレース:\n{traceback.format_exc()}")

            return {
                'error': str(e),
                'success': False,
                'error_details': error_details
            }
        
        # 性能評価
        metrics = self.metrics_evaluator.compute_all_metrics(
            X_means, true_states, X_covariances, test_data, likelihoods, verbose=False
        )
        
        if verbose:
            self._print_batch_summary(metrics, processing_time)
            
        return {
            'success': True,
            'processing_time': processing_time,
            'estimated_states': X_means,
            'covariances': X_covariances,
            'likelihoods': likelihoods,
            'metrics': metrics
        }
    
    def _analyze_online_filtering(
        self,
        inference_model,
        test_data: torch.Tensor,
        true_states: Optional[torch.Tensor],
        verbose: bool
    ) -> Dict[str, Any]:
        """オンラインフィルタリングの分析"""
        if verbose:
            print("\n📱 オンラインフィルタリング分析...")
            
        # フィルタ状態リセット（防御的チェック付き）
        if hasattr(inference_model, 'reset_state'):
            inference_model.reset_state()
        else:
            # フォールバック: 推論環境の再セットアップ
            if hasattr(inference_model, 'setup_inference') and inference_model.calibration_data is not None:
                inference_model.setup_inference(
                    calibration_data=inference_model.calibration_data,
                    method='data_driven'
                )
        
        # 逐次処理
        start_time = datetime.now()
        online_states = []
        online_covariances = []
        online_likelihoods = []
        step_times = []
        
        try:
            for t, observation in enumerate(test_data):
                step_start = datetime.now()
                
                # 1ステップフィルタリング
                x_hat, Sigma_x, likelihood = inference_model.filter_online(observation)
                
                step_end = datetime.now()
                step_time = (step_end - step_start).total_seconds()
                
                # 結果保存
                online_states.append(x_hat)
                online_covariances.append(Sigma_x)
                online_likelihoods.append(likelihood)
                step_times.append(step_time)
                
                # 進捗表示
                if verbose and (t + 1) % 100 == 0:
                    print(f"  📈 処理済み: {t+1}/{len(test_data)} (平均: {np.mean(step_times):.4f}秒/ステップ)")
                    
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            # 結果結合
            X_means_online = torch.stack(online_states)
            X_covariances_online = torch.stack(online_covariances)
            likelihoods_online = torch.tensor(online_likelihoods)
            
            if verbose:
                print(f"  オンラインフィルタリング完了 ({total_time:.2f}秒)")
                print(f"  📏 推定状態形状: {X_means_online.shape}")
                print(f"  ⚡ 平均ステップ時間: {np.mean(step_times):.4f}秒")
                
        except Exception as e:
            import traceback
            error_details = {
                'error_message': str(e),
                'error_type': type(e).__name__,
                'full_traceback': traceback.format_exc(),
                'inference_model_type': type(inference_model).__name__,
                'available_methods': [m for m in dir(inference_model) if not m.startswith('_')],
                'reset_methods': [m for m in dir(inference_model) if not m.startswith('_') and 'reset' in m.lower()],
                'streaming_methods': [m for m in dir(inference_model) if not m.startswith('_') and 'streaming' in m.lower()],
                'test_data_shape': list(test_data.shape),
                'test_data_type': str(test_data.dtype),
                'model_setup_status': getattr(inference_model, 'is_setup', 'unknown'),
                'streaming_estimator_exists': hasattr(inference_model, 'streaming_estimator') and inference_model.streaming_estimator is not None
            }

            if verbose:
                print(f"  ❌ オンラインフィルタリングエラー: {e}")
                print(f"  🔍 エラータイプ: {type(e).__name__}")
                print(f"  データ形状: {test_data.shape} (dtype: {test_data.dtype})")
                print(f"  モデルタイプ: {type(inference_model).__name__}")
                print(f"  🔧 利用可能なresetメソッド: {error_details['reset_methods']}")
                print(f"  🌊 利用可能なstreamingメソッド: {error_details['streaming_methods']}")
                print(f"  ⚙️  モデルセットアップ状況: {error_details['model_setup_status']}")
                print(f"  🔗 StreamingEstimator存在: {error_details['streaming_estimator_exists']}")
                print(f"  📝 詳細トレース:\n{traceback.format_exc()}")

            return {
                'error': str(e),
                'success': False,
                'error_details': error_details
            }
        
        # 性能評価
        metrics = self.metrics_evaluator.compute_all_metrics(
            X_means_online, true_states, X_covariances_online, 
            test_data, likelihoods_online, verbose=False
        )
        
        if verbose:
            self._print_online_summary(metrics, total_time, step_times)
            
        return {
            'success': True,
            'total_processing_time': total_time,
            'average_step_time': np.mean(step_times),
            'step_times': step_times,
            'estimated_states': X_means_online,
            'covariances': X_covariances_online,
            'likelihoods': likelihoods_online,
            'metrics': metrics
        }
    
    def _analyze_computational_efficiency(
        self,
        inference_model,
        test_data: torch.Tensor,
        verbose: bool
    ) -> Dict[str, Any]:
        """計算効率の分析"""
        if verbose:
            print("\n⚡ 計算効率分析...")
            
        efficiency_results = {}
        
        # バッチ推論時間測定
        batch_timing = self.computational_metrics.measure_inference_time(
            lambda data: inference_model.filter_sequence(data),
            test_data,
            n_trials=3,
            warmup=1
        )
        efficiency_results['batch_timing'] = batch_timing
        
        # オンライン推論時間測定
        def online_inference(data):
            inference_model.reset_state()
            for obs in data:
                inference_model.filter_online(obs)
                
        online_timing = self.computational_metrics.measure_inference_time(
            online_inference,
            test_data,
            n_trials=3,
            warmup=1
        )
        efficiency_results['online_timing'] = online_timing
        
        # メモリ使用量測定
        memory_usage = self.computational_metrics.measure_memory_usage(
            lambda data: inference_model.filter_sequence(data),
            test_data
        )
        efficiency_results['memory_usage'] = memory_usage
        
        if verbose:
            self._print_efficiency_summary(efficiency_results)
            
        return efficiency_results
    
    def _compare_filtering_methods(
        self,
        batch_results: Dict,
        online_results: Dict,
        verbose: bool
    ) -> Dict[str, Any]:
        """バッチ vs オンライン比較"""
        if not (batch_results.get('success') and online_results.get('success')):
            return {'comparison_available': False}
            
        if verbose:
            print("\n🔍 フィルタリング手法比較...")
            
        comparison = {
            'comparison_available': True,
            'time_comparison': {
                'batch_time': batch_results['processing_time'],
                'online_total_time': online_results['total_processing_time'],
                'online_avg_step_time': online_results['average_step_time'],
                'speed_ratio': online_results['total_processing_time'] / batch_results['processing_time']
            }
        }
        
        # 精度比較
        batch_metrics = batch_results.get('metrics', {})
        online_metrics = online_results.get('metrics', {})
        
        if 'accuracy' in batch_metrics and 'accuracy' in online_metrics:
            batch_acc = batch_metrics['accuracy']
            online_acc = online_metrics['accuracy']
            
            comparison['accuracy_comparison'] = {
                'mse_difference': online_acc['mse'] - batch_acc['mse'],
                'mae_difference': online_acc['mae'] - batch_acc['mae'],
                'correlation_difference': online_acc['correlation'] - batch_acc['correlation']
            }
            
        if verbose:
            self._print_comparison_summary(comparison)
            
        return comparison
    
    def _print_batch_summary(self, metrics: Dict, processing_time: float):
        """バッチ結果サマリ出力"""
        print(f"\n  バッチフィルタリング結果:")
        print(f"    処理時間: {processing_time:.4f}秒")
        
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            print(f"    MSE: {acc['mse']:.6f}")
            print(f"    MAE: {acc['mae']:.6f}")
            print(f"    相関係数: {acc['correlation']:.4f}")
            
        if 'uncertainty' in metrics:
            unc = metrics['uncertainty']
            print(f"    平均不確実性: {unc['mean_uncertainty']:.6f}")
            if 'coverage_95' in unc:
                print(f"    95%カバレッジ: {unc['coverage_95']:.4f}")
                
    def _print_online_summary(self, metrics: Dict, total_time: float, step_times: List[float]):
        """オンライン結果サマリ出力"""
        print(f"\n  📱 オンラインフィルタリング結果:")
        print(f"    総処理時間: {total_time:.4f}秒")
        print(f"    平均ステップ時間: {np.mean(step_times):.6f}秒")
        print(f"    ステップ時間標準偏差: {np.std(step_times):.6f}秒")
        
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            print(f"    MSE: {acc['mse']:.6f}")
            print(f"    MAE: {acc['mae']:.6f}")
            print(f"    相関係数: {acc['correlation']:.4f}")
            
    def _print_efficiency_summary(self, efficiency: Dict):
        """効率性サマリ出力"""
        print(f"\n  ⚡ 計算効率:")
        
        if 'batch_timing' in efficiency:
            batch = efficiency['batch_timing']
            print(f"    バッチ推論時間: {batch['mean_time']:.4f}±{batch['std_time']:.4f}秒")
            
        if 'online_timing' in efficiency:
            online = efficiency['online_timing']
            print(f"    オンライン推論時間: {online['mean_time']:.4f}±{online['std_time']:.4f}秒")
            
        if 'memory_usage' in efficiency and 'peak_memory_mb' in efficiency['memory_usage']:
            mem = efficiency['memory_usage']
            print(f"    ピークメモリ使用量: {mem['peak_memory_mb']:.1f}MB")
            
    def _print_comparison_summary(self, comparison: Dict):
        """比較結果サマリ出力"""
        if 'time_comparison' in comparison:
            time_comp = comparison['time_comparison']
            print(f"\n  🔍 手法比較:")
            print(f"    バッチ処理時間: {time_comp['batch_time']:.4f}秒")
            print(f"    オンライン処理時間: {time_comp['online_total_time']:.4f}秒")
            print(f"    速度比 (オンライン/バッチ): {time_comp['speed_ratio']:.2f}")
            
        if 'accuracy_comparison' in comparison:
            acc_comp = comparison['accuracy_comparison']
            print(f"    MSE差分 (オンライン-バッチ): {acc_comp['mse_difference']:+.6f}")
            print(f"    MAE差分 (オンライン-バッチ): {acc_comp['mae_difference']:+.6f}")
    
    def _save_analysis_results(self, analysis: Dict, experiment_name: str):
        """分析結果の保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON形式で保存
        json_path = self.output_dir / f"{experiment_name}_{timestamp}_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(self._make_json_serializable(analysis), f, indent=2)
            
        print(f"📁 分析結果保存: {json_path}")
        
        # CSV形式でメトリクス保存
        self._save_metrics_csv(analysis, experiment_name, timestamp)
        
        # 数値データ保存（NPZ形式）
        self._save_numerical_data(analysis, experiment_name, timestamp)
    
    def _save_metrics_csv(self, analysis: Dict, experiment_name: str, timestamp: str):
        """メトリクスをCSV形式で保存"""
        csv_path = self.output_dir / f"{experiment_name}_{timestamp}_metrics.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # ヘッダー
            writer.writerow([
                'method', 'mse', 'mae', 'rmse', 'correlation',
                'mean_uncertainty', 'coverage_95', 'processing_time'
            ])
            
            # バッチフィルタリング結果
            if analysis['batch_filtering'].get('success'):
                batch_metrics = analysis['batch_filtering']['metrics']
                if 'accuracy' in batch_metrics:
                    acc = batch_metrics['accuracy']
                    unc = batch_metrics.get('uncertainty', {})
                    writer.writerow([
                        'batch',
                        acc['mse'],
                        acc['mae'], 
                        acc['rmse'],
                        acc['correlation'],
                        unc.get('mean_uncertainty', ''),
                        unc.get('coverage_95', ''),
                        analysis['batch_filtering']['processing_time']
                    ])
            
            # オンラインフィルタリング結果
            if analysis['online_filtering'].get('success'):
                online_metrics = analysis['online_filtering']['metrics']
                if 'accuracy' in online_metrics:
                    acc = online_metrics['accuracy']
                    unc = online_metrics.get('uncertainty', {})
                    writer.writerow([
                        'online',
                        acc['mse'],
                        acc['mae'],
                        acc['rmse'], 
                        acc['correlation'],
                        unc.get('mean_uncertainty', ''),
                        unc.get('coverage_95', ''),
                        analysis['online_filtering']['total_processing_time']
                    ])
                    
        print(f"メトリクスCSV保存: {csv_path}")
    
    def _save_numerical_data(self, analysis: Dict, experiment_name: str, timestamp: str):
        """数値データをNPZ形式で保存"""
        npz_path = self.output_dir / f"{experiment_name}_{timestamp}_data.npz"
        
        save_data = {}
        
        # バッチフィルタリング結果
        if analysis['batch_filtering'].get('success'):
            batch_result = analysis['batch_filtering']
            if isinstance(batch_result['estimated_states'], torch.Tensor):
                save_data['batch_states'] = batch_result['estimated_states'].cpu().numpy()
                save_data['batch_covariances'] = batch_result['covariances'].cpu().numpy()
                if batch_result['likelihoods'] is not None:
                    save_data['batch_likelihoods'] = batch_result['likelihoods'].cpu().numpy()
        
        # オンラインフィルタリング結果  
        if analysis['online_filtering'].get('success'):
            online_result = analysis['online_filtering']
            if isinstance(online_result['estimated_states'], torch.Tensor):
                save_data['online_states'] = online_result['estimated_states'].cpu().numpy()
                save_data['online_covariances'] = online_result['covariances'].cpu().numpy()
                save_data['online_likelihoods'] = online_result['likelihoods'].cpu().numpy()
                save_data['step_times'] = np.array(online_result['step_times'])
        
        if save_data:
            np.savez(npz_path, **save_data)
            print(f"💾 数値データ保存: {npz_path}")
    
    def _make_json_serializable(self, obj):
        """JSON対応形式に変換"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        else:
            return obj


def create_filtering_analyzer(output_dir: str, device: str = 'cpu') -> FilteringAnalyzer:
    """フィルタリング分析器の作成"""
    return FilteringAnalyzer(output_dir, device)


def run_quick_filtering_analysis(
    inference_model,
    test_data: torch.Tensor,
    output_dir: str,
    experiment_name: str = "quick_analysis"
) -> Dict[str, Any]:
    """
    クイックフィルタリング分析
    
    簡単なAPIでフィルタリング分析を実行
    """
    analyzer = FilteringAnalyzer(output_dir)
    return analyzer.analyze_filtering_performance(
        inference_model, test_data, experiment_name=experiment_name
    )