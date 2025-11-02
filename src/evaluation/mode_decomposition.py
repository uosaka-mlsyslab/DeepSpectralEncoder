"""
モード分解・スペクトル分析モジュール

Koopman作用素理論に基づくDFIVスペクトル解析を提供。
学習済みV_A行列から固有値・モード分解を実行し、
連続時間スペクトル特性を抽出する。

実装機能:
- 基本スペクトル分析：固有値・連続時間変換
- 真値との MSE 評価（利用可能時）
- 学習済みモデルからのV_A抽出
- 結果保存（JSON, NPZ形式）
"""

import torch
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from pathlib import Path


class SpectrumAnalyzer:
    """
    Koopmanスペクトル分析クラス

    V_A行列からの固有値分解と連続時間スペクトル特性の抽出を実行。
    離散時間固有値から連続時間固有値への変換も含む。
    """

    def __init__(self, sampling_interval: float):
        """
        Args:
            sampling_interval: サンプリング間隔 Δt (連続時間変換用)
        """
        self.dt = sampling_interval

    def analyze_spectrum(self, V_A: torch.Tensor) -> Dict[str, Any]:
        """
        V_A からのスペクトル分析

        Args:
            V_A: 転送作用素行列 (d_A, d_A)

        Returns:
            Dict: スペクトル分析結果
                - eigenvalues_discrete: λ ∈ C^{d_A} (離散時間固有値)
                - eigenvalues_continuous: μ ∈ C^{d_A} (連続時間固有値)
                - growth_rates: Re(μ) (成長率/減衰率)
                - frequencies_hz: Im(μ)/(2π) (振動周波数 Hz)
                - dominant_indices: 主要モードインデックス
                - stable_indices: 安定モードインデックス
                - eigenvalues_magnitude: |λ| (離散時間振幅)
                - eigenvalues_phase: arg(λ) (離散時間位相)
        """
        # デバイス・形状情報をログ出力（デバッグ用）
        print(f"📋 V_A分析開始: shape={V_A.shape}, device={V_A.device}, dtype={V_A.dtype}")

        with torch.no_grad():
            # 固有値分解
            eigenvalues_discrete, eigenvectors = torch.linalg.eig(V_A)

            # 連続時間変換: μ = (1/Δt) * log(λ)
            eigenvalues_continuous = self._discrete_to_continuous_eigenvalues(
                eigenvalues_discrete
            )

            # スペクトル特性抽出
            growth_rates = eigenvalues_continuous.real
            frequencies_rad = eigenvalues_continuous.imag
            frequencies_hz = frequencies_rad / (2 * np.pi)

            # 離散時間特性
            eigenvalues_magnitude = torch.abs(eigenvalues_discrete)
            eigenvalues_phase = torch.angle(eigenvalues_discrete)

            # モード分類
            dominant_threshold = 0.1  # 設定可能にしたい場合は__init__に移動
            dominant_indices = self._find_dominant_modes(
                eigenvalues_magnitude, threshold=dominant_threshold
            )
            stable_indices = self._find_stable_modes(growth_rates)

            return {
                'eigenvalues_discrete': eigenvalues_discrete,
                'eigenvalues_continuous': eigenvalues_continuous,
                'eigenvectors': eigenvectors,
                'growth_rates': growth_rates,
                'frequencies_hz': frequencies_hz,
                'frequencies_rad': frequencies_rad,
                'eigenvalues_magnitude': eigenvalues_magnitude,
                'eigenvalues_phase': eigenvalues_phase,
                'dominant_indices': dominant_indices,
                'stable_indices': stable_indices,
                'n_stable_modes': len(stable_indices),
                'n_dominant_modes': len(dominant_indices),
                'spectral_radius': torch.max(eigenvalues_magnitude).item(),
                'sampling_interval': self.dt
            }

    def _discrete_to_continuous_eigenvalues(
        self,
        eigenvalues_discrete: torch.Tensor
    ) -> torch.Tensor:
        """
        離散時間固有値から連続時間固有値への変換

        μ = (1/Δt) * log(λ)

        Args:
            eigenvalues_discrete: 離散時間固有値 λ ∈ C^{d_A}

        Returns:
            torch.Tensor: 連続時間固有値 μ ∈ C^{d_A}
        """
        # log計算（ゼロ近似の場合の数値安定性考慮）
        eigenvalues_log = torch.log(eigenvalues_discrete + 1e-12)
        eigenvalues_continuous = eigenvalues_log / self.dt

        return eigenvalues_continuous

    def _find_dominant_modes(
        self,
        eigenvalues_magnitude: torch.Tensor,
        threshold: float = 0.1
    ) -> List[int]:
        """
        主要モードの特定

        Args:
            eigenvalues_magnitude: 固有値の絶対値 |λ|
            threshold: 閾値（スペクトル半径に対する比率）

        Returns:
            List[int]: 主要モードのインデックス
        """
        spectral_radius = torch.max(eigenvalues_magnitude)
        dominant_mask = eigenvalues_magnitude > threshold * spectral_radius
        return dominant_mask.nonzero(as_tuple=True)[0].tolist()

    def _find_stable_modes(self, growth_rates: torch.Tensor) -> List[int]:
        """
        安定モードの特定

        Args:
            growth_rates: 成長率 Re(μ)

        Returns:
            List[int]: 安定モード（Re(μ) < 0）のインデックス
        """
        stable_mask = growth_rates < 0
        return stable_mask.nonzero(as_tuple=True)[0].tolist()

    def evaluate_against_truth(
        self,
        predicted_eigenvalues: torch.Tensor,
        true_eigenvalues: torch.Tensor
    ) -> Dict[str, float]:
        """
        真値との比較評価（利用可能時）

        Args:
            predicted_eigenvalues: 推定固有値 μ_pred ∈ C^{d_A}
            true_eigenvalues: 真の固有値 μ_true ∈ C^{k}

        Returns:
            Dict: 評価指標
                - mse_real: 実部のMSE
                - mse_imag: 虚部のMSE
                - mse_magnitude: 絶対値のMSE
                - n_matched: マッチした固有値数
        """
        with torch.no_grad():
            # 最近接マッチング
            matched_pred, matched_true = self._match_eigenvalues(
                predicted_eigenvalues, true_eigenvalues
            )

            if len(matched_pred) == 0:
                warnings.warn("固有値マッチングに失敗しました")
                return {
                    'mse_real': float('inf'),
                    'mse_imag': float('inf'),
                    'mse_magnitude': float('inf'),
                    'n_matched': 0
                }

            # MSE計算
            mse_real = torch.mean((matched_pred.real - matched_true.real)**2).item()
            mse_imag = torch.mean((matched_pred.imag - matched_true.imag)**2).item()
            mse_magnitude = torch.mean(
                (torch.abs(matched_pred) - torch.abs(matched_true))**2
            ).item()

            return {
                'mse_real': mse_real,
                'mse_imag': mse_imag,
                'mse_magnitude': mse_magnitude,
                'n_matched': len(matched_pred)
            }

    def _match_eigenvalues(
        self,
        pred_eigenvalues: torch.Tensor,
        true_eigenvalues: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        固有値の最近接マッチング

        Args:
            pred_eigenvalues: 推定固有値
            true_eigenvalues: 真の固有値

        Returns:
            Tuple: (マッチした推定固有値, マッチした真の固有値)
        """
        matched_pred = []
        matched_true = []
        used_true_indices = set()

        for pred_val in pred_eigenvalues:
            # 最近接の真値を探索
            distances = torch.abs(true_eigenvalues - pred_val)
            best_idx = torch.argmin(distances).item()

            # 重複使用を避ける
            if best_idx not in used_true_indices:
                matched_pred.append(pred_val)
                matched_true.append(true_eigenvalues[best_idx])
                used_true_indices.add(best_idx)

        if len(matched_pred) > 0:
            return torch.stack(matched_pred), torch.stack(matched_true)
        else:
            return torch.tensor([]), torch.tensor([])


class TrainedModelSpectrumAnalysis:
    """
    学習済みモデルからのスペクトル解析クラス

    学習済みDFIVモデルからV_A行列を抽出し、
    スペクトル分析を実行する。
    """

    def __init__(self, sampling_interval: float):
        """
        Args:
            sampling_interval: サンプリング間隔
        """
        self.sampling_interval = sampling_interval
        self.analyzer = SpectrumAnalyzer(sampling_interval)

    def extract_transfer_matrix_from_model(self, model: Any) -> torch.Tensor:
        """
        学習済みモデルからV_A抽出

        Args:
            model: 学習済みDFIVモデル（DFStateLayerを含む）

        Returns:
            torch.Tensor: V_A行列 (d_A, d_A)
        """
        try:
            # モデル構造に応じてV_A抽出
            if hasattr(model, 'ssm') and hasattr(model.ssm, 'df_state_layer'):
                # 一般的なモデル構造
                V_A = model.ssm.df_state_layer.get_transfer_operator()
            elif hasattr(model, 'df_state_layer'):
                # 直接DFStateLayer
                V_A = model.df_state_layer.get_transfer_operator()
            elif hasattr(model, 'get_transfer_operator'):
                # DFStateLayerそのもの
                V_A = model.get_transfer_operator()
            elif isinstance(model, dict):
                # checkpointファイルから直接抽出
                if 'model_state_dict' in model and 'df_state' in model['model_state_dict']:
                    df_state_dict = model['model_state_dict']['df_state']
                    if 'V_A' in df_state_dict:
                        V_A = df_state_dict['V_A']
                    else:
                        raise ValueError("保存されたモデルにV_A行列が見つかりません")
                elif 'df_state' in model and 'V_A' in model['df_state']:
                    V_A = model['df_state']['V_A']
                else:
                    raise ValueError("モデル辞書からV_A行列を抽出できませんでした")
            else:
                raise ValueError("モデルからV_A行列を抽出できませんでした。モデル構造を確認してください。")

            return V_A

        except Exception as e:
            raise RuntimeError(f"V_A抽出エラー: {e}")

    def extract_transfer_matrix_from_path(self, model_path: str) -> torch.Tensor:
        """
        保存済みモデルファイルからV_A抽出

        Args:
            model_path: モデルファイルのパス

        Returns:
            torch.Tensor: V_A行列 (d_A, d_A)
        """
        try:
            # モデル読み込み
            checkpoint = torch.load(model_path, map_location='cpu')

            # V_A直接保存の場合
            if 'V_A' in checkpoint:
                return checkpoint['V_A']

            # state_dict内にV_Aがある場合
            if 'state_dict' in checkpoint and 'V_A' in checkpoint['state_dict']:
                return checkpoint['state_dict']['V_A']

            # モデル全体が保存されている場合
            if 'model' in checkpoint:
                return self.extract_transfer_matrix_from_model(checkpoint['model'])

            # checkpointがmodel自体の場合
            return self.extract_transfer_matrix_from_model(checkpoint)

        except Exception as e:
            raise RuntimeError(f"モデルファイル読み込みエラー: {e}")

    def perform_spectrum_analysis_from_model(self, model: Any) -> Dict[str, Any]:
        """
        学習済みモデルからスペクトル分析実行

        Args:
            model: 学習済みモデル

        Returns:
            Dict: スペクトル分析結果 + V_A行列
        """
        V_A = self.extract_transfer_matrix_from_model(model)
        spectrum_analysis = self.analyzer.analyze_spectrum(V_A)

        return {
            'spectrum': spectrum_analysis,
            'V_A': V_A,
            'V_A_shape': V_A.shape,
            'sampling_interval': self.sampling_interval
        }

    def perform_spectrum_analysis_from_path(self, model_path: str) -> Dict[str, Any]:
        """
        保存済みモデルファイルからスペクトル分析実行

        Args:
            model_path: モデルファイルパス

        Returns:
            Dict: スペクトル分析結果 + V_A行列
        """
        V_A = self.extract_transfer_matrix_from_path(model_path)
        spectrum_analysis = self.analyzer.analyze_spectrum(V_A)

        return {
            'spectrum': spectrum_analysis,
            'V_A': V_A,
            'V_A_shape': V_A.shape,
            'model_path': model_path,
            'sampling_interval': self.sampling_interval
        }


class SpectrumResultsSaver:
    """
    スペクトル分析結果の保存クラス

    JSON（設定・メタデータ）とNPZ（数値データ）形式での保存をサポート。
    """

    @staticmethod
    def save_results(
        results: Dict[str, Any],
        save_path: str,
        save_format: str = 'both'  # 'json', 'npz', 'both'
    ) -> None:
        """
        スペクトル分析結果の保存

        Args:
            results: 分析結果辞書
            save_path: 保存パス（拡張子なし）
            save_format: 保存形式
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_format in ['json', 'both']:
            SpectrumResultsSaver._save_json(results, save_path.with_suffix('.json'))

        if save_format in ['npz', 'both']:
            SpectrumResultsSaver._save_npz(results, save_path.with_suffix('.npz'))

    @staticmethod
    def _save_json(results: Dict[str, Any], json_path: Path) -> None:
        """JSON形式で保存（設定・メタデータ中心）"""
        # JSONシリアライズ可能な形式に変換
        json_data = {}

        for key, value in results.items():
            if key == 'spectrum':
                # スペクトル分析結果をJSON化
                spectrum = value
                json_spectrum = {}

                for spectrum_key, spectrum_value in spectrum.items():
                    if isinstance(spectrum_value, torch.Tensor):
                        if torch.is_complex(spectrum_value):
                            # 複素数を実部・虚部に分割
                            json_spectrum[f'{spectrum_key}_real'] = spectrum_value.real.tolist()
                            json_spectrum[f'{spectrum_key}_imag'] = spectrum_value.imag.tolist()
                        else:
                            json_spectrum[spectrum_key] = spectrum_value.tolist()
                    elif isinstance(spectrum_value, (int, float, str, list)):
                        json_spectrum[spectrum_key] = spectrum_value
                    else:
                        json_spectrum[spectrum_key] = str(spectrum_value)

                json_data['spectrum'] = json_spectrum

            elif isinstance(value, torch.Tensor):
                # その他のテンソル
                if torch.is_complex(value):
                    json_data[f'{key}_real'] = value.real.tolist()
                    json_data[f'{key}_imag'] = value.imag.tolist()
                else:
                    json_data[key] = value.tolist()

            elif isinstance(value, (int, float, str, list, tuple)):
                json_data[key] = value
            else:
                json_data[key] = str(value)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _save_npz(results: Dict[str, Any], npz_path: Path) -> None:
        """NPZ形式で保存（数値データ中心）"""
        npz_data = {}

        def _flatten_dict(d: Dict, prefix: str = '') -> None:
            for key, value in d.items():
                full_key = f"{prefix}{key}" if prefix else key

                if isinstance(value, dict):
                    _flatten_dict(value, f"{full_key}_")
                elif isinstance(value, torch.Tensor):
                    try:
                        if torch.is_complex(value):
                            # CUDA tensorの場合はCPUに移動してからnumpy変換
                            tensor_cpu = value.cpu() if value.is_cuda else value
                            npz_data[f'{full_key}_real'] = tensor_cpu.real.numpy()
                            npz_data[f'{full_key}_imag'] = tensor_cpu.imag.numpy()
                        else:
                            # CUDA tensorの場合はCPUに移動してからnumpy変換
                            tensor_cpu = value.cpu() if value.is_cuda else value
                            npz_data[full_key] = tensor_cpu.numpy()
                    except Exception as e:
                        print(f"Tensor変換エラー (key: {full_key}, shape: {value.shape}, device: {value.device}, dtype: {value.dtype}): {e}")
                        # エラー時はテンソル情報のみ保存
                        npz_data[f'{full_key}_error'] = f"Conversion failed: {str(e)}"
                elif isinstance(value, (list, tuple)):
                    try:
                        npz_data[full_key] = np.array(value)
                    except:
                        # 配列化できない場合はスキップ
                        pass
                elif isinstance(value, (int, float)):
                    npz_data[full_key] = np.array(value)

        _flatten_dict(results)
        np.savez(npz_path, **npz_data)

    @staticmethod
    def load_results(load_path: str, load_format: str = 'json') -> Dict[str, Any]:
        """
        保存済み結果の読み込み

        Args:
            load_path: 読み込みパス
            load_format: 読み込み形式 ('json' or 'npz')

        Returns:
            Dict: 読み込み結果
        """
        load_path = Path(load_path)

        if load_format == 'json':
            return SpectrumResultsSaver._load_json(load_path)
        elif load_format == 'npz':
            return SpectrumResultsSaver._load_npz(load_path)
        else:
            raise ValueError(f"未対応の読み込み形式: {load_format}")

    @staticmethod
    def _load_json(json_path: Path) -> Dict[str, Any]:
        """JSON読み込み"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _load_npz(npz_path: Path) -> Dict[str, Any]:
        """NPZ読み込み"""
        data = np.load(npz_path)
        return dict(data)


def compute_eigenvalue_mse(
    predicted_eigenvalues: torch.Tensor,
    true_eigenvalues: torch.Tensor
) -> Dict[str, float]:
    """
    スペクトル固有値の MSE 評価（独立関数版）

    Args:
        predicted_eigenvalues: 推定固有値 μ_pred ∈ C^{d_A}
        true_eigenvalues: 真の固有値 μ_true ∈ C^{k}

    Returns:
        Dict: MSE評価結果
    """
    analyzer = SpectrumAnalyzer(sampling_interval=1.0)  # サンプリング間隔は評価には不要
    return analyzer.evaluate_against_truth(predicted_eigenvalues, true_eigenvalues)


def create_spectrum_analyzer(sampling_interval: float) -> SpectrumAnalyzer:
    """
    スペクトル分析器の作成

    Args:
        sampling_interval: サンプリング間隔

    Returns:
        SpectrumAnalyzer: 初期化済み分析器
    """
    return SpectrumAnalyzer(sampling_interval)


def create_model_spectrum_analyzer(sampling_interval: float) -> TrainedModelSpectrumAnalysis:
    """
    モデルスペクトル分析器の作成

    Args:
        sampling_interval: サンプリング間隔

    Returns:
        TrainedModelSpectrumAnalysis: 初期化済み分析器
    """
    return TrainedModelSpectrumAnalysis(sampling_interval)