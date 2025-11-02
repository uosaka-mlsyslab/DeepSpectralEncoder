# src/utils/data_loader.py
"""
タスク3用統一データローダー

機能:
- 多様なファイル形式の統一読み込み (.npz, .npy, .csv, .json)
- データ品質チェック・前処理
- 正規化・標準化 
- 時系列順を保った学習/検証/テスト分割
- メタデータ管理

前提データ型: (T, d) 多変量時系列 torch.FloatTensor
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler


@dataclass
class DataMetadata:
    """データメタデータ"""
    original_shape: Tuple[int, int]
    feature_names: Optional[List[str]]
    time_index: Optional[List]
    sampling_rate: Optional[float]
    missing_ratio: float
    data_source: str
    normalization_method: str
    train_indices: Tuple[int, int]
    val_indices: Tuple[int, int]
    test_indices: Tuple[int, int]
    # ターゲットデータ関連フィールド追加
    has_target_data: bool = False
    target_shape: Optional[Tuple[int, int]] = None
    target_feature_names: Optional[List[str]] = None
    target_dtype: Optional[str] = None


class DataLoaderError(Exception):
    """データローダー専用例外"""
    pass


class UniversalTimeSeriesDataset(Dataset):
    """
    統一時系列データセット
    
    特徴:
    - 時系列順保持分割
    - 複数ファイル形式対応
    - 自動正規化
    - 欠損値処理
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        normalization: str = "standard",
        handle_missing: str = "interpolate",
        feature_names: Optional[List[str]] = None,
        experiment_mode: str = "reconstruction"  # 手動切り替え用パラメータ
    ):
        """
        Args:
            data_path: データファイルパス
            split: "train", "val", "test"
            train_ratio: 訓練データ比率
            val_ratio: 検証データ比率
            test_ratio: テストデータ比率
            normalization: "standard", "minmax", "none"
            handle_missing: "interpolate", "forward_fill", "remove"
            feature_names: 特徴量名リスト
            experiment_mode: "reconstruction" or "target_prediction"
        """
        super().__init__()
        
        # パラメータ検証
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise DataLoaderError(f"分割比率の合計が1.0でありません: {train_ratio + val_ratio + test_ratio}")
        
        if split not in ["train", "val", "test"]:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")
            
        # データ読み込み
        self.data_path = Path(data_path)
        self.split = split
        self.normalization = normalization
        self.experiment_mode = experiment_mode  # 実験モード保存
        
        # 生データ読み込み
        raw_data = self._load_raw_data()
        
        # データ検証・クリーニング
        cleaned_data = self._validate_and_clean(raw_data, handle_missing)
        
        # 正規化
        normalized_data, self.scaler = self._normalize_data(cleaned_data, normalization)
        
        # 時系列順分割
        split_data = self._split_time_series(normalized_data, train_ratio, val_ratio, test_ratio)
        
        # 分割データ取得
        self.data = split_data[split]
        self.length = self.data.shape[0]
        
        # メタデータ作成
        self._create_metadata(raw_data, split_data, feature_names)

    def _detect_target_data(self, data: dict) -> Dict[str, Any]:
        """ターゲットデータの自動検出（quad_linkデータ対応・キャッシュ対応）"""
        # クラスレベルキャッシュで重複処理を防ぐ
        # 理由: split="all"時に3つのインスタンス(train/val/test)が作成され、
        #      各インスタンスが同じファイルに対して_detect_target_dataを実行するため
        cache_key = str(self.data_path)
        if not hasattr(UniversalTimeSeriesDataset, '_class_target_cache'):
            UniversalTimeSeriesDataset._class_target_cache = {}

        if cache_key in UniversalTimeSeriesDataset._class_target_cache:
            return UniversalTimeSeriesDataset._class_target_cache[cache_key]

        target_info = {
            'has_target': False,
            'input_data': None,
            'target_data': None,
            'target_test_data': None
        }

        # quad_linkデータ形式のキーを優先的に検出
        target_keys_train = ['train_targets', 'y_train', 'target_train', 'labels_train']
        target_keys_test = ['test_targets', 'y_test', 'target_test', 'labels_test']
        input_keys_train = ['train_obs', 'X_train', 'input_train', 'obs_train']
        input_keys_test = ['test_obs', 'X_test', 'input_test', 'obs_test']

        # ターゲットデータ検出
        target_train = None
        target_test = None
        input_train = None
        input_test = None

        for key in target_keys_train:
            if key in data:
                candidate = data[key]
                # (1, T, d) → (T, d) へのreshape対応
                if candidate.ndim == 3 and candidate.shape[0] == 1:
                    target_train = candidate.reshape(candidate.shape[1], candidate.shape[2])
                elif candidate.ndim == 2:
                    target_train = candidate
                print(f"訓練ターゲットデータ検出: '{key}' → shape={target_train.shape}")
                break

        for key in target_keys_test:
            if key in data:
                candidate = data[key]
                # (1, T, d) → (T, d) へのreshape対応
                if candidate.ndim == 3 and candidate.shape[0] == 1:
                    target_test = candidate.reshape(candidate.shape[1], candidate.shape[2])
                elif candidate.ndim == 2:
                    target_test = candidate
                print(f"テストターゲットデータ検出: '{key}' → shape={target_test.shape}")
                break

        for key in input_keys_train:
            if key in data:
                input_train = data[key]
                print(f"訓練入力データ検出: '{key}' → shape={input_train.shape}")
                break

        for key in input_keys_test:
            if key in data:
                input_test = data[key]
                print(f"テスト入力データ検出: '{key}' → shape={input_test.shape}")
                break

        # ターゲットデータがある場合
        if target_train is not None and input_train is not None:
            target_info['has_target'] = True
            target_info['input_data'] = input_train
            target_info['target_data'] = target_train
            target_info['target_test_data'] = target_test  # テストターゲット（オプション）
            target_info['input_test_data'] = input_test    # テスト入力（オプション）

            print(f"ターゲットデータ構造確認完了:")
            print(f"   - 入力: {input_train.shape} ({input_train.dtype})")
            print(f"   - ターゲット: {target_train.shape} ({target_train.dtype})")
            if target_test is not None:
                print(f"   - テストターゲット: {target_test.shape} ({target_test.dtype})")

        # 結果をクラスレベルキャッシュに保存（同一ファイルの重複検出を防止）
        UniversalTimeSeriesDataset._class_target_cache[cache_key] = target_info
        return target_info

    def _load_raw_data(self) -> np.ndarray:
        """生データ読み込み（ターゲットデータ自動検出対応）"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"データファイルが存在しません: {self.data_path}")

        ext = self.data_path.suffix.lower()

        try:
            if ext == ".npz":
                data = np.load(self.data_path)

                # 手動切り替えに基づくデータ読み込み
                if self.experiment_mode == "target_prediction":
                    # ターゲット予測モード：ターゲットデータを強制的に検出
                    target_info = self._detect_target_data(data)

                    if target_info['has_target']:
                        self.has_target = True
                        self.target_data = target_info['target_data']
                        self.target_test_data = target_info.get('target_test_data', None)
                        self.input_test_data = target_info.get('input_test_data', None)
                        raw_data = target_info['input_data']
                        print(f"ターゲット予測モード: 入力{raw_data.shape} → ターゲット{self.target_data.shape}")
                    else:
                        raise DataLoaderError(
                            f"ターゲット予測モードが指定されましたが、ターゲットデータが見つかりません。\n"
                            f"利用可能なキー: {list(data.keys())}\n"
                            f"期待されるキー: train_targets, y_train, target_train など"
                        )
                else:
                    # 再構成モード：ターゲットデータを無視
                    self.has_target = False
                    self.target_data = None
                    self.target_test_data = None
                    self.input_test_data = None
                    print(f"再構成モード: ターゲットデータは使用しません")

                    # 柔軟なキー探索: 優先度順で適切なデータを自動選択（画像データ対応）
                    candidate_keys = ['Y', 'X', 'data', 'arr_0', 'train_obs', 'test_obs']
                    raw_data = None

                    # 優先度順でキーを探索
                    for key in candidate_keys:
                        if key in data:
                            candidate = data[key]
                            # 2次元時系列データまたは4次元画像データかチェック
                            if ((candidate.ndim == 2 and candidate.shape[0] > 1) or
                                (candidate.ndim == 4 and candidate.shape[0] > 1)):
                                raw_data = candidate
                                print(f"npzファイルからキー '{key}' を使用: shape={candidate.shape}")
                                break

                    # 優先キーがない場合、利用可能な全キーから最適なものを選択
                    if raw_data is None:
                        available_keys = list(data.keys())
                        for key in available_keys:
                            candidate = data[key]
                            # 2次元時系列または4次元画像データで時系列として妥当なサイズ
                            if (hasattr(candidate, 'ndim') and
                                ((candidate.ndim == 2 and candidate.shape[0] > 1 and candidate.shape[1] > 0) or
                                 (candidate.ndim == 4 and candidate.shape[0] > 1))):
                                raw_data = candidate
                                print(f"npzファイルから推定キー '{key}' を使用: shape={candidate.shape}")
                                break

                    # それでも見つからない場合はエラー
                    if raw_data is None:
                        available_info = []
                        for key in data.keys():
                            try:
                                shape = data[key].shape if hasattr(data[key], 'shape') else 'scalar'
                                dtype = data[key].dtype if hasattr(data[key], 'dtype') else type(data[key])
                                available_info.append(f"'{key}': shape={shape}, dtype={dtype}")
                            except:
                                available_info.append(f"'{key}': (読み込み不可)")

                        raise DataLoaderError(
                            f"npzファイルに適切なデータが見つかりません。\n"
                            f"利用可能なデータ: {', '.join(available_info)}\n"
                            f"期待される形式: (T, d) の時系列データ または (T, H, W, C) の画像データ"
                        )
                    
            elif ext == ".npy":
                raw_data = np.load(self.data_path)

                # npyファイルの柔軟な形状対応
                if raw_data.ndim == 1:
                    # 1次元の場合は単変量時系列として扱う
                    raw_data = raw_data.reshape(-1, 1)
                    print(f"npyファイル: 1次元データを2次元に変換 shape={raw_data.shape}")
                elif raw_data.ndim > 2:
                    # 3次元以上の場合は最初の2次元を使用
                    original_shape = raw_data.shape
                    raw_data = raw_data.reshape(raw_data.shape[0], -1)
                    print(f"npyファイル: {original_shape} → {raw_data.shape} に変換")
                elif raw_data.ndim == 2:
                    print(f"npyファイル: 2次元データを使用 shape={raw_data.shape}")
                else:
                    raise DataLoaderError(f"npyファイルのデータが0次元です: shape={raw_data.shape}")
                
            elif ext == ".csv":
                df = pd.read_csv(self.data_path, index_col=0 if 'time' in pd.read_csv(self.data_path, nrows=1).columns else None)
                raw_data = df.values
                self._csv_feature_names = df.columns.tolist()
                
            elif ext == ".json":
                with open(self.data_path, 'r') as f:
                    json_data = json.load(f)
                if 'data' in json_data:
                    raw_data = np.array(json_data['data'])
                    self._json_metadata = {k: v for k, v in json_data.items() if k != 'data'}
                else:
                    raw_data = np.array(json_data)
                    
            else:
                raise DataLoaderError(f"対応していないファイル形式: {ext}. 対応形式: .npz, .npy, .csv, .json")
                
        except Exception as e:
            raise DataLoaderError(f"データ読み込みエラー ({self.data_path}): {e}")
        
        return raw_data
    
    def _validate_and_clean(self, data: np.ndarray, handle_missing: str) -> np.ndarray:
        """データ検証・クリーニング（画像データ対応）"""
        # 形状チェック
        if data.ndim == 1:
            warnings.warn("1次元データを2次元に変換します")
            data = data.reshape(-1, 1)
        elif data.ndim == 2:
            # 標準的な時系列データ (T, d)
            pass
        elif data.ndim == 4:
            # 画像データ (T, H, W, C) - RKN画像データ対応
            T, H, W, C = data.shape
            print(f"画像データ検出: {data.shape} (T={T}, H={H}, W={W}, C={C})")
            # 画像データはそのまま保持（フラット化しない）
        else:
            raise DataLoaderError(f"サポートしていないデータ形状: {data.shape}. サポート形状: (T,), (T, d), (T, H, W, C)")

        # データ長チェック
        T = data.shape[0]  # 最初の次元は常に時系列長
        if T < 10:
            warnings.warn(f"データ長が短すぎます: T={T}. 最低10以上推奨")
        
        # 欠損値チェック（画像データ対応）
        missing_mask = np.isnan(data) | np.isinf(data)
        missing_ratio = missing_mask.sum() / data.size

        if missing_ratio > 0:
            warnings.warn(f"欠損値を検出: {missing_ratio:.1%}")

            if handle_missing == "interpolate":
                if data.ndim == 2:
                    # 2次元時系列データの線形補間
                    d = data.shape[1]
                    for j in range(d):
                        col_data = data[:, j]
                        missing_idx = np.isnan(col_data) | np.isinf(col_data)
                        if missing_idx.any():
                            valid_idx = ~missing_idx
                            if valid_idx.sum() > 1:
                                col_data[missing_idx] = np.interp(
                                    np.where(missing_idx)[0],
                                    np.where(valid_idx)[0],
                                    col_data[valid_idx]
                                )
                            else:
                                col_data[missing_idx] = 0.0
                elif data.ndim == 4:
                    # 4次元画像データの場合は0で置換（uint8画像データは通常欠損値なし）
                    print("画像データの欠損値を0で置換")
                    data = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)

            elif handle_missing == "forward_fill":
                if data.ndim == 2:
                    data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values
                else:
                    # 画像データの場合は0で置換
                    data = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)

            elif handle_missing == "remove":
                if data.ndim == 2:
                    valid_rows = ~missing_mask.any(axis=1)
                    data = data[valid_rows]
                    warnings.warn(f"欠損行を削除: {T} -> {data.shape[0]} 行")
                else:
                    # 画像データの場合は削除ではなく置換
                    data = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)

            # 無限大・NaNの最終チェック
            if np.isnan(data).any() or np.isinf(data).any():
                warnings.warn("欠損値処理後にもNaN/Infが残存。値で置換します。")
                if data.ndim == 4:  # 画像データ
                    data = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)
                else:  # 時系列データ
                    data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.missing_ratio = missing_ratio
        return data
    
    def _normalize_data(self, data: np.ndarray, method: str) -> Tuple[np.ndarray, Optional[object]]:
        """データ正規化（画像データ対応）"""
        if method == "none":
            return data, None

        elif method == "standard":
            # 標準化 (平均0, 標準偏差1)
            if data.ndim == 4:  # 画像データの場合は全画素で標準化
                data_flat = data.reshape(-1, data.shape[-1])  # (T*H*W, C)
                scaler = StandardScaler()
                normalized_flat = scaler.fit_transform(data_flat)
                normalized = normalized_flat.reshape(data.shape)  # 元の形状に戻す
            else:
                scaler = StandardScaler()
                normalized = scaler.fit_transform(data)

        elif method == "minmax":
            # Min-Max正規化 (0-1範囲)
            if data.ndim == 4:  # 画像データの場合
                data_flat = data.reshape(-1, data.shape[-1])
                scaler = MinMaxScaler()
                normalized_flat = scaler.fit_transform(data_flat)
                normalized = normalized_flat.reshape(data.shape)
            else:
                scaler = MinMaxScaler()
                normalized = scaler.fit_transform(data)

        elif method == "unit_scale":
            # Unit Scale正規化: [0, 255] → [0, 1] (画像用)
            print(f"Unit Scale正規化: {data.dtype} [{data.min()}, {data.max()}] → [0, 1]")
            if data.dtype == np.uint8:
                # uint8画像データ: [0, 255] → [0, 1]
                normalized = data.astype(np.float32) / 255.0
            else:
                # その他のデータ: 既に[0, 1]範囲と仮定
                normalized = data.astype(np.float32)
            scaler = None  # unit_scaleは逆変換用scalerを保持しない

        else:
            raise DataLoaderError(f"対応していない正規化方法: {method}. 使用可能: 'standard', 'minmax', 'unit_scale', 'none'")

        return normalized, scaler
    
    def _split_time_series(self, data: np.ndarray, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, np.ndarray]:
        """時系列順分割"""
        T = data.shape[0]
        
        # 分割点計算
        train_end = int(train_ratio * T)
        val_end = int((train_ratio + val_ratio) * T)
        
        splits = {
            "train": data[:train_end],
            "val": data[train_end:val_end],
            "test": data[val_end:]
        }
        
        # 分割インデックス記録
        self.split_indices = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, T)
        }
        
        return splits
    
    def _create_metadata(self, raw_data: np.ndarray, split_data: Dict[str, np.ndarray], feature_names: Optional[List[str]]):
        """メタデータ作成（ターゲットデータ対応）"""
        # 特徴量名の決定（画像データ対応）
        if feature_names:
            final_feature_names = feature_names
        elif hasattr(self, '_csv_feature_names'):
            final_feature_names = self._csv_feature_names
        else:
            if hasattr(raw_data, 'shape'):
                if raw_data.ndim == 2:
                    # 時系列データ: (T, d)
                    final_feature_names = [f"feature_{i}" for i in range(raw_data.shape[1])]
                elif raw_data.ndim == 4:
                    # 画像データ: (T, H, W, C)
                    T, H, W, C = raw_data.shape
                    final_feature_names = [f"image_pixel_{H}x{W}x{C}"]
                else:
                    final_feature_names = ["feature_0"]
            else:
                final_feature_names = ["feature_0"]

        # ターゲットデータ情報の準備
        has_target = getattr(self, 'has_target', False)
        target_shape = None
        target_feature_names = None
        target_dtype = None

        if has_target and hasattr(self, 'target_data') and self.target_data is not None:
            target_shape = self.target_data.shape
            target_dtype = str(self.target_data.dtype)
            target_feature_names = [f"target_{i}" for i in range(self.target_data.shape[1])]

        self.metadata = DataMetadata(
            original_shape=raw_data.shape,
            feature_names=final_feature_names,
            time_index=None,  # TODO: 必要に応じて実装
            sampling_rate=None,
            missing_ratio=self.missing_ratio,
            data_source=str(self.data_path),
            normalization_method=self.normalization,
            train_indices=self.split_indices["train"],
            val_indices=self.split_indices["val"],
            test_indices=self.split_indices["test"],
            # ターゲットデータ情報追加
            has_target_data=has_target,
            target_shape=target_shape,
            target_feature_names=target_feature_names,
            target_dtype=target_dtype
        )
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """データ取得（ターゲットデータ対応）"""
        sample = self.data[idx]  # shape: (d,) or (H, W, C)

        if hasattr(self, 'has_target') and self.has_target:
            # ターゲットデータありの場合: (入力, ターゲット) のタプルを返す
            if hasattr(self, 'target_data') and self.target_data is not None:
                # 分割されたターゲットデータから対応するインデックスを取得
                target_sample = self._get_target_for_split(idx)
                return (torch.from_numpy(sample).float(),
                       torch.from_numpy(target_sample).float())
            else:
                return torch.from_numpy(sample).float()
        else:
            # 従来通り：入力データのみ
            return torch.from_numpy(sample).float()

    def _get_target_for_split(self, idx: int) -> np.ndarray:
        """分割に対応するターゲットデータを取得"""
        if not hasattr(self, 'target_data') or self.target_data is None:
            raise ValueError("ターゲットデータが設定されていません")

        # 分割インデックスに基づいてターゲットデータをスライス
        split_start, split_end = self.split_indices[self.split]
        target_split_data = self.target_data[split_start:split_end]

        return target_split_data[idx]
    
    def get_full_data(self) -> torch.Tensor:
        """分割データ全体を取得"""
        return torch.from_numpy(self.data).float()
    
    def inverse_transform(self, normalized_data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """正規化逆変換"""
        if self.scaler is None:
            return normalized_data
            
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.cpu().numpy()
            
        return self.scaler.inverse_transform(normalized_data)


def load_experimental_data(
    data_path: str,
    config: Optional[Dict[str, Any]] = None,
    return_loaders: bool = False
) -> Union[Dict[str, torch.Tensor], Dict[str, DataLoader]]:
    """
    実験用データ読み込み・前処理パイプライン
    
    Args:
        data_path: データファイルパス
        config: 設定辞書（train_ratio, val_ratio, normalization等）
        return_loaders: DataLoaderを返すかTensorを返すか
        
    Returns:
        データ辞書またはDataLoader辞書
    """
    # デフォルト設定
    default_config = {
        'train_ratio': 0.7,
        'val_ratio': 0.2,
        'test_ratio': 0.1,
        'normalization': 'standard',
        'handle_missing': 'interpolate',
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': True
    }
    
    if config:
        default_config.update(config)
    
    # データセット作成
    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = UniversalTimeSeriesDataset(
            data_path=data_path,
            split=split,
            train_ratio=default_config['train_ratio'],
            val_ratio=default_config['val_ratio'],
            test_ratio=default_config['test_ratio'],
            normalization=default_config['normalization'],
            handle_missing=default_config['handle_missing']
        )
    
    if return_loaders:
        # DataLoader作成
        loaders = {}
        for split, dataset in datasets.items():
            # 時系列データはシャッフルしない
            loaders[split] = DataLoader(
                dataset,
                batch_size=default_config['batch_size'],
                shuffle=False,  # 時系列順保持
                num_workers=default_config['num_workers'],
                pin_memory=default_config['pin_memory']
            )
        return loaders
    else:
        # Tensor辞書作成
        data_dict = {split: dataset.get_full_data() for split, dataset in datasets.items()}
        # メタデータも含める
        data_dict['metadata'] = datasets['train'].metadata
        return data_dict


def create_data_loader_from_tensor(
    tensor: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = False,
    **dataloader_kwargs
) -> DataLoader:
    """
    Tensorから直接DataLoaderを作成
    
    Args:
        tensor: データテンソル (T, d)
        batch_size: バッチサイズ
        shuffle: シャッフルするか
        **dataloader_kwargs: DataLoader追加引数
        
    Returns:
        DataLoader
    """
    
    class TensorDataset(Dataset):
        def __init__(self, data: torch.Tensor):
            self.data = data
            
        def __len__(self) -> int:
            return self.data.shape[0]
            
        def __getitem__(self, idx: int) -> torch.Tensor:
            return self.data[idx]
    
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs)


# 既存互換性のための関数
def build_dataloaders(
    file_path: str,
    batch_size: int,
    split: str = "train",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    既存コードとの互換性のためのラッパー関数
    """
    dataset = UniversalTimeSeriesDataset(
        data_path=file_path,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


class QuadImageDataset(Dataset):
    """
    クアッドコプター画像データセット（画像再構成専用）
    画像のみのnpzファイル対応: 画像自己再構成学習

    データ形状:
    - 画像: (T, H, W, C) = (1500, 48, 48, 1)

    用途:
    - image_reconstruction: 画像 → 潜在表現 → 画像復元
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        image_normalization: str = "unit_scale",  # [0,255] → [0,1]
        **kwargs
    ):
        """
        Args:
            data_path: 画像npzファイルパス (train_obs, test_obs含む)
            split: "train", "val", "test"
            train_ratio: 訓練データ比率
            val_ratio: 検証データ比率
            test_ratio: テストデータ比率
            image_normalization: 画像正規化方法 ("unit_scale", "standard", "none")
        """
        super().__init__()

        # パラメータ検証
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise DataLoaderError(f"分割比率の合計が1.0でありません: {train_ratio + val_ratio + test_ratio}")

        if split not in ["train", "val", "test"]:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

        self.data_path = Path(data_path)
        self.split = split
        self.image_normalization = image_normalization

        # quad*.npz読み込み
        data = np.load(data_path)

        # データ選択（train/test）- 画像のみデータセット対応
        if split in ["train", "val"]:
            # trainデータを使用し、内部でtrain/val分割
            self.images = data['train_obs']      # (T, H, W, C) = (1500, 48, 48, 1)
        else:  # test
            self.images = data['test_obs']       # (T, H, W, C) = (1500, 48, 48, 1)

        # print(f"[DEBUG] データ読み込み完了: images={self.images.shape}")  # Resolved in Step 7

        # 時系列分割（train内でさらにtrain/val分割）
        T = len(self.images)
        if split == "train":
            start_idx = 0
            end_idx = int(T * train_ratio)
        elif split == "val":
            start_idx = int(T * train_ratio)
            end_idx = int(T * (train_ratio + val_ratio))
        else:  # test
            start_idx = 0  # testデータは全体を使用
            end_idx = T

        self.images = self.images[start_idx:end_idx]

        # print(f"[DEBUG] {split}分割後: images={self.images.shape}")  # Resolved in Step 7

        # 正規化
        self._normalize_data()

        # メタデータ作成
        self.metadata = DataMetadata(
            original_shape=(T, 48, 48, 1),
            feature_names=['image_features'],
            time_index=None,
            sampling_rate=None,
            missing_ratio=0.0,
            data_source=str(data_path),
            normalization_method=f"image:{image_normalization}",
            train_indices=(0, int(T * train_ratio)),
            val_indices=(int(T * train_ratio), int(T * (train_ratio + val_ratio))),
            test_indices=(int(T * (train_ratio + val_ratio)), T)
        )

    def _normalize_data(self):
        """データ正規化"""
        # 画像正規化
        if self.image_normalization == "unit_scale":
            # [0, 255] → [0, 1]
            self.images = self.images.astype(np.float32) / 255.0
        elif self.image_normalization == "standard":
            # 標準化
            mean = self.images.mean()
            std = self.images.std()
            self.images = (self.images - mean) / (std + 1e-8)
        elif self.image_normalization == "none":
            self.images = self.images.astype(np.float32)

        # 画像のみデータセットではターゲット正規化不要

    def __getitem__(self, idx):
        """
        データ取得: 画像再構成用

        Returns:
            (image, image) - 自己再構成ペア
        """
        image = torch.FloatTensor(self.images[idx])      # (H, W, C) = (48, 48, 1)

        # 画像再構成: 入力と出力が同じ
        return image, image

    def __len__(self):
        return len(self.images)

    def get_full_data(self):
        """分割データ全体を取得（時系列順）"""
        images = torch.FloatTensor(self.images)    # (T, H, W, C)
        return images





def load_experimental_data_with_architecture(
    data_path: str,
    config: Dict[str, Any],
    split: str = "train",
    return_dataloaders: bool = False,
    experiment_mode: Optional[str] = None  # 手動切り替え対応
) -> Union[Dataset, Dict[str, Dataset], DataLoader, Dict[str, DataLoader]]:
    """
    アーキテクチャに基づく統一データローダー関数（後方互換性保証）

    Args:
        data_path: データファイルパス
        config: 実験設定（model.encoder.type で判定）
        split: データ分割 ("train" | "val" | "test" | "all")
        return_dataloaders: DataLoaderを返すかDatasetを返すか
        experiment_mode: 実験モード ("reconstruction" | "target_prediction")
                        Noneの場合はconfig.experiment.modeから自動取得

    Returns:
        Dataset/DataLoader: 適切なデータセット/ローダークラス
    """
    # アーキテクチャタイプ判定
    encoder_type = config.get('model', {}).get('encoder', {}).get('type', 'time_invariant')
    data_config = config.get('data', {})

    # 実験モード判定（設定ファイルから自動取得 or 手動指定）
    if experiment_mode is None:
        experiment_mode = config.get('experiment', {}).get('mode', 'reconstruction')

    print(f"📋 実験モード: {experiment_mode} (encoder: {encoder_type})")

    if encoder_type == "rkn":
        # ★画像データローダー（UniversalTimeSeriesDataset使用で統一）

        # データセット・DataLoader・モデル パラメータ分離
        # 問題: config.data には Dataset用、DataLoader用、モデル用パラメータが混在
        # 原因: UniversalTimeSeriesDataset.__init__() が受け取らないパラメータで TypeError 発生
        # 解決: Dataset作成時は Dataset用パラメータのみを渡し、他は各用途で後使用
        # 除外対象:
        #   - DataLoader用: batch_size, num_workers, pin_memory
        #   - モデル用: image_shape, target_shape (モデル構築時に参照)
        dataset_params = {k: v for k, v in data_config.items()
                         if k not in ['batch_size', 'num_workers', 'pin_memory',
                                     'image_shape', 'target_shape']}

        if split == "all":
            # 全分割を返す
            datasets = {}
            for s in ["train", "val", "test"]:
                datasets[s] = UniversalTimeSeriesDataset(
                    data_path=data_path,
                    split=s,
                    experiment_mode=experiment_mode,  # 実験モードを渡す
                    **dataset_params  # Dataset用パラメータのみ渡す（batch_size等を除外）
                )

            if return_dataloaders:
                # DataLoader辞書を返す
                loaders = {}
                batch_size = data_config.get('batch_size', 16)  # 画像用に小さめ
                for s, dataset in datasets.items():
                    loaders[s] = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,  # 時系列順序保持のため全分割でシャッフル無効
                        num_workers=data_config.get('num_workers', 4),
                        pin_memory=data_config.get('pin_memory', True)
                    )
                return loaders
            else:
                return datasets
        else:
            # 単一分割を返す
            dataset = UniversalTimeSeriesDataset(
                data_path=data_path,
                split=split,
                experiment_mode=experiment_mode,  # 実験モードを渡す
                **dataset_params  # Dataset用パラメータのみ渡す（batch_size等を除外）
            )

            if return_dataloaders:
                batch_size = data_config.get('batch_size', 16)
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,  # 時系列順序保持のため全分割でシャッフル無効
                    num_workers=data_config.get('num_workers', 4),
                    pin_memory=data_config.get('pin_memory', True)
                )
            else:
                return dataset

    elif encoder_type in ["time_invariant", "tcn"]:
        # 既存の時系列データローダー（後方互換性保証）

        # データセット・DataLoader・モデル パラメータ分離（時系列データ用も同様の処理）
        dataset_params = {k: v for k, v in data_config.items()
                         if k not in ['batch_size', 'num_workers', 'pin_memory',
                                     'image_shape', 'target_shape']}

        if split == "all":
            datasets = {}
            for s in ["train", "val", "test"]:
                datasets[s] = UniversalTimeSeriesDataset(
                    data_path=data_path,
                    split=s,
                    experiment_mode=experiment_mode,  # 実験モードを渡す
                    **dataset_params  # Dataset用パラメータのみ渡す
                )

            if return_dataloaders:
                loaders = {}
                batch_size = data_config.get('batch_size', 32)
                for s, dataset in datasets.items():
                    loaders[s] = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,  # 時系列順序保持のため全分割でシャッフル無効
                        num_workers=data_config.get('num_workers', 4),
                        pin_memory=data_config.get('pin_memory', True)
                    )
                return loaders
            else:
                return datasets
        else:
            dataset = UniversalTimeSeriesDataset(
                data_path=data_path,
                split=split,
                experiment_mode=experiment_mode,  # 実験モードを渡す
                **dataset_params  # Dataset用パラメータのみ渡す
            )

            if return_dataloaders:
                batch_size = data_config.get('batch_size', 32)
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,  # 時系列順序保持のため全分割でシャッフル無効
                    num_workers=data_config.get('num_workers', 4),
                    pin_memory=data_config.get('pin_memory', True)
                )
            else:
                return dataset
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# 既存関数の後方互換性ラッパー
def load_experimental_data(
    data_path: str,
    config: Optional[Dict[str, Any]] = None,
    split: str = "all",
    return_dataloaders: bool = False
) -> Union[Dict[str, torch.Tensor], Dict[str, Dataset], Dict[str, DataLoader]]:
    """
    後方互換性保証のための既存関数ラッパー

    Args:
        data_path: データファイルパス
        config: 実験設定（Noneの場合は time_invariant と仮定）
        split: データ分割
        return_dataloaders: DataLoaderを返すか

    Returns:
        データ辞書: 既存形式との互換性を保証
    """
    if config is None:
        # 既存の動作: time_invariant前提
        config = {
            'model': {'encoder': {'type': 'time_invariant'}},
            'data': {'batch_size': 32}
        }

    # 新しい統一関数を使用
    result = load_experimental_data_with_architecture(
        data_path=data_path,
        config=config,
        split=split,
        return_dataloaders=return_dataloaders
    )

    # 既存形式への変換（必要に応じて）
    if not return_dataloaders and split == "all" and isinstance(result, dict):
        # Dataset辞書 → Tensor辞書変換（既存互換性）
        if all(hasattr(dataset, 'get_full_data') for dataset in result.values()):
            tensor_dict = {}
            for s, dataset in result.items():
                tensor_dict[s] = dataset.get_full_data()
            # メタデータも含める
            tensor_dict['metadata'] = result['train'].metadata
            return tensor_dict

    return result


if __name__ == "__main__":
    # 使用例とテスト
    print("統一データローダーのテスト")
    
    # テストデータ作成
    test_data = np.random.randn(100, 5)
    test_path = "test_data.npy"
    np.save(test_path, test_data)
    
    try:
        # データ読み込みテスト
        data_dict = load_experimental_data(test_path)
        
        print("データ分割結果:")
        for split, data in data_dict.items():
            if isinstance(data, torch.Tensor):
                print(f"  {split}: {data.shape}")
            else:
                print(f"  {split}: {type(data)}")
        
        print("\n統一データローダーのテストが完了しました")
        
    finally:
        # テストファイル削除
        if os.path.exists(test_path):
            os.remove(test_path)