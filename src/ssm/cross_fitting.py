# src/ssm/cross_fitting.py

import torch
import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any


class CrossFittingManager:
    """
    時系列データのクロスフィッティング管理クラス
    
    時系列を連続ブロックに分割し、各ブロックに対して
    out-of-fold推定を行うことで時間リークとself-fittingを防ぐ。
    
    資料の式(17)-(20)の実装に対応。
    """
    
    def __init__(self, T: int, n_blocks: int = 5, min_block_size: int = 10):
        """
        Args:
            T: 時系列長
            n_blocks: 分割ブロック数 K
            min_block_size: 最小ブロックサイズ（数値安定性のため）
        """
        self.T = T
        self.n_blocks = min(n_blocks, T // min_block_size)  # ブロック数制限
        self.min_block_size = min_block_size
        self.blocks = self._create_contiguous_blocks()
        
    def _create_contiguous_blocks(self) -> List[List[int]]:
        """
        時系列を連続ブロック {B_k}^K_{k=1} に分割
        
        Returns:
            List[List[int]]: 各ブロックのインデックスリスト
        """
        block_size = self.T // self.n_blocks
        remainder = self.T % self.n_blocks
        
        blocks = []
        start_idx = 0
        
        for k in range(self.n_blocks):
            # 余りを最初のいくつかのブロックに分配
            current_size = block_size + (1 if k < remainder else 0)
            end_idx = start_idx + current_size
            
            blocks.append(list(range(start_idx, min(end_idx, self.T))))
            start_idx = end_idx
            
        return blocks
    
    def get_out_of_fold_indices(self, block_k: int) -> List[int]:
        """
        ブロックk用のout-of-fold インデックス I_{-k} を取得
        
        Args:
            block_k: ブロック番号 (0-indexed)
            
        Returns:
            List[int]: I_{-k} = {0,...,T} /  B_k
        """
        if not (0 <= block_k < self.n_blocks):
            raise ValueError(f"block_k must be in [0, {self.n_blocks}), got {block_k}")
            
        in_fold = set(self.blocks[block_k])
        return [t for t in range(self.T) if t not in in_fold]
    
    def get_block_indices(self, block_k: int) -> List[int]:
        """
        ブロックkのインデックス B_k を取得
        
        Args:
            block_k: ブロック番号 (0-indexed)
            
        Returns:
            List[int]: B_k
        """
        if not (0 <= block_k < self.n_blocks):
            raise ValueError(f"block_k must be in [0, {self.n_blocks}), got {block_k}")
            
        return self.blocks[block_k].copy()
    
    def get_block_info(self) -> Dict[str, Any]:
        """
        ブロック分割の情報を返す（デバッグ用）
        
        Returns:
            Dict: ブロック分割情報
        """
        return {
            'T': self.T,
            'n_blocks': self.n_blocks, 
            'block_sizes': [len(block) for block in self.blocks],
            'blocks': self.blocks.copy()
        }


class TwoStageCrossFitter:
    """
    2段階回帰のクロスフィッティング実行クラス
    
    回帰アルゴリズムは外部から注入し、
    このクラスは時間分割とout-of-fold戦略のみを管理する。
    """
    
    def __init__(self, cf_manager: CrossFittingManager):
        """
        Args:
            cf_manager: CrossFittingManager インスタンス
        """
        self.cf_manager = cf_manager
        
    def cross_fit_stage1(
        self, 
        X_input: torch.Tensor, 
        Y_target: torch.Tensor,
        stage1_estimator: Callable,
        **estimator_kwargs
    ) -> List[torch.Tensor]:
        """
        Stage-1のクロスフィッティング実行
        
        各ブロックkに対してout-of-fold データで V^{(-k)} を推定する。
        推定方法は stage1_estimator に委譲。
        
        Args:
            X_input: 説明変数 (T, d_X)
            Y_target: 目的変数 (T, d_Y) 
            stage1_estimator: Stage-1推定関数
                signature: V = estimator(X_oof, Y_oof, **kwargs)
            **estimator_kwargs: estimatorの追加引数
            
        Returns:
            List[torch.Tensor]: [V^{(-0)}, V^{(-1)}, ..., V^{(-K-1)}]
        """
        if X_input.size(0) != self.cf_manager.T or Y_target.size(0) != self.cf_manager.T:
            raise ValueError(f"Input size mismatch: X={X_input.size(0)}, Y={Y_target.size(0)}, T={self.cf_manager.T}")
        
        V_list = []
        
        for k in range(self.cf_manager.n_blocks):
            # Out-of-fold インデックス I_{-k}
            oof_indices = self.cf_manager.get_out_of_fold_indices(k)
            
            if len(oof_indices) < self.cf_manager.min_block_size:
                raise ValueError(f"Out-of-fold size {len(oof_indices)} too small for block {k}")
            
            # Out-of-fold データでStage-1推定
            X_oof = X_input[oof_indices]  # (|I_{-k}|, d_X)
            Y_oof = Y_target[oof_indices]  # (|I_{-k}|, d_Y)
            
            # 推定方法は外部関数に委譲
            V_k = stage1_estimator(X_oof, Y_oof, **estimator_kwargs)
            V_list.append(V_k)
            
        return V_list
        
    def compute_out_of_fold_features(
        self,
        X_input: torch.Tensor,
        V_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Out-of-fold 特徴量 H^{(cf)} を計算
        
        各ブロックkに対して H^{(cf)}_{B_k} = V^{(-k)} X_{B_k} を計算し、
        元の時刻順に再構成する。
        
        Args:
            X_input: 入力特徴量 (T, d_X)
            V_list: cross_fit_stage1で得られた V^{(-k)} のリスト
            
        Returns:
            torch.Tensor: H^{(cf)} (T, d_V) - 元の時刻順に並んだクロスフィット特徴量
        """
        if len(V_list) != self.cf_manager.n_blocks:
            raise ValueError(f"V_list length {len(V_list)} != n_blocks {self.cf_manager.n_blocks}")
        
        if X_input.size(0) != self.cf_manager.T:
            raise ValueError(f"Input size mismatch: X={X_input.size(0)}, T={self.cf_manager.T}")
        
        device = X_input.device
        d_V = V_list[0].size(0)  # V の出力次元
        H_cf = torch.zeros(self.cf_manager.T, d_V, device=device, dtype=X_input.dtype)
        
        for k in range(self.cf_manager.n_blocks):
            # ブロック k のインデックス
            block_indices = self.cf_manager.get_block_indices(k)
            
            # V^{(-k)} @ X_{B_k}^T -> (H^{(cf)}_{B_k})^T
            X_block = X_input[block_indices]  # (|B_k|, d_X)
            H_block = (V_list[k] @ X_block.T).T  # (|B_k|, d_V)
            
            # 元の時刻位置に配置
            H_cf[block_indices] = H_block
            
        return H_cf
        
    def cross_fit_stage2(
        self,
        H_cf: torch.Tensor,
        Y_target: torch.Tensor,
        stage2_estimator: Callable,
        detach_features: bool = True,
        **estimator_kwargs
    ) -> torch.Tensor:
        """
        Stage-2推定（読み出し）
        
        クロスフィットされた特徴量H_cfを用いて最終推定を行う。
        推定方法は stage2_estimator に委譲。
        
        Args:
            H_cf: Out-of-fold 特徴量 (T, d_V)
            Y_target: ターゲット (T, d_Y)
            stage2_estimator: Stage-2推定関数
                signature: U = estimator(H, Y, **kwargs)  
            detach_features: 勾配を切断するかどうか（self-fitting防止）
            **estimator_kwargs: estimatorの追加引数
            
        Returns:
            torch.Tensor: Stage-2推定結果 U
        """
        if H_cf.size(0) != Y_target.size(0):
            raise ValueError(f"Feature-target size mismatch: H_cf={H_cf.size(0)}, Y={Y_target.size(0)}")
        
        if detach_features:
            H_cf = H_cf.detach()  # 勾配リーク防止
            
        return stage2_estimator(H_cf, Y_target, **estimator_kwargs)

    def cross_fit_stage2_matrix(
        self,
        H_cf: torch.Tensor,
        M_target: torch.Tensor,
        stage2_estimator: Callable,
        detach_features: bool = True,
        **estimator_kwargs
    ) -> List[torch.Tensor]:
        """
        Stage-2推定（多変量読み出し行列）

        各ブロックに対してout-of-fold推定を行い、U_Bリストを返す。

        Args:
            H_cf: Out-of-fold 特徴量 (T, d_B)
            M_target: 多変量ターゲット (T, m)
            stage2_estimator: Stage-2推定関数
                signature: U_B = estimator(H, M, **kwargs) -> (d_B, m)
            detach_features: 勾配を切断するかどうか
            **estimator_kwargs: estimatorの追加引数

        Returns:
            List[torch.Tensor]: 各ブロック用のU_Bリスト [(d_B, m), ...]
        """
        if H_cf.size(0) != M_target.size(0):
            raise ValueError(f"Feature-target size mismatch: H_cf={H_cf.size(0)}, M={M_target.size(0)}")

        U_B_list = []

        for k in range(self.cf_manager.n_blocks):
            # out-of-fold インデックス
            oof_indices = self.cf_manager.get_out_of_fold_indices(k)

            # out-of-fold データで推定
            H_oof = H_cf[oof_indices]  # (|I_{-k}|, d_B)
            M_oof = M_target[oof_indices]  # (|I_{-k}|, m)

            if detach_features:
                H_oof = H_oof.detach()

            # U_B推定
            U_B_k = stage2_estimator(H_oof, M_oof, **estimator_kwargs)
            U_B_list.append(U_B_k)

        return U_B_list


class CrossFittingError(Exception):
    """クロスフィッティング処理中のエラー"""
    pass


def validate_cross_fitting_setup(
    T: int, 
    n_blocks: int, 
    min_samples_per_fold: int = 10
) -> Dict[str, Any]:
    """
    クロスフィッティング設定の妥当性検証
    
    Args:
        T: 時系列長
        n_blocks: ブロック数
        min_samples_per_fold: fold当たり最小サンプル数
        
    Returns:
        Dict: 検証結果とメタ情報
        
    Raises:
        CrossFittingError: 設定が不適切な場合
    """
    if T < n_blocks * min_samples_per_fold:
        raise CrossFittingError(
            f"時系列長 {T} が短すぎます。n_blocks={n_blocks}, "
            f"min_samples={min_samples_per_fold} には最低 {n_blocks * min_samples_per_fold} 必要"
        )
    
    avg_block_size = T // n_blocks
    min_oof_size = T - avg_block_size - (T % n_blocks)  # 最小out-of-fold size
    
    if min_oof_size < min_samples_per_fold:
        raise CrossFittingError(
            f"Out-of-fold サイズ {min_oof_size} が小さすぎます。"
            f"最低 {min_samples_per_fold} 必要"
        )
    
    return {
        'valid': True,
        'T': T,
        'n_blocks': n_blocks,
        'avg_block_size': avg_block_size,
        'min_oof_size': min_oof_size,
        'max_oof_size': T - (T // n_blocks)
    }