# src/models/decoder.py (修正版)

import pkgutil
import importlib
from pathlib import Path
from . import architectures

# Scan architectures/ for any <name>Decoder and <name>_targetDecoder classes
_DECODERS: dict[str, type] = {}
_TARGET_DECODERS: dict[str, type] = {}
pkg_path = Path(architectures.__file__).parent

for module_info in pkgutil.iter_modules([str(pkg_path)]):
    name = module_info.name              # e.g. "rkn", "tcn", "time_invariant"
    module = importlib.import_module(f"{architectures.__package__}.{name}")

    # 通常のデコーダクラス検出: <name>Decoder
    cls_name = name + "Decoder"
    if hasattr(module, cls_name):
        _DECODERS[name] = getattr(module, cls_name)

    # ターゲット予測用デコーダクラス検出: <name>_targetDecoder
    target_cls_name = name + "_targetDecoder"
    if hasattr(module, target_cls_name):
        _TARGET_DECODERS[name] = getattr(module, target_cls_name)

def build_decoder(cfg, experiment_mode=None):
    """
    Factory for decoders with experiment mode support.
    Args:
      cfg: dict / Namespace with
        - type: key in _DECODERS (e.g. "tcn", "_mlp", "rkn")
        - other keys: kwargs for that decoder class
      experiment_mode: Optional str ("target_prediction" | "reconstruction")
                      Overrides decoder selection for target prediction
    Returns:
      An instance of the chosen Decoder class.
    """
    # cfg が dict ならキーアクセス、Namespace なら属性アクセスで 'type' を取得
    if isinstance(cfg, dict):
        cfg_type = cfg.get('type')
    else:
        cfg_type = getattr(cfg, 'type', None)
    if cfg_type is None:
        raise ValueError("decoder config must include 'type' key or attribute")

    # **新機能**: experiment_mode対応 - ターゲット予測時に<type>_targetDecoderを選択
    if experiment_mode == "target_prediction":
        if cfg_type in _TARGET_DECODERS:
            # <type>_targetDecoderが存在する場合は使用
            cls = _TARGET_DECODERS[cfg_type]
            print(f"ターゲット予測モード: {cfg_type}_targetDecoderを使用")
        else:
            # fallback: 通常のDecoderを使用（警告表示）
            if cfg_type in _DECODERS:
                cls = _DECODERS[cfg_type]
                print(f"警告: {cfg_type}_targetDecoderが見つかりません。通常の{cfg_type}Decoderを使用します")
            else:
                available_types = list(_DECODERS.keys())
                available_target_types = list(_TARGET_DECODERS.keys())
                raise ValueError(f"Unknown decoder type: {cfg_type}. Available types: {available_types}, Available target types: {available_target_types}")

        # Dictionary or Namespace に対応する
        if isinstance(cfg, dict):
            init_args = {k: v for k, v in cfg.items() if k != "type"}
        else:
            init_args = {k: getattr(cfg, k) for k in vars(cfg) if k != "type"}

        return cls(**init_args)
    
    # **修正**: tcn デコーダのサポート確認
    if cfg_type == "tcn" and "tcn" not in _DECODERS:
        raise ValueError("tcnDecoder が見つかりません。tcn.py が正しくインポートされているか確認してください")
    
    try:
        cls = _DECODERS[cfg_type]
    except KeyError:
        available_types = list(_DECODERS.keys())
        raise ValueError(f"Unknown decoder type: {cfg_type}. Available types: {available_types}")
    
    # Dictionary or Namespace に対応する
    # 残りのキーを初期化引数に
    if isinstance(cfg, dict):
        init_args = {k: v for k, v in cfg.items() if k != "type"}
    else:
        init_args = {k: getattr(cfg, k) for k in vars(cfg) if k != "type"}
    
    # **新機能**: TCNデコーダの自動設定
    if cfg_type == "tcn":
        if "output_dim" not in init_args:
            raise ValueError("tcnDecoder には output_dim の指定が必要です")
        print(f"[INFO] TCNデコーダを output_dim={init_args['output_dim']} で初期化します")
    
    return cls(**init_args)