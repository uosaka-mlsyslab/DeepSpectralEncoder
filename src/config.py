# src/config.py

import argparse       # コマンドライン引数の処理を行うため
import yaml           # YAML ファイルを読み込むため
from types import SimpleNamespace  # ネストされた辞書をドットでアクセスできるオブジェクトに変えるため

def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """
    再帰的にネストされた dict を SimpleNamespace に変換するヘルパー関数

    例えば:
      {"model": {"encoder": {"type":"mlp", "input_dim":10}}, "training": {"lr":0.001}}
    を渡すと、
      Namespace(
        model=Namespace(encoder=Namespace(type="mlp", input_dim=10)),
        training=Namespace(lr=0.001)
      )
    のようなオブジェクトを返す。
    """
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            # 値がさらに dict の場合は再帰呼び出しして深い階層も Namespace 化する
            setattr(ns, key, _dict_to_namespace(value))
        else:
            # dict でなければ、そのまま属性にセット
            setattr(ns, key, value)
    return ns

def load_cfg() -> SimpleNamespace:
    """
    1. argparse でコマンドラインの --config を読み取る
    2. 指定された YAML ファイルを読み込んで辞書化
    3. その辞書を _dict_to_namespace で Namespace に変換して返却
    4. visualization.output_dir がなければデフォルト値をセット

    使い方:
      python train.py --config configs/model.yaml
      のように起動して、train.py の中で次のように書く:
        cfg = load_cfg()
        # いまや cfg.model.encoder.type などで読み取れるようになる
    """
    # (1) コマンドライン引数定義
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    # ここで、args.config に例えば "configs/model.yaml" の文字列が入っている

    # (2) YAML ファイルを読み込んで辞書化
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    # たとえば cfg_dict = {"model": {...}, "ssm": {...}, ...} のような dict になる

    # (3) 辞書を Namespace に変換
    cfg = _dict_to_namespace(cfg_dict)
    # 例えば
    # cfg.model.encoder.type  というドットアクセスが可能に

    # (4) visualization に output_dir がなければデフォルトをセット
    if not hasattr(cfg, "visualization"):
        # もし設定ファイルに visualization セクション自体がなければ空の Namespace を作っておく
        cfg.visualization = _dict_to_namespace({})
    if not hasattr(cfg.visualization, "output_dir"):
        # visualization.output_dir が指定されていなければ "results/figs" をデフォルトに
        cfg.visualization.output_dir = "results/figs"

    return cfg
