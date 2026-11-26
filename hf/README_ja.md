# PLaMo-3-Translate-0.8B-XDNA1-NPU-runtime（日本語ガイド）

**Built with PLaMo.**

[English (README.md)](README.md) | [日本語](README_ja.md)

> **非公式のコミュニティプロジェクトです。** Preferred Networks（PFN）社および Advanced Micro Devices（AMD）社とは
> 無関係であり、両社の承認・推奨・サポートを受けていません。PLaMo は PFN 社のモデルファミリーです。AMD、Ryzen、Radeon、XDNA は
> AMD 社の商標です。**現状のまま（"AS IS"）・無保証**で提供されます。

## このリポジトリについて

このリポジトリは、Windows 11 搭載の AMD Ryzen "Phoenix" PC（XDNA1 NPU＋Radeon iGPU）上で
[PLaMo 3 Translate 0.8B](https://huggingface.co/pfnet/plamo-3-translate-0.8b) を実行するための
**NPU ランタイムペイロード**を格納しています。統合コマンド `translate_long.ps1` により **最大 256k コンテキスト（入力上限
262,143 トークン）** を扱い、2,048 トークン以下の入力は自動で純粋 NPU の厳密な投機的デコード経路へ委譲します。

単独で動作する Hugging Face Transformers モデルではなく、量子化重みイメージ、埋め込みテーブル、事前計算トークナイザテーブル、
投機的デコード用ドラフタ（`PLaMo-D20-M2`）、事前生成 NPU 命令列、CPython 3.13 拡張モジュール（`pyxrt.pyd`）を含みます。

利用にあたっては、以下の GitHub リポジトリのコードとインストーラを使用してください:
<https://github.com/nilpe/plamo3-translate-xdna1>

`install.ps1` が固定リビジョンから必要なファイルをダウンロードし、GitHub 側の SSH 署名付き `SHA256SUMS` と全ファイルを照合します。
続けて `python install_long.py --with-model` を実行すると、GitHub 側同梱の長文ファイルを検証したうえで公式リポジトリ
`pfnet/plamo-3-translate-0.8b` から BF16 重み `model.safetensors`（1.78 GB）を直接取得し、256k Long モード（Radeon iGPU
プリフィル＋NPU W4 デコード）が有効になります。ここから手動ダウンロードする必要はありません。

ベースモデル `pfnet/plamo-3-translate-0.8b` 本体はここには含まれません（`install.ps1` がトークナイザ等 3.6 MB、`install_long.py
--with-model` が `model.safetensors` 1.78 GB を、いずれもアップストリームのリビジョン `8b4d42855622093e610ba9b90c2d2b59f8d5c15d` から
直接取得します）。

## 性能の概要

作者の 1 台の PC（AMD Ryzen 7 8840U［XDNA1 NPU＋Radeon 780M iGPU］、16 GB DDR5-5600、Windows 11 build 26300、
NPU ドライバ 32.0.20101.3760、電源モード「Default」、turbo / keep-warm オフ）での実測値です。

**256k Long モード（`translate_long.ps1`、Radeon 780M iGPU プリフィル＋XDNA1 NPU W4 デコード）:**

| 入力長 | プリフィル（iGPU） | デコード（NPU） |
|---|---:|---:|
| 8k トークン | 約 8 秒 | 21〜22 ms/トークン |
| 32k トークン | 約 22 秒 | 24 ms/トークン |
| 262k トークン（上限 262,143） | 約 5 分（熱条件により 4〜5.5 分） | 約 50 ms/トークン |

**短文・通常コンテキスト（≤ 2,048 トークン、自動委譲される NPU `w4-m2` 厳密投機経路、新規プロセス交互 24 ペア中央値）:**

| 項目 | 実測値 |
|---|---|
| プロセス起動 -> 投機的デコード準備完了（spec-ready） | 0.69 秒 |
| 初回トークン生成時間（TTFT、入力 27〜31 / 515 トークン） | 0.11 秒 / 1.43 秒 |
| デコード速度（デフォルト `w4-m2`、入力 515 -> 出力 595 トークン） | **13.9 ms/トークン**（約 72 tokens/s） |
| 段落全体の生成時間（入力 515 -> 出力 595 トークン） | 9.7 秒 |
| 日常短文 / Tatoeba / 百科事典調セット（`-Service` モード） | 14.2 / 14.1 / 14.7 ms/トークン（1.63 / 1.56 / 1.47 tok/cycle） |

測定条件や注意点の詳細は [GitHub README（日本語版）](https://github.com/nilpe/plamo3-translate-xdna1/blob/main/README_ja.md) を参照してください。

## 収録ファイル構成

アーカイブ化せず、個別ファイルとして配置しています。

| パス | 内容 | ファイル数 | 目安サイズ |
|---|---|---:|---:|
| `plamo3-runtime/` | W4 / W8 重みイメージ、埋め込みテーブル、基本グラフの事前生成命令列 | 64 | 2.19 GB |
| `plamo3-verify-*/`, `plamo3-r*/` | 投機検証・ドラフタ・ホイスティング済みデコード・プリフィルヘッド・高速トークナイザ関連ペイロード | 29 | 2.54 GB |
| `plamo3-drafter-port-v7/` | 投機的デコード用ドラフタパッケージ（`PLaMo-D20-M2`、派生モデル） | 173 | 483 MB |
| `plamo3-default/` | 封印済みマニフェスト、コンパクトカタログ、リリース証明書 | 10 | 100 MB |
| `plamo3-drafter-port-v7-attribution/` | 学習データのレコード単位クレジットファイル（`PLaMo-ATTRIBUTION-D20.jsonl.gz`、本文なし） | 2 | 41 MB |
| `plamo3-prefill-r46/` | デュアルパイプライン・プリフィルグラフ | 11 | 7.6 MB |
| `plamo3-npu-cache/` | コンパイル済み NPU カーネル / イメージキャッシュ | 3 | 0.6 MB |
| `xrt/` | `xrt/python/pyxrt.pyd`（AMD XRT 2.21.0 の CPython 3.13 バインディング、Apache-2.0、pybind11 BSD-3-Clause を内包）および `xrt/license/LICENSE` | 2 | 0.6 MB |
| **必須ランタイムペイロード小計** | `install.ps1` がデフォルトでダウンロードする対象 | **297** | **約 5.35 GB** |
| `gate-data/` *（任意）* | リリースゲート検証用のリファレンス入力（`install.ps1 -WithGateData`、ライセンス分類は GitHub 側 `GATE_DATA_LICENSES.md` を参照） | 57 | 359 MB |
| **リポジトリ合計** | 必須ペイロード＋任意 `gate-data/` | **354** | **約 5.71 GB** |

*（`plamo3-verify-*/` は `r37`, `r39`, `r41`, `r42`, `r44`、`plamo3-r*/` は `r61`, `r67`, `r73`, `r74`, `r77`, `r80`, `r83` を指します。256k Long モードのコード・カーネル約 4.5 MB は GitHub リポジトリ側へ直接収録しています。）*

正確なファイル一覧・サイズ・SHA-256 ハッシュは、GitHub リポジトリ側の `PAYLOAD-MANIFEST.json` と `SHA256SUMS` に
置いています（Hugging Face 側の変更によってインストーラの期待値が書き換えられないようにするため、意図的に分離しています）。

## ライセンス

PLaMo 3 Translate に由来するすべてのファイルは、株式会社 Preferred Networks（PFN）の
[PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-ja) に基づく
**派生モデル（DERIVED MODEL）**です。全文の写しを同梱しています:
[LICENSE-PLaMo-ja.md](LICENSE-PLaMo-ja.md)（日本語原文）・[LICENSE-PLaMo-en.md](LICENSE-PLaMo-en.md)（英語）。

- 個人・学術研究などの非営利利用は無償です。
- **商用利用**（企業内利用、製品・サービスへの組み込み、ホスティング、API 提供などを含む）を行う場合は、
  **利用者自身が開始前に PFN へ直接登録する必要があり**、売上規模条件の適用を受けます。本リポジトリが登録を代行したり
  商用利用権を付与したりすることはありません。
- `xrt/python/pyxrt.pyd` は Apache-2.0（AMD XRT）で、pybind11（BSD-3-Clause）を組み込んでいます。詳細は GitHub 側の
  `NOTICE` および `xrt/license/LICENSE` を参照してください。
- 投機的デコード用ドラフタ（`PLaMo-D20-M2`）は、パーミッシブなライセンスのソース文を PLaMo 3 Translate 自身で翻訳した出力のみを
  使って学習しています（詳細は GitHub 側 `NOTICE` を参照）:
  - Tatoeba の例文（[CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/)、Tatoeba 上で CC0 のものは CC0、<https://tatoeba.org>）。
    全 Tatoeba 例文の投稿者情報は `plamo3-drafter-port-v7-attribution/PLaMo-ATTRIBUTION-D20.jsonl.gz`（41 MB、本文なし）に記録しています。
  - OpenAssistant OASST1 / OASST2（Apache-2.0）。
  - e-Gov 法令検索の法令データ（著作権法第 13 条第 1 号、デジタル庁・政府標準利用規約 2.0）。
  - 青空文庫および Project Gutenberg のパブリックドメイン作品。

  ドラフタ自体は派生モデルとして PLaMo Community License の下で提供されます。

## 想定用途と制限事項

- 日本語 <-> 英語翻訳専用。Long モード（iGPU プリフィル＋NPU デコード）で最大 **262,143 入力トークン**、NPU 単体投機経路で合計
  2,048 トークンまで対応します。
- **長文におけるベースモデル由来の制約:** ベースモデル自体（GPU 上の公式 BF16 を含む）が約 2k トークン超の一括入力で繰り返しループや
  翻訳停止を起こしやすいため、2,048 トークン超の翻訳品質は未検証です。Long モードではデフォルトで繰り返し停止が有効です（GitHub 側 `LONG.md` 参照）。
- 動作確認は Ryzen 7 8840U（XDNA1 NPU＋Radeon 780M iGPU、NPU ドライバ 32.0.20101.3760、Windows 11 build 26300）のみです。
  他の XDNA1 機種は未検証、XDNA2 NPU には非対応です。
- PLaMo 3 Translate 本来の制約やリスク（誤訳、抜け漏れ、ハルシネーション等）はそのまま引き継がれます。誤りが問題となる用途では
  必ず出力を確認してください。

## プライバシー

ランタイムは外部と一切通信せず、テレメトリも送信しません。本リポジトリが Hugging Face 標準の統計以外の情報を収集することはありません。

## 来歴と厳密性

派生ファイルはすべて `pfnet/plamo-3-translate-0.8b`（リビジョン `8b4d42855622093e610ba9b90c2d2b59f8d5c15d`）から
生成しています。W4 出力は独立した演算順序一致 CPU リファレンスに対し、標準 100 行＋長文 64 行でビット単位の一致を検証済みです。
投機的デコードによる近似は一切なく、また 256k Long モードのデコード経路も位置 262,143 までビット単位の一致を検証済みです。

## 関連リポジトリ

- **コード・インストーラ（GitHub）**: <https://github.com/nilpe/plamo3-translate-xdna1>
- **GPU 向け MTP ドラフタ（Hugging Face）**:
  [nilpe0555/PLaMo-3-Translate-0.8B-MTP-drafter-GPU](https://huggingface.co/nilpe0555/PLaMo-3-Translate-0.8B-MTP-drafter-GPU)
- **GPU 向け DSpark ドラフタ（Hugging Face）**:
  [nilpe0555/PLaMo-3-Translate-0.8B-DSpark-drafter-GPU](https://huggingface.co/nilpe0555/PLaMo-3-Translate-0.8B-DSpark-drafter-GPU)
