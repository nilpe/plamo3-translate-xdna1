# AMD XDNA1 NPU 向け PLaMo 3 Translate 推論ランタイム（最大 256k コンテキスト対応）

**Built with PLaMo.**

[English (README.md)](README.md) | [日本語](README_ja.md)

Windows 11 搭載の AMD Ryzen "Phoenix" ノート PC（XDNA1 NPU＋Radeon iGPU）上で
[PLaMo 3 Translate 0.8B](https://huggingface.co/pfnet/plamo-3-translate-0.8b) を完全ローカル実行する日本語 <-> 英語翻訳ランタイムです。
統合コマンド `translate_long.ps1` ひとつで **最大 256k トークン（入力上限 262,143 トークン）** まで対応します：

- **長文入力（2,048 トークン超〜最大 262,143 トークン・Long モード）:** **Radeon iGPU プリフィル＋XDNA1 NPU W4 デコード**
  （8k: プリフィル約 8 秒 / 21〜22 ms/トークン、32k: 約 22 秒 / 24 ms/トークン、262k: 約 5 分 / 約 50 ms/トークン）。
- **短文・通常入力（合計 2,048 トークン以下・自動高速経路）:** iGPU を使わず、純粋 NPU による **厳密な投機的デコード**
  （`translate_plamo3.ps1` / `w4-m2`）へ自動委譲 — 起動 **0.69 秒**、**13.9 ms/トークン（約 72 tokens/s）**、単発短文 **約 1.4 秒**、
  出力は同じ量子化モデルの通常貪欲デコードとトークン単位で完全一致します。

測定条件や長文におけるベースモデル由来の注意点については [性能](#性能) および [LONG.md](LONG.md) を参照してください。

> **非公式のコミュニティプロジェクトです。** Preferred Networks（PFN）社および Advanced Micro Devices（AMD）社とは
> 無関係であり、両社の承認・推奨・サポートを受けていません。PLaMo は PFN 社のモデルファミリーです。AMD、Ryzen、Radeon、XDNA は
> AMD 社の商標です。本ソフトウェアは **PLaMo Community License** の下で PLaMo 3 Translate 0.8B を実行します
> （[ライセンス](#ライセンスインストール前に必ずお読みください) を参照）。**現状のまま（"AS IS"）・無保証**で提供されます。

## クイックスタート

```powershell
git clone -c core.autocrlf=false https://github.com/nilpe/plamo3-translate-xdna1.git C:\PLaMo3-NPU
cd C:\PLaMo3-NPU
.\install.ps1                        # ライセンス確認・NPU ランタイム導入（5.4 GB）・自己テストを実行
python install_long.py --with-model  # 256k Long モードを有効化（長文ファイル検証＋公式 BF16 重み 1.78 GB を HF から取得）

# 入力長を問わず 1 つのコマンドで翻訳:
.\translate_long.ps1 -InputFile report.txt -SourceLang ja -TargetLang en
.\translate_long.ps1 "今日はいい天気ですね。" -SourceLang ja -TargetLang en
.\translate_long.ps1 "Thank you for your help." -SourceLang en -TargetLang ja
```

- **NPU 単体のみ導入する場合（2,048 トークン以下専用）:** 長文入力を使わない場合や Radeon 780M 相当の iGPU がない場合は、
  `install_long.py --with-model` を省いて `.\translate_plamo3.ps1` をそのまま利用できます（DL 約 5.4 GB・ディスク約 7 GB）。
- **常駐サービスモード**（2,048 トークン以下向け。モデルをロードしたまま維持し、1 行 1 JSON リクエストを処理）:

```powershell
.\translate_long.ps1 -Service
```

導入オプション・オフライン導入・SSH 署名検証・更新・アンインストールは [INSTALL.md](INSTALL.md)、256k Long モードの詳細は
[LONG.md](LONG.md)、全 CLI オプション・サービスプロトコル・環境変数・終了コードは [USAGE.md](USAGE.md) を参照してください。

## 提供されるモード・オプション

| コマンド / モード | 内容 |
|---|---|
| `translate_long.ps1` **（推奨・統合 CLI）** | 最大 **262,143 トークン**まで入力長を問わず翻訳。2,048 トークン以内（`-Long` なし）は高速な NPU 単体 `w4-m2` へ自動委譲し、超える入力（または `-Long` 指定時）は **Long モード**（Radeon iGPU プリフィル＋NPU W4 逐次デコード、最大 262,144 コンテキスト、繰り返し停止デフォルト有効）で実行します。 |
| `w4-m2`（短文高速経路、≤ 2,048 トークン） | 4-bit 重み本体＋8-bit 出力ヘッドに、小型ドラフタ（`PLaMo-D20-M2`）による**厳密な** 2 トークン投機的デコードを組み合わせた経路。出力はすべての位置で `w4-serial` とトークン単位で完全一致します。 |
| `w4-serial` | 同じ量子化モデルで 1 トークンずつ生成（≤ 2,048 トークン）。独立した演算順序一致 CPU リファレンスとビット完全一致を検証済み（標準 100 行＋長文 64 行）。 |
| `w8-serial` | 8-bit 重みで 1 トークンずつ生成（≤ 2,048 トークン）。 |
| 任意 `-Turbo`（`--turbo`） | 実行中のみ NPU 電源モードを turbo へ切替（管理者権限が必要な場合があります。権限不足時は通常モードで継続）。短文経路の長め出力で約 6% 高速化。 |
| 任意 `-KeepWarm`（`--keep-warm`） | 短文経路の生成中のみ CPU 論理コア 1 つを稼働させ C ステート復帰遅延を低減。約 4.5% 高速化（システム設定変更なし）。 |

すべてローカルで完結します。**テレメトリはありません**。翻訳実行時に外部へ通信することはありません（詳細は [SECURITY.md](SECURITY.md)）。

## 動作要件

- **PC / OS**: **AMD Ryzen XDNA1 NPU**（Phoenix / Hawk Point クラス）搭載の Windows 11（x64）PC と AMD NPU ドライバ
  （デバイスマネージャーに「NPU Compute Accelerator Device」が表示されていること）。
  - **256k Long モード用**: AMD Radeon 780M / 760M / 740M iGPU および AMD グラフィックスドライバ（**Radeon 780M のみ**検証済み）。
- **RAM / ディスク**: メモリ 16 GB（通常 NPU 経路で約 6 GB 空き推奨、Long モードで約 4 GB 空き必要）、**NTFS** ボリューム上に
  256k フル構成で約 **9 GB** の空き容量（NPU 単体のみの場合は約 **7 GB**、任意 `-WithGateData` 使用時はさらに約 0.5 GB）。
- **Python**: **64 ビット版 CPython 3.13**（python.org 公式インストーラ、`winget install Python.Python.3.13`、または conda-forge）。
  管理者権限は不要です。詳細は [INSTALL.md](INSTALL.md)。
- **インターネット接続（初回導入時のみ）**:
  - NPU ランタイム（`install.ps1`）: 約 **5.4 GB** — Hugging Face からランタイムペイロード 5.35 GB
    （[nilpe0555/PLaMo-3-Translate-0.8B-XDNA1-NPU-runtime](https://huggingface.co/nilpe0555/PLaMo-3-Translate-0.8B-XDNA1-NPU-runtime)）、
    [pfnet/plamo-3-translate-0.8b](https://huggingface.co/pfnet/plamo-3-translate-0.8b) からトークナイザ等 5 ファイル（3.6 MB）、
    PyPI から 2 つの wheel（`numpy`, `ml_dtypes`）。任意の `-WithGateData` はさらに約 359 MB。
  - 256k Long モード（`install_long.py --with-model`）: **1.78 GB**（[pfnet/plamo-3-translate-0.8b](https://huggingface.co/pfnet/plamo-3-translate-0.8b)
    から固定コミット・SHA-256 検証付きで `model.safetensors` を直接取得）。
- **Git**: リポジトリの clone および `SHA256SUMS` の OpenSSH 署名検証（Git for Windows 付属の `ssh-keygen` を自動検出。Windows 標準の
  OpenSSH クライアントは約 8 KB 超のファイル検証でハングするため使用しません。詳細は [INSTALL.md](INSTALL.md)）。

## 性能

作者の **1 台の環境** での実測値です（AMD Ryzen 7 8840U［XDNA1 NPU＋Radeon 780M iGPU］、16 GB DDR5-5600、Windows 11
build 26300、NPU ドライバ 32.0.20101.3760、NPU 電源モード「Default」、`-Turbo` / `-KeepWarm` オフ、キャッシュ済み）。

### 256k Long モード（`translate_long.ps1`、Radeon 780M iGPU プリフィル＋XDNA1 NPU デコード）

| 入力長 | プリフィル（iGPU） | デコード（NPU・W4 逐次） |
|---|---:|---:|
| 8k トークン | 約 8 秒 | 21〜22 ms/トークン |
| 32k トークン | 約 22 秒 | 24 ms/トークン |
| 262k トークン（上限 262,143） | 約 5 分（熱条件により 4〜5.5 分） | 約 50 ms/トークン |

パイプライン全体の数値正確性はゲート一式で検証済みです（ステンシル 14,336 ケース、最大 262,143 までの NPU デコード 10/10 ビット一致、
4k〜32k を跨ぐ連続コンパクション 15 ステップ ビット一致、最大 259k の分割／非分割 MD5 一致、iGPU プリフィル vLLM 基準 37/37 合格。詳細は
[LONG.md](LONG.md)）。

### 短文・通常コンテキスト（≤ 2,048 トークン、自動委譲される NPU `w4-m2` 厳密投機経路）

リリース判定ラン（開発版との交互 24 ペア・新規プロセス中央値、段落 = 入力 515 トークン・出力 595 トークン）:

| 項目 | 実測値 |
|---|---|
| プロセス起動 -> 投機的デコード準備完了（spec-ready） | 中央値 **0.69 秒** |
| 初回トークン生成時間（TTFT、入力 27〜31 / 515 トークン） | **0.11 秒** / **1.43 秒** |
| 595 トークン出力時のデコード速度（入力 515 トークン、デフォルト `w4-m2`） | **13.9 ms/トークン**（約 72 tokens/s） |
| 段落全体のリクエスト（入力 515 -> 出力 595 トークン）の生成時間 | **9.7 秒** |

同梱ドラフタ（`PLaMo-D20-M2`）・`-Service` モード・greedy デコード（各 128〜160 件）での実測:

| リクエストセット | デコード時間 (ms/トークン) | 確定トークン数 / サイクル |
|---|---:|---:|
| 日常短文 160 件（独自作成、日英双方向） | 14.2 | 1.63 |
| 日常文 128 件（Tatoeba、日英双方向） | 14.1 | 1.56 |
| 百科事典調の文・段落 128 件（日英双方向） | 14.7 | 1.47 |

同一カーネルを用いた開発ビルドでの参考値:

- 入力 1018 トークン時の TTFT: 約 2.9 秒
- 単発 CLI での 15 トークン短文（プロセス起動〜終了）: 約 1.35 秒
- 通常生成 `w4-serial`（文脈長 27 / 515 / 1018 トークン）: 16.95 / 17.56 / 18.20 ms/トークン
- 通常生成 `w8-serial`（短文脈 / 1k 文脈）: 約 23.6 / 26.5 ms/トークン
- 任意 `-Turbo` / `-KeepWarm` 有効時（短文経路の長め出力）: 実行時間 約 −6.4% / −4.5%

**測定値に関する注意点**:

- チップの一般ベンチマークではなく、特定 1 台の PC での実測値であり、他機種での性能保証はありません。
- 「spec-ready」は OS ファイルキャッシュが温まった状態での新規プロセス起動時間です。再起動直後の初回起動は遅くなる場合があります。
- 出力が 3〜8 トークン程度の極端な短文では、棄却された投機サイクルのコストが節約分を上回り、通常生成より遅くなることがあります。
- プロセス終了から約 0.3 秒以内に次のプロセスを起動すると、約 10 回に 1 回の割合でドライバ由来の約 0.5 秒の待ちが発生する場合があります。
  連続処理には `-Service` モードを使ってください。
- **メモリ帯域の下限について**: 4-bit 本体は 1 トークンあたり約 473 MB を読み込みます。本機で実測した shim DDR 実効帯域
  約 40 GB/s から導かれる通常生成の帯域下限は約 12 ms/トークン（`w4-serial` 実測は 16.95〜18.20 ms/トークン）です。
  厳密な投機的デコードにより出力を一切変えずに実効 13.9 ms/トークンまで短縮し、この理論下限に迫っています。

## 制限事項と長文におけるモデル特性

- **対象言語とコンテキスト長**: 日本語 <-> 英語のみ対応。Long モードは最大 **262,143 入力トークン**（合計コンテキスト 262,144）まで動作し、
  NPU 単体投機経路は合計 2,048 トークンまでです（`translate_long.ps1` では 2,048 超を自動で Long モードへ振り分け、`translate_plamo3.ps1`
  単体では 2,045 トークン超を拒否します）。
- **長文入力におけるベースモデル由来の繰り返し傾向（品質未検証）**: 256k パイプライン自体は最大 262,143 トークンまで正確に動作しますが、
  2,048 トークン超の翻訳品質はベースモデル自体の制約により未検証です。GPU 上の公式 BF16 モデル（`pfnet/plamo-3-translate-0.8b`）を含め、
  約 2k トークン超の一括入力では繰り返しループや翻訳停止を起こしやすい傾向があります（2.6k〜6k トークンの法令文書 16 件の検証では、
  公式 BF16 で 10/16 件、Long モードで 8/16 件がループしました。300〜1,000 トークン 40 件の検証では通常経路と同等の +0.33 chrF）。
  デフォルトで繰り返し停止が有効（`-NoRepeatStop` で無効化可）になっており、長文実行のたびに警告を出力します。詳細は [LONG.md](LONG.md)。
- **検証済み環境**: **Ryzen 7 8840U（XDNA1 NPU＋Radeon 780M iGPU）・NPU ドライバ 32.0.20101.3760・Windows 11 build 26300 のみ**で
  動作確認しています。他の XDNA1 / RDNA3 機種（Hawk Point / Ryzen 7 8040 等）でも動く可能性はありますが**未検証**、XDNA2 NPU
  （Strix Point 等）には非対応です。未検証のドライバや Windows バージョンは記録のうえ警告を出して継続します。
- **同時実行不可**: NPU ジョブは 1 つずつ実行されます。2 つ目のプロセスは待機するか使用中エラーで終了します。`--service` は
  2,048 トークン以下の経路専用です（Long モード非対応）。
- **インストール先パスは ASCII 文字のみ**: 半角スペースは利用可能ですが、日本語など非 ASCII 文字を含むフォルダパスでは XRT が
  Windows ANSI API 経由でファイルを開けないため、インストーラおよび両ランタイムとも終了コード 3 で拒否します（[INSTALL.md](INSTALL.md)）。
- **インストール後のフォルダ移動・コピー非推奨**: 移動・コピーすると NTFS ファイル ID が変わり、起動ごとに約 5.4 GB の全量ハッシュ検証
  （10〜20 秒）が走ります。移動した場合は `.\install.ps1 -Reattest` を実行してください。
- **厳格な整合性チェック**: 改変されたファイルがある場合は設計上起動を拒否します（[SECURITY.md](SECURITY.md)）。
- **ウイルス対策ソフト**: 独自のネイティブ `.exe` は配布していません（venv の `python.exe`、`pyxrt.pyd`、事前コンパイル済み HIP/NPU コードオブジェクトのみ）。

## ライセンス（インストール前に必ずお読みください）

- **本リポジトリのコード**: Apache License 2.0（[LICENSE](LICENSE)）。[amd/IRON](https://github.com/amd/IRON)（Apache-2.0）
  由来のファイルを含みます（詳細は [NOTICE](NOTICE)）。
- **モデル由来ファイル（ランタイムペイロード）**: Hugging Face からダウンロードされる量子化重みイメージ、埋め込みテーブル、
  事前計算トークナイザテーブル、投機的デコード用ドラフタ、事前生成 NPU 命令列は、PLaMo 3 Translate の**派生モデル
  （DERIVED MODEL）**にあたり、株式会社 Preferred Networks（PFN）の **PLaMo Community License** が適用されます:
  [LICENSE-PLaMo-ja.md](LICENSE-PLaMo-ja.md)（日本語原文）・[LICENSE-PLaMo-en.md](LICENSE-PLaMo-en.md)（英語）。
  インストーラ実行時に概要を表示し、同意を確認します。
- **商用利用について**: 個人・学術研究などの非営利目的での利用は無償です。**商用目的で利用する場合**（企業内での利用、
  製品・サービスへの組み込み、ホスティング、API 提供などを含みます）は、**利用者自身が開始前に PFN へ直接登録する必要があり**、
  ライセンスに定められた売上規模条件の適用を受けます。本プロジェクトが登録を代行することや、商用利用権を付与することはありません。
  必ずライセンス本文をご確認ください（本記載は要約であり法的助言ではありません）。
- **ベースモデル**: `pfnet/plamo-3-translate-0.8b` は本プロジェクトでは再配布しません。`install.ps1` がトークナイザ等 5 ファイル（3.6 MB）を、
  `install_long.py --with-model` が `model.safetensors`（1.78 GB）を固定リビジョン（`8b4d42855622093e610ba9b90c2d2b59f8d5c15d`）から直接取得し、
  ハッシュ検証します。
- **表示義務とサードパーティ表記**: ライセンス要件に従い「Built with PLaMo」を表示し、モデル名に「PLaMo」を含めています。
  サードパーティ表記（AMD IRON、mlir-aie、XRT/`pyxrt`、pybind11、データクレジット）は [NOTICE](NOTICE) を参照してください。
- **ドラフタ学習データのクレジット**（`PLaMo-D20-M2`、全文は [NOTICE](NOTICE) 第 6 節）。ドラフタはパーミッシブなライセンスのソース文を
  PLaMo 3 Translate 自身で翻訳した出力のみを用いて学習しており、ソース本文は再配布していません:
  - [Tatoeba](https://tatoeba.org) の例文（[CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/)、Tatoeba 上で CC0 の文は CC0）。
    全例文の投稿者情報はペイロード同梱の `plamo3-drafter-port-v7-attribution/PLaMo-ATTRIBUTION-D20.jsonl.gz` に記録されています。
  - OpenAssistant OASST1 / OASST2（Apache-2.0）。
  - e-Gov 法令検索の法令データ（著作権法第 13 条第 1 号、デジタル庁・政府標準利用規約 2.0）。
  - 青空文庫および Project Gutenberg のパブリックドメイン作品。

## プライバシーとセキュリティ

テレメトリ、アクセス解析、クラッシュレポート、自動アップデート確認は一切ありません。インストーラが通信するのは
`huggingface.co`（ペイロードおよびベースモデルファイル）と `pypi.org`（ハッシュ固定の 2 wheel）のみであり、翻訳本体は
一切通信しません。配布するすべてのファイルはメンテナの SSH 鍵で署名された `SHA256SUMS` と照合され（[INSTALL.md](INSTALL.md)）、
両ランタイムとも起動ごとに入力を再検証します。信頼モデルと脆弱性報告については [SECURITY.md](SECURITY.md) を参照してください。

## サポート

ベストエフォート対応（保証なし）です。不具合報告の際は Issue を作成し、`state\selftest-report.json` と
`state\env-record.json` を添付してください（個人情報や翻訳文は含まれず、プロファイルパスは `%USERPROFILE%` に置換されます）。

## 関連リポジトリ

- **NPU ランタイムペイロード（Hugging Face）**:
  [nilpe0555/PLaMo-3-Translate-0.8B-XDNA1-NPU-runtime](https://huggingface.co/nilpe0555/PLaMo-3-Translate-0.8B-XDNA1-NPU-runtime)
  — `install.ps1` が取得する量子化重み・命令列・`pyxrt.pyd`・`PLaMo-D20-M2` ドラフタ。
- **GPU 向け投機的デコード・ドラフタ（別系統、Hugging Face）**:
  - [nilpe0555/PLaMo-3-Translate-0.8B-MTP-drafter-GPU](https://huggingface.co/nilpe0555/PLaMo-3-Translate-0.8B-MTP-drafter-GPU)
    — NVIDIA GPU（BF16 本体）向け DeepSeek-V3 方式 Multi-Token Prediction（D=3）ドラフタ。
  - [nilpe0555/PLaMo-3-Translate-0.8B-DSpark-drafter-GPU](https://huggingface.co/nilpe0555/PLaMo-3-Translate-0.8B-DSpark-drafter-GPU)
    — NVIDIA GPU（BF16 本体）向け DSpark 並列ブロックドラフタ。

## リポジトリ構成

| パス | 内容 |
|---|---|
| `README.md`, `README_ja.md` | プロジェクト概要（英語 / 日本語） |
| `INSTALL.md`, `LONG.md`, `USAGE.md`, `SECURITY.md` | インストール手順、256k Long モード仕様、CLI / サービス仕様、セキュリティ・信頼モデル |
| `translate_long.ps1` | 推奨・統合 CLI（最大 262,143 トークン。2,048 以下は自動で `translate_plamo3.ps1` へ委譲） |
| `translate_plamo3.ps1` / `.cmd` | 純粋 NPU 短文経路 CLI（≤ 2,048 トークン） |
| `install.ps1`, `install.py`, `install_long.py`, `uninstall.ps1`, `selftest.py` | インストーラ（NPU ランタイム＋256k Long モード）、アンインストーラ、自己テスト |
| `long/`, `plamo3-long/`, `long-manifest.json`, `long-certificate.json` | 256k Long モードランタイム（iGPU HIP プリフィル＋NPU デコード、ハッシュ検証対象） |
| `IRON/`, `lib/` | 純粋 NPU ランタイムソースコード（ハッシュ検証対象） |
| `hf/` | Hugging Face ランタイムペイロード用モデルカード（`README.md`, `README_ja.md`） |
| `SHA256SUMS`, `SHA256SUMS.sig`, `PAYLOAD-MANIFEST.json`, `release.json`, `known-good-env.json` | リリースメタデータおよび署名 |
| `tools/` | リリースツール群（公開ツリー生成、スクラブ検証、Long モード検証ゲート等） |
| `tests/` | インストーラ・リリースツール・Long モードの単体テスト |

大容量の NPU ランタイムペイロード（`plamo3-*`, `xrt/`, 任意の `gate-data/`）は Hugging Face 側に配置されており、
`install.ps1` が本リポジトリと同じフォルダ内に配置・検証します。
