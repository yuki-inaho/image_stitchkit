# stitchkit — C++ 画像合成の統合実装

REW、NISwGSP、GES-GSP に基づく3種類の変形推定を、同じ C++17 ライブラリと
コマンドライン実行ファイルで選択できます。描画処理の CPU / CUDA 選択は、
合成手法の選択から分離しています。追加記事の LINEAE は第4の合成手法ではなく、
GES-GSP の構造制約へ線分を渡す検出器として組み込んでいます。

**これは論文の主要な変形モデルを独立に実装した研究用基盤です。
著者実装との出力一致や、論文の全評価手順の再現を主張するものではありません。**
2次元の相似変換事前分布、順序付き画像列、標準の輪郭抽出など、今回採用した範囲と
著者実装との差分は [実装範囲](ALGORITHM_COVERAGE.md) に列挙しています。

CPU は実際にビルドし、同梱画像で試験しています。CUDA と ONNX Runtime の追加ソースは
含みますが、この実行環境ではコンパイル・実機実行とも確認できていません。
LINEAE の前処理・出力解釈は、実モデルと独立した C++ 試験で確認しています。
現行の検証結果は [検証記録](VERIFICATION.md) を参照してください。

## 1. 必要なものと依存管理

CMake 3.24以上、C++17コンパイラ、Ninja、Gitを使用します。主要依存は
Eigen 3.4、libpng、libjpeg-turbo です。OpenCV と Python は実行時依存ではありません。
SIFT は、添付実装内にあった VLFeat 0.9.20 の必要部分を BSD ライセンスを保持して同梱しています。

vcpkg のマニフェスト方式を使用し、レジストリをコミット
`58845ed63eb19aff55e896ea1f5d51f2a0df5b66` に固定しています。
Eigen は `3.4.0#5` を指定しています。パッケージ管理の入口は `vcpkg.json` と
`CMakePresets.json` の2ファイルです。LINEAE を有効にしない限り ONNX Runtime は取得しません。
CUDA Toolkit は対応する環境へ別途導入します。

### Linux / macOS: vcpkg を使う場合

```bash
./tools/bootstrap_vcpkg.sh
export VCPKG_ROOT="$PWD/.tools/vcpkg"
cmake --preset vcpkg
cmake --build --preset vcpkg --parallel 2
ctest --preset vcpkg --output-on-failure
```

初回のみ依存の取得・ビルドが発生します。取得にはネットワーク接続が必要です。
スクリプトは既存の別リポジトリを勝手に書き換えず、コミットが異なる既存ディレクトリでは停止します。
既存の vcpkg を使う場合は、`VCPKG_ROOT` をその場所に設定してください。

### Windows

Visual Studio の「Developer PowerShell」で実行してください。Ninja が必要です。

```powershell
./tools/bootstrap_vcpkg.ps1
$env:VCPKG_ROOT = "$PWD/.tools/vcpkg"
cmake --preset vcpkg
cmake --build --preset vcpkg --parallel 2
ctest --preset vcpkg --output-on-failure
```

Windows / macOS の CI 定義は同梱していますが、この納品時点で実行済みではありません。
確認済みの OS・コンパイラは検証記録で区別しています。

### 既に依存を導入している環境

```bash
cmake --preset system
cmake --build --preset system --parallel 2
ctest --preset system --output-on-failure
```

Eigen の CMake 設定がない環境では、`cmake --preset system
-DEigen3_INCLUDE_DIR=/path/to/eigen3` のようにヘッダーの位置を渡せます。
今回実行したのはこちらの方式です。vcpkg による初回取得は、実行環境のネットワーク制約により
未実施であり、マニフェスト方式でのビルド成功とは混同していません。

## 2. 実行

入力順は、隣り合う画像に十分な重複領域がある順序にしてください。
2〜16枚を受け取り、任意順の画像集合から接続グラフを自動発見する処理は行いません。
PNG / JPEG を RGB で読み込み、出力は有効画素をアルファで表す PNG です。

```bash
./build/vcpkg/stitch \
  --method gesgsp --device cpu \
  --output output/library.png --report output/library.json \
  tests/assets/real/lib1.jpg tests/assets/real/lib2.jpg
```

`--method` は `rew` / `niswgsp` / `gesgsp` から選択します。
`--device` は `cpu` / `cuda` / `auto` です。別のモードでも同じ入力・出力形式を使用します。

```bash
./build/vcpkg/stitch \
  --method rew --device cpu --max-side 0 \
  --output output/three_views.png --report output/three_views.json \
  tests/assets/synthetic/view0.png \
  tests/assets/synthetic/view1.png \
  tests/assets/synthetic/view2.png
```

既定では、各入力の長辺を最大1000画素に縮小します。原寸処理は `--max-side 0` です。
JSON の入力寸法と幾何誤差は、この処理後の画像座標に対応します。原寸座標への自動変換はありません。
`--grid`、`--ratio`、`--ransac-threshold`、`--structure-weight` などは `--help` で確認できます。
API では `Options` の全項目を設定できます。未知の引数、数値末尾の不正文字、重複指定はエラーになります。

### GPU 描画

```bash
cmake --preset cuda -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset cuda --parallel 2
./build/cuda/stitch --capabilities
./build/cuda/stitch --method niswgsp --device cuda \
  --output output/gpu.png --report output/gpu.json \
  tests/assets/real/lib1.jpg tests/assets/real/lib2.jpg
ctest --preset cuda -R optional.cuda_parity --output-on-failure
```

CUDA 実装は、逆写像の標本化、双線形補間、境界距離に応じた加重合成を行います。
特徴点抽出、対応点の幾何検証、TPS / メッシュ最適化、逆写像の構築は CPU のままです。
GPU が必ず高速になるとは限らず、本一式では速度向上率を測定していません。
`native` はビルド機の GPU を対象にします。別機へ配布する場合は、対応する数値の
`CMAKE_CUDA_ARCHITECTURES` を明示してください。

`--device cuda` で CUDA が使えなければエラーになります。CPU で実行して成功したように見せません。
`--device auto` のときだけ、CUDA が検出されない場合に CPU を選び、警告と実際の選択を記録します。
GPU 選択後のメモリ不足・カーネル実行失敗はエラーとして伝播し、自動的には隠しません。
現行実装の GPU 番号は0です。

### LINEAE

LINEAE の ONNX モデルは同梱していません。モデルの配布元から取得し、ローカルパスで指定します。
外部モデルを実行時に自動ダウンロードする機能はありません。

固定した ONNX Runtime の vcpkg port が要求する
`ONNX_DISABLE_STATIC_REGISTRATION=ON` を ONNX 依存に渡す専用 triplet を用意しています。
Linux x64 の例です。

```bash
cmake --preset lineae -DVCPKG_TARGET_TRIPLET=x64-linux-lineae
cmake --build --preset lineae --parallel 2
./build/lineae/stitch \
  --method gesgsp --device cpu --structures lineae \
  --lineae-model /path/to/lineae_a.onnx --lineae-variant A \
  --lineae-device cpu --lineae-threshold 0.3 \
  --output output/lineae.png --report output/lineae.json \
  tests/assets/real/lib1.jpg tests/assets/real/lib2.jpg
```

Windows x64 は `x64-windows-lineae`、Apple Silicon は `arm64-osx-lineae` を指定します。
これらの追加依存の取得・ビルドは未検証です。ONNX Runtime から取得した共有ライブラリと
追加実行プロバイダーが、実行時のライブラリ検索パスに存在する必要があります。

CUDA Execution Provider を使用する場合は、別のビルドディレクトリで
`-DVCPKG_MANIFEST_FEATURES=lineae-cuda` を指定し、実行時に `--lineae-device cuda` を指定します。
対応する CUDA / cuDNN の組み合わせも必要です。明示した CUDA プロバイダーが使えない場合や、
CPU へのノード委譲が必要な場合には停止します。現行の `--lineae-device auto` は CPU を選びます。
描画用の `--device` と LINEAE 推論用の `--lineae-device` は独立しています。

対応するモデル仕様は、入力 `images: float32[1,3,H,W]`、
出力 `pred_logits: float32[1,K,2]` / `pred_lines: float32[1,K,4]` の固定形状です。
信頼度は **第0クラスの sigmoid**、線は **正規化した端点 (x1,y1,x2,y2)** として解釈します。
A/F/P/N/T と、それ以外の対応モデルでは平均・標準偏差を切り替えます。
モデル名から推測せず、`--lineae-variant` を必須にしています。
LINEAE が出す直線だけで曲線保護が失われないよう、標準の輪郭抽出も併用します。

実モデル試験を有効にする例:

```bash
export STITCHKIT_TEST_LINEAE_MODEL=/path/to/lineae_a.onnx
export STITCHKIT_TEST_LINEAE_VARIANT=A
ctest --preset lineae -R optional.lineae_model --output-on-failure
```

## 3. ライブラリとしての再利用

同一ビルド内では `add_subdirectory` で取り込み、`stitchkit::stitchkit` にリンクできます。
インストール後は `find_package(stitchkit CONFIG REQUIRED)` を使用します。
公開ヘッダーは Eigen / ONNX Runtime / CUDA の型を露出しません。

```bash
cmake --install build/vcpkg --prefix "$PWD/install"
cmake -S tests/consumer -B build/consumer -G Ninja \
  -DCMAKE_PREFIX_PATH="$PWD/install" \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DVCPKG_MANIFEST_MODE=OFF \
  -DVCPKG_INSTALLED_DIR="$PWD/build/vcpkg/vcpkg_installed"
cmake --build build/consumer
```

完全な利用例は `examples/basic.cpp`、外部プロジェクトからの使用例は `tests/consumer/` にあります。
`Stitcher` の別コンストラクターに対応点推定器、構造検出器、線形求解器を注入できます。
変形推定器と描画器も公開インターフェースを持ち、独立した試験と差し替えができます。

## 4. 試験

CTest は単体試験、資産を使用する一貫試験、コマンドライン試験を実行します。
Python は不要です。合成画像の再生成にだけ NumPy / Pillow が必要です。
画像の SHA-256 確認は標準 Python のみで実行できます。

```bash
ctest --preset vcpkg --output-on-failure
python tools/verify_assets.py
```

正解のある2枚・3枚の合成画像では、画素誤差、有効領域の被覆率、黒い有効画素を検証します。
さらに、局所視差、曲線を含む構造制約の効果、ゼロ変位 TPS、階数不足、反転メッシュ、
双線形の逆写像、非正方形の LINEAE 座標、壊れた画像、未知の指定などを試験します。
実画像3組も全3モードで通します。GPU / 実モデルがない試験は CTest の Skipped になります。
**Skipped は検証成功ではありません。**

GCC / Clang のメモリ検査用:

```bash
cmake --preset sanitize
cmake --build --preset sanitize --parallel 2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --preset sanitize --output-on-failure
```

## 5. 制限と資料

画像の露出補正、グラフカットによる継ぎ目選択、多重解像度合成は実装していません。
動く物体や遮蔽のある領域では二重像が残る可能性があります。入力 PNG の透明部分は白背景に合成して読み込み、
EXIF の回転指示は適用しません。再実行で既存の出力を上書きしますが、入力ファイルを出力先には指定できません。
画像出力と JSON 出力をまたぐ原子的な保存は行っていません。

詳細は [構成](ARCHITECTURE.md)、[数式と実装範囲](ALGORITHM_COVERAGE.md)、
[検証記録](VERIFICATION.md)、[第三者資料](THIRD_PARTY_NOTICES.md)、
[画像出所](tests/assets/PROVENANCE.md) に分けています。
