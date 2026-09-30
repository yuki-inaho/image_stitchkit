# 実行検証記録

検証日: 2026年9月29日。対象: stitchkit 0.1.0。
以下は実際に実行した結果と、実行できていない項目を分けた記録です。

## 検証した環境

| 項目 | 使用した環境 |
|---|---|
| OS / CPU | Debian GNU/Linux 13、Linux x86_64 |
| 通常ビルド | GCC / G++ 14.2、Release、C++17 |
| メモリ検査ビルド | Clang 17、Debug、`-O1 -g1`、AddressSanitizer + UndefinedBehaviorSanitizer |
| 構成 / ビルド | CMake 3.31.6、Ninja |
| Eigen | 3.4.0、既存のヘッダーを明示指定 |
| 画像入出力 | libpng 1.6.48、libjpeg-turbo 2.1.5、zlib 1.3.1 |
| GPU / 推論環境 | CUDA Toolkit・GPU・C++版 ONNX Runtime・LINEAE実モデルなし |

今回は既存の依存ライブラリを使用する `system` プリセットでビルドしました。
vcpkg のマニフェスト・コミット固定・追加機能・専用 triplet は同梱していますが、
実行環境から依存配布先へのネットワーク接続ができず、**vcpkg の初回取得とビルドは未検証**です。
マニフェストを置いたことを、依存の復元に成功したこととは扱っていません。

## 試験の内訳

| 対象 | 件数 | 内容 |
|---|---:|---|
| C++単体試験 | 13 | 畳み込み、型・範囲、格子、最小二乗、幾何推定、REW、構造制約、逆写像、画像入出力、輪郭、LINEAE入出力契約、装置選択、異常入力 |
| 画像を使う一貫試験 | 18 | 3手法 × 6場面。実際の特徴点抽出からPNG・JSON保存まで |
| コマンドライン試験 | 1 | 正常実行と未知の引数・不正数値・画像欠落等の終了コード |
| 画像整合性 | 1 | 同梱画像すべてのSHA-256をCMakeで検証 |
| GPUの比較試験 | 1 | 実行条件不足のため **Skipped** |
| LINEAE実モデル試験 | 1 | 実行条件不足のため **Skipped** |

通常ビルド・メモリ検査ビルドとも、**35件中33件が通過、失敗0件、未実行2件**です。
CTestの表示上の「100% tests passed」は、未実行2件を成功に数えた意味ではありません。
生ログとJUnit XMLは `verification/` に収録しています。

さらに、CTestとは別に `cmake --install` を実行し、`tests/consumer/` の独立プロジェクトで
`find_package(stitchkit CONFIG REQUIRED)`、コンパイル・リンク・2画像の合成・PNG保存まで確認しました。
公開ヘッダーを使う側にEigenのヘッダー位置を渡していません。
`consumer_configure.log`、`consumer_build.log`、`consumer_run.log` がその記録です。

## 正解画像がある試験

`translation` は既知の平行移動を持つ2枚、`chain` は同じ原画像から切り出した3枚です。
乱数種42で生成した固定画像を使い、黒い有効画素、被覆率、対応点誤差、PNG保存後のRGB一致を検査します。
出力座標は出力JSONの原点を使って正解と対応づけ、正解画像の周囲3画素を除いて比較します。

| 手法 | 2枚のRGB RMSE | 3枚のRGB RMSE |
|---|---:|---:|
| REW | 2.15623 | 4.21520 |
| NISwGSP | 1.70202 | 1.26045 |
| GES-GSP | 1.00827 | 1.13667 |

値は8ビットRGBの各成分に対する二乗平均平方根誤差です。全6試験で `RMSE < 6` を満たしました。
評価領域は2枚で157,276画素、3枚で206,596画素で、すべて有効画素に覆われています。
黒い確認領域は793画素が有効な黒として残り、透過扱いされていません。

これは同梱した合成画像上の回帰試験値であり、論文のベンチマーク値でも、
一般の実画像における手法間の優劣を示すものでもありません。
個別の測定値は `results/e2e/*_quality.json` に保存しています。

## 視差・構造制約

局所的な滑らかな変位を与えた画像対では、3手法とも対応点の残差が3画素未満となることを検査します。
REWの対応点誤差は各入力対の元画像座標、NIS/GESは共通パノラマ座標における値です。
異なる座標尺度の数値を、そのまま横並びの性能順位に使用してはいけません。

`unit.geometric_energy` は、既知の対応と、直線および閉曲線を与える独立試験です。
APAPを無効にし、構造重み20で幾何保存項が実際に効くことを確認しています。
構造関係の二乗残差和は、NIS側45.8181、GES側0.00165232でした。
これは追加した方程式の効果を意図的に分離した試験で、標準設定や実画像での精度比較ではありません。

## 実画像

添付資料の library / town_hall / ship の3組を、全3モードで処理しました。
試験時には各画像の長辺を最大640画素に縮小しています。同梱の画像原本は変更していません。
読み込み、特徴対応、各変形モデルの実行、構造方程式の有無、有限な結果、出力寸法の上限、
有効領域、PNGの保存・再読み込みを検査します。

これらの画像には正解パノラマがないため、**実画像について正解画素に対する精度を測定したとは主張しません**。
また、露出補正・継ぎ目探索・多重解像度合成は含んでいないため、二重像が残る領域があります。
全出力は `results/e2e/`、3手法の実行例一覧は `results/library_comparison.png` にあります。

## メモリ検査で見つけた問題と修正

同梱した旧VLFeatの畳み込み関数で、負の添字と符号なしストライドを混在させ、
配列の手前を指すポインターを形成する未定義動作が見つかりました。
検査を無効化せず、範囲内の整数添字による畳み込みに置き換えました。
零埋め・端値延長、転置、間引き、フィルターの適用順を回帰試験で確認しています。

修正後の全CPU試験では、ASan / UBSanのエラーは報告されていません。
このことは任意入力についての無欠陥性を証明するものではありません。
修正前のログは `verification/history/` に履歴として保存し、最終結果とは分離しています。
詳細は `third_party/vlfeat/PATCHES.md` を参照してください。

## 未検証・対象外

CUDA描画の `.cu` とONNX Runtimeアダプターの `.cpp` は同梱していますが、
**この環境では両方とも追加機能としてのコンパイル・リンク・実機実行を行っていません**。
GPU速度向上、CPU/GPU一致、実LINEAEモデルの精度・速度を確認済みとはしていません。
LINEAEの正規化、テンソル配列順、sigmoid、端点の復元・切り取りは、実モデルと独立したC++試験で確認しています。

Windows / macOSのCIは定義のみで未実行です。任意順画像集合、3次元カメラ推定、
著者実装との完全一致、全論文評価表の再現、長い画像列での品質も検証範囲に含みません。

## 実際に使用した主なコマンド

EigenのパスとClangのパスはこの検証環境のものです。他環境ではそれぞれの導入先に変更してください。

```bash
cmake --preset system \
  -DEigen3_INCLUDE_DIR=[PY_ENV]/lib/python3.13/site-packages/casadi/include/eigen3
cmake --build --preset system --parallel 2
ctest --preset system --parallel 2 --output-on-failure

cmake --preset sanitize \
  -DEigen3_INCLUDE_DIR=[PY_ENV]/lib/python3.13/site-packages/casadi/include/eigen3 \
  -DCMAKE_C_COMPILER=[CLANG_PATH] \
  -DCMAKE_CXX_COMPILER=[CLANG_PATH]++ \
  '-DCMAKE_C_FLAGS_DEBUG=-O1 -g1' '-DCMAKE_CXX_FLAGS_DEBUG=-O1 -g1'
cmake --build --preset sanitize --parallel 2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --preset sanitize --parallel 2 --output-on-failure
```

個々のコマンドの終了コードは `verification/*.status` です。
メモリ検査ではプロジェクト本体・VLFeat・試験コードを計装しています。
既存の共有libpng/libjpegそのものを、サニタイザー付きで再ビルドしたわけではありません。
