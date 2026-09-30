# 構成と契約 — 0.2

## 境界

```text
Python: immutable Options + RGB ndarray
    │ 検査、連続化、C++所有画像への複製
    ▼
nanobind adapter (bindings/module.cpp)
    │ ここでGILを解放
    ▼
Stitcher.prepare(images)
    ├─ IFeatureMatcher.match_sequence : 各画像の特徴抽出は1回
    └─ PreparedScene : 入力・対応点・位置合わせ設定を所有する変更不可の結果
          │ 同じ位置合わせ設定のみ再利用可能
          ▼
Stitcher.compose(scene)
    ├─ IStructureDetector : GESのみ、毎回計算
    ├─ IWarpEstimator : REW / NIS-GES共有のメッシュ推定
    │    └─ ILeastSquaresSolver
    ├─ WarpPlan : キャンバス + 逆写像
    └─ IRenderer : CPU / CUDA、または注入した描画器
          ▼
Result : RGB + valid mask + Diagnostics
    │ C++領域への参照を所有者カプセルで保持、Python復帰時はGIL取得済み
    ▼
read-only NumPy image/mask + report_json
```

従来の `stitch(images)` はprepare/composeの同じ内部処理を組み合わせます。ファイル読込・書込はこの流れの外です。CLIもPythonも同じC++処理を使用し、Pythonに第2の画像合成実装は置いていません。

## 責務

`types.hpp` は画像・設定・幾何情報、`interfaces.hpp` は差し替え可能な計算境界、`stitcher.hpp` は利用入口です。Eigen・CUDA・ONNX Runtime・nanobindの型を公開C++ヘッダーに露出させません。

`features.cpp` は抽出と隣接対応、`registration.cpp` は変換推定、`mesh_warp.cpp` はNIS/GESの共有計算、`rew.cpp` は弾性変形、`raster.cpp` / `render_cuda.cu` は描画です。`report.cpp` はCLI/Python共通のJSON、`io.cpp` は画像とファイル入出力です。

`python/stitchkit/__init__.py` は型・色順・配列の契約を扱う薄い入口です。Pythonの既定パラメーターはC++のOptionsから取得し、数値の既定値を重複管理しません。`benchmark.py` は計測と正解画像評価だけを扱い、描画はNotebookへ置いています。pandasやmatplotlibを通常の `import stitchkit` で読み込みません。

## PreparedSceneの互換条件

以下が準備時と一致しなければcomposeを拒否します。

`max_features`, `ratio_threshold`, `ransac_threshold`, `ransac_iterations`, `minimum_matches`, `seed`

手法、メッシュ設定、各エネルギーの重み、描画装置、構造検出設定は変更できます。画像や対応点の可変参照は公開しません。C++では移動済みオブジェクトのアクセスも例外にします。

既存の独自IFeatureMatcherには既定のmatch_sequence実装を用意し、対単位matchをそのまま利用できます。その場合、実際の特徴抽出回数を把握できなければ `-1` とし、仮の値を記録しません。

## 配列所有権

入力はC++所有で、通常の入力バッファの寿命に依存しません。配列の型・形状の検証と複製はGIL保持中に行います。時間のかかる計算の区間だけGILを解放し、カプセル・NumPy配列を作る前に再取得します。

出力はconstのC++データを参照します。NumPyの書込み可能フラグも無効で、所有者は結果への共有参照を持ちます。結果を削除しても配列だけを保持できます。一方、少数画素のビューだけを長期間保持しても元の結果全体が存続します。メモリを切り離したい場合はPythonでcopyを使います。

PythonのOptionsは変更不可、ネイティブ呼出しへ渡す設定も値として受け取ります。PythonオブジェクトをGILなしで操作する処理は追加していません。自由スレッドPythonや、別のネイティブスレッドが入力を書き換える使い方は未検証です。

## 計測契約

JSON schema_versionは2です。`timing_ms.registration` は当該処理の位置合わせ、`structure` は構造検出、`geometry` は変形の組立て・求解と逆写像構築、`render` は描画、`total` はC++パイプライン呼出し全体です。

`prepared_registration_ms` は準備結果を作った際の履歴です。再利用時の当該登録時間は0であり、履歴値を当該時間へ加えません。Pythonのwall timeには入力複製、設定生成、返却処理なども含まれます。PythonからC++への境界費用だけを隔離した測定ではありません。

## 構築・配布

C++ SDK、CLI、Python拡張の構築とインストールを個別に選択できます。wheelへ不要なSDKヘッダーや静的ライブラリを混入させません。実行時依存とビルド依存を分け、nanobind/scikit-build-coreはビルド依存です。

本プロジェクトは将来の計算装置を差し替えられる境界を持ちますが、GPUで実装していない処理が自動でGPUへ移る仕組みではありません。CUDAは補間・合成だけ、ONNX Runtimeは構造検出だけを担います。
