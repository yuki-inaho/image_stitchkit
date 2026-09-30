# 検証記録の読み方

`release_ctest.log` と `sanitize_ctest.log` が全35件の実行結果です。
各33件通過、2件Skipped、0件失敗です。JUnit形式は `release_tests.xml` / `sanitize_tests.xml` にあります。
SkippedはGPU描画比較とLINEAE実モデルで、成功扱いではありません。

`*_configure.log` / `*_build.log` は構成・コンパイルの記録です。
`consumer_*.log` はインストール済みライブラリを別のプロジェクトで利用した記録です。
`geometric_energy.log` は構造制約の制御試験です。
`capabilities.log` には、このビルドに含まれる追加機能の有無を記録しています。

各 `.status` は対応するコマンドの終了コードです。
`history/` は問題修正前の記録であり、最終結果ではありません。
全CPU試験後、任意GPU試験の画像を補色から非補色に変更し、補間誤差が相殺されないよう改善しました。
この任意試験のC++部分の再ビルドと、実行条件不足によるSkippedを `*_optional_test_*.log` に残しています。
GPUカーネルをコンパイル・実行した記録ではありません。

`source.sha256` は納品時のソース・構成ファイルのチェックサムです。
`summary.json` と `../docs/TEST_REPORT.md` は結果の要約です。
この環境固有のパスや共有ライブラリを含むビルド成果物そのものはZIPに同梱していません。
