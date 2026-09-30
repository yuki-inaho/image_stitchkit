# 修正前の検査記録

このディレクトリは開発中に検出した不具合の記録です。納品時点の検証結果ではありません。
`sanitize_before_convolution_fix.log` は旧 VLFeat のポインター演算を UBSan が検出したものです。
境界アクセスを修正し、専用の回帰試験を追加しました。修正後の結果は上のディレクトリにある
`sanitize_ctest.log` と `sanitize_tests.xml` を参照してください。
