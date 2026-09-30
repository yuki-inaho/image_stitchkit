# 第三者資料・権利の区分

## 同梱ソース

`third_party/vlfeat/` は、添付された GPU 版 NISwGSP のアーカイブ内にある
VLFeat 0.9.20 から、SIFT・スカラー画像処理に必要な部分のみを抽出したものです。
元の BSD ライセンス全文を `third_party/vlfeat/COPYING` に保持しています。
変更箇所とメモリ不足時の終了動作は `third_party/vlfeat/PATCHES.md` に記載しています。
著者の画像合成実装そのものをこのディレクトリへ転載したものではありません。

新規作成部分はルートの MIT ライセンスで提供します。元の NISwGSP / GES-GSP の
権利表示を置き換えたり、元リポジトリ全体を MIT として再配布したりしていません。
参照した数式・構成と、この実装との差分は `docs/ALGORITHM_COVERAGE.md` に記載しています。

## 画像

`tests/assets/real/` の6枚は、利用者が添付した DIP_Final_Project-master.zip の
`Seam Finder/` に由来します。元の画像に関する独立した許諾文書は今回の添付からは
確認できていません。画像の権利を新たに付与するものではありません。
本一式を第三者へ公開・配布する際は、これらの画像をそのまま公開できる条件を確認するか、
自ら権利を持つ画像へ置き換えてください。

`tests/assets/synthetic/` は本プロジェクト用に生成した画像です。
生成器と正解座標を同梱し、MIT ライセンスの対象にしています。

## 外部依存とモデル

Eigen、libpng、libjpeg-turbo、ONNX Runtime、CUDA は本ソース一式にバイナリを同梱していません。
それぞれの配布元の条件に従って取得してください。vcpkg のマニフェストは取得方法の定義です。
LINEAE の学習済みモデルも同梱しておらず、実行時に指定したローカルファイルのみを読み込みます。
モデルの利用条件は、その配布元に従ってください。

## Python binding build dependencies (0.2)

The Python extension statically includes nanobind 2.15.0 (BSD-3-Clause) and its
robin-map dependency (MIT). Their full notices are preserved in
`python/stitchkit/licenses/` and included in the wheel. nanobind and
scikit-build-core are build dependencies declared in pyproject.toml; development
installs also include them for local rebuilding and environment diagnostics.

The isolated validation environment could not access PyPI. Portable nanobind
Python/C++ sources were recovered from the MSYS2 2.15.0 package produced by the
merged update PR msys2/MINGW-packages#31091. All 89 retained upstream RECORD hashes
matched. No Windows binaries were used. The artifact digest and scikit-build-core
release artifact provenance are in `verification/review/dependency_provenance.json`.
No changes were made to nanobind's sources. These dependency archives are not
redistributed as part of the project source archive.
