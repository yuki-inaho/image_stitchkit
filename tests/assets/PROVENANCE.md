# 試験画像の出所

`manifest.json` に、同梱画像の相対パス、SHA-256、寸法、出所を記録しています。
`tools/verify_assets.py` はマニフェストの改変検出と読み込みを行います。

## 実画像

利用者の添付 `DIP_Final_Project-master.zip` から次をそのまま抽出しました。
リサイズは試験実行時だけ行い、同梱原本は変更していません。

| 同梱ファイル | ZIP内の相対パス | 寸法 |
|---|---|---|
| real/lib1.jpg | DIP_Final_Project-master/Seam Finder/lib1.jpg | 1000×620 |
| real/lib2.jpg | DIP_Final_Project-master/Seam Finder/lib2.jpg | 1000×620 |
| real/town_hall1.jpg | DIP_Final_Project-master/Seam Finder/town_hall1.jpg | 1280×720 |
| real/town_hall2.jpg | DIP_Final_Project-master/Seam Finder/town_hall2.jpg | 1280×720 |
| real/ship1.jpg | DIP_Final_Project-master/Seam Finder/ship1.jpg | 1830×1080 |
| real/ship2.jpg | DIP_Final_Project-master/Seam Finder/ship2.jpg | 1830×1080 |

実画像には正解の画素変換・パノラマが付いていません。これらに対する試験は、
実入力の読み込み、対応点検出、各手法の処理、出力、有限性と範囲の回帰試験です。
論文と同一の評価集合での精度比較、主観画質の保証、正解に対する誤差計測ではありません。

## 合成画像

生成器: `tools/generate_assets.py`。乱数種42。横760×縦280の原画像から
横400×縦280、開始x座標0・180・360の3視点を切り出しています。
正解画像は2枚用 `pair_truth.png`、3枚用 `scene.png` です。

局所視差画像 `parallax1.png` は、入力座標 (u,v) から原画像へ
`(u+180+10b, v+3b)`、`b=exp(-((u-130)/70)^2-((v-142)/65)^2)`
で逆写像して生成しています。黒い矩形は「黒を無効画素と誤判定しない」試験用です。
PNG再生成の圧縮バイト列は依存ライブラリの版により変わり得ます。
試験ではコミット済み画像を使うため、再生成は必須ではありません。
