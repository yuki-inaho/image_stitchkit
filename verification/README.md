# 検証記録の区別

`review/` は今回のレビュー・0.2の改修で取得した記録です。`baseline_*` のログも今回0.1を再構築して取得しています。

`baseline_v0.1/` と `history/` は前回納品から保存した資料です。今回実行した検証として件数を加算しません。

`review/notebook_first_export_issue.log` は、最初の全計算後のHTML出力で起きた環境設定の不備の記録です。出力先とテンプレート探索の修正後、全セル実行からHTML出力まで再実行した最終記録は `review/notebook_execution.log` です。

`review/uv_lock_offline.log` はネットワーク制限下で依存解決を試みた失敗記録です。uv.lockを正常生成したことを意味しません。
