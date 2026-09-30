# 参照した一次資料

確認日: 2026-09-29。採用したパッケージは最新追従ではなく、実際に構築・検証したバージョンに固定しています。

- nanobind: ndarrayの型・形状・配置制約、所有者カプセル、返却方針、GILの必要な操作。
  https://nanobind.readthedocs.io/en/latest/ndarray.html
- nanobind: パッケージ化。
  https://nanobind.readthedocs.io/en/latest/packaging.html
- scikit-build-core: pyproject.tomlとCMakeによるPython拡張の構築。
  https://scikit-build-core.readthedocs.io/en/latest/guide/getting_started.html
- uv: lockファイルと環境同期の違い、locked/frozen/no-syncの意味。
  https://docs.astral.sh/uv/concepts/projects/sync/
- uv: Jupyterとカーネル環境。
  https://docs.astral.sh/uv/guides/integration/jupyter/

依存配布物の取得経路とハッシュは `verification/review/dependency_provenance.json` に記載しています。
論文の手法に関する元の参照・実装との差異は `ALGORITHM_COVERAGE.md` にまとめています。
