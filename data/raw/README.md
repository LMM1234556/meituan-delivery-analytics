# 原始数据说明

原始数据应从美团官方仓库获取：

<https://github.com/meituan/Meituan-INFORMS-TSL-Research-Challenge>

请将以下文件放入 `data/raw/downloads/`：

- `all_waybill_info_meituan_0322.csv.zip`
- `courier_wave_info_meituan.csv`
- `dispatch_rider_meituan.csv`
- `dispatch_waybill_meituan.csv`

要求：

1. 不修改原始文件；
2. 不将原始数据提交到公开代码仓库；
3. 下载后通过数据清点脚本记录文件大小、哈希和覆盖范围；
4. 使用时遵守官方许可与致谢要求。

运行 `python scripts/run_pipeline.py` 时，程序会检查上述文件是否齐全，并自动将运单压缩包解压至分析所需的本地目录。原始数据目录已加入 `.gitignore`。

数据适用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.zh-hans)；未经许可不得对外分发原始数据，数据及其衍生作品仅限非商业用途。公开研究成果应注明：本研究由美团提供数据支持。
