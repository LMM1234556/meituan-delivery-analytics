# 美团外卖订单超时分析与高峰期骑手运力配置

[![Repository checks](https://github.com/LMM1234556/meituan-delivery-analytics/actions/workflows/repository-checks.yml/badge.svg)](https://github.com/LMM1234556/meituan-delivery-analytics/actions/workflows/repository-checks.yml)

基于美团官方公开即时配送研究数据完成的独立经营分析项目。围绕“高峰期有限增量运力应优先配置至哪些区域和订单”这一决策问题，完成数据粒度审计、订单级分析集市、履约指标诊断、分决策时点风险识别及约束条件下的离线策略回放，并沉淀可复现 SQL、Jupyter Notebook 与正式业务报告。

## 管理摘要

- 从 654,343 条运单尝试记录中识别 568,546 笔唯一成功运单对应订单；经营策略样本包含 436,062 笔有效完成即时单，超时率为 13.63%。
- 午晚用餐高峰（[10:30, 14:00)、[17:00, 21:00)）承载 71.70% 的订单并贡献 77.33% 的超时订单；前五个匿名商圈以 48.52% 的订单贡献 59.62% 的超时订单，履约风险呈现明显的时空集中性。
- 超时订单平均超时 4.60 分钟，P90 为 9.17 分钟；超时单相对准时单平均多用 17.32 分钟，其中 57.11% 的时长差来自取餐至送达阶段。
- 构建订单推送时与骑手接单时两阶段风险模型；跨期测试集 Lift@Top10% 分别为 2.10 和 2.48，均高于仅按配送距离排序的 1.53。
- 在六个午高峰派单快照中，每快照 30 个假设增量运力槽位的风险加权策略汇总覆盖 11.70% 的后验超时订单，高于供需量基线的 9.18%；逐快照比较为 4 胜、1 平、1 负。该指标用于衡量目标区域识别能力，不代表已实现的超时减少量。

## 核心结果

![用餐高峰订单与超时集中度](assets/figures/01_peak_concentration.png)

![跨期测试集模型风险排序能力](assets/figures/04_model_performance.png)

![风险加权策略与订单量基线离线回放](assets/figures/05_capacity_strategy.png)

## 决策输出

| 决策时点 | 分析输出 | 运营用途 |
|---|---|---|
| 高峰前 | 商圈 × 30 分钟订单规模、超时率与超时订单量 | 制定分层预备运力，避免按全城规模等比例增配 |
| 订单进入配送 | 仅使用当时可得特征的早期风险排序 | 建立高风险订单监控队列，提前识别风险聚集网格 |
| 骑手接单后 | 纳入已发生派单摩擦的更新风险 | 支持异常订单跟进，并为调度策略提供风险分层 |
| 派单快照 | 风险需求、局部候选骑手与预算约束下的网格优先级 | 形成离线试点方案，并通过随机化或准实验验证真实增量效果 |

## 交付物

- [00 数据清点与粒度核验](notebooks/00_data_inventory.ipynb)
- [01 订单宽表与样本规则](notebooks/01_order_mart_and_sample_rules.ipynb)
- [02 商圈、时段与履约阶段诊断](notebooks/02_fulfillment_metric_diagnostic.ipynb)
- [03 分决策时点风险模型与跨期验证](notebooks/03_late_risk_model.ipynb)
- [04 网格化运力优先级与离线策略回放](notebooks/04_peak_capacity_scenario.ipynb)
- [SQL 指标脚本](sql/)：核心 KPI、商圈时段表和阶段差分解
- [最终业务报告](docs/08_final_business_report.md)

## 数据来源与口径

- 官方仓库：<https://github.com/meituan/Meituan-INFORMS-TSL-Research-Challenge>
- 观察期：2022-10-17 至 2022-10-24
- 主要样本：完成链路有效、即时配送、匿名商圈已知
- 主指标：`is_late = arrive_time > estimate_arrived_time`

原始数据不纳入版本控制，许可和使用范围以官方仓库文件为准。8 天数据适合短周期履约诊断和跨日验证，不适合月度趋势、季节性预测或长期排班。

## 复现顺序

建议使用 Python 3.12。安装依赖并准备官方原始数据后，在项目根目录运行：

```powershell
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
```

如需分步骤执行，可依次运行：

```powershell
python scripts/profile_raw_data.py
python scripts/build_order_mart.py
python scripts/build_metric_marts.py
python scripts/train_late_risk_model.py
python scripts/build_capacity_scenario.py
python scripts/build_report_figures.py
python scripts/run_sql_checks.py
```

Notebook 已全部执行并保留输出，可直接在 GitHub 中审阅。完整流水线还会在本地 `outputs/notebook_previews/` 生成 HTML 预览，该目录不纳入版本控制。

原始数据、处理后数据、模型二进制及分析输出均不纳入版本控制。仓库保留已执行 Notebook 作为结果审阅载体，所有核心指标均可通过上述流程重新生成。

无需原始数据即可执行仓库完整性检查：

```powershell
python scripts/audit_repository.py
```

## 关键方法边界

- 运单表一行是一次派单尝试，不等于一个订单。
- 派单订单与候选骑手表是同一时点的候选集合，不可按时间直接做订单—骑手配对。
- 阶段时长差是描述性分解，不是因果归因。
- 骑手接单时模型具备更多过程信息，但对应更晚的运营处置时点；订单推送时模型信息较少，但更适合事前预警与资源准备。
- 运力槽位是假设性资源预算，不等同于骑手工时或真实排班；策略增量效果仍需通过线上对照试点验证。

## 数据许可与致谢

原始数据由美团提供，适用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.zh-hans)；未经许可不得对外分发原始数据，数据及其衍生作品仅限非商业用途。仓库仅保留分析代码、已执行 Notebook 和汇总结论，不包含原始数据、处理后明细数据或模型二进制。

本研究由美团提供数据支持。

## 文档导航

- [项目立项书](docs/01_project_charter.md)
- [指标字典](docs/02_kpi_dictionary.md)
- [分析路线](docs/03_analysis_roadmap.md)
- [原始数据清单](docs/04_raw_data_inventory.md)
- [样本与建表规则](docs/05_sample_and_table_rules.md)
- [模型设计](docs/06_model_design.md)
- [运力情景设计](docs/07_capacity_scenario.md)
- [最终业务报告](docs/08_final_business_report.md)
