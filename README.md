# 可执行的 Wasserstein-HMM 跨资产策略

本项目把 Boukardagha (2026)《Explainable Regime-Aware Investing》的核心思路转成严格因果、可回测、可执行的 ETF 配置策略。它追求更好的风险调整后收益与更低回撤，但**不保证高 Sharpe 或未来盈利**。

当前生产配置为
`config/paper_core_growth_gold20_r11_diversified_financing.yaml`（R11）。
策略管理资金先按 25% R11、75% R9 运行。R11 保留 R9 的市场环境判断、风险
缩放、恢复持仓和条件性 VIXY 保护；完整 R11 把每个非现金目标统一乘以 1.065，
现金承担差额，并保留因果 SMH 隔夜保护。它只把放大后黄金目标的最多 10% 用 GDE
表达，GDE 仍属于黄金袖套，不属于成长仓。Panel 未取得真实持仓，或下次开盘前
尚未取得上一交易日的完整收盘数据时，订单不可执行，不得把参考权重当成券商订单。

## 2026-07-27 R11 分散融资风险预算分阶段发布

R11 的主要改进来自对整个非现金组合的小幅放大，不再依赖大比例 GDE。由于当前
只让 25% 资金使用 R11，账户实际非现金目标倍数为
`75% × 1 + 25% × 1.065 = 1.01625`，即只比 R9 增加 1.625%。当前中心目标为
QQQ 43.5752%、SMH 19.8220%、GLD 22.0215%、GDE 0.4898%、现金/BIL
14.0915%。GDE 允许区间为 0%—0.9898%；如果账户当前没有 GDE，已经在区间内，
无需为了接近中心值而交易。

按当前成本假设，25% R11 在 2015 年至今、2006 年起长期替代数据、GDE 上市后
真实数据中的年化复合收益率分别比纯 R9 高 0.6196、0.3117、0.7208 个百分点，
历史最大回撤分别为 -15.58%、-16.83%、-15.58%。当普通交易成本、GDE 成本和
融资利差同时升高时，三个窗口的年化优势仍为 0.4501、0.2156、0.5548 个百分点，
最大回撤为 -16.80%、-17.13%、-15.77%。

R11 在普通数据 1,698 条和长期替代数据 1,808 条 SMH 保护参数路径中均排名第 1；
21、63、126 日三种相关性假设下，六个多重比较校正后概率为
3.74%—4.82%，全部低于 5%。三个独立模型编号在当前与压力成本下的收益差和
风险调整后收益差全部为正；逐年删除一个年份、逐次删除一次保护事件后，优势
最小值仍为正。5,000 次 GDE 跟踪偏差模拟在 25%、50%、100% 发布比例下均满足
收益为正、风险调整后收益不差于 R9、历史回撤不超过 18% 三项目标。

这里的严格检验只覆盖 SMH 因果保护参数族及登记的 R11 候选，不是项目历史上每个
想法的统一检验。初始只发布 25% 是为了前瞻验证真实融资利率、成交滑点、行情
时效和持仓流程；至少记录 63 个冻结后的交易日后，才审查是否扩大到 50% 或
100%。若实际融资利差高于每年 1.50%、GDE 单边成交成本高于 0.75%，或出现数据
完整性问题，不得扩大。

复现当前目标与最终审计：

```bash
PYTHONPATH=src:.:tools .venv/bin/python tools/refresh_r11_production.py
PYTHONPATH=src:.:tools .venv/bin/python tools/build_r11_production_overlay.py
PYTHONPATH=src:.:tools .venv/bin/python tools/evaluate_r11_final_candidate_audit.py
PYTHONPATH=src:.:tools .venv/bin/python tools/evaluate_r11_final_candidate_robustness.py
```

日常生产刷新统一使用第一条命令，并安排在美东时间 16:15 之后运行。该流程会刷新
核心资产、GDE 和已审计个股，重算 R9、R11 与个股市场状态，再自动导出 Panel
快照。程序在美股盘中拒绝生成生产信号，并要求核心、GDE、R9、R11 和个股数据都
对齐到最近已完整收盘的交易日；核心与 GDE 最近五个交易日不连续时也会失败。
个别非核心资产缺值若采用代理，日期和方法必须写入行情元数据。目前 2026-07-27
的 VIXY 缺值使用前一收盘价作平价代理；QQQ、SMH、GLD、现金及市场环境判断没有
使用该代理。

用户自行选择成长或半导体个股时，默认生产叠加配置为
`config/r9_individual_stock_overlay_v2.yaml`。它不改变 R9 核心，也不自动选股；只有
市场处于安静/正常上涨、所选篮子过去 63/126 日都跑赢对应 ETF、且股票相对 ETF
的 63 日收益分化不在过去三年最高三分之一时，才允许个股替代 QQQ/SMH。之后再用
60/252 日差异风险、单只 5%、合计 20% 三道上限控制仓位。v1 保留为需要真实选股
记录支持的高捕获模式。最终验证见
`output/individual_stock_regime_final_validation/validation_summary.md`。

## 2026-07-27 SMH 周内保护第三轮验证

R9 生产配置不变。完全因果的“前日极端跳空、次日开盘降风险”中间版在
2015 至今回测提高约 0.93 个百分点年化收益，上涨捕获率约 99.65%，但
2,240 组参数的家族级选择偏差校正概率约为 47%–50%，没有通过生产证据门槛。
新增的 2002 年起真实隔夜代理还显示：未过滤版本在唯一的 2011 早期事件中
为负贡献；加入半导体相对 QQQ 弱势过滤后，早期变成零触发，不能算正向独立
验证。历史尾部预算、转入 QQQ 和 1–2 日快速恢复也均未通过。因此同开盘保护
与次日因果规则都只保留为影子/研究路径，不生成生产订单。完整规则、数据哈希、
成本压力和被否决方向见
`research/smh_intraweek_risk_optimization_round3_2026-07-27.md`。

## 2026-07-27 个股与 ETF 切换层再验证

继续检验了四类改进：按相对强弱连续决定个股仓位、QQQ 与半导体分别计算分化、
取两组中更高分化作为系统警报，以及 0.25%–1.5% 的个股层不交易区间。没有候选
同时改善 2015–2021 开发期和 2022–2025 近期的较差 10% 组合；因此生产配置继续
使用 63/126 日相对强弱确认和全股票池 67% 分化硬门槛，不因本轮研究改动交易。

截至 2026-07-24，QQQ 成长股组分化位于过去约三年的 56.75%，半导体组为
87.96%，但“分别放行”在历史上恶化了近期尾部和最大回撤，所以两个数字只用于
诊断，不能取代当前全局门控。完整审计位于
`output/individual_stock_regime_optimization_20260727/optimization_summary.md`。

## 2026-07-26 R9 个股与 ETF 环境切换层发布

新默认模式把“市场是否适合承担成长风险”与“是否值得承担个股特有风险”分开。40
个随机篮子的 2022—2025 中位年化相对对数收益由旧规则的 -0.42% 改善为 +0.06%，
较差 10% 由 -1.38% 改善为 -0.18%；2015—2025 中位年化复合收益为 19.57%，较差
10% 为 19.42%，中位最大回撤为 -16.31%。21 日成块重复抽样区间仍跨过零，因此
它是默认防错与表达控制，不是未来额外收益承诺。当前股票分化位于过去三年的约
87% 分位，所以证据门控模式使用 ETF；下一次月末审查为 2026-07-31 收盘后。

## 2026-07-26 R9 个股受控替代层发布

旧 Panel 用个股总波动相对 ETF 的“风险倍数”逐只扣减 ETF，并把未用风险额度留成
现金。该方法重复削减了 R9 本来就要承担的市场风险，40 个随机股票篮子的中位
2015—2025 年化复合收益率只有 16.38%。新版本改为控制股票篮子收益减去对应 ETF
收益后的差异波动，并与 ETF 等金额替换，不额外留现金。

最终规则在 40 个随机篮子中的中位年化复合收益率为 19.78%，10% 较差结果为
18.84%；中位最大回撤为 -16.36%，10% 较差结果为 -17.46%。这些结果只验证风险
框架，不验证自动选股。2022—2025 的中位随机篮子需要约 4.22% 的额外年化选股
收益才能追平纯 R9，较差 10% 篮子需要约 14.74%。因此该叠加层只适合用户确有
选股优势的前提，不能被描述成无条件提高收益。

## 2026-07-25 R9 分阶段重入发布

R9 的改动只发生在“成长仓位已经接近零、市场开始恢复、主模型仍处于防守”这一小段时间。最初只建立 20% QQQ；恢复信号连续十个交易日后，才允许增加到 35% QQQ。这样把两个错误成本分开处理：先降低完全错过快速反弹的成本，再用时间确认限制假反弹中的损失。

2026-07-25 数据审计后，VIX 与 VIX3M 改为逐日采用 Cboe 官方历史，调仓与模型更新时间也改为不依赖数据起点的固定工作日序号。按修复后的共同日程，2015–2025 年化复合收益为 19.40%，R8 为 18.85%；最大回撤为 -16.57%，R8 为 -17.52%。2006–2025 长期替代数据的年化复合收益为 14.56%，R8 为 14.07%；最大回撤为 -17.65%，R8 为 -19.52%。完整来源、文件指纹、官方对账、修正前后隔离实验和剩余限制见 `output/r9_data_audit_2026-07-25/数据审计报告.md`。

在下一交易日开盘成交并假设每单位调仓金额产生 0.15% 成本时，修复后的 R9 年化复合收益为 17.63%，R8 为 17.26%。滚动三年仍有短暂落后 R8 的窗口，最差年化差约为 -1.45 个百分点。日程修复后尚未重做覆盖全部历史候选的多重比较统计检验，因此旧版该项统计结论不再视为当前证据。

## 2026-07-25 R8 分段生产发布

R8 在冻结的 2015–2025 窗口 CAGR 19.05%、Sharpe 1.241、最大回撤 -14.23%，旧生产分别为 16.69%、1.149、-13.78%；15bps 成本下 CAGR 为 17.57% 对 15.46%，2012–2025 为 17.38% 对 14.89%，2006–2025 代理为 13.56% 对 12.35%。37 个候选家族的 selection-adjusted p 值最高 3.88%，一日执行延迟后仍有 +1.26 个百分点 CAGR 优势，剔除最佳重入事件后的正常和代理样本增量均为正。20 年代理在 15bps 下仍有 +1.03 个百分点 CAGR 优势，但 MDD 为 -18.69%，所以 18% 不是高成本或未来保证；滚动 3 年跑赢率 82.7%，最差年化相对收益 -2.87%，2008–09 与 2011 是已知相对失效区间。

旧生产到 R8 的当前单边迁移为 35.54%，因此不允许一次性全量切换。四个连续交易日各向当日重算目标移动 25%，每段模型单边换手约 8.88%，15bps 假设总显性成本约 5.33bp NAV。历史任意上线日重放的活跃路径 5% 分位机会成本约 -0.48%，最差约 -2.78%；这说明分段降低操作冲击但不是免费择时。Cboe 官方 VIX3M 现在用于按日期精确补齐 Yahoo 缺口，数据来源、补值数量与最终剩余缺值写入每次 `run_metadata.json`。完整准入、迁移与不可变哈希见 `output/forward_monitoring/freeze_manifest_2026-07-25-r8-v3.json`。

## 2026-07-23 生产替换

新策略按预先固定的门槛替换旧策略，而不是按最高 CAGR 直接上线。2015–2025 开盘可执行代理（7.5 bps）CAGR 16.60%、Sharpe 1.14、最大回撤 -13.40%，旧策略分别为 14.56%、1.10、-16.96%。开发期 CAGR 改善 +2.15pp，2022–2025 留出期 +1.83pp，15 bps 成本下 +1.56pp，2012 扩展起点 +1.63pp。18 个合格候选的 family-wise Reality Check p=6.34%，通过 10% 门槛；21/63/126 日区块重采样突破 18% 回撤的频率为 62.8%/44.7%/17.0%，仍说明 18% 不是未来保证。

生产规则只使用少量、具有经济含义的离散设定：6 个月行业相对动量、12 个月双趋势加仓、20/60 日压力波动、最多 1.10 倍敞口、期限结构倒挂时 4% VIXY。没有继续围绕通过者微调比例或回看期；2026-07-23 起的新成交必须作为前瞻记录，不能再用于回改本版本。

项目提供多套目标不同的冻结配置：

- `config/default.yaml`：低波动、低回撤优先。
- `config/cagr_first.yaml`：CAGR 优先，固定 80% QQQ/SMH 等权核心与 20% Wasserstein-HMM 跨资产卫星，不使用杠杆。
- `config/paper_core_growth.yaml`：HMM 条件收益控制 QQQ/SMH 增长核心，其余时间持有防御资产。
- `config/paper_core_dd18.yaml`：在其余参数不变时，把论文核心版的预测波动目标机械提高至 22%，用于检验放宽至 18% 回撤预算后的结果。
- `config/paper_core_sticky.yaml`：风险退出即时、只在 HMM 重估日重新进入的换手实验；已因回撤超限而淘汰。
- `config/paper_core_hmm_only.yaml`：删除额外趋势过滤器的纯 HMM 实验；已因回撤超限而淘汰。
- `config/paper_core_risk_parity.yaml`：用收缩后的 HMM 条件波动对 QQQ/SMH 做逆波动配置；未产生实质改善。
- `config/paper_core_macro_equal.yaml`：只用论文原始宏观资产定义 HMM 状态、QQQ/SMH 固定等权；已因回撤超限而淘汰。
- `config/paper_core_macro_risk_parity.yaml`：宏观 HMM 与 QQQ/SMH 逆波动配置的组合实验；已因回撤超限而淘汰。
- `config/paper_core_smh_satellite.yaml`：20% SMH 长趋势/波动目标卫星；改善留出期但未达到事前定义的全周期改善门槛。
- `config/paper_core_smh_bridge.yaml`：只在主策略 risk-off、SMH 长趋势已转正时启用卫星；效果过小，未升级为推荐配置。
- `config/paper_core_vix_guard.yaml`：VIX/VIX3M 倒挂时立即禁止 risk-on；收益与回撤均恶化，已淘汰。
- `config/paper_core_vix_recovery.yaml`：倒挂结束但 HMM 尚未重新进入时启用 20% QQQ/SMH 恢复桥；未产生实质改善。
- `config/paper_core_vrp_feature.yaml`：把滞后 VRP 加入 HMM；开发期不稳且回撤超限，已淘汰。
- `config/paper_core_bipower_guard.yaml`：用日频 bipower 代理区分跳跃与持续波动；信号增量过少，未产生改善。
- `config/paper_core_vix_trend_recovery.yaml`：只在期限结构恢复且增长趋势为正时启用恢复桥；未产生改善。
- `config/paper_core_smh_high_vol_budget.yaml`：SMH 进入四年波动率最高 10% 时改用逆波动核心；与基准近似等价，未升级。
- `config/paper_core_robust_vol_guarded_floor_ensemble.yaml`：资产级重入优化的稳健比较基线；三个固定随机初始化等权、总风险敞口封顶 100%，并使用急性波动保护的 20% risk-off 增长底仓。
- `config/paper_core_robust_vol_guarded_floor_ensemble_2012.yaml`：同一规则的 2012 起点稳定性版本。
- `config/paper_core_zero_entry_growth_reallocation_ensemble.yaml`：无杠杆增长基准；仅在零仓位高风险重入时把部分 SMH 转成 QQQ，保持总增长敞口。
- `config/paper_core_growth_gold20_daily_risk_ensemble.yaml`：上一代生产策略；QQQ/SMH/GLD 战略核心，6 个月 QQQ/SMH 相对动量，双资产 252 日趋势为正时最高 1.10 倍总敞口，每日跳跃感知压力波控，并在 VIX/VIX3M 倒挂时配置 4% VIXY。
- `config/paper_core_growth_gold20_jump_aware_daily_risk_ensemble.yaml`：机制驱动的改进候选；用 20 日标准波动处理突发跳空、60 日双幂变差识别持续波动，须经 2026 纸面期确认后才能替换操作默认。
- `config/paper_core_growth_gold20_lev110_target25_jump_stresscorr_daily_risk_ensemble.yaml`：2026-07-23 文献整合后的高 CAGR 影子候选；风险开启时最高 1.10 倍总敞口，并用跳跃感知波动与 20/60 日保守相关性控制 QQQ/SMH 联合风险。它通过预设点估计门槛，但家族 Reality Check 尚未通过，因此不替换当前 Panel。
- `config/paper_core_growth_gold20_lev110_target25_jump_stresscorr_industrymom_daily_risk_ensemble.yaml`：新的最佳影子候选；用过去 126 个交易日的行业相对动量在 QQQ/SMH 内迁移增长风险，不改变增长总预算。2015–2025 开盘代理 CAGR 16.26%、MDD -16.00%，相对生产提高 1.70pp CAGR；20 候选家族 Reality Check p=0.0851，但区块重采样尾部仍弱于生产，因此不替换当前 Panel。完整审计见 `output/industry_momentum_optimization_2026-07-23.md`。
- `config/paper_core_zero_entry_growth_gold20_lev110_ensemble.yaml`：轻杠杆研究对照；收盘口径较强，但开盘执行代理 MDD 为 -21.43%，已从生产候选降级。
- `config/paper_core_zero_entry_growth_gold20_lev110_accountcap_ensemble.yaml`：全账户 20% 压力波动接入实验；用于量化零仓首次建仓保护的长期代价，不替代综合主候选。
- `config/paper_core_zero_entry_state_switch_ensemble.yaml`：按清仓期 SMH/QQQ 相对表现选择转 QQQ 或 BIL 的探索候选；点估计更高但更依赖 2024。
- `config/paper_core_vx_futures_guard.yaml`：使用 Cboe 官方 VX1/VX2 结算曲线做精确倒挂硬否决；显著恶化，已淘汰。
- `config/paper_core_relative_momentum.yaml`：保持总增长敞口不变，只按6/12个月相对动量在QQQ/SMH内部倾斜；完整年度退化，已淘汰。
- `config/paper_core_growth_floor.yaml`：risk-off时保留20% QQQ/SMH等权底仓；目前最优研究候选，但未达到事前 +1pp 门槛。
- `config/paper_core_growth_gold20_dual_reentry_inverse_momentum_netted_ensemble.yaml`：R8 的直接对照与已晋升前身；把零仓 20% 桥接与已有仓 40% floor 分开，并在高波动期把相对动量切换为 60 日逆波动。单独版本未通过多重检验，但其机制与当前目标被 R8 保留。
- `config/paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear20_inverse_momentum_netted_ensemble.yaml`：R8 回退配置；在 Dual 上使用事先测试过的 45% 已有仓位保护，并仅在高波动、20 日下行变差占比不超过 50%、252 日超额收益不低于 -20% 时允许 20% 零仓桥接。
- `config/paper_core_growth_gold20_r9_staged_reentry_netted_ensemble.yaml`：R11 的 75% 核心和回退配置；从接近零仓位恢复时先配置 20% QQQ，信号连续十个交易日后才允许提高到 35%。
- `config/paper_core_growth_gold20_r10_capital_efficient_guard.yaml`：上一版分阶段生产配置；因严格 SMH 参数族检验未通过而被 R11 替代。
- `config/paper_core_growth_gold20_r11_diversified_financing.yaml`：当前分阶段生产配置；25% 策略管理资金统一放大 R9 非现金目标并保留小比例 GDE 与 SMH 因果隔夜保护，其余 75% 保留 R9。
- `config/paper_probability_dd15.yaml`：HMM 有利模板的后验概率连续控制增长敞口，并按 12%-15% 最大回撤预算设置硬风控。

## CAGR-first 版本

```bash
python scripts/run_backtest.py --config config/cagr_first.yaml --refresh
```

2015-01-02 至 2026-07-21、7.5 bps 成交成本下：CAGR 21.96%、年化波动 21.63%、零利率 Sharpe 1.03、相对 BIL Sharpe 0.94、最大回撤 -32.46%。同期 SPY CAGR 为 13.74%、最大回撤 -33.72%。成本翻倍至 15 bps 后，CAGR 为 21.75%、最大回撤为 -32.56%。

固定窗口诊断：2015-2021 CAGR 22.02%（SPY 14.84%）；2022-2025 CAGR 16.84%（SPY 11.05%）。策略没有使用参数网格搜索，QQQ/SMH 固定等权，80/20 核心卫星比例在首次运行前冻结，结果不用于反向修改参数。

必须注意：QQQ/SMH 是在已经知道其长期高收益后选入的，因此资产选择本身仍存在事后选择偏差。2022-2025 只能视为固定窗口诊断，不是学术意义上的完全未见留出集。真正的抗过拟合证据需要从现在开始冻结配置，进行至少 6-12 个月前瞻纸面交易。

输出位于 `output/cagr_first/`，双倍成本结果位于 `output/cagr_first_cost15/`。

## 12%-15% 最大回撤约束

```bash
python scripts/run_backtest.py --config config/paper_probability_dd15.yaml --refresh
```

该配置在首次运行前固定：HMM 有利状态概率经 0.12 EMA 平滑，线性映射为 25%-125% 的 QQQ/SMH 等权增长敞口；条件协方差施加 22% 预测波动率上限；组合回撤达到 5%/7.5%/10%/12% 后，风险仓位依次乘以 75%/50%/25%/0%，其余配置 BIL。所有状态估计和持仓均为因果计算。

2015-01-02 至 2026-07-21 的首次冻结结果为：CAGR 6.36%、年化波动 10.41%、相对 BIL Sharpe 0.46、最大回撤 -14.08%。同期 SPY CAGR 13.74%、最大回撤 -33.72%。该版本满足历史回撤目标，但**没有满足大幅跑赢 SPY 的收益目标**，不能把它包装为目标已实现。主要代价来自硬回撤闸门：37.0% 的再平衡点为零风险。

现有未经筛选版本呈现的风险收益前沿如下。它是约束冲突的证据，不是从多组参数中挑选出的最好结果：

| 冻结配置 | CAGR | 最大回撤 | 结论 |
| --- | ---: | ---: | --- |
| `cagr_first.yaml` | 21.96% | -32.46% | 达到高 CAGR，回撤不达标 |
| `paper_core_growth.yaml` | 15.11% | -16.43% | 接近回撤上限，只小幅跑赢 SPY |
| `paper_core_dd18.yaml` | 14.35% | -17.71% | 回撤达标，但提高风险后 CAGR 反而下降 |
| `paper_core_sticky.yaml` | 14.26% | -19.06% | 降低换手但错过反弹，淘汰 |
| `paper_core_hmm_only.yaml` | 14.36% | -22.51% | 删除辅助趋势风控后回撤超限，淘汰 |
| `paper_core_risk_parity.yaml` | 14.57% | -16.76% | 回撤达标，但 CAGR 与 Sharpe 均低于固定等权 |
| `paper_core_macro_equal.yaml` | 11.04% | -22.04% | 宏观状态对科技压力反应较慢，淘汰 |
| `paper_core_macro_risk_parity.yaml` | 11.10% | -22.20% | 风险平价未修复宏观信号错配，淘汰 |
| `paper_core_smh_satellite.yaml` | 15.46% | -15.73% | 风险调整指标改善，但完整年度 CAGR 增幅不足 |
| `paper_core_smh_bridge.yaml` | 15.14% | -16.43% | 只修复少数重入窗口，效果过小 |
| `paper_core_vix_guard.yaml` | 11.71% | -19.27% | 倒挂时机械退出导致反弹追涨，淘汰 |
| `paper_core_vix_recovery.yaml` | 14.50% | -16.82% | 恐慌消退不等于增长趋势恢复，未通过门槛 |
| `paper_probability_dd15.yaml` | 6.36% | -14.08% | 回撤达标，CAGR 不达标 |

在仅使用长仓 ETF、每周交易和论文所述 HMM 框架的范围内，当前证据不支持同时声称“CAGR 大幅跑赢 SPY”与“最大回撤 12%-15%”。若继续围绕同一历史区间搜索状态标签、阈值和杠杆，最可能增加的是过拟合，而不是可复制收益。

放宽到 15%-18% 后，`paper_core_growth.yaml` 是现有结果中更稳健的论文核心候选。机械提高风险预算的 `paper_core_dd18.yaml` 在 7.5 bps 成本下取得 14.35% CAGR、-17.71% 最大回撤；但 15 bps 成本压力下退化为 12.93% CAGR、-19.24% 最大回撤。固定分段同样不支持稳定超额：2015-2021 CAGR 12.83%（SPY 14.84%），2022-2025 CAGR 10.00%（SPY 11.05%）。该配置因此不应作为优选版本上线。

进一步优化采用结构消融而非参数网格。原版的 46 次风险状态切换贡献了 61% 的总换手，因此测试了“退出即时、重入等待 HMM 重估”；它虽将年化成本拖累从 0.94% 降至 0.86%，但 CAGR 降至 14.26%、最大回撤扩大至 -19.06%。随后删除论文以外的趋势过滤器，CAGR 为 14.36%、最大回撤恶化至 -22.51%。两项改动均未通过预先设定的 18% 回撤门槛，未进入双倍成本复核。原 `paper_core_growth.yaml` 因此保持推荐状态，而不是从连续参数搜索中挑出的局部最优值。

QQQ/SMH 暴露优化使用预先固定的 2×2 消融，而非搜索历史最佳比例。条件逆波动版本把 QQQ 的平均核心权重从 50% 提高至 57.5%，但因两者高度相关，组合年化波动只从 16.07% 降至 15.92%；同时降低了 SMH 收益贡献并增加动态调权成本，最终 CAGR 14.57%、最大回撤 -16.76%、相对 BIL Sharpe 0.81，未超过固定 50/50。宏观 HMM 版本的风险开启比例从 70.4% 升至 79.7%，对科技专属压力退出过慢，两个宏观版本的最大回撤均超过 22%，2022-2025 CAGR 约 5%。三项候选均未达到事前定义的改善门槛，因此没有进入双倍成本复核，也没有用于反调权重公式。

SMH 的进一步利用采用独立、严格因果的 200 日趋势与 60 日波动率目标，而没有修改 HMM 阈值。固定 20% 账户卫星在趋势为正时按 20% 年化波动率缩放 SMH、否则持有 BIL，并把账户 SMH 目标权重限制在 70%。它把 2015-2025 CAGR 从 12.83% 提高到 13.22%、最大回撤从 -16.43% 改善到 -15.73%；2022-2025 CAGR 从 10.92% 提高到 12.44%。但开发期 CAGR 从 13.93% 降至 13.66%，全周期增幅也未达到事前要求的 1 个百分点，故不升级。

随后只做了一项结构性复核：卫星仅在主策略仍为 risk-off、但 SMH 趋势已经转正时启用，避免平时稀释主策略。该桥接仓在再平衡点的平均有效比例只有 2.75%，2015-2025 CAGR 仅升至 12.93%，2022-2025 仅升至 11.23%，最大回撤仍为 -16.43%，开发期最大回撤反而由 -15.26% 扩大到 -16.16%。这个结果说明可被简单 SMH 趋势修复的 HMM 重入滞后不是主要收益瓶颈；继续搜索趋势天数、卫星比例或启用阈值会明显提高过拟合风险。因此该轮历史结论仍保留 `paper_core_growth.yaml`。

## 波动率 Meta-Survey 扩展

《Equity Volatility Regimes: A Meta-Survey》不是一份披露完整回测方法的单一策略论文，而是对已实现波动率、隐含波动率期限结构、三状态 SWARCH 和市场微观结构的综述。最可操作的三项结论是：低/中波动状态通常持续，高波动状态约占历史的 10% 且突发、短暂；`VIX > VIX3M` 的 backwardation 代表近期急性恐慌；短期已实现波动率上升会触发波控、CTA 和风险平价的机械去杠杆。现有 Wasserstein-HMM 已经覆盖状态持续性和已实现波动率，因此扩展只检验期限结构，不把 VIX 指数加入可交易资产，也不做波动率卖空。

第一项冻结实验在上一收盘 `VIX > VIX3M` 时立即否决 QQQ/SMH risk-on，没有搜索阈值。2008-2026 的日度倒挂比例为 10.22%，与文章所称的稀有状态方向一致；但该规则把 2015-2025 CAGR 从 12.83% 降到 10.06%，最大回撤从 -16.43% 扩大到 -19.27%。2022 年结果完全不变，因为原 HMM/趋势规则当时已经退出；损失主要来自 2017、2020 和 2024-2026 的急跌后反弹。换手成本也因反复退出/追入而上升。这正是文章所警告的顺周期波控行为，故硬闸门淘汰。

第二项实验利用“高波动状态短暂”的结论：VIX 曲线从倒挂恢复 contango、但原 HMM 仍为 risk-off 时，暂时把 20% 账户预算配置到 QQQ/SMH 等权；HMM risk-on 或再次倒挂即结束。20% 沿用此前冻结的卫星预算，没有设置持续天数或搜索曲线幅度。桥接在 16.35% 的再平衡点启用，平均账户敞口为 3.27%；2015-2025 CAGR 为 12.21%、最大回撤 -16.82%，开发期 CAGR 13.10%，2022-2025 CAGR 10.66%，均未超过基线。期限结构正常化只说明急性恐慌消退，并不证明科技趋势已恢复。

两项波动率扩展均未通过事前门槛（2015-2025 CAGR 至少增加 1 个百分点、最大回撤不超过 18%、开发期和留出期不出现方向性恶化），所以没有进入双倍成本测试，也没有继续搜索 VIX 比率、确认天数或桥接比例。第二篇文章增强了对尾部风险和顺周期去杠杆的解释，但该轮数据不支持把 VIX 期限结构升级为额外交易开关；该轮历史基准仍为 `paper_core_growth.yaml`。

## 五个冻结验证方向

在上述两项规则失败后，进一步检验了五个互相独立、运行前即冻结的方向。所有持仓继续只使用上一完整收盘的信息；没有阈值网格、结果后调参或从相邻版本中挑最好值：

1. **VRP HMM 特征**：增加 `VIX² - 21 日年化已实现方差`，在进入 HMM 前滞后一天；不添加正负阈值。
2. **跳跃/持续波动**：用日收益的 bipower variation 作为持续波动代理，保留原 20 日、30% 压力阈值。完整 bipower jump 检验需要日内高频数据；Yahoo 5 分钟历史只覆盖最近 60 天，因此该项明确只是 2015 年以来可复现的日频代理筛查，不冒充论文级日内验证。
3. **趋势确认的 VIX 恢复桥**：只有曲线从倒挂恢复 contango、基准仍为 risk-off、且现有 QQQ/SMH 多周期趋势平均分数不低于零时，才启用冻结的 20% 恢复桥。
4. **SMH 状态依赖风险预算**：正常状态保持 QQQ/SMH 50/50；SMH 60 日波动率进入过去 1008 日的最高 10% 时才改为两资产逆波动。10% 来自文章对高波动状态频率的描述，不从回测搜索。
5. **精确 VX 期货曲线**：从 Cboe 官方逐月合约 CSV 构造每个交易日最前两张未到期合约的结算价；`VX1 > VX2` 时沿用原硬否决，用来隔离 VIX/VIX3M 近似误差。到期日合约在当日曲线中排除，2015-2025 对 ETF 交易日零缺口。

事前验收门槛是：完整 2015-2025 CAGR 至少比基准高 1 个百分点，即不低于 13.83%；最大回撤不超过 18%；开发期 2015-2021 与留出期 2022-2025 的 CAGR 都不得低于基准。结果如下：

| 冻结方向 | 2015-2025 CAGR | Sharpe | 最大回撤 | 开发期 CAGR | 留出期 CAGR | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 基准 `paper_core_growth` | 12.83% | 0.85 | -16.43% | 13.93% | 10.92% | 比较基准 |
| VRP HMM 特征 | 11.42% | 0.78 | -23.26% | 9.60% | 14.69% | 样本不稳定且回撤失败 |
| 日频 bipower 代理 | 12.07% | 0.82 | -16.86% | 13.33% | 9.89% | 收益退化 |
| 趋势确认恢复桥 | 12.51% | 0.84 | -16.43% | 13.43% | 10.92% | 没有增量 alpha |
| SMH 高波动预算 | 12.81% | 0.85 | -16.57% | 13.78% | 11.14% | 与基准无法区分 |
| 精确 VX 倒挂硬否决 | 8.62% | 0.66 | -19.37% | 9.05% | 7.87% | 收益与回撤均失败 |

机制统计解释了失败原因。VRP 改变了 234/581 次 HMM 状态数选择和 47 次最终风险决策；它在开发期大幅失效、留出期改善，说明把一个方向性风险溢价变量直接交给无监督状态聚类会重排状态，却不能保证正确的收益方向。日频 bipower 只改变 8 次压力标记和 5 次 risk-on 决策，增量过于稀疏。趋势确认把恢复桥压缩到 12 次再平衡，但这些交易仍未带来正超额。SMH 高波动预算触发 118 次，触发时 SMH 核心平均只从 50% 降至 43.34%；组合波动目标随后提高总增长敞口，抵消了部分风险削减，所以 CAGR 和回撤几乎不变。

精确 VX 曲线反而强化了原来的否定结论：581 次再平衡中，精确 `VX1 > VX2` 出现 105 次，VIX/VIX3M 近似只出现 48 次，两种测量有 59 次不一致；精确倒挂与基准 risk-on 冲突 42 次，近似规则只有 16 次。期货曲线更频繁地表示短端压力，却仍不提供市场方向；机械硬否决因此更频繁地错过急跌后的风险溢价和反弹，并把完整期年化平均总成本从 0.97% 提高到 1.16%。

为减少“单一历史路径碰巧”的误判，还对候选与基准的日收益差做了固定 21 个交易日区块、10,000 次配对 bootstrap。SMH 高波动预算的年化相对收益 95% 区间为 -0.36% 至 +0.31%，改善概率 48.0%，本质上等于零；精确 VX 的区间为 -6.17% 至 -1.39%，改善概率仅 0.06%。其余三个方向的区间也都包含零且点估计为负。五项均未过第一道 CAGR 门槛，因此按事前规则不做 15 bps 二次成本测试，也不搜索相邻参数。

结论是保留 `paper_core_growth.yaml`，不把五项中的任何一项并入生产候选。完整结果在 `output/five_direction_validation/`；复现命令为：

```bash
PYTHONPATH=src .venv/bin/python tools/fetch_vx_curve.py --first-year 2014 --last-year 2027
.venv/bin/python -m regime_strategy.cli --config config/paper_core_vrp_feature.yaml
.venv/bin/python -m regime_strategy.cli --config config/paper_core_bipower_guard.yaml
.venv/bin/python -m regime_strategy.cli --config config/paper_core_vix_trend_recovery.yaml
.venv/bin/python -m regime_strategy.cli --config config/paper_core_smh_high_vol_budget.yaml
.venv/bin/python -m regime_strategy.cli --config config/paper_core_vx_futures_guard.yaml
PYTHONPATH=src .venv/bin/python tools/evaluate_five_directions.py
```

## 第一性原理增长底仓候选

五个波动率方向失败后，策略设计从“更精确地预测何时退出”改为“减少二元退出造成的风险溢价损失”。先冻结并检验了一个QQQ/SMH相对动量覆盖：HMM和20%波动目标仍决定QQQ+SMH总敞口，只用跳过最近21日的126/252日相对动量把SMH在核心中的比例连续限制于30%-70%。该版本截至2026-07-21的全期CAGR由15.11%升至15.49%，但完整2015-2025 CAGR从12.83%降至12.61%，最大回撤扩大到-17.92%，开发期和留出期都退化。全期改善来自2026年后的SMH强势，不能据此升级；该实验未调整窗口或倾斜幅度。

第二个、结构上独立的冻结候选是 `paper_core_growth_floor.yaml`。它不预测QQQ与SMH的相对强弱，也不改变HMM风险状态：

- HMM risk-on时完全沿用原策略；
- HMM risk-off时，80%账户继续持有原防御目标，20%配置QQQ/SMH等权；
- 随后仍受20%预测波动率上限和原有-10%/-15%/-20%回撤乘数约束；
- 每5个交易日评估，单边换手低于1%不交易，成本保持7.5 bps；
- 20%沿用此前已冻结的账户卫星预算，没有搜索10%/20%/30%。

它解决的是动作空间不对称：原二元策略在29.6%的再平衡点完全放弃增长风险溢价，永久底仓则保留小部分反弹参与度。581次再平衡中底仓启用172次；受波动目标、回撤闸门和实际权重漂移影响，启用时QQQ+SMH平均实际仓位为16.77%。风险状态本身与基准零分歧。

| 指标 | 原 `paper_core_growth` | 20% risk-off增长底仓 | 差异 |
| --- | ---: | ---: | ---: |
| 2015-2025 CAGR | 12.83% | 13.67% | +0.84pp |
| 2015-2025 Sharpe | 0.85 | 0.89 | +0.04 |
| 2015-2025最大回撤 | -16.43% | -17.35% | -0.92pp |
| 2015-2021 CAGR | 13.93% | 15.11% | +1.17pp |
| 2022-2025 CAGR | 10.92% | 11.18% | +0.26pp |
| 全期至2026-07-21 CAGR | 15.11% | 15.83% | +0.72pp |
| 全期最大回撤 | -16.43% | -17.35% | -0.92pp |

候选在11个完整年度中有9年高于基准，但收益并非均匀：2022年比基准低5.02个百分点，2019、2023和2025分别高4.06、4.18和3.34个百分点。上涨/下跌日记账归因显示，它在QQQ上涨日增加10.45%的年化相对对数收益，同时在下跌日损失9.71%，净增约0.74%；这与失败的VX硬否决方向相反，但安全边际很小。由于风险切换幅度减少，年化平均总成本还从约0.97%降至0.87%。

固定21日区块、10,000次配对bootstrap给出的年化相对收益点估计为+0.74%，95%区间为-0.90%至+2.40%，正改善概率81.31%。完整2015-2025 CAGR增幅距离事前+1个百分点门槛仍差0.16个百分点，因此 `overall_pass=0`：没有继续搜索底仓比例，也没有按通过者规则运行15 bps版本。它在该轮“最大回撤不超过18%”的约束目标下是较优研究候选，但证据尚不足以替换当时的生产基准。

复现命令与结果：

```bash
.venv/bin/python -m regime_strategy.cli --config config/paper_core_relative_momentum.yaml
PYTHONPATH=src .venv/bin/python tools/evaluate_relative_momentum.py
.venv/bin/python -m regime_strategy.cli --config config/paper_core_growth_floor.yaml
PYTHONPATH=src .venv/bin/python tools/evaluate_growth_floor.py
```

相对动量结果位于 `output/relative_momentum_validation/`，增长底仓结果位于 `output/growth_floor_validation/`。该轮结论是 `paper_core_growth.yaml` 保持基准、增长底仓只用于纸面跟踪；顶部“当前结论”优先于此处历史记录。

## 第一性原理稳健组合：多初始化、无杠杆、波动保护底仓

进一步审计发现，单一 HMM 的低回撤结论会随 EM 随机初始化改变。两次随机重启时，种子 7/42/123 的最大回撤分别可达到约 -27.6%/-17.0%/-17.4%；增加到八次重启仍不能收敛到相同交易风险。原因不是标签排列，而是若干近似似然局部解会在关键日期给出不同的二元 risk-on 决策。少量错误开启日若同时使用超过 100% 的 QQQ/SMH 敞口，会放大尾部；错误关闭又会错过恢复。继续选择“最好种子”或增加状态阈值属于结果驱动过拟合。

该轮首选研究候选 `paper_core_robust_vol_guarded_floor_ensemble.yaml` 因此使用以下固定结构：

- 独立运行种子 7、42、123 三个 HMM 子账户，每个占账户三分之一；种子集合在稳健性测试前固定，不按结果选优。
- 每个成员使用当前 HMM 拟合的后验混合收益矩、全局日历锚定的 5/21/63 日交易/重估/选阶计划，并把 QQQ+SMH 总敞口封顶为 100%，不借款加杠杆。
- risk-on 时持有 QQQ/SMH 等权增长核心并受 20% 预测波动目标约束；risk-off 时保留固定 20% 的 QQQ/SMH 等权底仓，以减少二元退出丢失长期风险溢价。
- 当 QQQ 过去 20 日年化实现波动率超过原策略已经冻结的 30% 压力阈值时，20% 底仓关闭；慢速 200 日趋势为负本身不关闭底仓。这样长期底仓不会覆盖急性波动风控，也没有新增阈值。
- 三个子账户每 21 个交易日尝试恢复等权，偏离不足 1% 不交易；成员内部和外层都按双边成交额计 7.5 bps。当前样本中外层偏离一直小于 1%，因此外层成本近似为零，但代码仍显式计费。

截至 2026-07-21 的净成本后结果如下。`完整期` 固定为 2015-2025，避免用 2026 年 SMH 强势反向选择模型：

| 指标 | 原 `paper_core_growth` | 稳健组合 | 差异 |
| --- | ---: | ---: | ---: |
| 2015-2025 CAGR | 12.83% | 14.93% | +2.10pp |
| 2015-2025 Sharpe | 0.85 | 1.01 | +0.16 |
| 2015-2025 最大回撤 | -16.43% | -16.86% | -0.43pp |
| 2015-2021 CAGR | 13.93% | 13.92% | -0.01pp |
| 2022-2025 CAGR | 10.92% | 16.73% | +5.81pp |
| 全期至 2026-07-21 CAGR | 15.11% | 16.97% | +1.86pp |
| 全期至 2026-07-21 Sharpe | 0.96 | 1.10 | +0.14 |
| 15 bps、2015-2025 CAGR | 11.44% | 14.16% | +2.72pp |
| 15 bps 最大回撤 | -16.83% | -16.88% | -0.05pp |

从 2012 开始运行时，候选至 2025 年 CAGR 为 14.31%、Sharpe 1.00、最大回撤 -16.86%；原 2012 基准分别为 9.82%、0.71、-22.51%。2012 与 2015 两种起点在共同区间的日收益相关性为 0.9967、最大回撤相同，共同区间 CAGR 分别为 16.50% 和 16.97%。这显著弱化了原实现的起点路径依赖。

统计证据仍需保守解释。2015-2025 的 21 日配对区块 bootstrap 给出年化相对收益 +1.86%，95% 区间 -1.75% 至 +5.79%，正改善概率 84.0%；2012-2025 的区间为 +0.65% 至 +7.77%，正改善概率 99.1%。但在 24 个正式且回撤不超过 18% 的候选中，Reality Check 家族 p 值为 0.450。候选还因开发期 CAGR 比基准低约 1.1 个基点，没有通过事前设定的“任何分段不得退化”零容忍布尔门槛。因此它只是该轮风险回报较强、工程上可执行的**纸面候选**，不是已证明的可复制 alpha。

以 2026-07-21 收盘数据计算的下一交易日理论等权成员目标为 QQQ 37.56%、SMH 37.56%、BIL 24.88%，其余为零。实际下单应使用 `next_target_weights.csv` 中的 `ensemble_current_sleeve_weight` 列，以反映三个子账户的实时净值漂移。复现命令：

```bash
# 日常生成下一交易日目标；联网刷新调整后价格
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli --config config/paper_core_robust_vol_guarded_floor_ensemble.yaml --refresh

# 冻结缓存复现与验证
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli --config config/paper_core_robust_vol_guarded_floor_ensemble.yaml
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli --config config/paper_core_robust_vol_guarded_floor_ensemble.yaml --cost-bps 15 --output-dir output/paper_core_robust_vol_guarded_floor_ensemble_cost15
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli --config config/paper_core_robust_vol_guarded_floor_ensemble_2012.yaml
PYTHONPATH=src:tools .venv/bin/python tools/evaluate_robust_ensemble.py
```

完整统计位于 `output/robust_vol_guarded_floor_validation/`。

## 资产级重入风险：SMH 高 beta，而不是与 QQQ 失去相关性

2026 年 7 月事件暴露了原稳健组合的动作缺口。三个成员在 6 月 30 日至 7 月 14 日都持有零增长仓位，但 7 月 15 日同时恢复 risk-on，把 QQQ+SMH 从 0% 一步提高到约 79.3%。7 月 15–17 日组合损失 -4.34%。用 7 月 14 日前一完整收盘计算，QQQ/SMH 的 20/60 日压力波动分别约 28.7%/60.9%，60 日相关系数约 0.94，SMH 对 QQQ 的 beta 约 2.0。因此问题不是相关性脱钩，而是原 QQQ 压力闸门在 QQQ 波动刚降回 30% 以下时重新开放，而 SMH 的两倍因子敏感度仍未恢复。

先检验了三种广泛波控：全时逆波动加 20% 硬上限、只在月度重配时使用、以及每日只降不升。它们把全样本最大回撤压到约 -13% 至 -14%，但 CAGR 降到约 14.9%–15.2%，原因是高波动不等于负期望收益，机械降仓长期放弃了 SMH 风险溢价；每日规则还把年化成本拖累提高到约 1.12%。这些版本全部淘汰。

该轮更窄的生产候选 `paper_core_zero_entry_growth_reallocation_ensemble.yaml` 只处理一个状态转换：

- 只有上一实际 QQQ+SMH 权重为零、当前首次转为 risk-on，且原目标的 20/60 日压力实现波动超过既有 20% 风险预算时才激活；
- 保持 QQQ+SMH 总敞口不变，不额外降杠杆；
- 首次重入按压力逆波动把部分 SMH 转成 QQQ，使两者账户风险贡献接近相等；
- 到下一个既有 5 日评估即恢复正常模型目标，不新增确认天数或优化阈值；
- 其余所有日期与稳健基线完全相同。

固定比较如下。所有收益均已计 7.5 bps 双边成交额成本；`7 月重入损失` 是 2026-07-15 至 2026-07-17。表内全样本点估计是研究快照，不用于替代实时 Panel：

| 版本 | 2015–2025 CAGR | Sharpe | MDD | 开发期 CAGR | 留出期 CAGR | 全样本 CAGR | 7 月重入损失 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 稳健基线 | 14.93% | 1.011 | -16.86% | 13.92% | 16.73% | 17.01% | -4.34% |
| 增长优先：SMH 转 QQQ | 14.99% | 1.015 | -16.86% | 13.96% | 16.80% | 17.08% | -3.78% |
| 风险优先：SMH 转 BIL | 14.96% | 1.016 | -16.86% | 13.92% | 16.81% | 17.10% | -2.77% |
| 相对状态分流 | 15.06% | 1.020 | -16.86% | 13.94% | 17.06% | 17.19% | -2.77% |

增长优先版是该轮主候选，因为它在完整期、开发期、留出期、2012 起点和 15 bps 双倍成本下都没有方向性退化，而且逐年留一的最差相对年化收益仅约 -0.004 个百分点。2012–2025 CAGR 从 14.31% 提高到 14.41%，15 bps 的 2015–2025 CAGR 从 14.16% 提高到 14.20%。20/60 日窗口改成 10/40 或 40/120、20% 激活预算改成 18% 或 22% 时，完整期 CAGR 都维持在约 15.04%–15.06%，说明结果不是单点参数尖峰。

`paper_core_zero_entry_state_switch_ensemble.yaml` 是探索候选：清仓期 SMH 相对 QQQ 为负时把超额 SMH 风险转 BIL，否则转 QQQ。它的点估计更高且对当前事件保护更强，但剔除 2024 年后相对年化收益约为 -0.02%，不能因样本内更好而取代更朴素的主候选。

统计证据仍然不足。增长优先版固定 21 日区块 bootstrap 的正改善概率约 76.6%，95% 区间跨零；把当前目录中 38 个回撤可行候选纳入 family-wise Reality Check 后 p 值约 0.98。资产级重入规则应理解为机制明确、长期代价很小的风险修复，不是已证明的新 alpha。完整审计位于 `output/zero_entry_tail_relative_risk_validation/`。

## 开盘执行修正与当前可执行候选

早期结果把前一完整收盘形成的信号作用于“前收盘到当日收盘”收益，相当于能在信号形成的同一收盘价成交。真实次日开盘无法避开隔夜跳空。复权 Open/Close 审计把每天拆成“旧仓隔夜收益、开盘调仓、新仓盘中收益”，发现原无杠杆增长版的 2015–2025 MDD 从 -16.86% 恶化为约 -22.99%；20% GLD+1.10× 版本也从 -16.44% 恶化为约 -21.43%。这些收盘口径候选因此不能再视为满足 18% 目标。

当前可执行候选 `paper_core_growth_gold20_daily_risk_ensemble.yaml` 使用 40% QQQ、40% SMH、20% GLD risk-on 核心，不使用杠杆；在每个交易日只允许 20/60 日压力波动规则降低现有 QQQ/SMH 仓位，不因波动下降而在非计划日加仓。GLD 提供结构分散，日波控处理 2020 年 risk-off 后仍保留的增长底仓风险。GLD 不是逐日崩盘保险：QQQ/SMH 最差 5% 日里，GLD 平均约 +0.02%，只有 45.8% 的日期为正，其价值来自约 0.03 的 252 日相关系数中位数和跨周期再平衡。

净 7.5 bps 成本的开盘执行代理结果如下；这仍不是未来保证：

| 口径 | CAGR | Sharpe | MDD | 同期 SPY CAGR | SPY MDD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2015–2025 | 14.56% | 1.101 | -16.96% | 13.45% | -33.72% |
| 2015–2025，15 bps 成本 | 13.80% | 1.050 | -17.09% | 13.45% | -33.72% |
| 2012–2025 | 13.42% | 1.053 | -16.96% | 14.90% | -33.72% |

因此只能得到两个分开的结论：候选在 2015 起点、正常成本下历史跑赢 SPY 约 1.11 个百分点并把 MDD 降到 18% 内；15 bps 下超额仅约 0.35 个百分点，估算成本盈亏平衡约为 18.5 bps；2012 起点则落后 SPY 约 1.47 个百分点，更没有“大幅跑赢” QQQ/SMH。高 CAGR 轻杠杆版虽把 2015–2025 开盘代理 CAGR 提到约 14.74%，MDD 也升到约 -17.30%，只换来约 0.18 个百分点，不足以补偿杠杆、融资和换手复杂度，故降级为研究对照。

区块重采样进一步限制了 18% 数字的解释。固定 20% 压力波动目标在 21/63/126 日区块下，替代路径突破 18% 回撤的频率约为 63.8%/55.3%/37.7%；它显著优于 SPY 的尾部，但不能提供“最大回撤不会超过 18%”的保证。相邻 18%/20%/22% 波动目标的历史开盘 CAGR 为 14.04%/14.56%/14.81%，MDD 为 -16.11%/-16.96%/-17.70%，说明 20% 不是孤立参数尖峰；22% 距硬约束仅余约 0.30 个百分点，因此不作为默认。

跳跃感知版来自波动率综述中的跳跃-扩散分解，而非收益曲线搜索：它把 60 日慢波动从标准差改为双幂变差，同时保留 20 日标准差以便突发时立即减仓。冻结窗开盘 CAGR 为 14.65%，15 bps 下为 13.90%，2012 起点为 13.50%，MDD 与基准相同；三种区块长度下相对 CAGR 改善概率约 93.7%–95.4%，但 95% 区间仍轻微跨零，绝对改善只有约 0.10 个百分点。纳入本轮 12 个历史回撤可行候选后，它相对 SPY 的年化对数超额约 +1.06%，名义单侧 p=0.389、家族 Reality Check p=0.438，排名第 5，因此只能视为机制更干净的工程修复，不是统计上已证明的 alpha。完整审计位于 `output/jump_aware_validation/` 和 `output/open_family_reality_check/`。

更保守的风险容量前沿也没有消除路径不确定性。把增长压力波动目标从 22% 依次降到 20%/18%/16%/14%，冻结窗开盘 CAGR 为 14.81%/14.56%/14.04%/13.57%/12.80%，历史 MDD 为 -17.70%/-16.96%/-16.11%/-14.84%/-13.56%。即便 14% 目标，21 日区块重采样突破 18% 的频率仍约 37.4%，且 CAGR 已落后 SPY；14%–22% 没有一点同时通过“所有区块突破概率低于 25%”与稳健基准超额。完整前沿位于 `output/risk_capacity_frontier/`。

扩大无交易带也没有实质改善。跳跃感知版的 1%/2%/3% 无交易带对应开盘 CAGR 14.65%/14.68%/14.70%，MDD -16.96%/-17.10%/-17.19%；换手只减少 1.3%/2.3%，Sharpe 反而略降。因此保留 1%，不按最高样本 CAGR 继续细搜阈值。审计位于 `output/no_trade_band_validation/`。

2026 数据已经用于问题诊断和当前 Panel，不能再伪装成完全未见留出集。跳跃感知规则在 2026-07-22 收盘前冻结；真正前瞻的成交与信号日志从 2026-07-23 开始累计，至少观察 8–12 周后才允许根据实盘证据升级结论。在此之前，2026 报告只能用于检查机制是否发生明显反向失效，不能用于宣称新 alpha。

发布 `2026-07-22-v1` 已把六个核心执行模块、两份候选配置、验收结果与当前 Panel 的 SHA-256 写入 `output/forward_monitoring/freeze_manifest_2026-07-22-v1.json`；同名发布无法覆盖。标准版与跳跃感知版的 2026-07-22 信号也已写入不可变 `signal_log.csv`，执行日为 2026-07-23。任何核心逻辑变更都必须使用新发布名并重启前瞻计时，不能回写本版本。

零仓首次接入再加一层操作保护：先按压力逆波动把部分 SMH 转为 QQQ，再对 QQQ+SMH+GLD 的 20/60 日联合压力协方差执行 20% 全账户波动上限。它是首个持仓周期的风险预算，不是回撤保证；下一既有评估点才决定是否向长期模型目标增仓。开盘执行证据位于 `output/open_execution_validation/`，结构分散证据位于 `output/structural_diversification_validation/` 和 `output/gold_hedge_mechanism_validation/`。

当前零仓位 70 万美元账户可生成位置感知面板：

```bash
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli \
  --config config/paper_core_growth_gold20_daily_risk_ensemble.yaml \
  --refresh
PYTHONPATH=src .venv/bin/python tools/build_operational_panel.py \
  --account-value 700000 \
  --current-qqq 0 --current-smh 0 --current-gld 0 --current-bil 0
```

刷新必须在纽约交易日 16:15 以后进行；盘中日线会被丢弃，盘中写入的旧缓存即使在盘后读取也仍视为不完整。操作文件位于 `output/current_operational_panel/`，应以其中 `panel.md` 和 `order_plan.csv` 的最新时间状态为准：只有 `READY_FOR_OPEN` 或 `UPCOMING` 且 `executable=True` 的订单才可进入下一步人工复核；`MISSED`、`DRAFT_*` 或 `executable=False` 表示执行窗口已经失效，必须等新的完整收盘后刷新重算，禁止按旧价追单。

零仓位不能把模型的 `HOLD` 解读为继续空仓；Panel 默认生成全账户 20% 压力波动接入目标，同时显示一周 1σ 金额。后者不是最大损失。用户若不能承受超过开盘代理历史 -16.96%、甚至重采样路径中更深的组合回撤，不应仅靠分批买入掩盖风险承受能力不匹配。

复现命令：

```bash
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli \
  --config config/paper_core_growth_gold20_daily_risk_ensemble.yaml
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli \
  --config config/paper_core_growth_gold20_daily_risk_ensemble_2012.yaml
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli \
  --config config/paper_core_growth_gold20_daily_risk_ensemble.yaml \
  --cost-bps 15 \
  --output-dir output/paper_core_growth_gold20_daily_risk_ensemble_cost15
PYTHONPATH=src .venv/bin/python tools/evaluate_open_execution.py --refresh
PYTHONPATH=src:tools .venv/bin/python tools/evaluate_executable_candidate.py
PYTHONPATH=src:tools .venv/bin/python tools/evaluate_jump_aware_overlay.py
PYTHONPATH=src:tools .venv/bin/python tools/evaluate_live_execution.py
PYTHONPATH=src:tools .venv/bin/python tools/evaluate_risk_capacity_frontier.py
PYTHONPATH=src:tools .venv/bin/python tools/evaluate_no_trade_bands.py
PYTHONPATH=src:tools .venv/bin/python tools/evaluate_open_family_reality_check.py
PYTHONPATH=src .venv/bin/python tools/snapshot_forward_signal.py
PYTHONPATH=src .venv/bin/python tools/freeze_strategy_release.py
PYTHONPATH=src .venv/bin/python tools/evaluate_structural_diversification.py
PYTHONPATH=src:tools .venv/bin/python \
  tools/evaluate_zero_entry_tail_relative_risk.py
```

## 论文结论与复现边界

论文报告的 OOS Sharpe 为 2.18、最大回撤为 -5.43%，但没有披露 ticker、训练/测试切分、HMM 和优化器的完整超参数，也未明确给出滑点。论文公式还把 15 维特征分布参数直接写入 5 资产 MVO，存在维度缺口。因此这里不声称“复现 2.18”，而是采用可审计的实现：HMM 识别特征状态，历史后验概率估计各状态的**下一期资产收益矩**，再经 Wasserstein 模板平滑后进入 MVO。

## 默认交易规则

- 资产：SPY（股票）、IEF（美债）、GLD（黄金）、DBC（商品/油价代理）、UUP（美元），BIL 作为现金替代。
- 特征：上一交易日收益、滞后 60 日波动率、滞后 20 日均值；任何 `t` 日持仓都只使用 `t-1` 收盘前可得信息。
- 模型：4 年滚动窗口；HMM 状态数在 2/3/4 中按过去 126 日预测似然选择；每 21 个交易日重估，每 63 日重选状态数。
- 身份：对角高斯的 2-Wasserstein 距离映射到 4 个持久模板，模板以 0.12 EMA 更新。
- 组合：长仓、满仓、单一风险资产不超过 40%，条件均值/协方差收缩，L1 换手惩罚。
- 稳健先验：最终优化仓位向五个风险资产的 `1/N` 组合收缩 50%，降低均值估计误差和模型失效风险。
- 稳健收益预测：HMM 条件均值占 35%，固定的 1/3/6/12 月波动调整趋势占 65%；该比例和周期不在 OOS 上搜索。
- 风控：9% 年化波动率目标；SPY 200 日绝对动量为负或 20 日年化波动超过 25% 时，将 SPY/商品仓位降至 25%；组合从历史峰值回撤 4%/7%/10% 时，将全部风险仓位依次降至 75%/50%/25%，余款转入 BIL。
- 执行：每 5 个交易日评估一次；低于 1% 的单边换手不交易；按总成交名义金额收取 7.5 bps 成本。

信号在收盘后生成，下一交易日开盘附近执行。先计算 `目标权重 - 当前实际权重`，若单边总换手低于 1% 则不交易；否则使用限价单或参与率算法。实盘系统必须把实际成交回写为下一次优化的 `previous_weights`，不能假设理论目标全部成交。

## 运行

```bash
source .venv/bin/activate
python -m pip install -e .
python scripts/run_backtest.py --refresh
```

第二次运行可省略 `--refresh`，使用 `data/prices.csv` 缓存。输出位于 `output/backtest_v3/`；`output/backtest/` 和 `output/backtest_v2/` 保留前两版基线：

- `metrics.csv`：净成本后零利率 Sharpe、相对 BIL 的 `cash_excess_sharpe`、Sortino、最大回撤等；
- `annual_metrics.csv` / `paper_window_metrics.csv`：逐年和论文同窗口的稳健性拆分；
- `daily_returns.csv`：每日收益、成本、换手和回撤；
- `weights.csv`：每日开盘时持仓；
- `regimes.csv`：HMM 状态数、模板概率和预测波动；
- `performance.png`：净值、回撤和权重图。
- `next_target_weights.csv`：最新完整收盘后的下一交易日动作与权重。`HOLD` 表示非计划再平衡日，不应下单；`REBALANCE` 才计算目标差额。
- `next_signal_diagnostics.csv`：最新 HMM 状态、模板概率、预测波动和风控开关。
- `run_metadata.json`：本次模型使用的完整收盘截止日、生成时间和配置路径；Panel 会拒绝与价格截止日不一致的旧模型输出。

测试：

```bash
.venv/bin/python -m pytest -q
```

## 上线前门槛

不要仅凭全样本 Sharpe 上线。至少要求：多个市场子区间 Sharpe 同号；成本翻倍后仍有正 Calmar；参数在相邻取值下结果连续；纸面交易 8-12 周的信号、成交和回写无误。当前可执行候选不使用杠杆；新账户应使用 Panel 的全账户 20% 压力波动接入目标，并禁止在 `MISSED` 或 `STALE_MODEL` 状态追单。由于 2012 起点仍未跑赢 SPY、15 bps 下超额很薄，不应把它宣传为稳定的基准超额收益策略。

## 文献驱动的 residual / dynamic-correlation 复核

基于 Moreira–Muir、Cederburg 等、Engle–Sheppard、Residual Momentum、Risk-Constrained Kelly、PBO 与 Deflated Sharpe Ratio，又冻结检验了两个无参数搜索候选。36 个月 beta + 12–1 月残差动量候选的 2015–2025 开盘 CAGR 为 14.40%、MDD -16.96%，低于生产基线的 14.56% 且没有改善回撤；13 个风险可行候选的 familywise p=0.460。使用 20/60 日压力相关性、超预算时先削 SMH 的动态因子风险候选，开盘 CAGR 为 14.46%、MDD -16.90%，只改善 0.07pp 回撤并牺牲 0.09pp CAGR，也不升级。完整论文矩阵、机制诊断与 bootstrap 证据位于 `output/literature_integration_validation/decision.md`。
