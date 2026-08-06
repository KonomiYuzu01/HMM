import { stockRiskData } from "./stock-risk-data";
import { strategyLiveData } from "./strategy-live-data";

const formatPercent = (value: number, digits = 2) =>
  `${(value * 100).toFixed(digits)}%`;
const stockDispersion = stockRiskData.activeDispersionPercentile;
const stockDispersionEligible =
  stockDispersion <= stockRiskData.activeDispersionMaximumPercentile;
const decisionAuthority = strategyLiveData.decisionAuthority;
const relativeDamageVeto = strategyLiveData.relativeDamageVeto;
const panelStatus = strategyLiveData.panelStatus;
const recursiveTrendCushion = strategyLiveData.recursiveTrendCushion;
const pputProtectedCapacity = strategyLiveData.pputProtectedCapacity;
const activeStrategy = panelStatus.activeStrategy;
const r39Active = activeStrategy === "R39";
const volatilityAccelerationBlock =
  "volatilityAccelerationBlock" in strategyLiveData.riskBudget
    ? strategyLiveData.riskBudget.volatilityAccelerationBlock
    : false;
const stateActiveMultiplier =
  "stateActiveMultiplier" in strategyLiveData.riskBudget
    ? strategyLiveData.riskBudget.stateActiveMultiplier
    : 1.3;
const generationMode =
  "generationMode" in strategyLiveData
    ? strategyLiveData.generationMode
    : "STANDARD_REFRESH";
const executionWindowStatus =
  "executionWindowStatus" in strategyLiveData
    ? strategyLiveData.executionWindowStatus
    : "UPCOMING";
const marketEnvironmentLabel = {
  incomplete: "数据不完整",
  healthy: "健康趋势",
  acute_liquidity_shock: "急性流动性冲击",
  ordinary_correction: "普通修正",
  structural_damage: "结构性破坏",
  early_repair: "早期修复",
}[decisionAuthority.marketEnvironment] ?? "状态未知";
const r39FallbackUserMessage =
  "R39 的额外半导体保护暂未启用；当前继续使用已验证的 R38 组合。";

export const strategySnapshot = {
  release: strategyLiveData.release,
  authorityRelease: decisionAuthority.release,
  priceAsOf: strategyLiveData.priceAsOf,
  generatedAt: strategyLiveData.generatedAtJst,
  nextExecutionEt: strategyLiveData.nextExecutionEt,
  nextExecutionJst: strategyLiveData.nextExecutionJst,
  nextExecutionWeekdayZh: strategyLiveData.nextExecutionWeekdayZh,
  generationMode,
  executionWindowStatus,
  accountBuildTiming: strategyLiveData.accountBuildTiming,
  totalOneWayTurnover: 0,
  trancheTurnoverLimit: 0.1,
  noTradeThreshold: 0.01,
  estimatedMigrationCostRate: 0,
  stagedGdeCenter: strategyLiveData.stagedTarget.GDE,
  stagedGdeLower: strategyLiveData.stagedLower.GDE,
  stagedGdeUpper: strategyLiveData.stagedUpper.GDE,
  goldSleeveTarget: strategyLiveData.goldSleeveTarget,
  stockReplacementShare: 0.5,
  nextStockIncreaseReview: "2026-07-31 收盘后",
  stockIncreaseAllowedNow: stockDispersionEligible,
  relativeDamageVeto,
  panelStatus,
  activeStrategy,
  requestedStrategy: panelStatus.requestedStrategy,
  fallbackActive: panelStatus.fallbackActive,
  fallbackReason: panelStatus.fallbackActive
    ? r39FallbackUserMessage
    : panelStatus.fallbackReason,
  fallbackTechnicalReason: panelStatus.fallbackReason,
  recursiveTrendCushion: strategyLiveData.recursiveTrendCushion,
  pputProtectedCapacity: strategyLiveData.pputProtectedCapacity,
  r11Reference: strategyLiveData.r11Reference,
} as const;

export const productionTargets = {
  staged: strategyLiveData.stagedTarget,
  r11: strategyLiveData.r11Reference,
} as const;

export const accountBuildRules = {
  requiredNewCompletedCloses:
    strategyLiveData.accountBuildTiming.requiredNewCompletedCloses,
  remainingGapThreshold: 0.01,
  fractionOfRemainingGap: 0.25,
  maximumOneWayTurnover: 0.10,
  r38RolloutMinimumSessions: 63,
} as const;

export const referencePrices = {
  priceDate: strategyLiveData.priceAsOf,
  gdePriceDate: strategyLiveData.priceAsOf,
  ...strategyLiveData.referencePrices,
} as const;

const growthTarget =
  strategyLiveData.stagedTarget.QQQ + strategyLiveData.stagedTarget.SMH;

export const decisionPipeline = [
  {
    stage: "01 · 状态输入",
    source: "R9 / HMM",
    title: `${strategyLiveData.riskOnVotes} / ${strategyLiveData.riskOnVoteTotal} 个模型处于风险偏好`,
    evidence: `市场被解释为“${marketEnvironmentLabel}”，但 R9 当前动作是 ${strategyLiveData.r9Action}；QQQ 63 日收益 ${formatPercent(decisionAuthority.qqq63Return)}，SMH 63 / 126 日收益 ${formatPercent(decisionAuthority.semis63Return)} / ${formatPercent(decisionAuthority.semis126Return)}。`,
    rule: "HMM 状态与 R9 动作是上游输入，不直接等于最终组合；后续风险层仍会继续限制仓位。",
    result: `保留 ${formatPercent(growthTarget)} 成长仓，而不是全部转成现金`,
    plainLanguage: "市场的大方向仍被判断为可参与上涨，所以系统没有把资金全部撤到现金；但这还不是加仓命令。",
    counterfactual: "若 HMM 或 R9 状态转弱，基础组合会先重算；不会继续沿用今天的成长仓比例。",
  },
  {
    stage: "02 · 总风险",
    source: "R11 + R38",
    title: strategyLiveData.riskBudget.oneSessionShockActive
      ? "单日异常波动刹车覆盖趋势加仓"
      : "趋势与波动共同决定总风险",
    evidence: `双 200 日趋势许可为${strategyLiveData.riskBudget.trendPermission ? "开" : "关"}；短期波动加速为${volatilityAccelerationBlock ? "是" : "否"}；单日刹车为${strategyLiveData.riskBudget.oneSessionShockActive ? "开" : "关"}。`,
    rule: strategyLiveData.riskBudget.oneSessionShockActive
      ? `1.070 × ${stateActiveMultiplier.toFixed(3)} 的趋势请求被单日刹车压到 ${strategyLiveData.riskBudget.acceptedAbsoluteMultiplier.toFixed(3)}；下一完整收盘重新判断。`
      : `基础倍数 1.070 × 状态倍数 ${stateActiveMultiplier.toFixed(3)}，再受现金不得低于 −20% 的硬限制。`,
    result: `现金 / BIL ${formatPercent(strategyLiveData.stagedTarget.cash)}，VIXY ${formatPercent(strategyLiveData.stagedTarget.VIXY)}`,
    plainLanguage: strategyLiveData.riskBudget.oneSessionShockActive
      ? "方向仍偏多，但市场最近太颠簸；今天可以靠近目标，不能利用趋势信号把风险进一步放大。"
      : "趋势和短期波动都没有阻止风险放大，因此系统允许较高的非现金参与度。",
    counterfactual: strategyLiveData.riskBudget.oneSessionShockActive
      ? `若下一完整收盘后单日刹车不再触发，才会重新评估 ${stateActiveMultiplier.toFixed(3)} 状态倍数是否可用。`
      : "若单日刹车触发，会立刻覆盖趋势放大，将绝对倍数压到 1.000。",
  },
  {
    stage: "03 · 成长仓结构",
    source: r39Active ? "R39" : "R38",
    title: r39Active
      ? "SMH 相对损失触发账户级集中度保护"
      : "使用 R38 的慢速 QQQ / SMH 锚点",
    evidence: r39Active
      ? `SMH 过去 ${relativeDamageVeto.lookbackTradingDays} 日相对 QQQ 回落 ${formatPercent(relativeDamageVeto.relativeLoss)}；保护前 SMH 权重 ${formatPercent(strategyLiveData.parentStagedTarget.SMH)}。`
      : r39FallbackUserMessage,
    rule: r39Active
      ? `${formatPercent(strategyLiveData.parentStagedTarget.SMH)} × ${formatPercent(relativeDamageVeto.relativeLoss)} = ${formatPercent(relativeDamageVeto.proposedAccountRelativeLoss)}，高于 ${formatPercent(relativeDamageVeto.accountRelativeLossBudget)} 账户预算；硬保护把 SMH 限制为成长仓的 ${formatPercent(relativeDamageVeto.maximumActiveSemisGrowthShare)}。`
      : "R38 只向 126 日相对强弱端点移动 10%，不扩大成长仓总额。",
    result: r39Active
      ? `QQQ ${formatPercent(strategyLiveData.stagedTarget.QQQ)} · SMH ${formatPercent(strategyLiveData.stagedTarget.SMH)}；压力估算降至 ${formatPercent(relativeDamageVeto.implementedAccountRelativeLoss)}`
      : `QQQ ${formatPercent(strategyLiveData.stagedTarget.QQQ)} · SMH ${formatPercent(strategyLiveData.stagedTarget.SMH)}`,
    plainLanguage: r39Active
      ? "成长仓的总金额不变，只是在 QQQ 和 SMH 之间重新分配，避免半导体单独受伤时拖累过大。"
      : "当前只使用经过验证的 R38：半导体仍是成长仓主力，但系统只小幅向中性比例靠拢。",
    counterfactual: r39Active
      ? "若相对损失保护解除，QQQ / SMH 会回到 R38 慢速锚点，不会自动增加成长仓总额。"
      : "只有 R39 的额外保护完成全部安全验证后，才会重新评估是否将它加入当前目标。",
  },
  {
    stage: "04 · 长期熊市保护",
    source: "R40 账户净值安全垫",
    title: recursiveTrendCushion.productionEligible && recursiveTrendCushion.sameDate
      ? "R40 已进入私有账户执行链"
      : "R40 未通过同日资格，阻断账户草稿",
    evidence: `历史 1931–2007 代理最深回撤 ${formatPercent(recursiveTrendCushion.historicalPre2008MaxDrawdown)}；现代 CAGR ${formatPercent(recursiveTrendCushion.modernCagr)}，相对基准 ${formatPercent(recursiveTrendCushion.modernCagrDelta)}。`,
    rule: `保护底线 = 账户高水位 × ${formatPercent(1 + recursiveTrendCushion.floorDrawdown, 0)}；非现金请求上限 = ${recursiveTrendCushion.dualTrendPositive ? recursiveTrendCushion.bullMultiplier : recursiveTrendCushion.bearMultiplier} × 安全垫 ÷ 当前净值，再向下取整到 ${formatPercent(recursiveTrendCushion.tierSize, 0)} 档。`,
    result: recursiveTrendCushion.productionEligible && recursiveTrendCushion.sameDate
      ? "私有面板用真实账户净值计算；触发后关闭 R38 增量风险并增加现金"
      : "不生成账户调整草稿",
    plainLanguage: "它看的不是市场是否像熊市，而是你的策略资金距离自身历史高点还剩多少亏损空间；空间越小，允许持有的非现金资产越少。",
    counterfactual: "若账户回到新高，安全垫恢复并允许完整 R38；若双趋势转弱，同样安全垫会使用更低的 9.5 倍，减仓更快。",
  },
  {
    stage: "05 · R41 保护性容量",
    source: "R41 · SPYM 5% 虚值长期看跌期权",
    title: pputProtectedCapacity.productionEligible && pputProtectedCapacity.sameDate
      ? pputProtectedCapacity.active
        ? "R41 保护已确认，120% 上限可以参与计算"
        : "R41 已通过资格，但尚未持有合格保护；继续使用 R40"
      : "R41 尚未通过同日生产资格；继续使用 R40",
    evidence: `现代净 CAGR ${formatPercent(pputProtectedCapacity.modernCagr)}、最大回撤 ${formatPercent(pputProtectedCapacity.modernMaxDrawdown)}；1931–2007 代理最深回撤 ${formatPercent(pputProtectedCapacity.historicalPre2008MaxDrawdown)}。研究已按保护名义每年 ${formatPercent(0.02)} 的额外实施拖累扣费。`,
    rule: `期权覆盖 = SPYM 价格 × 100 × 合约数 ÷ 策略资金，目标 ${formatPercent(pputProtectedCapacity.targetCoverage, 0)}，只接受 ${formatPercent(pputProtectedCapacity.minimumCoverage, 0)}–${formatPercent(pputProtectedCapacity.maximumCoverage, 0)}。50 万美元、SPYM 90.52 美元时选择 ${pputProtectedCapacity.example500kContracts} 张，实际覆盖 ${formatPercent(pputProtectedCapacity.example500kCoverage, 2)}。`,
    result: pputProtectedCapacity.active
      ? `双趋势为正且安全垫为正常状态时，非现金上限可到 ${formatPercent(pputProtectedCapacity.normalNonCashCap, 0)}`
      : "当前不启用 120% 扩展；R40 仍是实际账户保护层",
    plainLanguage: "保护资格和实际持仓是两件事。只有账户里确实存在合格长仓看跌期权，系统才用这份保护换取额外上涨容量；没有就按原 R40 行事。",
    counterfactual: `若实时报价、期权权限、整数合约覆盖或已持仓任一项未确认，立即失效关闭，不生成期权订单，也不把非现金上限提高到 ${formatPercent(pputProtectedCapacity.normalNonCashCap, 0)}。`,
  },
  {
    stage: "06 · 执行门控",
    source: "账户建仓规则",
    title: strategyLiveData.ordersExecutable
      ? "目标与真实持仓都已完整"
      : "模型目标已生成，但订单仍不可执行",
    evidence: strategyLiveData.ordersExecutable
      ? "生产快照已包含完整真实持仓。"
      : "生产快照缺少完整真实持仓及实际 GDE 比例，无法计算买卖差额。",
    rule: "订单数量 = 模型目标市值 − 当前真实市值；任何一侧缺失，都不能用猜测补齐。",
    result: strategyLiveData.ordersExecutable
      ? "可以生成订单草稿，仍须人工复核"
      : "先录入并核对持仓，不生成过期或猜测订单",
    plainLanguage: "策略知道理想比例，但只有知道你实际持有什么，才能判断该买、卖还是不动；它不会替你猜。",
    counterfactual: "真实持仓、现金与 GDE 比例核对完成后，系统只会生成调整草稿；仍不会自动向券商提交订单。",
  },
] as const;

export const rejectedAlternatives = [
  {
    question: "为什么不是全部转成现金？",
    answer: `HMM 是 ${strategyLiveData.riskOnVotes} / ${strategyLiveData.riskOnVoteTotal} 风险偏好，双 200 日趋势许可为${strategyLiveData.riskBudget.trendPermission ? "开" : "关"}，市场环境为“${marketEnvironmentLabel}”。完整规则仍给出 ${formatPercent(growthTarget)} 成长仓；单日刹车只取消融资，不自动清空风险资产。`,
  },
  {
    question: "为什么不继续维持高 SMH？",
    answer: r39Active
      ? `保护前账户相对损失估算 ${formatPercent(relativeDamageVeto.proposedAccountRelativeLoss)} 已超过 ${formatPercent(relativeDamageVeto.accountRelativeLossBudget)} 预算；R39 因此把 QQQ / SMH 调为成长仓内 50 / 50。`
      : "R39 的额外半导体保护尚未启用；本次继续使用已验证的 R38 成长仓结构。",
  },
  {
    question: "为什么现在不能生成订单？",
    answer: "目标比例不是订单。没有真实股数、现金与 GDE 实际比例，就不知道应买卖多少；系统宁可阻断，也不部署猜测订单。",
  },
] as const;

export const decisionChangeConditions = [
  "HMM 状态或双 200 日趋势改变时，重新计算总风险",
  "下一完整收盘后，重新判断单日刹车与波动加速",
  "SMH 相对损失或集中度门槛解除时，重新计算 QQQ / SMH 比例",
  "真实持仓与实际 GDE 比例完整后，才计算可复核订单草稿",
] as const;

export const readinessChecks = [
  {
    label: "生产策略",
    value: `${activeStrategy} 已完成安全验证`,
    detail: panelStatus.fallbackActive
      ? `${panelStatus.activeQualification} 项检查通过；R39 额外保护当前待命`
      : `${panelStatus.activeQualification} 项检查通过；底层仍是 25% R38 与 75% R11`,
    tone: panelStatus.fallbackActive ? "caution" : "good",
  },
  {
    label: "市场数据",
    value: `截至 ${strategyLiveData.priceAsOf}`,
    detail:
      executionWindowStatus === "MISSED"
        ? "使用最新完整收盘数据部署；对应开盘窗口已过，当前不会生成订单"
        : "核心资产与 GDE 使用同一个完整交易日，近期交易日连续",
    tone: executionWindowStatus === "MISSED" ? "caution" : "good",
  },
  {
    label: "趋势风险预算",
    value: strategyLiveData.riskBudget.trendPermission
      ? "允许提高上涨参与"
      : "只使用基础风险",
    detail: strategyLiveData.riskBudget.oneSessionShockActive
      ? "单日异常波动刹车正在生效"
      : volatilityAccelerationBlock
        ? `短期波动正在加速，使用原 R38 的 1.30 相对倍数`
        : `短期波动未加速，当前允许 1.375 相对倍数`,
    tone: strategyLiveData.riskBudget.oneSessionShockActive ? "caution" : "good",
  },
  {
    label: "现金硬下限",
    value: strategyLiveData.riskBudget.cashHardLimitActive
      ? "正在限制风险"
      : "未触及 −20%",
    detail: `完整 R38 在 GDE 替代前现金为 ${formatPercent(strategyLiveData.riskBudget.fullPreGdeCashWeight)}`,
    tone: "good",
  },
  {
    label: "个股替代层",
    value: stockDispersionEligible ? "可逐只验证" : "高分化，暂缓替代",
    detail: stockDispersionEligible
      ? `市场分化位于过去三年的 ${formatPercent(stockDispersion, 0)} 分位；股票仍须同时跑赢对应 ETF 的 63 日和 126 日收益`
      : `市场分化位于过去三年的 ${formatPercent(stockDispersion, 0)} 分位，高于 ${formatPercent(stockRiskData.activeDispersionMaximumPercentile, 0)} 上限；证据门控个股当前不替代 ETF`,
    tone: stockDispersionEligible ? "good" : "caution",
  },
  {
    label: "账户持仓",
    value: "尚未录入",
    detail: "不知道真实持仓，不能计算可提交订单",
    tone: "blocked",
  },
  {
    label: "订单",
    value: "不可执行",
    detail: "当前页面只能生成复核建议，不会直接连接券商下单",
    tone: "blocked",
  },
] as const;

export const allocationRows = [
  {
    asset: "QQQ",
    name: "纳斯达克 100",
    oldTarget: strategyLiveData.r11Reference.QQQ,
    firstStage: strategyLiveData.stagedTarget.QQQ,
    finalTarget: strategyLiveData.stagedTarget.QQQ,
    role: "主要成长资产",
    color: "#5a67d8",
  },
  {
    asset: "SMH",
    name: "半导体",
    oldTarget: strategyLiveData.r11Reference.SMH,
    firstStage: strategyLiveData.stagedTarget.SMH,
    finalTarget: strategyLiveData.stagedTarget.SMH,
    role: relativeDamageVeto.active
      ? "账户级相对损失保护正在限制半导体集中度"
      : "在 R9 原比例附近做受限相对强弱调整",
    color: "#0c8f73",
  },
  {
    asset: "GLD",
    name: "黄金",
    oldTarget: strategyLiveData.r11Reference.GLD,
    firstStage: strategyLiveData.stagedTarget.GLD,
    finalTarget: strategyLiveData.stagedTarget.GLD,
    role: "独立黄金风险袖套",
    color: "#b7791f",
  },
  {
    asset: "GDE",
    name: "股票 + 黄金资本效率基金",
    oldTarget: strategyLiveData.r11Reference.GDE,
    firstStage: strategyLiveData.stagedTarget.GDE,
    finalTarget: strategyLiveData.stagedTarget.GDE,
    role: `允许 ${formatPercent(strategyLiveData.stagedLower.GDE)}–${formatPercent(strategyLiveData.stagedUpper.GDE)}；区间内不交易`,
    color: "#7c5ac7",
  },
  {
    asset: "BIL / 现金",
    name: "短期国债、现金与融资",
    oldTarget: strategyLiveData.r11Reference.cash,
    firstStage: strategyLiveData.stagedTarget.cash,
    finalTarget: strategyLiveData.stagedTarget.cash,
    role: "当前分阶段目标仍保留正现金",
    color: "#64748b",
  },
  {
    asset: "VIXY",
    name: "波动率对冲",
    oldTarget: strategyLiveData.r11Reference.VIXY,
    firstStage: strategyLiveData.stagedTarget.VIXY,
    finalTarget: strategyLiveData.stagedTarget.VIXY,
    role: "仅由原生产核心按条件启用",
    color: "#d14d72",
  },
] as const;

const fullGrowthWeight =
  strategyLiveData.fullTarget.QQQ + strategyLiveData.fullTarget.SMH;
const fullGoldSleeveWeight =
  strategyLiveData.fullTarget.GLD + strategyLiveData.fullTarget.GDE;

export const targetDerivation = {
  r11Share: 0.75,
  r38Share: 0.25,
  risk: {
    baseMultiplier: 1.07,
    stateMultiplier: stateActiveMultiplier,
    requestedMultiplier: 1.07 * stateActiveMultiplier,
    acceptedMultiplier: strategyLiveData.riskBudget.acceptedAbsoluteMultiplier,
    shockActive: strategyLiveData.riskBudget.oneSessionShockActive,
  },
  semiconductor: {
    baseShare: strategyLiveData.semiconductorOverlay.baseShare,
    fullSignalShare: strategyLiveData.semiconductorOverlay.fullSignalShare,
    overlayFraction: strategyLiveData.semiconductorOverlay.overlayFraction,
    implementedShare: strategyLiveData.semiconductorOverlay.implementedShare,
  },
  gde: {
    goldSleeve: fullGoldSleeveWeight,
    growthWeight: fullGrowthWeight,
    substitutionFraction: 0.1,
    growthGate: Math.min(fullGrowthWeight / 0.8, 1),
    fullCenter: strategyLiveData.fullTarget.GDE,
    stagedLower: strategyLiveData.stagedLower.GDE,
    stagedCenter: strategyLiveData.stagedTarget.GDE,
    stagedUpper: strategyLiveData.stagedUpper.GDE,
  },
  rows: [
    {
      asset: "QQQ",
      r11: strategyLiveData.r11Reference.QQQ,
      r38: strategyLiveData.fullTarget.QQQ,
      staged: strategyLiveData.stagedTarget.QQQ,
      reason: "保留成长总额；R38 只调整 QQQ / SMH 内部比例。",
    },
    {
      asset: "SMH",
      r11: strategyLiveData.r11Reference.SMH,
      r38: strategyLiveData.fullTarget.SMH,
      staged: strategyLiveData.stagedTarget.SMH,
      reason: "相对强弱锚点经 10% 有界覆盖后，低于 R11 基准。",
    },
    {
      asset: "GLD",
      r11: strategyLiveData.r11Reference.GLD,
      r38: strategyLiveData.fullTarget.GLD,
      staged: strategyLiveData.stagedTarget.GLD,
      reason: "与 GDE 合成黄金袖套；GDE 的配置从 GLD 等额转出。",
    },
    {
      asset: "GDE",
      r11: strategyLiveData.r11Reference.GDE,
      r38: strategyLiveData.fullTarget.GDE,
      staged: strategyLiveData.stagedTarget.GDE,
      reason: "受黄金替换比例和成长仓门控约束；区间内不强制交易。",
    },
    {
      asset: "BIL / 现金",
      r11: strategyLiveData.r11Reference.cash,
      r38: strategyLiveData.fullTarget.cash,
      staged: strategyLiveData.stagedTarget.cash,
      reason: "单日冲击保护把 R38 完整层降至 1.00 倍风险，留下更多现金。",
    },
    {
      asset: "VIXY",
      r11: strategyLiveData.r11Reference.VIXY,
      r38: strategyLiveData.fullTarget.VIXY,
      staged: strategyLiveData.stagedTarget.VIXY,
      reason: "期限结构条件未要求波动率对冲。",
    },
  ],
} as const;

export const whiteboxRuleLedger = [
  {
    layer: "策略选择",
    observed: "R39 的额外保护尚未满足启用条件；R38 10 / 10 已验证",
    threshold: "额外保护只有完成全部安全验证后才可启用",
    outcome: "使用 R38",
    effect: "不采用未经资格确认的 R39 快照。",
    tone: "caution",
  },
  {
    layer: "数据新鲜度",
    observed: `R9、R11、R38、核心与 GDE 均截至 ${strategyLiveData.priceAsOf}`,
    threshold: "同一完整美股交易日且近期连续",
    outcome: "通过",
    effect: "允许计算下一交易日目标。",
    tone: "good",
  },
  {
    layer: "总风险",
    observed: `${strategyLiveData.riskOnVotes} / ${strategyLiveData.riskOnVoteTotal} 风险偏好；双 200 日趋势许可开启`,
    threshold: "HMM 与趋势只是风险参与的上游许可",
    outcome: "允许参与",
    effect: "不因状态直接清仓；仍需经过后续保护层。",
    tone: "good",
  },
  {
    layer: "单日冲击",
    observed: "20 日 QQQ / SEMIS 年化波动 44.16%",
    threshold: "高于因果 90 分位阈值 36.97%",
    outcome: "已触发",
    effect: "本交易日把绝对风险倍数压至 1.000。",
    tone: "caution",
  },
  {
    layer: "SMH 跳空保护",
    observed: `绝对缺口 ${formatPercent(strategyLiveData.smhSignalGap)}；相对 QQQ ${formatPercent(strategyLiveData.smhRelativeSignalGap)}`,
    threshold: "绝对 ≤ −2.50% 且相对 ≤ −0.50%，并且事前高度集中",
    outcome: "未触发",
    effect: "不把 SMH 强制降至 15%。",
    tone: "good",
  },
  {
    layer: "个股替代",
    observed: `市场分化 ${formatPercent(stockDispersion, 2)}`,
    threshold: `不得高于 ${formatPercent(stockRiskData.activeDispersionMaximumPercentile, 0)}`,
    outcome: stockDispersionEligible ? "允许逐只验证" : "不允许新增替代",
    effect: stockDispersionEligible
      ? "符合额外个股条件时可等额替换 ETF。"
      : "不能增加个股替代，维持 ETF 为默认实现。",
    tone: stockDispersionEligible ? "good" : "caution",
  },
] as const;

export const migrationStages = [1, 2, 3, 4].map((stage) => ({
  stage,
  qqq: strategyLiveData.stagedTarget.QQQ,
  smh: strategyLiveData.stagedTarget.SMH,
  gold: strategyLiveData.stagedTarget.GLD,
  gde: strategyLiveData.stagedTarget.GDE,
  cash: strategyLiveData.stagedTarget.cash,
})) as readonly {
  stage: number;
  qqq: number;
  smh: number;
  gold: number;
  gde: number;
  cash: number;
}[];

export const decisionDrivers = [
  {
    title: `市场环境：${marketEnvironmentLabel}`,
    value: `${strategyLiveData.riskOnVotes} / ${strategyLiveData.riskOnVoteTotal} 风险偏好`,
    explanation:
      `三个核心模型给出风险偏好；QQQ 过去 63 日 ${formatPercent(decisionAuthority.qqq63Return)}，SMH 过去 63 / 126 日 ${formatPercent(decisionAuthority.semis63Return)} / ${formatPercent(decisionAuthority.semis126Return)}。环境标签负责解释，实际仓位仍由完整规则链生成。`,
    status: "趋势成立",
    tone: "positive",
  },
  {
    title: "趋势成立且波动未加速时提高非现金目标",
    value: `绝对倍数 ${strategyLiveData.riskBudget.acceptedAbsoluteMultiplier.toFixed(3)}`,
    explanation:
      volatilityAccelerationBlock
        ? "QQQ 与 SMH 的长期趋势仍成立，但至少一个资产的 21 日波动高于 63 日波动；策略因此不使用新增容量，回到原 R38 的 1.070 × 1.30。"
        : "QQQ 与 SMH 的长期趋势成立，且两者 21 日波动都没有高于 63 日波动；策略允许 1.070 × 1.375，实际总投资仍受现金不得低于 −20% 的硬限制。",
    status: volatilityAccelerationBlock
      ? "新增容量关闭"
      : "新增容量开启",
    tone: volatilityAccelerationBlock
      ? "neutral"
      : "positive",
  },
  {
    title: "R38 半导体慢速锚点只做 10% 偏移",
    value: `${formatPercent(strategyLiveData.semiconductorOverlay.baseShare)} → ${formatPercent(strategyLiveData.semiconductorOverlay.implementedShare)}`,
    explanation:
      `R9 原本让 SMH 占成长仓 ${formatPercent(strategyLiveData.semiconductorOverlay.baseShare)}。完整相对强弱信号端点为 ${formatPercent(strategyLiveData.semiconductorOverlay.fullSignalShare)}；R38 只向该端点移动十分之一，作为慢速配置锚点。`,
    status: "底层锚点",
    tone: "positive",
  },
  {
    title: r39Active
      ? "账户级相对损失保护正在生效"
      : "R39 额外保护当前待命",
    value: r39Active
      ? `${formatPercent(relativeDamageVeto.proposedAccountRelativeLoss)} → ${formatPercent(relativeDamageVeto.implementedAccountRelativeLoss)}`
      : "当前使用 R38",
    explanation: r39Active
      ? `SMH 在过去 ${relativeDamageVeto.lookbackTradingDays} 个交易日相对 QQQ 回落 ${formatPercent(relativeDamageVeto.relativeLoss)}。R39 先用 ${formatPercent(relativeDamageVeto.accountRelativeLossBudget)} 账户预算连续减仓；波动加速解除且拟议损失达到预算的 ${relativeDamageVeto.maximumShareActivationMultiple.toFixed(2)} 倍后，再把 SMH 限制到成长仓 ${formatPercent(relativeDamageVeto.maximumActiveSemisGrowthShare)}。成长仓总额保持不变。`
      : r39FallbackUserMessage,
    status: r39Active ? "正在生效" : "已阻断",
    tone: r39Active ? "caution" : "neutral",
  },
  {
    title: strategyLiveData.riskBudget.oneSessionShockActive
      ? "异常波动刹车正在生效"
      : "异常波动刹车没有启动",
    value: strategyLiveData.riskBudget.oneSessionShockActive
      ? "绝对倍数：1.000"
      : "单日刹车：关闭",
    explanation:
      strategyLiveData.riskBudget.oneSessionShockActive
        ? "当前因果高波动状态已经触发，下一目标取消融资；只持续一个交易日，之后由最新完整收盘重新判断。"
        : "若因果高波动状态触发，下一目标会把融资降到 0，只持续一个交易日，再由完整状态重新判断。",
    status: strategyLiveData.riskBudget.oneSessionShockActive
      ? "正在生效"
      : "待命",
    tone: strategyLiveData.riskBudget.oneSessionShockActive
      ? "caution"
      : "neutral",
  },
  {
    title: "SMH 隔夜缺口保护没有启动",
    value: `缺口 ${formatPercent(strategyLiveData.smhSignalGap)}`,
    explanation:
      "缺口条件虽然较弱，但还必须满足触发前 SMH 高度集中；当前不满足集中度前提，所以不会因为一次普通急跌机械卖出。",
    status: "待命",
    tone: "neutral",
  },
  {
    title: "底层 R38 先替换四分之一 R11",
    value: "25% R38 · 75% R11",
    explanation:
      "R38 的研究与生产资格审计已通过，但真实融资、滑点和操作仍需要前瞻记录。分阶段发布限制了首次策略变化，不改变历史候选本身的定义。",
    status: "受控发布",
    tone: "caution",
  },
  {
    title: "个股仍是 ETF 的受控替代",
    value: `市场分化 ${formatPercent(stockDispersion, 0)}`,
    explanation:
      "个股不会叠加在 QQQ/SMH 之上。只有股票的 63 日和 126 日收益都优于对应 ETF，且组合差异风险未超预算时，才等金额替代一部分 ETF。",
    status: "逐只验证",
    tone: "positive",
  },
] as const;

export const keyParameters = [
  {
    label: "R41 保护性容量",
    value: pputProtectedCapacity.productionEligible && pputProtectedCapacity.sameDate
      ? pputProtectedCapacity.active
        ? "已激活"
        : "资格通过 · 未激活"
      : "阻断",
    currentUse: `目标覆盖 ${formatPercent(pputProtectedCapacity.targetCoverage, 0)}；合格带 ${formatPercent(pputProtectedCapacity.minimumCoverage, 0)}–${formatPercent(pputProtectedCapacity.maximumCoverage, 0)}`,
    why: "用小额、明确的尾部保护交换正常趋势状态下的额外容量；没有真实保护就自动退回 R40",
    tone: pputProtectedCapacity.active ? "active" : "standby",
  },
  {
    label: "R40 账户非现金上限",
    value: recursiveTrendCushion.productionEligible && recursiveTrendCushion.sameDate
      ? "已进入私有执行链"
      : "阻断",
    currentUse: `底线 ${formatPercent(recursiveTrendCushion.floorDrawdown)}；分档 ${formatPercent(recursiveTrendCushion.tierSize, 0)}`,
    why: "用真实账户高水位约束长期熊市中的反复重入；账户数据只保存在登录后的私有空间",
    tone: recursiveTrendCushion.productionEligible && recursiveTrendCushion.sameDate ? "active" : "standby",
  },
  {
    label: "当前生产资格",
    value: `${activeStrategy} ${panelStatus.activeQualification} 通过`,
    currentUse: panelStatus.fallbackActive
      ? "R39 额外保护当前待命；当前目标来自已验证的 R38"
      : "R39 账户级相对损失保护已进入生产配置",
    why: panelStatus.fallbackActive
      ? "当前组合优先采用已完成安全验证的 R38；R39 的额外保护待满足启用条件后再评估。"
      : "参数、日期、溯源和订单阻断均已复核",
    tone: panelStatus.fallbackActive ? "standby" : "active",
  },
  {
    label: "账户相对损失预算",
    value: formatPercent(relativeDamageVeto.accountRelativeLossBudget),
    currentUse: relativeDamageVeto.active
      ? `当前将 ${formatPercent(relativeDamageVeto.proposedAccountRelativeLoss)} 压到 ${formatPercent(relativeDamageVeto.implementedAccountRelativeLoss)}`
      : "当前未触发",
    why: `使用 ${relativeDamageVeto.lookbackTradingDays} 个交易日的 SMH 相对 QQQ 损失，防止慢速锚点吃满快速回撤`,
    tone: relativeDamageVeto.active ? "active" : "standby",
  },
  {
    label: "底层 R38 发布比例",
    value: "25%",
    currentUse: "其余 75% 继续按 R11 运行",
    why: r39Active
      ? "R39 只在底层成长仓内部重配，不扩大成长仓总额"
      : "R39 额外保护待命时直接使用已验证的 R38，不应用该额外保护层",
    tone: "active",
  },
  {
    label: "基础 / 加速 / 稳定绝对倍数",
    value: "1.070 / 1.391 / 1.471",
    currentUse: `当前使用 ${strategyLiveData.riskBudget.acceptedAbsoluteMultiplier.toFixed(3)}`,
    why: "长期趋势决定是否加风险；短期波动加速时只保留原 R38 容量",
    tone: "active",
  },
  {
    label: "短期波动加速",
    value: volatilityAccelerationBlock
      ? "是"
      : "否",
    currentUse: volatilityAccelerationBlock
      ? "新增 0.075 相对倍数当前关闭"
      : "新增 0.075 相对倍数当前允许",
    why: "任一 QQQ / SMH 的 21 日波动高于 63 日波动，就立即回到原 R38",
    tone: volatilityAccelerationBlock
      ? "active"
      : "standby",
  },
  {
    label: "现金硬下限",
    value: "−20%",
    currentUse: `完整目标当前为 ${formatPercent(strategyLiveData.riskBudget.fullPreGdeCashWeight)}`,
    why: "把最大总投资比例真正写入生产执行，而不是只写在说明中",
    tone: strategyLiveData.riskBudget.cashHardLimitActive ? "active" : "standby",
  },
  {
    label: "半导体相对信号占比",
    value: "10%",
    currentUse: `SMH 成长仓份额 ${formatPercent(strategyLiveData.semiconductorOverlay.implementedShare)}`,
    why: "保留原模型的大部分配置，只做小幅、因果、可审计的调整",
    tone: "active",
  },
  {
    label: "单日异常波动刹车",
    value: "触发时绝对倍数 1.00",
    currentUse: strategyLiveData.riskBudget.oneSessionShockActive
      ? "当前正在生效"
      : "当前未触发",
    why: "处理一周评估之间的突然冲击，同时避免长期错过反弹",
    tone: strategyLiveData.riskBudget.oneSessionShockActive ? "active" : "standby",
  },
  {
    label: "GDE 替代",
    value: "最多替代黄金目标的 10%",
    currentUse: `账户中心 ${formatPercent(strategyLiveData.stagedTarget.GDE)}；允许 ${formatPercent(strategyLiveData.stagedLower.GDE)}–${formatPercent(strategyLiveData.stagedUpper.GDE)}`,
    why: "黄金仍独立判断；GDE 只提高少量资金效率",
    tone: "active",
  },
  {
    label: "每次最多调动资金",
    value: "策略管理资金的 10%",
    currentUse: "真实持仓录入后计算",
    why: "防止策略切换和持仓误差造成一次性大额交易",
    tone: "active",
  },
  {
    label: "个股组合差异风险预算",
    value: "年化 3%",
    currentUse: "按 60 日与 252 日较高值控制",
    why: "限制选股相对 QQQ / SMH 新增的风险",
    tone: "active",
  },
] as const;

export const parameterGroups = [
  {
    name: "R41 保护性容量（资格通过，当前未激活）",
    summary: `中央现代净 CAGR ${formatPercent(pputProtectedCapacity.modernCagr)}；当前仍由 R40 执行`,
    parameters: [
      ["保护工具", "SPYM 月度长期看跌期权", "最高不超过现货 95% 的已上市行权价"],
      ["目标 / 合格覆盖", `${formatPercent(pputProtectedCapacity.targetCoverage, 0)} / ${formatPercent(pputProtectedCapacity.minimumCoverage, 0)}–${formatPercent(pputProtectedCapacity.maximumCoverage, 0)}`, "按标的名义金额，不是按期权费占账户比例"],
      ["50 万美元示例", `${pputProtectedCapacity.example500kContracts} 张 = ${formatPercent(pputProtectedCapacity.example500kCoverage, 2)}`, "以 SPYM 90.52 美元和每张 100 股计算"],
      ["正常状态上限", formatPercent(pputProtectedCapacity.normalNonCashCap, 0), "仅在双 200 日趋势为正、安全垫正常且真实保护已确认时启用"],
      ["最早建立窗口", `${strategyLiveData.nextExecutionEt}`, "最近完整收盘复核后，在下一常规开盘用实时盘口人工复核"],
      ["失效关闭", "回退 R40", "权限、实时价差、整数覆盖或持仓缺一项都不激活 R41"],
      ["不可消除的差异", "SPYM 与 PPUT/SPX 存在基差", "美式实物交割、价差、跟踪、税务与滚动时点可能偏离基准"],
    ],
  },
  {
    name: "R40 长期熊市账户保护",
    summary: recursiveTrendCushion.productionEligible && recursiveTrendCushion.sameDate
      ? "生产资格已通过；账户上限在私有持仓中计算"
      : "同日资格未通过，账户草稿被阻断",
    parameters: [
      ["历史回撤底线", formatPercent(recursiveTrendCushion.floorDrawdown), "高水位乘以 81% 得到底线净值"],
      ["双趋势为正乘数", recursiveTrendCushion.bullMultiplier.toFixed(1), "接近高水位时保持完整上涨参与"],
      ["其他状态乘数", recursiveTrendCushion.bearMultiplier.toFixed(1), "趋势转弱后更快压低非现金上限"],
      ["非现金分档", formatPercent(recursiveTrendCushion.tierSize, 0), "请求值向下取整，避免假精度"],
      ["受控状态底稿", "R11", "关闭 R38 增量风险后，同比缩放全部非现金资产"],
      ["首次启用", "当前确认净值 = 初始高水位", "不倒填启用前损益"],
      ["历史限制", "20% 不是未来保证", "跳空、滑点和代理数据都可能造成更深实际损失"],
    ],
  },
  {
    name: "R9 核心市场环境",
    summary: "三个模型均处于风险偏好环境",
    parameters: [
      ["历史观察", "1,008 个交易日", "用于估计市场状态"],
      ["模型验证", "126 个交易日", "用于选择更稳定的模型"],
      ["独立模型编号", "7、42、123", "降低单次估计偶然性"],
      ["常规重估间隔", "21 个交易日", "约每月正式更新一次"],
      ["最近数据", strategyLiveData.priceAsOf, "只使用已经完整收盘的交易日"],
    ],
  },
  {
    name: "R38 上涨参与",
    summary: `趋势许可开启；完整目标现金 ${formatPercent(strategyLiveData.riskBudget.fullPreGdeCashWeight)}`,
    parameters: [
      ["基础非现金倍数", "1.070", "无趋势许可时使用"],
      ["原 R38 趋势相对倍数", "1.30", "波动加速时保留，不会进一步减仓"],
      ["波动稳定时趋势相对倍数", "1.375", "QQQ 与 SMH 的 21 日波动都不高于各自 63 日波动时使用"],
      ["当前状态倍数", stateActiveMultiplier.toFixed(3), volatilityAccelerationBlock ? "短期波动正在加速，新增容量关闭" : "短期波动未加速，新增容量开启"],
      ["当前请求绝对倍数", strategyLiveData.riskBudget.acceptedAbsoluteMultiplier.toFixed(3), "随后仍受现金硬下限限制"],
      ["现金最低", "−20%", "非现金总权重最多 120%"],
      ["融资利差主 / 压力假设", "每年 1.00% / 1.50%", "回测已从净收益中扣除"],
    ],
  },
  {
    name: "QQQ / SMH 内部配置",
    summary: `R38 锚点 ${formatPercent(relativeDamageVeto.baseSemisGrowthShare)}；R39 当前 ${formatPercent(relativeDamageVeto.implementedSemisGrowthShare)}`,
    parameters: [
      ["相对强弱窗口", "126 个交易日", "只使用目标日前的历史收盘"],
      ["更新间隔", "21 个交易日", "减少反复切换"],
      ["R38 覆盖比例", "10%", "90% 保留原配置，10% 来自完整相对信号"],
      ["高波动端点", "QQQ / SMH 各 50%", "完整信号在最高波动状态回到中性"],
      ["成长预算", "保持不变", "只在 QQQ 和 SMH 之间移动"],
    ],
  },
  {
    name: r39Active
      ? "R39 账户级相对损失保护"
      : "R39 额外半导体保护（当前待命）",
    summary: r39Active && relativeDamageVeto.active
      ? `正在生效：${formatPercent(relativeDamageVeto.proposedAccountRelativeLoss)} → ${formatPercent(relativeDamageVeto.implementedAccountRelativeLoss)}`
      : "R39 的额外保护当前待命；使用已验证的 R38 组合。",
    parameters: [
      ["快速观察窗口", `${relativeDamageVeto.lookbackTradingDays} 个交易日`, "比 126 日慢速锚点更快识别持续相对回撤"],
      ["SMH 相对 QQQ 损失", formatPercent(relativeDamageVeto.relativeLoss), "只使用上一完整收盘可得信息"],
      ["账户相对损失预算", formatPercent(relativeDamageVeto.accountRelativeLossBudget), "以 SMH 权重乘相对损失衡量账户级暴露"],
      ["R38 提议暴露", formatPercent(relativeDamageVeto.proposedAccountRelativeLoss), "超预算时才触发重配"],
      ["R39 实施暴露", formatPercent(relativeDamageVeto.implementedAccountRelativeLoss), relativeDamageVeto.maximumShareGuardActive ? "50/50 硬上限比 3% 预算更严格，因此当前低于预算" : "按 3% 账户预算连续限制"],
      ["触发前 SMH 最低成长仓份额", formatPercent(relativeDamageVeto.triggerMinimumSemisGrowthShare), "只有底层配置高度集中时才启动；不是调整后的仓位下限"],
      ["SMH 成长仓硬上限", formatPercent(relativeDamageVeto.maximumActiveSemisGrowthShare), `拟议损失达到预算 ${relativeDamageVeto.maximumShareActivationMultiple.toFixed(2)} 倍，且波动加速解除后才允许`],
      ["当前硬上限状态", relativeDamageVeto.maximumShareGuardActive ? "已启用" : "未启用", relativeDamageVeto.maximumShareGuardPermitted ? "波动加速已解除" : "波动仍在加速；只保留连续预算控制"],
      ["等额转移", `${formatPercent(relativeDamageVeto.removedSemisWeight)} SMH → QQQ`, "成长仓总额不变，不把风险转成现金"],
    ],
  },
  {
    name: "快速风险控制",
    summary: relativeDamageVeto.active
      ? "账户级相对损失保护生效；其他快速控制当前待命"
      : "快速控制当前均待命",
    parameters: [
      ["异常波动刹车", "最多一个交易日", "触发时取消融资，绝对倍数降到 1.00"],
      ["SMH 绝对缺口", "≤ −2.50%", "使用开盘相对前收盘"],
      ["SMH 相对 QQQ 缺口", "≤ −0.50%", "两个缺口条件必须同时满足"],
      ["触发前 SMH 最低权重", "64.2%", "只保护高度集中的半导体仓位"],
      ["触发后 SMH 上限", "15%", "信号次日执行，多余资金转现金"],
      ["恢复", "2 次相对上涨或最多 4 日", "避免一次反弹就全部追回"],
    ],
  },
  {
    name: "成本与分阶段发布",
    summary: "25% R38 / 75% R11；订单仍等待真实持仓",
    parameters: [
      ["普通资产单边成本", "0.075%", "用于主回测"],
      ["压力成本", "0.15%", "用于保守压力检验"],
      ["GDE 主 / 压力成本", "0.40% / 0.75%", "GDE 只允许限价单"],
      ["首次 R38 比例", "25%", "剩余 75% 继续使用 R11"],
      ["扩大前最低记录", "63 个交易日", "检查真实融资、滑点、数据和操作"],
      ["当前阻断", "缺少完整真实持仓与 GDE 比例", "Panel 不会直接下单"],
    ],
  },
  {
    name: "个股受控替代",
    summary: "保留原有持仓记忆和风险倍数功能",
    parameters: [
      ["角色", "替代部分 QQQ / SMH", "不会额外增加总成长仓"],
      ["双窗口确认", "63 日和 126 日", "两段都跑赢对应 ETF 才允许替代"],
      ["差异风险预算", "年化 3%", "衡量股票组合减去 ETF 的波动"],
      ["单只 / 合计上限", "5% / 20%", "以策略管理资金为分母"],
      ["高捕获模式", "最多替代对应 ETF 的 50%", "只适合有真实选股记录的用户"],
    ],
  },
] as const;

export const productionEvidence = [
  {
    label: "当前策略选择依据",
    candidate: `${activeStrategy}：${panelStatus.activeQualification} 项安全检查通过`,
    previous: `候选升级层：${panelStatus.requestedStrategy}`,
    note: panelStatus.fallbackActive
      ? "当前采用已验证的 R38；R39 的额外保护待满足启用条件后再评估。"
      : "R39 参数、日期、溯源和订单阻断全部通过",
  },
  {
    label: "R39 独立最终审计",
    candidate: panelStatus.r39ResearchQualificationPass
      ? "通过"
      : "未通过",
    previous: `R39 数据截至 ${panelStatus.r39PriceAsOf}`,
    note: panelStatus.r39ResearchQualificationPass
      ? "重新构建逐日路径，并检查参数邻域、事件窗口、因果性和数据完整性"
      : "失败的候选不会覆盖同日合格的 R38 生产快照",
  },
  {
    label: "当前账户相对损失暴露",
    candidate: formatPercent(relativeDamageVeto.implementedAccountRelativeLoss),
    previous: `R38 提议 ${formatPercent(relativeDamageVeto.proposedAccountRelativeLoss)}`,
    note: relativeDamageVeto.maximumShareGuardActive
      ? `预算为 ${formatPercent(relativeDamageVeto.accountRelativeLossBudget)}；50/50 硬上限更严格，当前低于预算`
      : `预算为 ${formatPercent(relativeDamageVeto.accountRelativeLossBudget)}；当前按连续预算控制`,
  },
  {
    label: "成长仓总额守恒",
    candidate: formatPercent(relativeDamageVeto.growthBudgetAfter),
    previous: formatPercent(relativeDamageVeto.growthBudgetBefore),
    note: "R39 只把 SMH 等额转到 QQQ，不额外加仓，也不机械转成现金",
  },
  {
    label: "完整样本年化收益拖累",
    candidate: "−0.23 个百分点",
    previous: "事前上限 −0.50",
    note: "为降低集中损伤付出的历史机会成本，不能解释为未来保证",
  },
  {
    label: "事件窗口回撤变化",
    candidate: "最差 −0.38 个百分点",
    previous: "事前上限 −0.50",
    note: "所有声明事件窗口均在风险容忍范围内",
  },
] as const;

export const knownRisks = [
  r39Active
    ? "R39 只约束 SMH 相对 QQQ 的账户级损失暴露；若 QQQ 与 SMH 同时下跌，它不会消除成长仓的绝对亏损。"
    : "R39 当前未通过资格审计；账户目标不包含其相对损失保护，使用同日合格的 R38。",
  "21 日窗口是因果的快速保护，不是预测器；震荡反转时可能先减 SMH、随后错过反弹。",
  "当前保护把 SMH 转到 QQQ，并未降低 48.11% 的成长仓总额；系统性科技风险仍然存在。",
  "波动加速度会关闭 R38 新增容量，并暂缓 R39 的 50/50 硬切；R39 的 3% 连续预算仍生效。它不能预判所有突然跳空。",
  "完整 R38 当前会使用融资；现金硬下限是 −20%。真实融资利差若持续高于每年 1.50%，不得扩大发布比例。",
  "R38 只先替换 25% 的 R11。至少记录 63 个冻结后的交易日，且数据、成交、融资和操作没有事故后，才审查扩大。",
  "半导体覆盖只得到半导体领域的证据，不应解释为适用于所有行业或所有资产。",
  "R38 的 SMH 相对信号只改变 QQQ / SMH 内部比例，不会因为 SMH 强势就自动增加成长总额。",
  "单日冲击刹车只能降低突发风险，无法保证避开所有盘中暴跌或连续跳空。",
  "GDE 自 2022 年才有真实历史；更早结果使用融资近似。买卖价差超过压力假设时应取消订单。",
  "个股层不自动选股；它只管理用户已选股票何时适合替代 ETF，以及最多替代多少。",
  "历史价格供应商可能追溯修订数据；生产运行必须继续保存每日输入哈希和目标。",
] as const;
