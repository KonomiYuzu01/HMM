import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { registerHooks } from "node:module";
import test from "node:test";

globalThis.__cloudflareTestEnv = {};
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === "cloudflare:workers") {
      return {
        url: "data:text/javascript,export const env = globalThis.__cloudflareTestEnv;",
        shortCircuit: true,
      };
    }
    if (
      specifier === "./stock-risk-data" &&
      context.parentURL?.endsWith("/app/risk-multiple.ts")
    ) {
      return {
        url: new URL("../app/stock-risk-data.ts", import.meta.url).href,
        shortCircuit: true,
      };
    }
    return nextResolve(specifier, context);
  },
});

const templateRoot = new URL("../", import.meta.url);
const {
  calculateManagedStockPlan,
  stockRiskData,
} = await import("../app/risk-multiple.ts");
const { strategyLiveData } = await import("../app/strategy-live-data.ts");
const {
  calculateRecursiveTrendCushion,
  protectedAccountTarget,
} = await import("../app/recursive-trend-cushion.ts");
const formatPercent = (value) => `${(value * 100).toFixed(2)}%`;

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${pathname}`, {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("does not expose saved holdings without a signed-in user", async () => {
  const response = await render("/api/holdings");
  assert.equal(response.status, 401);
  assert.deepEqual(await response.json(), {
    error: "需要登录后才能读取持仓。",
  });
});

test("server-renders the currently qualified strategy operating dashboard", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>策略运行驾驶舱<\/title>/i);
  assert.match(html, /http:\/\/localhost\/og-r39\.png/);
  assert.match(html, /策略已冻结：75% R11 \+ 25% R38/);
  assert.match(html, /先录入真实持仓，暂时不要下单/);
  assert.match(html, />录入真实持仓<\/button>/);
  assert.match(html, /为什么现在这样做/);
  assert.match(html, /从市场状态到行动的完整决策链/);
  assert.match(html, /这对你意味着/);
  assert.match(html, /若情况改变/);
  assert.match(html, /规则核对表/);
  assert.match(html, /当前读数、门槛和实际影响/);
  assert.match(html, /R9 \/ HMM 状态/);
  assert.match(html, /R11 基础组合/);
  assert.match(html, /R38 25% 冻结/);
  assert.match(html, /R39–R41 阻断/);
  assert.match(html, /账户执行门控/);
  assert.match(
    html,
    new RegExp(
      `${strategyLiveData.riskOnVotes} \/ ${strategyLiveData.riskOnVoteTotal} 个模型处于风险偏好`,
    ),
  );
  assert.match(html, new RegExp(`R9 当前动作是 ${strategyLiveData.r9Action}`));
  assert.match(html, /HMM 状态与 R9 动作是上游输入/);
  if (strategyLiveData.panelStatus.activeStrategy === "R39") {
    assert.match(
      html,
      new RegExp(formatPercent(strategyLiveData.relativeDamageVeto.relativeLoss)),
    );
    assert.match(
      html,
      new RegExp(
        `${formatPercent(strategyLiveData.relativeDamageVeto.proposedAccountRelativeLoss)}(?:<!-- -->)?，高于 ${formatPercent(strategyLiveData.relativeDamageVeto.accountRelativeLossBudget)}`,
      ),
    );
  } else {
    assert.match(html, /R38 只向 126 日相对强弱端点移动 10%/);
  }
  assert.match(html, /订单数量 = 模型目标市值 − 当前真实市值/);
  assert.match(html, /被排除的替代动作/);
  assert.match(html, /为什么不是全部转成现金/);
  assert.match(html, /为什么不继续维持高 SMH/);
  assert.match(html, /为什么现在不能生成订单/);
  assert.match(html, /哪些变化会让结论改变/);
  assert.match(html, new RegExp(strategyLiveData.nextExecutionEt));
  assert.match(html, /下一笔建仓最早窗口/);
  assert.match(html, /这就是第二笔的时间/);
  assert.match(
    html,
    new RegExp(
      strategyLiveData.accountBuildTiming.followingTrancheEarliestExecutionEt,
    ),
  );
  assert.match(html, new RegExp(strategyLiveData.priceAsOf));
  if (strategyLiveData.executionWindowStatus === "MISSED") {
    assert.match(html, /本次开盘窗口已过/);
    assert.match(html, /不生成盘中追单/);
  }
  assert.match(html, /最近 5 个交易日连续/);
  assert.match(html, /订单不可执行/);
  assert.match(
    html,
    new RegExp(`${strategyLiveData.panelStatus.activeStrategy} 冻结运行中`),
  );
  assert.match(html, /相关参数的最新情况/);
  assert.match(
    html,
    new RegExp(
      `${strategyLiveData.panelStatus.activeStrategy}(?:<!-- -->)? 当前如何配置它管理的资金`,
    ),
  );
  assert.match(html, /GDE 与核心同日/);
  assert.match(
    html,
    new RegExp(
      `${formatPercent(strategyLiveData.stagedLower.GDE)}–${formatPercent(
        strategyLiveData.stagedUpper.GDE,
      )}`,
    ),
  );
  assert.match(html, /资格、前瞻冻结与数据日期均已独立复核/);
  assert.match(html, /趋势成立且波动未加速时提高非现金目标/);
  assert.match(html, /短期波动加速时只保留原 R38 容量/);
  assert.match(html, /1\.070 × 1\.375/);
  assert.match(html, /R38 半导体慢速锚点只做 10% 偏移/);
  if (strategyLiveData.panelStatus.fallbackActive) {
    assert.match(html, /R38 已冻结在策略资金的 25%/);
    assert.match(html, /R39 至 R41 不参与生产晋级/);
    assert.match(html, /前瞻数据只做通过或停止判断/);
    assert.match(html, /R39–R41 已阻断/);
  } else {
    assert.match(html, /账户级相对损失保护正在生效/);
    assert.match(html, /21 日相对损失按账户 3% 预算限制集中度/);
    assert.match(html, /波动加速解除/);
    assert.equal(strategyLiveData.relativeDamageVeto.maximumShareGuardActive, true);
    assert.equal(strategyLiveData.relativeDamageVeto.maximumShareGuardPermitted, true);
  }
  assert.match(html, /当前合格规则已启用/);
  assert.match(html, /现金不得低于 −20%/);
  assert.match(html, /底层 R38 先替换四分之一 R11/);
  assert.match(
    html,
    stockRiskData.activeDispersionPercentile <=
      stockRiskData.activeDispersionMaximumPercentile
      ? /可逐只验证/
      : /高分化，暂缓替代/,
  );
  assert.match(html, /当前 VIXY 目标为/);
  assert.match(html, /每一个百分点都可以倒推回规则/);
  assert.match(html, /90% ×/);
  assert.ok(html.includes("75% × R11 + 25% × 完整 R38"));
  assert.match(html, /当前目标权重推导表/);
  assert.match(html, /当前策略选择依据/);
  assert.match(html, new RegExp(strategyLiveData.panelStatus.activeQualification));
  assert.match(html, /2026-07-29-r11-authority-v1/);
  assert.doesNotMatch(html, /从旧生产参考，分四段迁移到 R9/);
  assert.doesNotMatch(html, /成长仓下限没有决定当前仓位/);
  assert.doesNotMatch(html, /零仓重入规则没有触发/);
  assert.doesNotMatch(html, /Shadow/);
});

test("states the account build cadence separately from the R38 rollout", async () => {
  const panelSource = await readFile(
    new URL("../app/control-panel.tsx", import.meta.url),
    "utf8",
  );
  assert.match(panelSource, /不是固定等待 63 个交易日/);
  assert.match(panelSource, /等待.*新的完整收盘/s);
  assert.match(panelSource, /63 个交易日仅用于审查 R38 生产占比/);
  assert.match(panelSource, /remainingGapThreshold/);
  assert.match(panelSource, /visibleDecisionPipeline/);
  assert.match(panelSource, /生产快照本身不会读取这份私人数据/);
  assert.equal(
    strategyLiveData.accountBuildTiming.requiredNewCompletedCloses,
    1,
  );
});

test("allows a same-date R40 private draft while promotion remains frozen", async () => {
  const r40 = strategyLiveData.recursiveTrendCushion;
  assert.equal(r40.productionQualificationPass, true);
  assert.equal(r40.sameDate, true);
  assert.equal(r40.draftEligible, true);
  assert.equal(r40.promotionAllowed, false);
  assert.equal(r40.productionEligible, false);

  const response = await render();
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /R40 同日资格通过，私有草稿可用（晋级冻结）/);

  const panelSource = await readFile(
    new URL("../app/control-panel.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    panelSource,
    /const eligible = strategySnapshot\.recursiveTrendCushion\.draftEligible/,
  );
  assert.doesNotMatch(
    panelSource,
    /const eligible = strategySnapshot\.recursiveTrendCushion\.productionEligible/,
  );

  const state = calculateRecursiveTrendCushion({
    priorEquity: 85,
    priorPeak: 100,
    dualTrendPositive: false,
    parameters: {
      floorDrawdown: -0.19,
      bullMultiplier: 100,
      bearMultiplier: 9.5,
      tierSize: 0.05,
    },
  });
  assert.equal(state.acceptedNonCashCap, 0.4);
  const target = protectedAccountTarget({
    stagedTarget: strategyLiveData.stagedTarget,
    r11Target: strategyLiveData.r11Reference,
    acceptedNonCashCap: state.acceptedNonCashCap,
  });
  const nonCash = target.QQQ + target.SMH + target.GLD + target.GDE + target.VIXY;
  assert.ok(Math.abs(nonCash - 0.4) < 1e-12);
  assert.ok(Math.abs(Object.values(target).reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
});

test("explains the fail-closed R41 protection layer", async () => {
  const response = await render("/");
  const html = await response.text();
  const r41 = strategyLiveData.pputProtectedCapacity;
  const expectedTitle = "R41 已撤回资格：Put 不再用于跨过 20% 回撤门槛";
  assert.equal(r41.active, false);
  assert.equal(r41.productionEligible, false);
  assert.equal(r41.example500kContracts, 3);
  assert.ok(Math.abs(r41.example500kCoverage - 0.054312) < 1e-12);
  assert.ok(html.includes(expectedTitle));
  assert.match(html, /3 张/);
  assert.match(html, /5\.43%/);
  assert.match(html, /不买 Put，不启用 120% 上限/);
});

test("exposes the anti-overfit forward freeze", async () => {
  const response = await render("/");
  const html = await response.text();
  const governance = strategyLiveData.overfitGovernance;
  assert.equal(governance.currentR38Share, 0.25);
  assert.equal(governance.successorPromotionAllowed, false);
  assert.ok(governance.forwardSessions < governance.minimumForwardSessions);
  assert.match(html, /反过拟合治理（当前最高优先级）/);
  assert.match(html, new RegExp(`${governance.forwardSessions} / ${governance.minimumForwardSessions}`));
  assert.match(html, /所有旧历史统一视为样本内/);
  assert.match(html, /达到 63 日也不会自动晋级/);
});

test("ships the exact social preview dimensions", async () => {
  const preview = await readFile(new URL("../public/og-r39.png", import.meta.url));
  assert.equal(preview.subarray(1, 4).toString("ascii"), "PNG");
  assert.equal(preview.readUInt32BE(16), 1200);
  assert.equal(preview.readUInt32BE(20), 630);
});

test("ships a same-date qualified strategy target", () => {
  assert.equal(
    strategyLiveData.release,
    strategyLiveData.panelStatus.activeRelease,
  );
  assert.equal(strategyLiveData.priceAsOf, strategyLiveData.expectedPriceAsOf);
  assert.equal(
    strategyLiveData.panelStatus.activeStrategy,
    strategyLiveData.panelStatus.fallbackActive ? "R38" : "R39",
  );
  const targetSum = Object.values(strategyLiveData.stagedTarget).reduce(
    (sum, value) => sum + value,
    0,
  );
  assert.ok(Math.abs(targetSum - 1) < 1e-12);
  if (strategyLiveData.panelStatus.fallbackActive) {
    assert.equal(strategyLiveData.panelStatus.r39ResearchQualificationPass, false);
    assert.equal(strategyLiveData.relativeDamageVeto.active, false);
  }
  assert.equal(strategyLiveData.ordersExecutable, false);
});

test("removes the disposable starter preview", async () => {
  await assert.rejects(
    access(new URL("app/_sites-preview", templateRoot)),
  );
});

test("supports explicit individual-stock treatment choices", async () => {
  const source = await readFile(
    new URL("../app/control-panel.tsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /添加个股/);
  assert.match(source, /保留并排除在策略之外/);
  assert.match(source, /逐步卖出并转入策略/);
  assert.match(source, /替代一部分 QQQ \/ SMH/);
  assert.match(source, /整个个股组合的差异风险/);
  assert.match(source, /证据门控（默认）/);
  assert.match(source, /高捕获/);
  assert.match(source, /当前分化位于过去三年的/);
  assert.match(source, /策略实际管理资金/);
  assert.match(source, /\| "GDE"/);
  assert.match(source, /goldSleeveOnlyAdjustment/);
});

test("sizes a supported basket within active-risk and position caps", () => {
  const result = calculateManagedStockPlan({
    managedCapital: 100_000,
    targetWeights: { QQQ: 0.4, SMH: 0.2 },
    replacementShare: 0.5,
    selectionMode: "high_capture",
    stocks: [
      { id: "1", ticker: "MSFT", benchmark: "QQQ", value: 10_000 },
      { id: "2", ticker: "AMZN", benchmark: "QQQ", value: 10_000 },
      { id: "3", ticker: "NVDA", benchmark: "SMH", value: 10_000 },
      { id: "4", ticker: "AVGO", benchmark: "SMH", value: 10_000 },
    ],
  });

  assert.equal(result.status, "ready");
  assert.ok(result.combinedTrackingError <= 0.03 + 1e-12);
  assert.ok(result.totalStockTargetDollars <= 20_000 + 1e-8);
  assert.ok(
    result.targets.every((stock) => stock.targetDollars <= 5_000 + 1e-8),
  );
  assert.ok(
    Math.abs(
      result.budgets.QQQ.remainingEtfDollars +
        result.budgets.QQQ.stockTargetDollars -
        40_000,
    ) < 1e-8,
  );
});

test("does not use unaudited stocks as ETF replacements", () => {
  const result = calculateManagedStockPlan({
    managedCapital: 100_000,
    targetWeights: { QQQ: 0.4, SMH: 0.2 },
    replacementShare: 0.5,
    stocks: [
      { id: "1", ticker: "UNKNOWN", benchmark: "QQQ", value: 10_000 },
    ],
  });

  assert.equal(result.status, "unsupported");
  assert.deepEqual(result.unsupportedTickers, ["UNKNOWN"]);
  assert.equal(result.totalStockTargetDollars, 0);
});

test("sets managed stocks to zero when the regime allows no replacement", () => {
  const result = calculateManagedStockPlan({
    managedCapital: 100_000,
    targetWeights: { QQQ: 0.4, SMH: 0.2 },
    replacementShare: 0,
    selectionMode: "high_capture",
    stocks: [
      { id: "1", ticker: "MSFT", benchmark: "QQQ", value: 10_000 },
    ],
  });

  assert.equal(result.status, "ready");
  assert.equal(result.totalStockTargetDollars, 0);
  assert.equal(result.budgets.QQQ.remainingEtfDollars, 40_000);
  assert.equal(stockRiskData.asOf, strategyLiveData.priceAsOf);
  assert.equal(stockRiskData.marketStateAsOf, strategyLiveData.priceAsOf);
});

test("blocks evidence-gated stocks when dispersion is above its cap", () => {
  const currentDispersion = stockRiskData.activeDispersionPercentile;
  stockRiskData.activeDispersionPercentile =
    stockRiskData.activeDispersionMaximumPercentile + 0.01;

  try {
    const result = calculateManagedStockPlan({
      managedCapital: 100_000,
      targetWeights: { QQQ: 0.4, SMH: 0.2 },
      replacementShare: 0.5,
      stocks: [
        { id: "1", ticker: "AAPL", benchmark: "QQQ", value: 10_000 },
        { id: "2", ticker: "AMD", benchmark: "SMH", value: 10_000 },
      ],
    });

    assert.equal(result.selectionMode, "evidence_gated");
    assert.equal(result.marketStateEligible, true);
    assert.equal(result.activeDispersionEligible, false);
    assert.equal(result.totalStockTargetDollars, 0);
    assert.match(result.blockers.join("；"), /高于 67% 上限/);
  } finally {
    stockRiskData.activeDispersionPercentile = currentDispersion;
  }
});

test("persists holdings for the signed-in private-panel user", async () => {
  const [panelSource, routeSource, hostingConfig] = await Promise.all([
    readFile(new URL("../app/control-panel.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/holdings/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../.openai/hosting.json", import.meta.url), "utf8"),
  ]);

  assert.match(panelSource, /fetch\("\/api\/holdings"/);
  assert.match(panelSource, /已保存在此私有 Panel/);
  assert.match(panelSource, /stockSelectionMode/);
  assert.match(panelSource, /删除已保存持仓/);
  assert.match(routeSource, /getChatGPTUser/);
  assert.match(routeSource, /export async function DELETE/);
  assert.match(routeSource, /"GDE"/);
  assert.equal(JSON.parse(hostingConfig).d1, "DB");
});
