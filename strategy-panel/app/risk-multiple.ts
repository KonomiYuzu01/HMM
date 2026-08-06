import { stockRiskData } from "./stock-risk-data";

export type GrowthBenchmark = "QQQ" | "SMH";
export type StockSelectionMode = "evidence_gated" | "high_capture";

export type ManagedStockInput = {
  id: string;
  ticker: string;
  benchmark: GrowthBenchmark;
  value: number;
};

export type ManagedStockTarget = ManagedStockInput & {
  targetDollars: number;
};

export type ActiveRiskBudget = {
  benchmark: GrowthBenchmark;
  etfTargetDollars: number;
  maximumReplacementDollars: number;
  stockTargetDollars: number;
  remainingEtfDollars: number;
  trackingError60: number;
  trackingError252: number;
};

export type ManagedStockPlan = {
  status: "ready" | "unsupported" | "duplicate";
  unsupportedTickers: string[];
  duplicateTickers: string[];
  targets: ManagedStockTarget[];
  budgets: Record<GrowthBenchmark, ActiveRiskBudget>;
  combinedTrackingError: number;
  activeRiskBudget: number;
  maximumNameWeight: number;
  maximumTotalWeight: number;
  totalStockTargetDollars: number;
  dataAsOf: string;
  selectionMode: StockSelectionMode;
  marketState: string;
  marketStateEligible: boolean;
  activeDispersionPercentile: number;
  activeDispersionEligible: boolean;
  benchmarkEligibility: Record<
    GrowthBenchmark,
    {
      activeLogReturn63: number;
      activeLogReturn126: number;
      relativeStrengthConfirmed: boolean;
      effectiveReplacementShare: number;
    }
  >;
  blockers: string[];
};

type ManagedStockPlanInput = {
  managedCapital: number;
  targetWeights: Record<GrowthBenchmark, number>;
  replacementShare: number;
  stocks: ManagedStockInput[];
  activeRiskBudget?: number;
  maximumNameWeight?: number;
  maximumTotalWeight?: number;
  selectionMode?: StockSelectionMode;
};

const assets = [...stockRiskData.assets];
const supportedStocks = new Set<string>(stockRiskData.supportedStocks);
const assetIndex = new Map<string, number>(
  assets.map((asset, index) => [asset, index]),
);

function quadraticTrackingError(
  vector: number[],
  covariance: readonly (readonly number[])[],
) {
  let variance = 0;
  for (let row = 0; row < vector.length; row += 1) {
    for (let column = 0; column < vector.length; column += 1) {
      variance += vector[row] * covariance[row][column] * vector[column];
    }
  }
  return Math.sqrt(Math.max(0, variance) * stockRiskData.annualization);
}

function basketTrackingError(
  stocks: ManagedStockInput[],
  benchmark: GrowthBenchmark,
  covariance: readonly (readonly number[])[],
) {
  if (stocks.length === 0) return 0;
  const vector = Array.from({ length: assets.length }, () => 0);
  const totalValue = stocks.reduce(
    (sum, stock) => sum + Math.max(0, stock.value),
    0,
  );
  stocks.forEach((stock) => {
    const weight =
      totalValue > 0 ? Math.max(0, stock.value) / totalValue : 1 / stocks.length;
    vector[assetIndex.get(stock.ticker) ?? -1] += weight;
  });
  vector[assetIndex.get(benchmark) ?? -1] -= 1;
  return quadraticTrackingError(vector, covariance);
}

function basketActiveLogReturn(
  stocks: ManagedStockInput[],
  values: Readonly<Record<string, number>>,
) {
  if (stocks.length === 0) return 0;
  const totalValue = stocks.reduce(
    (sum, stock) => sum + Math.max(0, stock.value),
    0,
  );
  return stocks.reduce((sum, stock) => {
    const weight =
      totalValue > 0 ? Math.max(0, stock.value) / totalValue : 1 / stocks.length;
    return sum + weight * (values[stock.ticker] ?? 0);
  }, 0);
}

function emptyBudget(
  benchmark: GrowthBenchmark,
  managedCapital: number,
  targetWeight: number,
  replacementShare: number,
): ActiveRiskBudget {
  const etfTargetDollars =
    Math.max(0, managedCapital) * Math.max(0, targetWeight);
  return {
    benchmark,
    etfTargetDollars,
    maximumReplacementDollars:
      etfTargetDollars * Math.max(0, replacementShare),
    stockTargetDollars: 0,
    remainingEtfDollars: etfTargetDollars,
    trackingError60: 0,
    trackingError252: 0,
  };
}

export function calculateManagedStockPlan({
  managedCapital,
  targetWeights,
  replacementShare,
  stocks,
  activeRiskBudget = 0.03,
  maximumNameWeight = 0.05,
  maximumTotalWeight = 0.20,
  selectionMode = "evidence_gated",
}: ManagedStockPlanInput): ManagedStockPlan {
  const safeCapital = Math.max(0, managedCapital);
  const normalizedStocks = stocks.map((stock) => ({
    ...stock,
    ticker: stock.ticker.trim().toUpperCase(),
    value: Math.max(0, stock.value),
  }));
  const unsupportedTickers = [
    ...new Set(
      normalizedStocks
        .map((stock) => stock.ticker)
        .filter((ticker) => !supportedStocks.has(ticker)),
    ),
  ];
  const tickerCounts = normalizedStocks.reduce<Record<string, number>>(
    (counts, stock) => ({
      ...counts,
      [stock.ticker]: (counts[stock.ticker] ?? 0) + 1,
    }),
    {},
  );
  const duplicateTickers = Object.entries(tickerCounts)
    .filter(([, count]) => count > 1)
    .map(([ticker]) => ticker);
  const marketStateEligible = (
    stockRiskData.stockFriendlyMarketStates as readonly string[]
  ).includes(stockRiskData.marketState);
  const activeDispersionEligible =
    stockRiskData.activeDispersionPercentile <=
    stockRiskData.activeDispersionMaximumPercentile;
  const benchmarkEligibility = Object.fromEntries(
    (["QQQ", "SMH"] as const).map((benchmark) => {
      const matching = normalizedStocks.filter(
        (stock) => stock.benchmark === benchmark,
      );
      const activeLogReturn63 = basketActiveLogReturn(
        matching,
        stockRiskData.activeLogReturn63,
      );
      const activeLogReturn126 = basketActiveLogReturn(
        matching,
        stockRiskData.activeLogReturn126,
      );
      const relativeStrengthConfirmed =
        matching.length > 0 &&
        activeLogReturn63 > 0 &&
        activeLogReturn126 > 0;
      const evidenceConfirmed =
        marketStateEligible &&
        activeDispersionEligible &&
        relativeStrengthConfirmed;
      return [
        benchmark,
        {
          activeLogReturn63,
          activeLogReturn126,
          relativeStrengthConfirmed,
          effectiveReplacementShare:
            selectionMode === "high_capture" || evidenceConfirmed
              ? Math.max(0, replacementShare)
              : 0,
        },
      ];
    }),
  ) as ManagedStockPlan["benchmarkEligibility"];
  const blockers = [
    ...(selectionMode === "evidence_gated" && !marketStateEligible
      ? [`当前市场状态 ${stockRiskData.marketState} 不是稳定上涨状态`]
      : []),
    ...(selectionMode === "evidence_gated" && !activeDispersionEligible
      ? [
          `股票分化位于过去三年的 ${(
            stockRiskData.activeDispersionPercentile * 100
          ).toFixed(0)}% 分位，高于 67% 上限`,
        ]
      : []),
    ...(["QQQ", "SMH"] as const)
      .filter(
        (benchmark) =>
          selectionMode === "evidence_gated" &&
          normalizedStocks.some((stock) => stock.benchmark === benchmark) &&
          !benchmarkEligibility[benchmark].relativeStrengthConfirmed,
      )
      .map(
        (benchmark) =>
          `${benchmark} 个股篮子没有同时跑赢过去 63 日和 126 日`,
      ),
  ];
  const budgets = Object.fromEntries(
    (["QQQ", "SMH"] as const).map((benchmark) => [
      benchmark,
      emptyBudget(
        benchmark,
        safeCapital,
        targetWeights[benchmark],
        benchmarkEligibility[benchmark].effectiveReplacementShare,
      ),
    ]),
  ) as Record<GrowthBenchmark, ActiveRiskBudget>;
  const baseResult = {
    unsupportedTickers,
    duplicateTickers,
    targets: normalizedStocks.map((stock) => ({
      ...stock,
      targetDollars: 0,
    })),
    budgets,
    combinedTrackingError: 0,
    activeRiskBudget,
    maximumNameWeight,
    maximumTotalWeight,
    totalStockTargetDollars: 0,
    dataAsOf: stockRiskData.asOf,
    selectionMode,
    marketState: stockRiskData.marketState,
    marketStateEligible,
    activeDispersionPercentile: stockRiskData.activeDispersionPercentile,
    activeDispersionEligible,
    benchmarkEligibility,
    blockers,
  };
  if (unsupportedTickers.length > 0) {
    return { ...baseResult, status: "unsupported" };
  }
  if (duplicateTickers.length > 0) {
    return { ...baseResult, status: "duplicate" };
  }
  if (
    safeCapital <= 0 ||
    normalizedStocks.length === 0 ||
    replacementShare <= 0
  ) {
    return { ...baseResult, status: "ready" };
  }

  const activeBenchmarks = (["QQQ", "SMH"] as const).filter((benchmark) =>
    normalizedStocks.some((stock) => stock.benchmark === benchmark) &&
    benchmarkEligibility[benchmark].effectiveReplacementShare > 0,
  );
  const totalMaximumReplacementWeight = activeBenchmarks.reduce(
    (sum, benchmark) =>
      sum +
      Math.max(0, targetWeights[benchmark]) *
        benchmarkEligibility[benchmark].effectiveReplacementShare,
    0,
  );
  const provisionalTargets = new Map<string, number>();

  activeBenchmarks.forEach((benchmark) => {
    const matching = normalizedStocks.filter(
      (stock) => stock.benchmark === benchmark,
    );
    const desiredTotal = matching.reduce((sum, stock) => sum + stock.value, 0);
    const mixes = matching.map((stock) =>
      desiredTotal > 0 ? stock.value / desiredTotal : 1 / matching.length,
    );
    const trackingError60 = basketTrackingError(
      matching,
      benchmark,
      stockRiskData.covariance60,
    );
    const trackingError252 = basketTrackingError(
      matching,
      benchmark,
      stockRiskData.covariance252,
    );
    const worstTrackingError = Math.max(
      trackingError60,
      trackingError252,
      1e-12,
    );
    const maximumReplacementWeight =
      Math.max(0, targetWeights[benchmark]) *
      benchmarkEligibility[benchmark].effectiveReplacementShare;
    const allocatedTrackingError =
      totalMaximumReplacementWeight > 0
        ? (activeRiskBudget * maximumReplacementWeight) /
          totalMaximumReplacementWeight
        : 0;
    const maximumMix = Math.max(...mixes);
    const stockTargetWeight = Math.min(
      maximumReplacementWeight,
      allocatedTrackingError / worstTrackingError,
      maximumNameWeight / maximumMix,
    );
    matching.forEach((stock, index) => {
      provisionalTargets.set(
        stock.id,
        stockTargetWeight * mixes[index] * safeCapital,
      );
    });
    const stockTargetDollars = stockTargetWeight * safeCapital;
    budgets[benchmark] = {
      ...budgets[benchmark],
      stockTargetDollars,
      remainingEtfDollars: Math.max(
        0,
        budgets[benchmark].etfTargetDollars - stockTargetDollars,
      ),
      trackingError60,
      trackingError252,
    };
  });

  const provisionalTotal = [...provisionalTargets.values()].reduce(
    (sum, value) => sum + value,
    0,
  );
  const aggregateScale =
    provisionalTotal > safeCapital * maximumTotalWeight
      ? (safeCapital * maximumTotalWeight) / provisionalTotal
      : 1;
  const targets = normalizedStocks.map((stock) => ({
    ...stock,
    targetDollars: (provisionalTargets.get(stock.id) ?? 0) * aggregateScale,
  }));
  (["QQQ", "SMH"] as const).forEach((benchmark) => {
    const stockTargetDollars = targets
      .filter((stock) => stock.benchmark === benchmark)
      .reduce((sum, stock) => sum + stock.targetDollars, 0);
    budgets[benchmark] = {
      ...budgets[benchmark],
      stockTargetDollars,
      remainingEtfDollars: Math.max(
        0,
        budgets[benchmark].etfTargetDollars - stockTargetDollars,
      ),
    };
  });

  const combinedVector = Array.from({ length: assets.length }, () => 0);
  targets.forEach((stock) => {
    combinedVector[assetIndex.get(stock.ticker) ?? -1] +=
      stock.targetDollars / safeCapital;
  });
  (["QQQ", "SMH"] as const).forEach((benchmark) => {
    combinedVector[assetIndex.get(benchmark) ?? -1] -=
      budgets[benchmark].stockTargetDollars / safeCapital;
  });
  const combinedTrackingError = Math.max(
    quadraticTrackingError(combinedVector, stockRiskData.covariance60),
    quadraticTrackingError(combinedVector, stockRiskData.covariance252),
  );
  const totalStockTargetDollars = targets.reduce(
    (sum, stock) => sum + stock.targetDollars,
    0,
  );

  return {
    status: "ready",
    unsupportedTickers: [],
    duplicateTickers: [],
    targets,
    budgets,
    combinedTrackingError,
    activeRiskBudget,
    maximumNameWeight,
    maximumTotalWeight,
    totalStockTargetDollars,
    dataAsOf: stockRiskData.asOf,
    selectionMode,
    marketState: stockRiskData.marketState,
    marketStateEligible,
    activeDispersionPercentile: stockRiskData.activeDispersionPercentile,
    activeDispersionEligible,
    benchmarkEligibility,
    blockers,
  };
}

export { stockRiskData };
