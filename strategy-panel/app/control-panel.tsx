"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  accountBuildRules,
  allocationRows,
  decisionChangeConditions,
  decisionDrivers,
  decisionPipeline,
  keyParameters,
  knownRisks,
  migrationStages,
  parameterGroups,
  productionEvidence,
  productionTargets,
  referencePrices,
  rejectedAlternatives,
  readinessChecks,
  strategySnapshot,
  targetDerivation,
  whiteboxRuleLedger,
} from "./strategy-data";
import {
  calculateRecursiveTrendCushion,
  protectedAccountTarget,
  type Allocation,
} from "./recursive-trend-cushion";
import {
  calculateManagedStockPlan,
  stockRiskData,
  type GrowthBenchmark,
  type StockSelectionMode,
} from "./risk-multiple";

const percent = (value: number, digits = 1) =>
  `${(value * 100).toFixed(digits)}%`;

const dollars = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

type HoldingKey =
  | "accountValue"
  | "cash"
  | "QQQ"
  | "SMH"
  | "GLD"
  | "GDE"
  | "BIL"
  | "VIXY";

type HoldingInputs = Record<HoldingKey, string>;

type StockTreatment = "exclude" | "liquidate" | "review";

type IndividualStock = {
  id: string;
  ticker: string;
  shares: string;
  price: string;
  treatment: StockTreatment;
  benchmark: GrowthBenchmark;
};

type SavedHoldingsPayload = {
  holdings: HoldingInputs;
  individualStocks: IndividualStock[];
  accountNotes: string;
  stockSelectionMode?: StockSelectionMode;
  protectionState?: ProtectionState | null;
};

type ProtectionState = {
  highWater: string;
  asOf: string;
  activatedAt: string;
};

type PersistenceStatus = "loading" | "ready" | "saving" | "saved" | "error";

const emptyHoldings: HoldingInputs = {
  accountValue: "",
  cash: "",
  QQQ: "",
  SMH: "",
  GLD: "",
  GDE: "",
  BIL: "",
  VIXY: "",
};

const holdingAssets = [
  { key: "QQQ", label: "QQQ 股数", price: referencePrices.QQQ, target: migrationStages[3].qqq },
  { key: "SMH", label: "SMH 股数", price: referencePrices.SMH, target: migrationStages[3].smh },
  { key: "GLD", label: "GLD 股数", price: referencePrices.GLD, target: migrationStages[3].gold },
  { key: "GDE", label: "GDE 股数", price: referencePrices.GDE, target: migrationStages[3].gde },
  { key: "BIL", label: "BIL 股数", price: referencePrices.BIL, target: migrationStages[3].cash },
  { key: "VIXY", label: "VIXY 股数", price: referencePrices.VIXY, target: 0 },
] as const;

const stockTreatmentLabels: Record<StockTreatment, string> = {
  exclude: "保留并排除在策略之外",
  liquidate: "逐步卖出并转入策略",
  review: "替代一部分 QQQ / SMH",
};

function allocationWeight(
  target: Allocation,
  key: (typeof holdingAssets)[number]["key"],
) {
  return key === "BIL" ? target.cash : target[key];
}

function normalizeIndividualStock(
  stock: Partial<IndividualStock> &
    Pick<IndividualStock, "id" | "ticker" | "shares" | "price" | "treatment">,
): IndividualStock {
  return {
    ...stock,
    benchmark: stock.benchmark === "SMH" ? "SMH" : "QQQ",
  };
}

const holdingsTemplate = `账户总价值：
可用现金：
QQQ：___ 股
SMH：___ 股
GLD：___ 股
GDE：___ 股
BIL：___ 股
VIXY：___ 股
其他持仓或账户限制：
数据记录时间：`;

function StatusDot({ tone }: { tone: string }) {
  return <span className={`status-dot ${tone}`} aria-hidden="true" />;
}

function TargetAllocationBar({
  finalTarget,
  color,
}: {
  finalTarget: number;
  color: string;
}) {
  return (
    <div className="target-bar" aria-hidden="true">
      <div
        className="target-bar-fill"
        style={{
          width: `${finalTarget * 100}%`,
          background: color,
        }}
      />
    </div>
  );
}

export function ControlPanel() {
  const [expandedParameter, setExpandedParameter] = useState("");
  const [showHoldingsEntry, setShowHoldingsEntry] = useState(false);
  const [holdings, setHoldings] = useState<HoldingInputs>(emptyHoldings);
  const [individualStocks, setIndividualStocks] = useState<IndividualStock[]>([]);
  const [accountNotes, setAccountNotes] = useState("");
  const [stockSelectionMode, setStockSelectionMode] =
    useState<StockSelectionMode>("evidence_gated");
  const [holdingError, setHoldingError] = useState("");
  const [draftReady, setDraftReady] = useState(false);
  const [templateCopyState, setTemplateCopyState] = useState("复制空白模板");
  const [validatedCopyState, setValidatedCopyState] = useState("复制已校验的持仓数据");
  const [persistenceReady, setPersistenceReady] = useState(false);
  const [persistenceStatus, setPersistenceStatus] =
    useState<PersistenceStatus>("loading");
  const [hasSavedHoldings, setHasSavedHoldings] = useState(false);
  const [lastSavedAt, setLastSavedAt] = useState("");
  const [protectionState, setProtectionState] =
    useState<ProtectionState | null>(null);
  const lastSavedPayload = useRef("");

  const hasHoldingData =
    Object.values(holdings).some(Boolean) ||
    individualStocks.length > 0 ||
    Boolean(accountNotes);

  const visibleDecisionPipeline = decisionPipeline.map((step) => {
    if (step.stage !== "05 · 执行门控" || !hasSavedHoldings) return step;
    if (draftReady) {
      return {
        ...step,
        title: "私有持仓已校验，调整草稿已生成",
        evidence: "已保存持仓已加载，并通过账户完整性检查。",
        result: "可以人工复核调整草稿；系统仍不会自动向券商提交订单。",
        plainLanguage: "你已经跨过了“系统不知道你持有什么”这一关；下一步是核对价格和外部执行约束，而不是重新输入仓位。",
      };
    }
    return {
      ...step,
      title: "私有持仓已恢复，等待与券商核对",
      evidence: "本私有 Panel 已加载你保存的股数和现金；生产快照本身不会读取这份私人数据。",
      result: "核对账户总值、股数、现金与 GDE 后，可以生成调整草稿。",
      plainLanguage: "系统已经有你的上次记录，但需要你确认它仍与券商一致；这与生产系统永不自动下单是两件不同的事。",
    };
  });

  useEffect(() => {
    const controller = new AbortController();

    async function loadSavedHoldings() {
      try {
        const response = await fetch("/api/holdings", {
          signal: controller.signal,
          cache: "no-store",
        });
        const result = (await response.json()) as {
          holdings?: SavedHoldingsPayload | null;
          updatedAt?: string | null;
          error?: string;
        };
        if (!response.ok) throw new Error(result.error || "无法读取已保存持仓。");

        if (result.holdings) {
          setHoldings({ ...emptyHoldings, ...result.holdings.holdings });
          setIndividualStocks(
            result.holdings.individualStocks.map(normalizeIndividualStock),
          );
          setAccountNotes(result.holdings.accountNotes);
          setStockSelectionMode(
            result.holdings.stockSelectionMode === "high_capture"
              ? "high_capture"
              : "evidence_gated",
          );
          setProtectionState(result.holdings.protectionState ?? null);
          lastSavedPayload.current = JSON.stringify(result.holdings);
          setHasSavedHoldings(true);
          setLastSavedAt(result.updatedAt ?? "");
          setPersistenceStatus("saved");
        } else {
          setPersistenceStatus("ready");
        }
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setPersistenceStatus("error");
      } finally {
        if (!controller.signal.aborted) setPersistenceReady(true);
      }
    }

    void loadSavedHoldings();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!persistenceReady || !hasHoldingData) return;

    const payload: SavedHoldingsPayload = {
      holdings,
      individualStocks,
      accountNotes,
      stockSelectionMode,
      protectionState,
    };
    const serialized = JSON.stringify(payload);
    if (serialized === lastSavedPayload.current) return;

    setPersistenceStatus("saving");
    const timer = window.setTimeout(async () => {
      try {
        const response = await fetch("/api/holdings", {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: serialized,
        });
        const result = (await response.json()) as {
          updatedAt?: string;
          error?: string;
        };
        if (!response.ok) throw new Error(result.error || "无法保存持仓。");

        setHasSavedHoldings(true);
        lastSavedPayload.current = serialized;
        setLastSavedAt(result.updatedAt ?? new Date().toISOString());
        setPersistenceStatus("saved");
      } catch {
        setPersistenceStatus("error");
      }
    }, 800);

    return () => window.clearTimeout(timer);
  }, [
    accountNotes,
    hasHoldingData,
    holdings,
    individualStocks,
    persistenceReady,
    stockSelectionMode,
    protectionState,
  ]);

  const accountProtection = useMemo(() => {
    const accountValue = Number(holdings.accountValue);
    const excludedStockValue = individualStocks
      .filter((stock) => stock.treatment === "exclude")
      .reduce(
        (sum, stock) =>
          sum + Number(stock.shares || 0) * Number(stock.price || 0),
        0,
      );
    const managedEquity = accountValue - excludedStockValue;
    const savedHighWater = Number(protectionState?.highWater ?? 0);
    const stateDateValid =
      !protectionState || protectionState.asOf <= referencePrices.priceDate;
    const highWater = Math.max(
      managedEquity,
      Number.isFinite(savedHighWater) && savedHighWater > 0
        ? savedHighWater
        : managedEquity,
    );
    const eligible =
      strategySnapshot.recursiveTrendCushion.productionEligible &&
      strategySnapshot.recursiveTrendCushion.sameDate;
    const calculation =
      eligible && stateDateValid && managedEquity > 0
        ? calculateRecursiveTrendCushion({
            priorEquity: managedEquity,
            priorPeak: highWater,
            dualTrendPositive:
              strategySnapshot.recursiveTrendCushion.dualTrendPositive,
            parameters: {
              floorDrawdown:
                strategySnapshot.recursiveTrendCushion.floorDrawdown,
              bullMultiplier:
                strategySnapshot.recursiveTrendCushion.bullMultiplier,
              bearMultiplier:
                strategySnapshot.recursiveTrendCushion.bearMultiplier,
              tierSize: strategySnapshot.recursiveTrendCushion.tierSize,
            },
          })
        : null;
    const target = calculation
      ? protectedAccountTarget({
          stagedTarget: productionTargets.staged,
          r11Target: productionTargets.r11,
          acceptedNonCashCap: calculation.acceptedNonCashCap,
        })
      : productionTargets.staged;
    return {
      eligible,
      stateDateValid,
      managedEquity,
      highWater,
      calculation,
      target,
      initialized: protectionState !== null,
    };
  }, [holdings.accountValue, individualStocks, protectionState]);

  const holdingCalculation = useMemo(() => {
    const accountValue = Number(holdings.accountValue);
    const cash = Number(holdings.cash || 0);
    const assetValues = Object.fromEntries(
      holdingAssets.map((asset) => [
        asset.key,
        Number(holdings[asset.key] || 0) * asset.price,
      ]),
    ) as Record<(typeof holdingAssets)[number]["key"], number>;
    const stockValues = individualStocks.map((stock) => {
      const value =
        Number(stock.shares || 0) * Number(stock.price || 0);
      return {
        ...stock,
        normalizedTicker: stock.ticker.trim().toUpperCase(),
        numericShares: Number(stock.shares || 0),
        numericPrice: Number(stock.price || 0),
        value,
      };
    });
    const individualStockValue = stockValues.reduce((sum, stock) => sum + stock.value, 0);
    const excludedStockValue = stockValues
      .filter((stock) => stock.treatment === "exclude")
      .reduce((sum, stock) => sum + stock.value, 0);
    const liquidateStocks = stockValues.filter((stock) => stock.treatment === "liquidate");
    const reviewStocks = stockValues.filter((stock) => stock.treatment === "review");
    const estimatedTotal =
      cash +
      Object.values(assetValues).reduce((sum, value) => sum + value, 0) +
      individualStockValue;
    const mismatch =
      accountValue > 0 ? Math.abs(estimatedTotal - accountValue) / accountValue : 1;
    const managedCapital = accountValue - excludedStockValue;
    const managedStockPlan = calculateManagedStockPlan({
      managedCapital,
      targetWeights: {
        QQQ: accountProtection.target.QQQ,
        SMH: accountProtection.target.SMH,
      },
      replacementShare: strategySnapshot.stockReplacementShare,
      selectionMode: stockSelectionMode,
      stocks: reviewStocks.map((stock) => ({
        id: stock.id,
        ticker: stock.normalizedTicker,
        benchmark: stock.benchmark,
        value: stock.value,
      })),
    });
    const operationalStockTargets = new Map(
      reviewStocks.map((stock) => {
        const fullTarget =
          managedStockPlan.targets.find((target) => target.id === stock.id)
            ?.targetDollars ?? 0;
        return [
          stock.id,
          strategySnapshot.stockIncreaseAllowedNow
            ? fullTarget
            : Math.min(stock.value, fullTarget),
        ];
      }),
    );
    const operationalStockByBenchmark = Object.fromEntries(
      (["QQQ", "SMH"] as const).map((benchmark) => [
        benchmark,
        reviewStocks
          .filter((stock) => stock.benchmark === benchmark)
          .reduce(
            (sum, stock) =>
              sum + (operationalStockTargets.get(stock.id) ?? 0),
            0,
          ),
      ]),
    ) as Record<GrowthBenchmark, number>;
    const operationalEtfTargets = {
      QQQ: Math.max(
        0,
        managedCapital * accountProtection.target.QQQ -
          operationalStockByBenchmark.QQQ,
      ),
      SMH: Math.max(
        0,
        managedCapital * accountProtection.target.SMH -
          operationalStockByBenchmark.SMH,
      ),
    };
    const protectionControlled =
      (accountProtection.calculation?.acceptedNonCashCap ?? 1) < 1;
    const gdeLowerValue = managedCapital * (
      protectionControlled
        ? accountProtection.target.GDE
        : strategySnapshot.stagedGdeLower
    );
    const gdeUpperValue = managedCapital * (
      protectionControlled
        ? accountProtection.target.GDE
        : strategySnapshot.stagedGdeUpper
    );
    const operationalGdeTarget =
      assetValues.GDE < gdeLowerValue
        ? gdeLowerValue
        : assetValues.GDE > gdeUpperValue
          ? gdeUpperValue
          : assetValues.GDE;
    const operationalGoldTarget =
      managedCapital *
        (accountProtection.target.GLD + accountProtection.target.GDE) -
      operationalGdeTarget;
    const targetCashSleeveValue =
      managedCapital * accountProtection.target.cash;

    const rawRows = [
      ...holdingAssets
        .filter((asset) => asset.key !== "BIL")
        .map((asset) => {
          const currentValue = assetValues[asset.key];
          const targetValue =
            asset.key === "QQQ" || asset.key === "SMH"
              ? operationalEtfTargets[asset.key]
              : asset.key === "GDE"
                ? operationalGdeTarget
                : asset.key === "GLD"
                  ? operationalGoldTarget
                  : managedCapital * allocationWeight(accountProtection.target, asset.key);
          const difference = targetValue - currentValue;
          return {
            rowKey: asset.key,
            asset: asset.key,
            currentValue,
            fullStageTargetWeight: allocationWeight(accountProtection.target, asset.key),
            rawDifference: difference,
            approximateShares: difference / asset.price,
            combinedCashSleeve: false,
            sourceShares: Number(holdings[asset.key] || 0),
            price: asset.price,
          };
        }),
      {
        rowKey: "BIL-cash",
        asset: "BIL / 现金",
        currentValue: assetValues.BIL + cash,
        fullStageTargetWeight:
          managedCapital > 0 ? targetCashSleeveValue / managedCapital : 0,
        rawDifference: targetCashSleeveValue - assetValues.BIL - cash,
        approximateShares: 0,
        combinedCashSleeve: true,
        sourceShares: Number(holdings.BIL || 0),
        price: referencePrices.BIL,
      },
      ...reviewStocks.map((stock) => {
        const targetValue = operationalStockTargets.get(stock.id) ?? 0;
        return {
          rowKey: stock.id,
          asset: `${stock.normalizedTicker}（替代 ${stock.benchmark}）`,
          currentValue: stock.value,
          fullStageTargetWeight:
            managedCapital > 0 ? targetValue / managedCapital : 0,
          rawDifference: targetValue - stock.value,
          approximateShares: (targetValue - stock.value) / stock.numericPrice,
          combinedCashSleeve: false,
          sourceShares: stock.numericShares,
          price: stock.numericPrice,
        };
      }),
      ...liquidateStocks.map((stock) => ({
        rowKey: stock.id,
        asset: stock.normalizedTicker,
        currentValue: stock.value,
        fullStageTargetWeight: 0,
        rawDifference: -stock.value,
        approximateShares: -stock.numericShares,
        combinedCashSleeve: false,
        sourceShares: stock.numericShares,
        price: stock.numericPrice,
      })),
    ];
    const uncappedOneWayTurnover =
      managedCapital > 0
        ? rawRows.reduce((sum, row) => sum + Math.abs(row.rawDifference), 0) /
          (2 * managedCapital)
        : 1;
    const nonGoldSleeveTurnover =
      managedCapital > 0
        ? rawRows
            .filter((row) => row.rowKey !== "GLD" && row.rowKey !== "GDE")
            .reduce((sum, row) => sum + Math.abs(row.rawDifference), 0) /
          (2 * managedCapital)
        : 1;
    const goldSleeveOnlyAdjustment =
      nonGoldSleeveTurnover < strategySnapshot.noTradeThreshold;
    const turnoverCapFactor =
      uncappedOneWayTurnover < strategySnapshot.noTradeThreshold
        ? 0
        : Math.min(
            goldSleeveOnlyAdjustment ? 1 : 0.25,
            strategySnapshot.trancheTurnoverLimit / uncappedOneWayTurnover,
            1,
          );
    const rows = rawRows.map((row) => {
      const difference = row.rawDifference * turnoverCapFactor;
      const resultingValue = row.currentValue + difference;
      return {
        ...row,
        difference,
        approximateShares: row.combinedCashSleeve
          ? 0
          : difference / row.price,
        currentWeight: managedCapital > 0 ? row.currentValue / managedCapital : 0,
        targetWeight: managedCapital > 0 ? resultingValue / managedCapital : 0,
      };
    });
    const appliedOneWayTurnover = uncappedOneWayTurnover * turnoverCapFactor;

    return {
      accountValue,
      cash,
      assetValues,
      stockValues,
      individualStockValue,
      excludedStockValue,
      liquidateStocks,
      reviewStocks,
      managedStockPlan,
      operationalStockTargets,
      estimatedTotal,
      mismatch,
      managedCapital,
      uncappedOneWayTurnover,
      goldSleeveOnlyAdjustment,
      turnoverCapFactor,
      appliedOneWayTurnover,
      rows,
    };
  }, [accountProtection, holdings, individualStocks, stockSelectionMode]);

  function openHoldingsEntry() {
    setShowHoldingsEntry(true);
    window.setTimeout(
      () => document.getElementById("holdings-entry")?.scrollIntoView({ behavior: "smooth" }),
      0,
    );
  }

  function updateHolding(key: HoldingKey, value: string) {
    setHoldings((current) => ({ ...current, [key]: value }));
    setDraftReady(false);
    setHoldingError("");
  }

  function addIndividualStock() {
    setIndividualStocks((current) => [
      ...current,
      {
        id: `stock-${Date.now()}-${current.length}`,
        ticker: "",
        shares: "",
        price: "",
        treatment: "exclude",
        benchmark: "QQQ",
      },
    ]);
    setDraftReady(false);
    setHoldingError("");
  }

  function updateIndividualStock<K extends keyof Omit<IndividualStock, "id">>(
    id: string,
    field: K,
    value: IndividualStock[K],
  ) {
    setIndividualStocks((current) =>
      current.map((stock) =>
        stock.id === id ? { ...stock, [field]: value } : stock,
      ),
    );
    setDraftReady(false);
    setHoldingError("");
  }

  function updateIndividualStockTicker(id: string, value: string) {
    const normalized = value.trim().toUpperCase();
    const latestPrice = (
      stockRiskData.latestPrices as Record<string, number>
    )[normalized];
    setIndividualStocks((current) =>
      current.map((stock) =>
        stock.id === id
          ? {
              ...stock,
              ticker: value,
              price:
                stock.price || latestPrice === undefined
                  ? stock.price
                  : latestPrice.toString(),
            }
          : stock,
      ),
    );
    setDraftReady(false);
    setHoldingError("");
  }

  function removeIndividualStock(id: string) {
    setIndividualStocks((current) => current.filter((stock) => stock.id !== id));
    setDraftReady(false);
    setHoldingError("");
  }

  async function clearSavedHoldings() {
    if (
      hasSavedHoldings &&
      !window.confirm("确定删除已保存的全部真实持仓吗？删除后无法恢复。")
    ) {
      return;
    }

    if (hasSavedHoldings) {
      setPersistenceStatus("saving");
      try {
        const response = await fetch("/api/holdings", { method: "DELETE" });
        if (!response.ok) throw new Error("无法删除已保存持仓。");
      } catch {
        setPersistenceStatus("error");
        setHoldingError("删除失败，已保存的持仓仍然保留。请稍后重试。");
        return;
      }
    }

    setHoldings(emptyHoldings);
    setIndividualStocks([]);
    setAccountNotes("");
    setStockSelectionMode("evidence_gated");
    setProtectionState(null);
    setDraftReady(false);
    setHoldingError("");
    setHasSavedHoldings(false);
    setLastSavedAt("");
    lastSavedPayload.current = "";
    setPersistenceStatus("ready");
  }

  function generateDraft(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const values = Object.values(holdings).map((value) => Number(value || 0));
    if (!Number.isFinite(holdingCalculation.accountValue) || holdingCalculation.accountValue <= 0) {
      setHoldingError("请填写大于 0 的账户总价值。");
      return;
    }
    if (values.some((value) => !Number.isFinite(value) || value < 0)) {
      setHoldingError("所有金额和股数都必须是大于或等于 0 的数字。");
      return;
    }
    const invalidStock = holdingCalculation.stockValues.find(
      (stock) =>
        !stock.normalizedTicker ||
        !Number.isFinite(stock.numericShares) ||
        stock.numericShares < 0 ||
        !Number.isFinite(stock.numericPrice) ||
        stock.numericPrice <= 0,
    );
    if (invalidStock) {
      setHoldingError("请完整填写每只个股的代码、股数和大于 0 的当前价格。");
      return;
    }
    if (holdingCalculation.mismatch > 0.02) {
      setHoldingError(
        `按参考价估算的资产合计为 ${dollars.format(holdingCalculation.estimatedTotal)}，与账户总价值相差 ${percent(holdingCalculation.mismatch, 1)}。请检查是否漏填持仓；差异降至 2% 以内后再生成调整建议。`,
      );
      return;
    }
    if (holdingCalculation.managedCapital <= 0) {
      setHoldingError("排除的个股价值不能等于或超过账户总价值。");
      return;
    }
    if (holdingCalculation.managedStockPlan.status === "unsupported") {
      setHoldingError(
        `暂时没有 ${holdingCalculation.managedStockPlan.unsupportedTickers.join("、")} 的已审计风险数据。请先选择“保留并排除”或“逐步卖出”，不要用不完整数据替代 QQQ / SMH。`,
      );
      return;
    }
    if (holdingCalculation.managedStockPlan.status === "duplicate") {
      setHoldingError(
        `同一股票请只录入一次：${holdingCalculation.managedStockPlan.duplicateTickers.join("、")}。`,
      );
      return;
    }
    if (!accountProtection.eligible) {
      setHoldingError("R40 账户保护层尚未通过同日生产资格，不能生成调整草稿。");
      return;
    }
    if (!accountProtection.stateDateValid) {
      setHoldingError("保存的账户保护日期晚于当前行情日期，已阻断草稿；请等待生产数据追上后再试。");
      return;
    }
    setProtectionState({
      highWater: accountProtection.highWater.toFixed(2),
      asOf: referencePrices.priceDate,
      activatedAt:
        protectionState?.activatedAt ?? new Date().toISOString(),
    });
    setHoldingError("");
    setDraftReady(true);
    window.setTimeout(
      () => document.getElementById("holding-draft")?.scrollIntoView({ behavior: "smooth" }),
      0,
    );
  }

  async function copyHoldingsTemplate() {
    try {
      await navigator.clipboard.writeText(holdingsTemplate);
      setTemplateCopyState("已复制空白模板");
      window.setTimeout(() => setTemplateCopyState("复制空白模板"), 1800);
    } catch {
      setTemplateCopyState("请手动复制下方模板");
    }
  }

  async function copyValidatedHoldings() {
    const normalized = `${strategySnapshot.activeStrategy} 真实持仓录入
账户总价值：${holdings.accountValue}
可用现金：${holdings.cash || "0"}
QQQ：${holdings.QQQ || "0"} 股
SMH：${holdings.SMH || "0"} 股
GLD：${holdings.GLD || "0"} 股
GDE：${holdings.GDE || "0"} 股
BIL：${holdings.BIL || "0"} 股
VIXY：${holdings.VIXY || "0"} 股
其他个股：
${holdingCalculation.stockValues.length === 0
  ? "无"
  : holdingCalculation.stockValues
      .map(
        (stock) =>
          `${stock.normalizedTicker}：${stock.numericShares} 股，价格 ${stock.numericPrice}，处理方式 ${stockTreatmentLabels[stock.treatment]}${
            stock.treatment === "review"
              ? `，替代 ${stock.benchmark}，目标市值 ${
                  holdingCalculation.managedStockPlan.targets
                    .find((target) => target.id === stock.id)
                    ?.targetDollars.toFixed(2) ?? "0"
                }`
              : ""
          }`,
      )
      .join("\n")}
账户限制：${accountNotes || "无"}
估值参考日期：${referencePrices.priceDate}
按参考价估算合计：${holdingCalculation.estimatedTotal.toFixed(2)}
排除在策略之外的个股价值：${holdingCalculation.excludedStockValue.toFixed(2)}
策略实际管理资金：${holdingCalculation.managedCapital.toFixed(2)}
个股使用方式：${stockSelectionMode === "evidence_gated" ? "证据门控（默认）" : "高捕获（需要真实记录支持）"}
当前个股使用判断：${holdingCalculation.managedStockPlan.blockers.length === 0 ? "允许个股替代" : `ETF 优先：${holdingCalculation.managedStockPlan.blockers.join("；")}`}
个股组合预计差异波动：${percent(holdingCalculation.managedStockPlan.combinedTrackingError, 2)}
QQQ 替代个股目标市值：${holdingCalculation.managedStockPlan.budgets.QQQ.stockTargetDollars.toFixed(2)}
QQQ ETF 剩余额度：${holdingCalculation.managedStockPlan.budgets.QQQ.remainingEtfDollars.toFixed(2)}
SMH 替代个股目标市值：${holdingCalculation.managedStockPlan.budgets.SMH.stockTargetDollars.toFixed(2)}
SMH ETF 剩余额度：${holdingCalculation.managedStockPlan.budgets.SMH.remainingEtfDollars.toFixed(2)}
与账户总价值差异：${percent(holdingCalculation.mismatch, 2)}`;
    try {
      await navigator.clipboard.writeText(normalized);
      setValidatedCopyState("已复制，可以发给我");
      window.setTimeout(() => setValidatedCopyState("复制已校验的持仓数据"), 1800);
    } catch {
      setHoldingError("浏览器未允许复制。请手动记录表单内容。");
    }
  }

  return (
    <main id="top">
      <header className="app-header">
        <a className="wordmark" href="#top" aria-label="返回顶部">
          <span>{strategySnapshot.activeStrategy}</span>
          <strong>策略运行驾驶舱</strong>
        </a>
        <nav aria-label="页面导航">
          <a href="#action">下一步</a>
          <a href="#decision-chain">决策链</a>
          <a href="#allocation">目标仓位</a>
          <a href="#weight-proof">权重理由</a>
          <a href="#drivers">当前信号</a>
          <a href="#parameters">参数</a>
          <a href="#risk">风险</a>
        </nav>
        <div className="header-version">
          <span>生产版本</span>
          <strong>{strategySnapshot.release}</strong>
        </div>
      </header>

      <div className="page-shell">
        <section className="page-intro">
          <div>
            <p className="overline">每日只需要先看这一页</p>
            <h1>
              策略已冻结：75% R11 + 25% R38。
            </h1>
            <p className="intro-copy">
              {`当前已记录 ${strategySnapshot.overfitGovernance.forwardSessions} / ${strategySnapshot.overfitGovernance.minimumForwardSessions} 个完整前瞻交易日；这些新数据只用于通过或停止判断，不用于调参。当前目标与参考价截至 ${strategySnapshot.priceAsOf}。`}
            </p>
          </div>
          <div className="as-of-card">
            <span>最近完整市场数据</span>
            <strong>{strategySnapshot.priceAsOf}</strong>
            <small>{strategySnapshot.generatedAt} 生成</small>
            <i><StatusDot tone="good" />GDE 与核心同日，最近 5 个交易日连续</i>
          </div>
        </section>

        <section className="action-layout" id="action">
          <article className="primary-action">
            <div className="action-state">
              <span className="action-number">01</span>
              <div>
                <p className="overline">你现在应该做什么</p>
                <h2>
                  {hasSavedHoldings
                    ? "先核对已保存持仓，暂时不要下单"
                    : "先录入真实持仓，暂时不要下单"}
                </h2>
              </div>
              <span className="blocked-pill">订单不可执行</span>
            </div>

            <p className="action-explanation">
              {hasSavedHoldings
                ? `系统已经自动恢复上次保存的持仓。请对照券商检查股数、现金和个股价格；确认无误后，使用截至 ${strategySnapshot.priceAsOf} 的生产目标计算调整草稿。`
                : `系统还不知道你的真实持仓，因此现在无法判断需要买卖什么。请先录入 QQQ、SMH、GLD、GDE、BIL、VIXY、其他个股的实际股数和可用现金；行情本身已更新到 ${strategySnapshot.priceAsOf}。`}
            </p>

            <section className="whitebox-panel" id="decision-chain">
              <div className="whitebox-heading">
                <div>
                  <p className="overline">为什么现在这样做</p>
                  <h3>从市场状态到行动的完整决策链</h3>
                </div>
                <span>白匣子 · 数据截至 {strategySnapshot.priceAsOf}</span>
              </div>

              <div className="strategy-lineage" aria-label="策略层级">
                <span>R9 / HMM 状态</span>
                <i>→</i>
                <span>R11 基础组合</span>
                <i>→</i>
                <span>R38 25% 冻结</span>
                <i>→</i>
                <span>R39–R41 阻断</span>
                <i>→</i>
                <span>账户执行门控</span>
              </div>

              <div className="decision-pipeline">
                {visibleDecisionPipeline.map((step) => (
                  <article key={step.stage}>
                    <div>
                      <span>{step.stage}</span>
                      <b>{step.source}</b>
                    </div>
                    <h4>{step.title}</h4>
                    <p className="decision-plain"><strong>这对你意味着：</strong>{step.plainLanguage}</p>
                    <p><strong>输入：</strong>{step.evidence}</p>
                    <p><strong>规则：</strong>{step.rule}</p>
                    <small><strong>结果：</strong>{step.result}</small>
                    <p className="decision-counterfactual"><strong>若情况改变：</strong>{step.counterfactual}</p>
                  </article>
                ))}
              </div>

              <section className="rule-ledger" aria-label="当前规则核对表">
                <div className="rule-ledger-heading">
                  <p className="overline">规则核对表</p>
                  <span>每一项均显示当前读数、门槛和实际影响</span>
                </div>
                <div className="rule-ledger-head" aria-hidden="true">
                  <span>层级</span>
                  <span>当前读数</span>
                  <span>门槛</span>
                  <span>结果与影响</span>
                </div>
                {whiteboxRuleLedger.map((rule) => (
                  <article className={`rule-ledger-row ${rule.tone}`} key={rule.layer}>
                    <strong>{rule.layer}</strong>
                    <span>{rule.observed}</span>
                    <span>{rule.threshold}</span>
                    <p><b>{rule.outcome}</b>{rule.effect}</p>
                  </article>
                ))}
              </section>

              <div className="counterfactuals">
                <p className="overline">被排除的替代动作</p>
                <div>
                  {rejectedAlternatives.map((alternative) => (
                    <article key={alternative.question}>
                      <strong>{alternative.question}</strong>
                      <p>{alternative.answer}</p>
                    </article>
                  ))}
                </div>
              </div>

              <details className="change-conditions">
                <summary>哪些变化会让结论改变？</summary>
                <ul>
                  {decisionChangeConditions.map((condition) => (
                    <li key={condition}>{condition}</li>
                  ))}
                </ul>
              </details>
            </section>

            <ol className="next-steps">
              <li className="now">
                <span>现在</span>
                <div>
                  <strong>
                    {hasSavedHoldings ? "核对已保存的真实持仓" : "提供真实账户持仓"}
                  </strong>
                  <small>
                    {hasSavedHoldings
                      ? "确认股数、现金、账户总价值和个股价格仍与券商一致"
                      : "股数、可用现金、账户总价值，以及不能交易或不希望卖出的限制"}
                  </small>
                </div>
              </li>
              <li>
                <span>随后</span>
                <div>
                  <strong>计算下一次调整建议</strong>
                  <small>每次向建议组合前进一部分，最多调动策略管理资金的 10%</small>
                </div>
              </li>
              <li>
                <span>最后</span>
                <div>
                  <strong>在开盘前人工复核</strong>
                  <small>确认价格仍新鲜、订单状态变为“可以执行”，再提交券商</small>
                </div>
              </li>
            </ol>

            <div className="action-buttons">
              <button
                className="primary-button"
                onClick={openHoldingsEntry}
                disabled={!persistenceReady}
              >
                {!persistenceReady
                  ? "正在读取持仓"
                  : hasSavedHoldings
                    ? "检查已保存持仓"
                    : "录入真实持仓"}
              </button>
              <a className="secondary-button" href="#allocation">
                查看模型建议组合
              </a>
            </div>
          </article>

          <aside className="execution-window">
            <p className="overline">下一笔建仓最早窗口</p>
            <strong>
              {strategySnapshot.executionWindowStatus === "MISSED"
                ? "本次开盘窗口已过"
                : `${strategySnapshot.nextExecutionWeekdayZh}开盘`}
            </strong>
            <time>
              {strategySnapshot.executionWindowStatus === "MISSED"
                ? `原计划：${strategySnapshot.nextExecutionEt}`
                : strategySnapshot.nextExecutionEt}
            </time>
            <small>
              {strategySnapshot.executionWindowStatus === "MISSED"
                ? `当前版本使用截至 ${strategySnapshot.priceAsOf} 的完整收盘数据部署，不生成盘中追单。`
                : strategySnapshot.nextExecutionJst}
            </small>
            <div className="window-rule">
              <span>这就是第二笔的时间</span>
              <p>
                前提是第一笔已经成交，并已把成交后的真实股数和现金更新到面板；
                届时仍须用最新完整收盘重新计算，而不是照抄第一笔之后预估的订单。
              </p>
            </div>
            <div className="window-rule">
              <span>本笔完成后</span>
              <p>
                {strategySnapshot.accountBuildTiming.afterCurrentTrancheReviewEt}
                （{strategySnapshot.accountBuildTiming.afterCurrentTrancheReviewJst}）复核；
                若全部门控通过，再下一笔最早为{" "}
                {strategySnapshot.accountBuildTiming.followingTrancheEarliestExecutionEt}
                （{strategySnapshot.accountBuildTiming.followingTrancheEarliestExecutionJst}）。
              </p>
            </div>
            <div className="authority-summary">
              <span>{strategySnapshot.activeStrategy} 当前合格规则已启用</span>
              <ul>
                <li>双 200 日趋势决定是否提高上涨参与</li>
                <li>QQQ / SMH 短期波动加速时只保留原 R38 容量</li>
                <li>半导体信号只调整 QQQ / SMH 内部比例</li>
                <li>
                  {strategySnapshot.activeStrategy === "R39"
                    ? "21 日相对损失按账户 3% 预算限制集中度"
                    : "R39 额外保护当前待命，本次继续使用 R38 目标"}
                </li>
                <li>异常波动最多取消融资一个交易日</li>
                <li>现金不得低于 −20%</li>
              </ul>
              <small>{strategySnapshot.authorityRelease}</small>
            </div>
            <div className="window-rule">
              <span>前提</span>
              <p>真实持仓已录入，且用最新完整收盘数据重新计算。</p>
            </div>
            <div className="hold-explainer">
              <span>为什么模型今天没有改变建议组合？</span>
              <p>
                模型认为当前建议比例仍然有效；这不代表你的真实账户已经达到这些比例。
              </p>
            </div>
          </aside>
        </section>

        {showHoldingsEntry && (
          <section className="holdings-entry" id="holdings-entry">
            <div className="holdings-entry-heading">
              <div>
                <p className="overline">真实持仓录入</p>
                <h2>输入账户当前实际数据</h2>
                <p>
                  输入后会自动保存在这个私有 Panel 中，并在下次打开时恢复。只有登录此私有
                  Panel 的账户可以读取；你可以随时删除。
                </p>
              </div>
              <div className="holding-entry-controls">
                <div
                  className={`persistence-state ${persistenceStatus}`}
                  role="status"
                  aria-live="polite"
                >
                  <StatusDot
                    tone={persistenceStatus === "error" ? "blocked" : "good"}
                  />
                  <span>
                    {persistenceStatus === "loading"
                      ? "正在读取已保存持仓"
                      : persistenceStatus === "saving"
                        ? "正在自动保存"
                        : persistenceStatus === "saved"
                          ? "已保存在此私有 Panel"
                          : persistenceStatus === "error"
                            ? "自动保存失败"
                            : "尚未保存"}
                  </span>
                  {lastSavedAt && persistenceStatus === "saved" && (
                    <small>
                      最近保存：
                      {new Date(lastSavedAt).toLocaleString("zh-CN", {
                        dateStyle: "short",
                        timeStyle: "short",
                      })}
                    </small>
                  )}
                </div>
                <button
                  className="close-entry"
                  onClick={() => {
                    setShowHoldingsEntry(false);
                    setDraftReady(false);
                    setHoldingError("");
                  }}
                  aria-label="关闭真实持仓录入"
                >
                  关闭
                </button>
              </div>
            </div>

            <form onSubmit={generateDraft}>
              <div className="account-fields">
                <label>
                  <span>账户总价值（美元）</span>
                  <input
                    type="number"
                    min="0"
                    step="0.01"
                    inputMode="decimal"
                    placeholder="例如 700000"
                    value={holdings.accountValue}
                    onChange={(event) => updateHolding("accountValue", event.target.value)}
                    required
                  />
                  <small>使用券商显示的账户净值</small>
                </label>
                <label>
                  <span>可用现金（美元）</span>
                  <input
                    type="number"
                    min="0"
                    step="0.01"
                    inputMode="decimal"
                    placeholder="例如 25000"
                    value={holdings.cash}
                    onChange={(event) => updateHolding("cash", event.target.value)}
                  />
                  <small>不包括 BIL 的市场价值</small>
                </label>
              </div>

              <div className="share-fields">
                {holdingAssets.map((asset) => (
                  <label key={asset.key}>
                    <span>{asset.label}</span>
                    <input
                      type="number"
                      min="0"
                      step="0.000001"
                      inputMode="decimal"
                      placeholder="0"
                      value={holdings[asset.key]}
                      onChange={(event) => updateHolding(asset.key, event.target.value)}
                    />
                    <small>
                      {referencePrices.priceDate} 参考价 {dollars.format(asset.price)}
                    </small>
                  </label>
                ))}
              </div>

              <div className="individual-stock-section">
                <div className="individual-stock-heading">
                  <div>
                    <strong>其他个股</strong>
                    <p>
                      默认选择“保留并排除在策略之外”。系统会从账户总价值中扣除这些个股，
                      再计算策略实际管理的资金。
                    </p>
                  </div>
                  <button className="secondary-button" type="button" onClick={addIndividualStock}>
                    ＋ 添加个股
                  </button>
                </div>

                <div className="stock-selection-mode">
                  <div>
                    <strong>什么时候使用个股？</strong>
                    <p>
                      默认先确认市场稳定、所选股票持续跑赢 ETF，并避开极端高分化。
                      不满足时保留 QQQ / SMH。
                    </p>
                  </div>
                  <div className="stock-mode-options">
                    <label>
                      <input
                        type="radio"
                        name="stock-selection-mode"
                        value="evidence_gated"
                        checked={stockSelectionMode === "evidence_gated"}
                        onChange={() =>
                          setStockSelectionMode("evidence_gated")
                        }
                      />
                      <span>
                        <strong>证据门控（默认）</strong>
                        <small>优先减少错误选股；当前生产推荐</small>
                      </span>
                    </label>
                    <label>
                      <input
                        type="radio"
                        name="stock-selection-mode"
                        value="high_capture"
                        checked={stockSelectionMode === "high_capture"}
                        onChange={() => setStockSelectionMode("high_capture")}
                      />
                      <span>
                        <strong>高捕获</strong>
                        <small>仅适合有至少两年真实选股记录并接受更大个股风险</small>
                      </span>
                    </label>
                  </div>
                  <div
                    className={`stock-mode-decision ${
                      holdingCalculation.managedStockPlan.blockers.length > 0
                        ? "etf-first"
                        : "stock-ready"
                    }`}
                  >
                    <span>当前判断</span>
                    <strong>
                      {holdingCalculation.managedStockPlan.blockers.length > 0
                        ? "ETF 优先"
                        : "可以使用受控个股替代"}
                    </strong>
                    <p>
                      {holdingCalculation.managedStockPlan.blockers.length > 0
                        ? holdingCalculation.managedStockPlan.blockers.join(
                            "；",
                          )
                        : stockSelectionMode === "high_capture"
                          ? "你已选择高捕获模式；系统仍执行 3% 差异风险和仓位上限。"
                          : "市场状态、股票分化和所选篮子相对强弱均已通过。"}
                    </p>
                    <small>
                      当前分化位于过去三年的{" "}
                      {percent(
                        holdingCalculation.managedStockPlan
                          .activeDispersionPercentile,
                        0,
                      )}{" "}
                      分位；默认上限为 67%
                    </small>
                  </div>
                </div>

                {individualStocks.length === 0 ? (
                  <div className="individual-stock-empty">
                    还没有录入个股。如果账户只有 QQQ、SMH、GLD、GDE、BIL、VIXY 和现金，可以直接继续。
                  </div>
                ) : (
                  <div className="individual-stock-list">
                    {individualStocks.map((stock, index) => {
                      const value =
                        Number(stock.shares || 0) * Number(stock.price || 0);
                      const normalizedTicker = stock.ticker.trim().toUpperCase();
                      const isSupported = (
                        stockRiskData.supportedStocks as readonly string[]
                      ).includes(normalizedTicker);
                      const plannedTarget =
                        holdingCalculation.managedStockPlan.targets.find(
                          (target) => target.id === stock.id,
                        )?.targetDollars ?? 0;
                      const stockEligibility =
                        holdingCalculation.managedStockPlan
                          .benchmarkEligibility[stock.benchmark];
                      return (
                        <div className="individual-stock-card" key={stock.id}>
                          <div className="individual-stock-row">
                            <span className="stock-row-number">{index + 1}</span>
                            <label>
                              <span>股票代码</span>
                              <input
                                type="text"
                                inputMode="text"
                                placeholder="例如 NVDA"
                                value={stock.ticker}
                                onChange={(event) =>
                                  updateIndividualStockTicker(
                                    stock.id,
                                    event.target.value,
                                  )
                                }
                              />
                            </label>
                            <label>
                              <span>股数</span>
                              <input
                                type="number"
                                min="0"
                                step="0.000001"
                                inputMode="decimal"
                                placeholder="0"
                                value={stock.shares}
                                onChange={(event) =>
                                  updateIndividualStock(stock.id, "shares", event.target.value)
                                }
                              />
                            </label>
                            <label>
                              <span>当前价格（美元）</span>
                              <input
                                type="number"
                                min="0"
                                step="0.01"
                                inputMode="decimal"
                                placeholder="0.00"
                                value={stock.price}
                                onChange={(event) =>
                                  updateIndividualStock(stock.id, "price", event.target.value)
                                }
                              />
                            </label>
                            <label className="stock-treatment">
                              <span>处理方式</span>
                              <select
                                value={stock.treatment}
                                onChange={(event) =>
                                  updateIndividualStock(
                                    stock.id,
                                    "treatment",
                                    event.target.value as StockTreatment,
                                  )
                                }
                              >
                                <option value="exclude">保留并排除在策略之外</option>
                                <option value="liquidate">逐步卖出并转入策略</option>
                                <option value="review">替代一部分 QQQ / SMH</option>
                              </select>
                            </label>
                            <div className="stock-value">
                              <span>估算价值</span>
                              <strong>{dollars.format(value)}</strong>
                            </div>
                            <button
                              className="remove-stock"
                              type="button"
                              onClick={() => removeIndividualStock(stock.id)}
                              aria-label={`删除第 ${index + 1} 只个股`}
                            >
                              删除
                            </button>
                          </div>

                          {stock.treatment === "review" && (
                            <div className="stock-risk-config">
                              <div className="stock-risk-intro">
                                <div>
                                  <strong>自动计算整个个股组合的差异风险</strong>
                                  <p>
                                    系统比较个股组合与被替代 ETF 的每日收益差，
                                    同时看最近 60 和 252 个交易日，并用较高的风险估计控制仓位。
                                  </p>
                                </div>
                                <span>
                                  风险数据截至 {stockRiskData.asOf}
                                </span>
                              </div>

                              <div className="stock-risk-fields">
                                <label>
                                  <span>替代哪个 ETF</span>
                                  <select
                                    value={stock.benchmark}
                                    onChange={(event) =>
                                      updateIndividualStock(
                                        stock.id,
                                        "benchmark",
                                        event.target.value as GrowthBenchmark,
                                      )
                                    }
                                  >
                                    <option value="QQQ">QQQ</option>
                                    <option value="SMH">SMH</option>
                                  </select>
                                </label>
                              </div>

                              <div
                                className={`stock-risk-result ${
                                  isSupported ? "ready" : "incomplete"
                                }`}
                              >
                                <div>
                                  <span>风险数据</span>
                                  <strong>
                                    {isSupported ? "已覆盖" : "暂未覆盖"}
                                  </strong>
                                </div>
                                <div>
                                  <span>完整目标市值</span>
                                  <strong>
                                    {dollars.format(plannedTarget)}
                                  </strong>
                                </div>
                                <p>
                                  {isSupported
                                    ? stockSelectionMode ===
                                        "evidence_gated" &&
                                      stockEligibility
                                        .effectiveReplacementShare === 0
                                      ? `当前完整目标为 0。${holdingCalculation.managedStockPlan.blockers.join("；")}。`
                                      : !strategySnapshot.stockIncreaseAllowedNow &&
                                      plannedTarget > value
                                      ? `目标由整个篮子的联合风险决定；本次不新增，等到 ${strategySnapshot.nextStockIncreaseReview} 再审查。`
                                      : "目标由整个篮子的联合风险决定，不再逐只股票按自身总波动单独扣减。"
                                    : "当前没有经过审计的历史数据；请改为“保留并排除”或“逐步卖出”。"}
                                </p>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}

                {holdingCalculation.reviewStocks.length > 0 && (
                  <div className="risk-budget-summary">
                    <div className="risk-budget-heading">
                      <div>
                        <strong>个股替代额度</strong>
                        <p>
                          先判断当前应该使用个股还是 ETF；通过后个股与 ETF
                          等金额替换，再应用 3% 差异风险、单只 5% 和合计 20%
                          三道上限。
                        </p>
                      </div>
                      <span>
                        组合预计差异波动{" "}
                        {percent(
                          holdingCalculation.managedStockPlan.combinedTrackingError,
                          2,
                        )}
                      </span>
                    </div>
                    <div className="risk-budget-grid">
                      {(["QQQ", "SMH"] as const).map((benchmark) => {
                        const budget =
                          holdingCalculation.managedStockPlan.budgets[benchmark];
                        return (
                          <article key={benchmark}>
                            <div>
                              <strong>{benchmark}</strong>
                              <span>
                                ETF 原目标 {dollars.format(budget.etfTargetDollars)}
                              </span>
                            </div>
                            <dl>
                              <div>
                                <dt>最多允许替代</dt>
                                <dd>{dollars.format(budget.maximumReplacementDollars)}</dd>
                              </div>
                              <div>
                                <dt>个股完整目标</dt>
                                <dd>{dollars.format(budget.stockTargetDollars)}</dd>
                              </div>
                              <div>
                                <dt>ETF 还应持有</dt>
                                <dd>{dollars.format(budget.remainingEtfDollars)}</dd>
                              </div>
                            </dl>
                            <p>
                              该篮子每投入 100 美元的历史年化差异波动约为{" "}
                              {percent(
                                Math.max(
                                  budget.trackingError60,
                                  budget.trackingError252,
                                ),
                                1,
                              )}
                              。
                            </p>
                          </article>
                        );
                      })}
                    </div>
                  </div>
                )}

                <div className="stock-treatment-guide">
                  <div>
                    <strong>保留并排除</strong>
                    <span>个股不交易；策略只管理剩余资金。适合长期持有或暂不想卖的股票。</span>
                  </div>
                  <div>
                    <strong>逐步卖出</strong>
                    <span>系统会把这只个股逐步降到零，但每次调整仍受 10% 上限约束。</span>
                  </div>
                  <div>
                    <strong>替代 QQQ / SMH</strong>
                    <span>系统计算组合目标并等额减少对应 ETF；剩余额度继续持有 ETF。</span>
                  </div>
                </div>
              </div>

              <label className="account-notes">
                <span>账户限制或备注</span>
                <textarea
                  rows={3}
                  placeholder="例如：只能交易整数股、GLD 不卖、需要保留 10,000 美元现金"
                  value={accountNotes}
                  onChange={(event) => setAccountNotes(event.target.value)}
                />
                <small>这些限制不会自动改变建议，但正式执行前必须应用。</small>
              </label>

              {holdingError && (
                <div className="holding-error" role="alert">
                  <StatusDot tone="blocked" />
                  {holdingError}
                </div>
              )}

              <div className="entry-actions">
                <button className="primary-button" type="submit">
                  检查并生成下一次调整建议
                </button>
                <button className="secondary-button" type="button" onClick={copyHoldingsTemplate}>
                  {templateCopyState}
                </button>
                <button
                  className="text-button"
                  type="button"
                  onClick={clearSavedHoldings}
                >
                  {hasSavedHoldings ? "删除已保存持仓" : "清空表单"}
                </button>
              </div>

              {templateCopyState === "请手动复制下方模板" && (
                <pre className="copy-fallback">{holdingsTemplate}</pre>
              )}
            </form>

            {draftReady && (
              <div className="holding-draft" id="holding-draft">
                <div className="draft-heading">
                  <div>
                    <span className="draft-label">下一次调整建议</span>
                    <h3>录入数据已通过完整性检查</h3>
                  </div>
                  <span className="draft-scope">所有仓位比例均以策略实际管理资金为分母</span>
                </div>

                <div className="draft-capital-summary">
                  <div>
                    <span>账户估算总价值</span>
                    <strong>{dollars.format(holdingCalculation.estimatedTotal)}</strong>
                    <small>与输入账户总价值相差 {percent(holdingCalculation.mismatch, 2)}</small>
                  </div>
                  <div>
                    <span>保留并排除的个股</span>
                    <strong>{dollars.format(holdingCalculation.excludedStockValue)}</strong>
                    <small>{holdingCalculation.stockValues.filter((stock) => stock.treatment === "exclude").length} 只</small>
                  </div>
                  <div>
                    <span>策略实际管理资金</span>
                    <strong>{dollars.format(holdingCalculation.managedCapital)}</strong>
                    <small>账户价值减去排除的个股</small>
                  </div>
                  <div>
                    <span>本次需要调动的资金</span>
                    <strong>{percent(holdingCalculation.appliedOneWayTurnover, 2)}</strong>
                    <small>
                      {holdingCalculation.uncappedOneWayTurnover <
                      strategySnapshot.noTradeThreshold
                        ? "总差异低于 1%，本次无需调整"
                        : holdingCalculation.uncappedOneWayTurnover * 0.25 >
                            strategySnapshot.trancheTurnoverLimit
                          ? `完整差距为 ${percent(holdingCalculation.uncappedOneWayTurnover, 2)}，本次受 10% 上限约束`
                          : `完整差距为 ${percent(holdingCalculation.uncappedOneWayTurnover, 2)}，本次前进四分之一`}
                    </small>
                  </div>
                </div>

                {accountProtection.calculation && (
                  <div className="draft-warning">
                    <strong>R40 长期熊市保护已进入账户执行链</strong>
                    <p>
                      账户高水位 {dollars.format(accountProtection.highWater)}；
                      当前策略资金 {dollars.format(accountProtection.managedEquity)}，
                      回撤 {percent(accountProtection.calculation.priorDrawdown, 2)}。
                      保护底线 = 高水位 × 81% = {dollars.format(accountProtection.calculation.floorEquity)}；
                      可承受安全垫 = 当前资金 − 保护底线 = {dollars.format(accountProtection.calculation.cushion)}。
                    </p>
                    <p>
                      请求的非现金上限 = {accountProtection.calculation.multiplier.toFixed(1)} ×
                      安全垫 ÷ 当前资金 = {percent(accountProtection.calculation.requestedNonCashCap, 2)}；
                      再向下取整到 5% 档，实际非现金上限为 {percent(accountProtection.calculation.acceptedNonCashCap, 0)}。
                      {accountProtection.calculation.acceptedNonCashCap < 1
                        ? " 当前已关闭 R38 增量风险，以 R11 为结构底稿后同比缩放全部非现金资产。"
                        : " 当前安全垫充足，继续使用同日合格的 R38 目标。"}
                    </p>
                    <p>
                      {accountProtection.initialized
                        ? `高水位状态已保存至 ${referencePrices.priceDate}；以后只能上调，不能因回撤下调。`
                        : "这是首次启用：以今天确认的策略资金初始化高水位，不倒填启用前损益。"}
                      历史回测的 20% 是收盘到收盘结果，不是对未来跳空或成交滑点的保证。
                    </p>
                  </div>
                )}

                <div className="draft-table" role="table" aria-label="下一次调整建议">
                  <div className="draft-table-head" role="row">
                    <span role="columnheader">资产</span>
                    <span role="columnheader">当前仓位（策略资金）</span>
                    <span role="columnheader">本次调整后（策略资金）</span>
                    <span role="columnheader">金额差额</span>
                    <span role="columnheader">近似股数差额</span>
                  </div>
                  {holdingCalculation.rows.map((row) => (
                    <div className="draft-table-row" role="row" key={row.rowKey}>
                      <strong role="cell">{row.asset}</strong>
                      <span role="cell">
                        {percent(row.currentWeight, 2)}
                        <small>{dollars.format(row.currentValue)}</small>
                      </span>
                      <span role="cell">{percent(row.targetWeight, 2)}</span>
                      <b
                        className={row.difference > 50 ? "buy" : row.difference < -50 ? "sell" : ""}
                        role="cell"
                      >
                        {row.difference >= 0 ? "+" : "−"}
                        {dollars.format(Math.abs(row.difference))}
                      </b>
                      <span role="cell">
                        {row.combinedCashSleeve
                          ? "由执行系统拆分"
                          : `${row.approximateShares >= 0 ? "+" : "−"}${Math.abs(row.approximateShares).toFixed(3)} 股`}
                      </span>
                    </div>
                  ))}
                </div>

                <div className="draft-warning">
                  <strong>这仍然不是可提交订单。</strong>
                  <p>
                    QQQ、SMH、GLD、BIL、VIXY 的参考价来自 {referencePrices.priceDate} 收盘；
                    GDE 参考价也截至 {referencePrices.gdePriceDate}，两者已同日核对；
                    最近 5 个交易日连续。当前草稿仍不能作为订单，因为必须先确认真实持仓、
                    账户实际 GDE 比例和开盘前实时报价；已审计个股会自动带入最近收盘价，
                    但仍需对照券商核对；BIL 与现金仍需由正式执行系统统一处理。
                    正式提交前还必须应用整股、税务和账户限制。
                  </p>
                </div>

                <div className="draft-warning">
                  <strong>下一笔什么时候：不是固定等待 63 个交易日。</strong>
                  <p>
                    本笔成交后先更新真实持仓，等待{" "}
                    {accountBuildRules.requiredNewCompletedCloses} 个新的完整收盘；
                    {strategySnapshot.accountBuildTiming.afterCurrentTrancheReviewEt}
                    （{strategySnapshot.accountBuildTiming.afterCurrentTrancheReviewJst}）重新计算。
                    若数据完整、风险门控允许且剩余差距仍不少于{" "}
                    {percent(accountBuildRules.remainingGapThreshold, 0)}，再下一笔最早在{" "}
                    {strategySnapshot.accountBuildTiming.followingTrancheEarliestExecutionEt}
                    （{strategySnapshot.accountBuildTiming.followingTrancheEarliestExecutionJst}）执行。
                    63 个交易日仅用于审查 R38 生产占比是否由 25% 提高到 50%，
                    不用于拖延账户分批建仓。
                  </p>
                </div>

                <div className="draft-actions">
                  <button className="primary-button" type="button" onClick={copyValidatedHoldings}>
                    {validatedCopyState}
                  </button>
                  <span>复制后可直接发给我，用正式执行流程生成可复核订单。</span>
                </div>
              </div>
            )}
          </section>
        )}

        <section className="section-block health-section">
          <div className="section-heading">
            <div>
              <p className="overline">运行情况</p>
              <h2>五项状态，一眼判断系统能否行动</h2>
            </div>
            <span className="summary-state">
              <StatusDot tone="blocked" />
              {draftReady
                ? "1 项仍需正式复核"
                : hasSavedHoldings
                  ? "先核对已保存持仓"
                  : "2 项待处理"}
            </span>
          </div>
          <div className="readiness-grid">
            {readinessChecks.map((sourceItem) => {
              const item =
                draftReady && sourceItem.label === "账户持仓"
                  ? {
                      ...sourceItem,
                      value: "已录入并校验",
                      detail: "已保存在此私有 Panel；仍应在交易前核对",
                      tone: "good",
                    }
                  : hasSavedHoldings && sourceItem.label === "账户持仓"
                    ? {
                        ...sourceItem,
                        value: "已恢复，待核对",
                        detail: "请确认是否仍与券商中的真实持仓一致",
                      }
                    : draftReady && sourceItem.label === "订单"
                      ? {
                          ...sourceItem,
                          value: "调整建议已生成",
                          detail: "仍需最新价格和正式执行系统复核",
                        }
                      : sourceItem;
              return (
                <article className={`readiness-card ${item.tone}`} key={item.label}>
                  <div>
                    <StatusDot tone={item.tone} />
                    <span>{item.label}</span>
                  </div>
                  <strong>{item.value}</strong>
                  <p>{item.detail}</p>
                </article>
              );
            })}
          </div>
        </section>

        <section className="section-block" id="allocation">
          <div className="section-heading">
            <div>
              <p className="overline">模型建议组合</p>
              <h2>{strategySnapshot.activeStrategy} 当前如何配置它管理的资金</h2>
              <p>
                下面是希望最终达到的组合，不是要求你现在一次买到这个比例。
                选择“替代 QQQ / SMH”的个股会等额减少相应 ETF，不会改变黄金和现金目标；
                保留并排除的个股不包含在这些比例中。
              </p>
            </div>
            <span className="plain-badge">数据截至 {strategySnapshot.priceAsOf}</span>
          </div>

          <div className="target-allocation-layout">
            <div className="target-allocation-list">
              {allocationRows.map((item) => (
                <article className="target-allocation-row" key={item.asset}>
                  <div>
                    <strong>{item.asset}</strong>
                    <span>{item.name}</span>
                  </div>
                  <TargetAllocationBar {...item} />
                  <b>{percent(item.finalTarget, 2)}</b>
                  <small>{item.role}</small>
                </article>
              ))}
            </div>

            <aside className="target-summary">
              <p className="overline">组合结构</p>
              <div>
                <span>成长资产</span>
                <strong>
                  {percent(
                    allocationRows
                      .filter((item) => item.asset === "QQQ" || item.asset === "SMH")
                      .reduce((total, item) => total + item.finalTarget, 0),
                    2,
                  )}
                </strong>
                <small>QQQ 与 SMH</small>
              </div>
              <div>
                <span>黄金袖套</span>
                <strong>
                  {percent(
                    allocationRows
                      .filter((item) => item.asset === "GLD" || item.asset === "GDE")
                      .reduce((total, item) => total + item.finalTarget, 0),
                    2,
                  )}
                </strong>
                <small>GLD + GDE，总预算不变</small>
              </div>
              <div>
                <span>短期国债与现金</span>
                <strong>
                  {percent(
                    allocationRows.find((item) => item.asset === "BIL / 现金")?.finalTarget ?? 0,
                    2,
                  )}
                </strong>
                <small>资金缓冲</small>
              </div>
              <p>
                当前 VIXY 目标为{" "}
                {percent(
                  allocationRows.find((item) => item.asset === "VIXY")?.finalTarget ?? 0,
                  2,
                )}
                ；该保护仓只由原生产核心按期限结构条件启用。
              </p>
            </aside>
          </div>

          <section className="target-proof" id="weight-proof">
            <div className="target-proof-heading">
              <div>
                <p className="overline">白匣子 · 权重推导</p>
                <h3>每一个百分点都可以倒推回规则</h3>
              </div>
              <span>R11 75% ＋ R38 25%</span>
            </div>

            <div className="target-proof-cards">
              <article>
                <span>01 · 总风险</span>
                <strong>
                  {targetDerivation.risk.baseMultiplier.toFixed(3)} × {targetDerivation.risk.stateMultiplier.toFixed(3)} = {targetDerivation.risk.requestedMultiplier.toFixed(3)}
                </strong>
                <p>
                  {targetDerivation.risk.shockActive
                    ? `单日冲击保护生效，因此本次完整 R38 目标改用 ${targetDerivation.risk.acceptedMultiplier.toFixed(3)}，不是 ${targetDerivation.risk.requestedMultiplier.toFixed(3)}。`
                    : `没有单日冲击保护，完整 R38 目标采用 ${targetDerivation.risk.acceptedMultiplier.toFixed(3)}。`}
                </p>
              </article>
              <article>
                <span>02 · QQQ / SMH 内部比例</span>
                <strong>
                  90% × {percent(targetDerivation.semiconductor.baseShare, 2)} + 10% × {percent(targetDerivation.semiconductor.fullSignalShare, 2)} = {percent(targetDerivation.semiconductor.implementedShare, 2)}
                </strong>
                <p>成长仓总额不变；这一步只把 QQQ 与 SMH 之间的比例向完整信号移动十分之一。</p>
              </article>
              <article>
                <span>03 · GDE / GLD</span>
                <strong>
                  10% × {percent(targetDerivation.gde.goldSleeve, 2)} × min({percent(targetDerivation.gde.growthWeight, 2)} / 80%, 1) = {percent(targetDerivation.gde.fullCenter, 2)}
                </strong>
                <p>GDE 从 GLD 等额替换，不增加黄金袖套；账户层在 {percent(targetDerivation.gde.stagedLower, 2)}–{percent(targetDerivation.gde.stagedUpper, 2)} 内不强制交易。</p>
              </article>
              <article>
                <span>04 · 分阶段发布</span>
                <strong>
                  75% × R11 + 25% × 完整 R38
                </strong>
                <p>当前显示的是这两个合格层的加权组合；25% 是策略发布比例，不是本次交易额。</p>
              </article>
            </div>

            <div className="target-proof-table" role="table" aria-label="当前目标权重推导表">
              <div className="target-proof-table-head" role="row">
                <span role="columnheader">资产</span>
                <span role="columnheader">R11 基准</span>
                <span role="columnheader">完整 R38</span>
                <span role="columnheader">当前目标算式</span>
                <span role="columnheader">为什么</span>
              </div>
              {targetDerivation.rows.map((row) => (
                <div className="target-proof-table-row" role="row" key={row.asset}>
                  <strong role="cell">{row.asset}</strong>
                  <span role="cell">{percent(row.r11, 2)}</span>
                  <span role="cell">{percent(row.r38, 2)}</span>
                  <b role="cell">
                    75% × {percent(row.r11, 2)} + 25% × {percent(row.r38, 2)} = {percent(row.staged, 2)}
                  </b>
                  <small role="cell">{row.reason}</small>
                </div>
              ))}
            </div>
          </section>

          <div className="execution-protection">
            <div className="execution-protection-heading">
              <div>
                <p className="overline">如何从你的持仓走向建议组合</p>
                <h3>{draftReady ? "你的下一次调整建议已在上方生成" : "先录入真实持仓，再计算下一次调整"}</h3>
              </div>
              {!draftReady && (
                <button className="primary-button" type="button" onClick={openHoldingsEntry}>
                  录入真实持仓
                </button>
              )}
            </div>
            <div className="protection-rules">
              <div>
                <span>01</span>
                <strong>每次只前进一部分</strong>
                <p>
                  普通调整一次最多完成差距的四分之一；若只是在 GLD/GDE
                  内移动且总规模低于 10%，直接到最近允许边界。
                </p>
              </div>
              <div>
                <span>02</span>
                <strong>限制调整规模</strong>
                <p>如果需要买卖的资金过多，本次最多调整策略管理资金的 10%。</p>
              </div>
              <div>
                <span>03</span>
                <strong>过滤无意义的小调整</strong>
                <p>如果需要调整的资金不足策略管理资金的 1%，本次不交易。</p>
              </div>
              <div>
                <span>04</span>
                <strong>每次成交后重新计算</strong>
                <p>不会预先生成后续连续订单，避免使用已经过期的价格和持仓。</p>
              </div>
            </div>

            <details className="audit-details">
              <summary>查看策略版本切换的内部审计信息（通常不需要）</summary>
              <div>
                <p>
                  底层 R38 当前先替换 25% 的 R11，并继续负责趋势、波动加速度、
                  现金下限与 126 日慢速相对强弱锚点。{" "}
                  {strategySnapshot.activeStrategy === "R39"
                    ? `R39 账户保护层已通过资格检查并进入当前目标。`
                    : `${strategySnapshot.fallbackReason} 当前目标没有应用 R39 额外保护层。`}
                  实际下一步仍必须根据真实持仓与最新价格重新计算。
                </p>
                <span>
                  历史预估总调整规模 {percent(strategySnapshot.totalOneWayTurnover, 2)}；
                  在每单位调仓金额产生 0.15% 成本的保守假设下，预估成本约占账户价值
                  {percent(strategySnapshot.estimatedMigrationCostRate, 3)}。
                </span>
              </div>
            </details>
          </div>
        </section>

        <section className="section-block" id="drivers">
          <div className="section-heading">
            <div>
              <p className="overline">为什么是这个目标</p>
              <h2>当前目标由这些可解释的判断共同形成</h2>
            </div>
            <span className="plain-badge">不是黑箱结论</span>
          </div>
          <div className="driver-list">
            {decisionDrivers.map((driver, index) => (
              <article className={`driver-row ${driver.tone}`} key={driver.title}>
                <span className="driver-index">0{index + 1}</span>
                <div className="driver-copy">
                  <h3>{driver.title}</h3>
                  <p>{driver.explanation}</p>
                </div>
                <strong>{driver.value}</strong>
                <span className="driver-status">{driver.status}</span>
              </article>
            ))}
          </div>
        </section>

        <section className="section-block" id="parameters">
          <div className="section-heading">
            <div>
              <p className="overline">相关参数的最新情况</p>
              <h2>先看会影响下一步行动的关键参数</h2>
            </div>
            <span className="plain-badge">版本 {strategySnapshot.release}</span>
          </div>

          <div className="key-parameter-grid">
            {keyParameters.map((parameter) => (
              <article className="key-parameter" key={parameter.label}>
                <div>
                  <span>{parameter.label}</span>
                  <i className={parameter.tone}>
                    {parameter.tone === "active" ? "正在生效" : "当前待命"}
                  </i>
                </div>
                <strong>{parameter.value}</strong>
                <p>{parameter.currentUse}</p>
                <small>{parameter.why}</small>
              </article>
            ))}
          </div>

          <div className="all-parameters">
            <div className="all-parameters-heading">
              <div>
                <h3>完整生产参数</h3>
                <p>只有需要审计或理解模型时才展开；日常决策不需要逐项阅读。</p>
              </div>
              <span>
                当前实际运行 {strategySnapshot.activeStrategy}；未通过资格的更高层不会进入目标
              </span>
            </div>
            {parameterGroups.map((group) => {
              const isOpen = expandedParameter === group.name;
              return (
                <article className={`parameter-group ${isOpen ? "open" : ""}`} key={group.name}>
                  <button
                    onClick={() => setExpandedParameter(isOpen ? "" : group.name)}
                    aria-expanded={isOpen}
                  >
                    <span className="expand-symbol">{isOpen ? "−" : "+"}</span>
                    <strong>{group.name}</strong>
                    <small>{group.summary}</small>
                  </button>
                  {isOpen && (
                    <div className="parameter-table">
                      {group.parameters.map(([label, value, meaning]) => (
                        <div key={label}>
                          <span>{label}</span>
                          <strong>{value}</strong>
                          <p>{meaning}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        </section>

        <section className="section-block evidence-section">
          <div className="section-heading">
            <div>
              <p className="overline">当前生产层级为什么可以使用</p>
              <h2>资格、前瞻冻结与数据日期均已独立复核</h2>
            </div>
          </div>
          <div className="evidence-table">
            <div className="evidence-head">
              <span>检查项目</span>
              <span>当前结果</span>
              <span>请求层级 / 比较基准</span>
              <span>如何理解</span>
            </div>
            {productionEvidence.map((item) => (
              <div className="evidence-row" key={item.label}>
                <strong>{item.label}</strong>
                <b>{item.candidate}</b>
                <span>{item.previous}</span>
                <p>{item.note}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="risk-layout" id="risk">
          <article className="risk-card">
            <p className="overline">需要持续记住的风险</p>
            <h2>更高的上涨参与，也意味着放弃一部分防守</h2>
            <ul>
              {knownRisks.map((risk) => <li key={risk}>{risk}</li>)}
            </ul>
          </article>
          <article className="definitions-card">
            <p className="overline">页面用语</p>
            <h2>所有术语都可以直接解释</h2>
            <dl>
              <div>
                <dt>受控生产发布</dt>
                <dd>
                  底层仍是 25% R38、75% R11。只有 R39 资格通过时才允许其在现有
                  QQQ / SMH 成长仓内部重配；未通过时自动保持 R38。至少积累 63 个
                  冻结后的真实交易日记录，才审查底层 R38 的发布比例。
                </dd>
              </div>
              <div>
                <dt>账户级相对损失预算</dt>
                <dd>
                  这是 R39 候选层的账户保护预算。它只有在研究与生产资格同时通过时
                  才能进入实际目标；当前未通过时，页面明确回退到同日合格 R38。
                </dd>
              </div>
              <div>
                <dt>GDE</dt>
                <dd>
                  一只同时持有美国大盘股和黄金期货的基金。它不是成长仓里的黄金；
                  R38 从黄金袖套中等额划出，当前 GLD 与 GDE 合计目标为{" "}
                  {percent(strategySnapshot.goldSleeveTarget, 2)}。
                </dd>
              </div>
              <div>
                <dt>GDE 不交易区间</dt>
                <dd>
                  当前账户的 GDE 只要在 {percent(strategySnapshot.stagedGdeLower, 2)}–
                  {percent(strategySnapshot.stagedGdeUpper, 2)} 之间就不交易；超出时也只调到最近
                  边界，不需要为精确接近{" "}
                  {percent(strategySnapshot.stagedGdeCenter, 2)} 中心值支付较宽买卖价差。
                </dd>
              </div>
              <div>
                <dt>全研究库多重比较</dt>
                <dd>
                  把项目中 900 条独特且符合事前纳入规则的收益路径放在一起比较，
                  检查领先结果是否可能只是反复尝试造成。R38 在 21、63、126 日三种
                  连续区块假设下的校正后概率都低于 5%。
                </dd>
              </div>
              <div>
                <dt>短期波动加速</dt>
                <dd>
                  任一 QQQ 或 SMH 的前一收盘 21 日波动高于其 63 日波动。发生时只
                  撤掉本次新增的风险容量，回到原 R38；不会因此把成长仓降到零。
                </dd>
              </div>
              <div>
                <dt>需要调动的资金占比</dt>
                <dd>为接近建议组合，需要买卖的资金占策略管理资金的比例。买入和卖出不重复计算。</dd>
              </div>
              <div>
                <dt>风险偏好环境</dt>
                <dd>模型认为成长资产的预期回报相对风险仍值得持有；不等于市场不会下跌。</dd>
              </div>
              <div>
                <dt>VIX ÷ VIX3M</dt>
                <dd>短期市场预期波动与未来三个月预期波动之比。大于 1 通常表示近期压力更急迫。</dd>
              </div>
              <div>
                <dt>长期替代数据</dt>
                <dd>某些基金成立前，用性质相近的、更早历史数据代替，仅用于压力检查。</dd>
              </div>
              <div>
                <dt>差异风险</dt>
                <dd>
                  个股篮子收益减去被替代 QQQ / SMH 收益后的波动。它衡量选股给
                  策略新增了多少风险，不会把原本就要承担的市场风险重复扣除。
                </dd>
              </div>
              <div>
                <dt>股票分化</dt>
                <dd>
                  同一时期不同股票相对 QQQ / SMH 表现的差距。分化很高意味着选对和
                  选错的结果差得更远，也更容易发生排名反转，不等于更容易赚钱。
                </dd>
              </div>
              <div>
                <dt>相对强弱确认</dt>
                <dd>
                  用户所选股票篮子在过去 63 日和 126 日都跑赢对应 ETF。它只确认已经
                  发生的表现，不保证之后继续跑赢。
                </dd>
              </div>
            </dl>
          </article>
        </section>

        <footer>
          <div>
            <strong>{strategySnapshot.activeStrategy} 策略运行驾驶舱</strong>
            <span>数据截至 {strategySnapshot.priceAsOf}</span>
          </div>
          <p>
            这是生产策略与执行准备状态的快照。没有真实持仓和最新完整收盘数据时，任何参考仓位都不是可提交订单。
          </p>
          <a href="#top">返回顶部 ↑</a>
        </footer>
      </div>
    </main>
  );
}
