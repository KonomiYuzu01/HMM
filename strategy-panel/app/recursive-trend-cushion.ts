export type Allocation = {
  QQQ: number;
  SMH: number;
  GLD: number;
  GDE: number;
  cash: number;
  VIXY: number;
};

export type RecursiveTrendCushionParameters = {
  floorDrawdown: number;
  bullMultiplier: number;
  bearMultiplier: number;
  tierSize: number;
};

export function calculateRecursiveTrendCushion({
  priorEquity,
  priorPeak,
  dualTrendPositive,
  parameters,
}: {
  priorEquity: number;
  priorPeak: number;
  dualTrendPositive: boolean;
  parameters: RecursiveTrendCushionParameters;
}) {
  if (!(priorEquity > 0) || !(priorPeak > 0)) {
    throw new Error("账户净值和高水位必须大于 0。");
  }
  if (priorEquity > priorPeak + 1e-10) {
    throw new Error("账户高水位不能低于当前净值。");
  }
  const floorEquity = priorPeak * (1 + parameters.floorDrawdown);
  const cushion = Math.max(priorEquity - floorEquity, 0);
  const multiplier = dualTrendPositive
    ? parameters.bullMultiplier
    : parameters.bearMultiplier;
  const requestedNonCashCap = Math.min(
    Math.max((multiplier * cushion) / priorEquity, 0),
    1,
  );
  const acceptedNonCashCap =
    requestedNonCashCap >= 1 - 1e-12
      ? 1
      : Math.min(
          Math.max(
            Math.floor(
              (requestedNonCashCap + 1e-12) / parameters.tierSize,
            ) * parameters.tierSize,
            0,
          ),
          1,
        );
  return {
    priorDrawdown: priorEquity / priorPeak - 1,
    floorEquity,
    cushion,
    multiplier,
    requestedNonCashCap,
    acceptedNonCashCap,
    state:
      acceptedNonCashCap <= 1e-12
        ? "floor"
        : acceptedNonCashCap < 1
          ? "controlled"
          : "normal",
  } as const;
}

function capTotalNonCash(target: Allocation, cap: number): Allocation {
  const nonCash = target.QQQ + target.SMH + target.GLD + target.GDE + target.VIXY;
  if (nonCash <= cap + 1e-12) return { ...target };
  const scale = cap / nonCash;
  return {
    QQQ: target.QQQ * scale,
    SMH: target.SMH * scale,
    GLD: target.GLD * scale,
    GDE: target.GDE * scale,
    VIXY: target.VIXY * scale,
    cash: 1 - nonCash * scale,
  };
}

export function protectedAccountTarget({
  stagedTarget,
  r11Target,
  acceptedNonCashCap,
}: {
  stagedTarget: Allocation;
  r11Target: Allocation;
  acceptedNonCashCap: number;
}) {
  const base = acceptedNonCashCap < 1 - 1e-12 ? r11Target : stagedTarget;
  return capTotalNonCash(base, acceptedNonCashCap);
}
