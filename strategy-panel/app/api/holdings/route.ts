import { eq } from "drizzle-orm";
import { getDb } from "../../../db";
import { savedHoldings } from "../../../db/schema";
import { getChatGPTUser } from "../../chatgpt-auth";

const holdingKeys = [
  "accountValue",
  "cash",
  "QQQ",
  "SMH",
  "GLD",
  "GDE",
  "BIL",
  "VIXY",
] as const;

const treatments = new Set(["exclude", "liquidate", "review"]);
const benchmarks = new Set(["QQQ", "SMH"]);
const optionalStockFields = [
  "historyDays",
  "beta252",
  "volatilityRatio60",
  "volatilityRatio252",
  "previousRiskMultiple",
] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isShortString(value: unknown, maximumLength = 64): value is string {
  return typeof value === "string" && value.length <= maximumLength;
}

function isValidPayload(value: unknown) {
  if (!isRecord(value)) return false;
  const holdings = value.holdings;
  if (!isRecord(holdings)) return false;
  if (!holdingKeys.every((key) => isShortString(holdings[key]))) return false;
  if (!Array.isArray(value.individualStocks) || value.individualStocks.length > 100) {
    return false;
  }
  if (!isShortString(value.accountNotes, 5000)) return false;
  if (
    "protectionState" in value &&
    value.protectionState !== null &&
    (!isRecord(value.protectionState) ||
      !isShortString(value.protectionState.highWater) ||
      !isShortString(value.protectionState.asOf) ||
      !isShortString(value.protectionState.activatedAt))
  ) {
    return false;
  }

  return value.individualStocks.every(
    (stock) =>
      isRecord(stock) &&
      isShortString(stock.id) &&
      isShortString(stock.ticker) &&
      isShortString(stock.shares) &&
      isShortString(stock.price) &&
      typeof stock.treatment === "string" &&
      treatments.has(stock.treatment) &&
      (!("benchmark" in stock) ||
        (typeof stock.benchmark === "string" &&
          benchmarks.has(stock.benchmark))) &&
      optionalStockFields.every(
        (field) => !(field in stock) || isShortString(stock[field]),
      ),
  );
}

async function authenticatedEmail() {
  const user = await getChatGPTUser();
  return user?.email.toLowerCase() ?? null;
}

export async function GET() {
  const userEmail = await authenticatedEmail();
  if (!userEmail) {
    return Response.json({ error: "需要登录后才能读取持仓。" }, { status: 401 });
  }

  const db = getDb();
  const [row] = await db
    .select()
    .from(savedHoldings)
    .where(eq(savedHoldings.userEmail, userEmail))
    .limit(1);

  if (!row) return Response.json({ holdings: null, updatedAt: null });

  try {
    return Response.json({
      holdings: JSON.parse(row.payload),
      updatedAt: row.updatedAt,
    });
  } catch {
    return Response.json(
      { error: "已保存的持仓数据无法读取，请删除后重新录入。" },
      { status: 500 },
    );
  }
}

export async function PUT(request: Request) {
  const userEmail = await authenticatedEmail();
  if (!userEmail) {
    return Response.json({ error: "需要登录后才能保存持仓。" }, { status: 401 });
  }

  const payload: unknown = await request.json();
  if (!isValidPayload(payload)) {
    return Response.json({ error: "持仓数据格式不正确。" }, { status: 400 });
  }

  const serialized = JSON.stringify(payload);
  if (serialized.length > 100_000) {
    return Response.json({ error: "持仓数据过大。" }, { status: 413 });
  }

  const updatedAt = new Date().toISOString();
  const db = getDb();
  await db
    .insert(savedHoldings)
    .values({ userEmail, payload: serialized, updatedAt })
    .onConflictDoUpdate({
      target: savedHoldings.userEmail,
      set: { payload: serialized, updatedAt },
    });

  return Response.json({ saved: true, updatedAt });
}

export async function DELETE() {
  const userEmail = await authenticatedEmail();
  if (!userEmail) {
    return Response.json({ error: "需要登录后才能删除持仓。" }, { status: 401 });
  }

  const db = getDb();
  await db.delete(savedHoldings).where(eq(savedHoldings.userEmail, userEmail));
  return Response.json({ deleted: true });
}
