import type { Metadata } from "next";
import { headers } from "next/headers";
import { ControlPanel } from "./control-panel";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host =
    requestHeaders.get("x-forwarded-host") ??
    requestHeaders.get("host") ??
    "localhost";
  const protocol =
    requestHeaders.get("x-forwarded-proto") ??
    (host.startsWith("localhost") ? "http" : "https");
  const previewImage = `${protocol}://${host}/og-r39.png`;

  return {
    title: "策略运行驾驶舱",
    description:
      "一页查看当前合格策略是否正常、为什么持有当前目标，以及下一步应该做什么。",
    openGraph: {
      title: "策略运行驾驶舱",
      description: "策略运行、执行准备、目标仓位与下一步行动。",
      images: [{ url: previewImage, width: 1200, height: 630 }],
    },
    twitter: {
      card: "summary_large_image",
      title: "策略运行驾驶舱",
      description: "策略运行、执行准备、目标仓位与下一步行动。",
      images: [previewImage],
    },
  };
}

export default function Home() {
  return <ControlPanel />;
}
