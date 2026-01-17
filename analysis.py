#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis.py
基于清洗后的数据做分析（单独文件版）/ Analysis on cleaned data (standalone)

功能 / What it does:
- 自动识别/构造 Total（销售额）列 / Auto-resolve or compute Total (sales)
- promo-proxy（品类内低价分位数当疑似促销）/ Promo-proxy using within-category low-price quantile
- KPI 对比：promo vs non-promo / KPI comparison
- 按品类分层 / Breakdown by product line
- 价格弹性：log-log 回归 / Price elasticity via log-log regression
- 输出 CSV + 图 + 文本总结 / Export CSVs + plots + summary.txt

Usage / 用法:
  python analysis.py --in clean.csv --outdir outputs --promo_q 0.25
"""

import argparse
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# 0) Small utilities / 小工具
# ----------------------------

def _safe_mkdir(path: str) -> None:
    """确保输出目录存在 / Ensure output directory exists."""
    os.makedirs(path, exist_ok=True)


def _require_columns(df: pd.DataFrame, cols: list) -> None:
    """检查必需列是否存在 / Validate required columns exist."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """安全转数值 / Safe numeric conversion."""
    return pd.to_numeric(df[col], errors="coerce")


# -------------------------------------------------
# 1) Column resolvers / 列名自动映射（更稳）
# -------------------------------------------------

def resolve_column(df: pd.DataFrame, target: str, candidates: list) -> pd.DataFrame:
    """
    把候选列映射成统一列名 target / Map one of candidate columns to target name.
    If target already exists, do nothing.
    """
    df = df.copy()
    if target in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            df[target] = df[c]
            return df
    return df


def resolve_total_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    自动解析销售额列 Total / Resolve Total (sales) automatically

    优先：从已有列映射 / Prefer mapping existing columns:
      Total, Sales, Amount, Revenue...
    否则：Total = Unit price * Quantity / Else compute from price*qty.
    """
    df = df.copy()

    # 先尝试把“总额列”映射为 Total / Try mapping existing sales columns to Total
    total_candidates = [
        "Total", "total", "TOTAL",
        "Sales", "sales", "SALES",
        "Amount", "amount",
        "Total amount", "Total Amount",
        "Revenue", "revenue"
    ]
    df = resolve_column(df, "Total", total_candidates)

    # 如果还没有 Total，就用 price*qty 计算 / If still missing, compute from price*qty
    if "Total" not in df.columns:
        if ("Unit price" in df.columns) and ("Quantity" in df.columns):
            df["Total"] = _to_numeric(df, "Unit price") * _to_numeric(df, "Quantity")
        # 如果连价格/数量都没有，就留给后面报错 / If even price/qty missing, let later checks fail

    # 强制转数值 / Ensure numeric
    if "Total" in df.columns:
        df["Total"] = pd.to_numeric(df["Total"], errors="coerce")

    return df


# -------------------------------------------------
# 2) Core analytics / 核心分析逻辑
# -------------------------------------------------

def add_promo_proxy(df: pd.DataFrame, promo_q: float = 0.25) -> pd.DataFrame:
    """
    promo_flag = 1 if Unit price <= quantile(Unit price | Product line)
    用“品类内底部分位数价格”作为促销代理 / Promo proxy using within-category low price quantile
    """
    df = df.copy()
    _require_columns(df, ["Product line", "Unit price"])

    threshold = df.groupby("Product line")["Unit price"].transform(lambda s: s.quantile(promo_q))
    df["promo_flag"] = (df["Unit price"] <= threshold).astype(int)
    df["promo_q"] = float(promo_q)
    return df


def kpi_promo_vs_nonpromo(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 对比 / KPI comparison between promo_flag groups."""
    df = df.copy()
    _require_columns(df, ["promo_flag", "Unit price", "Quantity", "Total"])

    out = (
        df.groupby("promo_flag", as_index=False)
          .agg(
              rows=("promo_flag", "size"),
              qty_sum=("Quantity", "sum"),
              qty_mean=("Quantity", "mean"),
              price_mean=("Unit price", "mean"),
              total_sum=("Total", "sum"),
              total_mean=("Total", "mean"),
              total_median=("Total", "median"),
          )
    )
    out["share"] = out["rows"] / out["rows"].sum()
    out["aov_proxy"] = out["total_mean"]  # 这里把每行均值当作“代理客单价” / AOV proxy
    return out


def kpi_by_product_line(df: pd.DataFrame) -> pd.DataFrame:
    """按品类分层 / Breakdown by Product line and promo_flag."""
    df = df.copy()
    _require_columns(df, ["Product line", "promo_flag", "Unit price", "Quantity", "Total"])

    out = (
        df.groupby(["Product line", "promo_flag"], as_index=False)
          .agg(
              rows=("promo_flag", "size"),
              qty_sum=("Quantity", "sum"),
              qty_mean=("Quantity", "mean"),
              price_mean=("Unit price", "mean"),
              total_sum=("Total", "sum"),
              total_mean=("Total", "mean"),
          )
    )
    out["share_within_product_line"] = out["rows"] / out.groupby("Product line")["rows"].transform("sum")
    return out.sort_values(["Product line", "promo_flag"])


def elasticity_loglog(df: pd.DataFrame) -> Dict[str, float]:
    """
    价格弹性（相关性，不是因果）/ Price elasticity (correlation, not causal)
    log(Q) = a + b*log(P), b ~ elasticity
    """
    _require_columns(df, ["Unit price", "Quantity"])

    x = pd.to_numeric(df["Unit price"], errors="coerce")
    y = pd.to_numeric(df["Quantity"], errors="coerce")

    m = (x > 0) & (y > 0) & x.notna() & y.notna()
    x = x[m].to_numpy()
    y = y[m].to_numpy()

    if len(x) < 20:
        return {"ok": 0.0, "n": float(len(x)), "elasticity_b": np.nan, "r2": np.nan}

    lx = np.log(x)
    ly = np.log(y)

    X = np.column_stack([np.ones_like(lx), lx])
    beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
    a, b = beta[0], beta[1]

    yhat = X @ beta
    ss_res = float(np.sum((ly - yhat) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"ok": 1.0, "n": float(len(x)), "intercept_a": float(a), "elasticity_b": float(b), "r2": float(r2)}


# -------------------------------------------------
# 3) Plots / 画图
# -------------------------------------------------

def plot_price_vs_quantity(df: pd.DataFrame, outpath: str) -> None:
    """散点图：价格 vs 销量 / Scatter: price vs quantity."""
    plt.figure()
    plt.scatter(df["Unit price"], df["Quantity"], alpha=0.25)
    plt.xlabel("Unit price")
    plt.ylabel("Quantity")
    plt.title("Unit price vs Quantity")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_promo_share_by_product_line(by_pl: pd.DataFrame, outpath: str) -> None:
    """柱状图：各品类 promo-proxy 占比 / Bar: promo share within product line."""
    promo = by_pl[by_pl["promo_flag"] == 1].copy()
    promo = promo.sort_values("share_within_product_line", ascending=False)

    plt.figure()
    plt.bar(promo["Product line"].astype(str), promo["share_within_product_line"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Promo-proxy share (within product line)")
    plt.title("Promo-proxy share by Product line")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def write_summary_txt(path: str, kpi: pd.DataFrame, el: Dict[str, float], promo_q: float) -> None:
    """写一份中英摘要 / Write a bilingual summary."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Project Summary / 项目摘要 ===\n")
        f.write(f"Promo proxy quantile / 促销代理分位数: bottom {promo_q:.2f}\n\n")

        f.write("=== KPI: Promo vs Non-Promo (proxy) / 指标对比：低价组 vs 非低价组 ===\n")
        f.write(kpi.to_string(index=False))
        f.write("\n\n")

        f.write("=== Price Elasticity (log-log) / 价格弹性（对数回归）===\n")
        if el.get("ok", 0.0) == 1.0:
            b = el["elasticity_b"]
            f.write(f"Elasticity b: {b:.4f}\n")
            f.write(f"Interpretation: price +1% -> quantity {b*100:.2f}% (approx)\n")
            f.write(f"R^2: {el['r2']:.4f}, N: {int(el['n'])}\n")
            f.write("Note: correlational, not causal. / 注意：相关性，不代表因果。\n")
        else:
            f.write(f"Elasticity estimation failed (N={int(el.get('n', 0))}).\n")


# -------------------------------------------------
# 4) Main / 主函数
# -------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True, help="清洗后CSV路径 / Cleaned CSV path")
    parser.add_argument("--outdir", default="outputs", help="输出目录 / Output directory")
    parser.add_argument("--promo_q", type=float, default=0.25, help="促销代理分位数 / Promo quantile (bottom)")
    args = parser.parse_args()

    infile = args.infile
    outdir = args.outdir
    promo_q = args.promo_q

    if not os.path.exists(infile):
        print(f"[ERROR] Input file not found: {infile}")
        sys.exit(1)

    _safe_mkdir(outdir)

    # 读取数据 / Load
    df = pd.read_csv(infile)
    df.columns = [c.strip() for c in df.columns]

    # 统一关键列名（更兼容不同版本数据集）/ Normalize key column names for compatibility
    df = resolve_column(df, "Product line", ["Product line", "Product Line", "product line", "Category", "category"])
    df = resolve_column(df, "Unit price", ["Unit price", "Unit Price", "unit price", "Price", "price"])
    df = resolve_column(df, "Quantity", ["Quantity", "Qty", "quantity", "QTY"])

    # 先检查“品类/价格/数量”是否存在 / Validate product line, price, quantity
    _require_columns(df, ["Product line", "Unit price", "Quantity"])

    # 数值化 / Coerce numeric
    df["Unit price"] = pd.to_numeric(df["Unit price"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    # 自动映射/计算 Total / Resolve or compute Total
    df = resolve_total_column(df)
    _require_columns(df, ["Total"])  # 现在必须有 Total

    # 丢掉关键缺失 / Drop missing essentials
    df = df.dropna(subset=["Product line", "Unit price", "Quantity", "Total"])

    # 添加 promo proxy / Add promo proxy
    df = add_promo_proxy(df, promo_q=promo_q)

    # KPI / Metrics
    kpi = kpi_promo_vs_nonpromo(df)
    by_pl = kpi_by_product_line(df)

    # Elasticity / 弹性
    el = elasticity_loglog(df)

    # 输出 CSV / Export CSV
    df.to_csv(os.path.join(outdir, "analysis_dataset.csv"), index=False)
    kpi.to_csv(os.path.join(outdir, "kpi_promo_vs_nonpromo.csv"), index=False)
    by_pl.to_csv(os.path.join(outdir, "kpi_by_product_line.csv"), index=False)

    # 输出图 / Export plots
    plot_price_vs_quantity(df, os.path.join(outdir, "price_vs_quantity.png"))
    plot_promo_share_by_product_line(by_pl, os.path.join(outdir, "promo_share_by_product_line.png"))

    # 输出摘要 / Export summary
    write_summary_txt(os.path.join(outdir, "summary.txt"), kpi=kpi, el=el, promo_q=promo_q)

    # Console
    print("[DONE] Analysis finished.")
    print(f"Outputs saved to: {outdir}")
    print("Files: analysis_dataset.csv, kpi_promo_vs_nonpromo.csv, kpi_by_product_line.csv, summary.txt, plots")


if __name__ == "__main__":
    main()
