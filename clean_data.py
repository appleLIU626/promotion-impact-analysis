#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 20:56:30 2026

@author: shihaoliu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-  # 指定文件编码（UTF-8） / Set file encoding (UTF-8)

"""
clean_data.py
更严格的数据清洗脚本（单独文件版）/ A stricter data cleaning script (single-file)

用法 / Usage:
  python clean_data.py --in raw.csv --out clean.csv --report cleaning_report.txt

如果文件名有空格 / If filename has spaces:
  python clean_data.py --in "supermarket_sales new.csv" --out clean.csv
"""

import argparse  # 命令行参数解析 / Parse command-line arguments
import os  # 文件路径与目录操作 / File path & directory utilities
import sys  # 系统退出与输出 / System utilities (exit, stdout)
from typing import Tuple, Dict  # 类型提示 / Type hints

import numpy as np  # 数值计算 / Numerical computing
import pandas as pd  # 数据处理 / Data processing


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """标准化列名：去空格、统一大小写风格 / Normalize column names: strip spaces, unify style."""
    df = df.copy()  # 复制数据避免修改原对象 / Copy to avoid mutating input
    df.columns = [c.strip() for c in df.columns]  # 去掉列名两侧空格 / Strip column name spaces
    return df  # 返回处理后的 DataFrame / Return normalized DataFrame


def coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """把指定列安全转换为数值，无法转换的变为 NaN / Convert columns to numeric safely (invalid -> NaN)."""
    df = df.copy()  # 复制数据 / Copy dataframe
    for c in cols:  # 遍历列名 / Iterate columns
        if c in df.columns:  # 若列存在 / If column exists
            df[c] = pd.to_numeric(df[c], errors="coerce")  # 强制转数值 / Coerce to numeric
    return df  # 返回 / Return


def trim_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """去掉文本列两侧空格，统一空字符串为 NaN / Trim text columns; empty strings -> NaN."""
    df = df.copy()  # 复制 / Copy
    for c in df.columns:  # 遍历所有列 / Loop over columns
        if df[c].dtype == "object":  # 只处理文本列 / Only object (string-like) columns
            df[c] = df[c].astype(str).str.strip()  # 转字符串并去空格 / Cast to str and strip
            df.loc[df[c].isin(["", "nan", "None", "NULL", "null"]), c] = np.nan  # 常见空值统一 / Normalize empties
    return df  # 返回 / Return


def parse_datetime_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    """
    如果存在 Date/Time 字段，尝试解析并生成 datetime 列 / If Date/Time exist, parse into a datetime column.
    支持常见超市数据格式：Date + Time / Support common supermarket format: Date + Time.
    """
    df = df.copy()  # 复制 / Copy

    # 常见列名 / Common column names
    has_date = "Date" in df.columns  # 是否有 Date 列 / Whether Date column exists
    has_time = "Time" in df.columns  # 是否有 Time 列 / Whether Time column exists

    if has_date and has_time:  # 若同时有 Date 和 Time / If both exist
        dt_str = df["Date"].astype(str) + " " + df["Time"].astype(str)  # 拼接成日期时间字符串 / Combine date+time
        df["Datetime"] = pd.to_datetime(dt_str, errors="coerce")  # 解析成 datetime / Parse to datetime
    elif has_date and not has_time:  # 只有 Date / Date only
        df["Datetime"] = pd.to_datetime(df["Date"], errors="coerce")  # 直接解析 / Parse date
    else:  # 没有日期字段 / No date fields
        pass  # 不做处理 / Do nothing

    # 如果解析出了 Datetime，就拆分出常用字段 / If Datetime parsed, derive useful fields
    if "Datetime" in df.columns:  # Datetime 列存在 / If Datetime exists
        df["Year"] = df["Datetime"].dt.year  # 年 / Year
        df["Month"] = df["Datetime"].dt.month  # 月 / Month
        df["Day"] = df["Datetime"].dt.day  # 日 / Day
        df["Hour"] = df["Datetime"].dt.hour  # 小时 / Hour

    return df  # 返回 / Return


def fix_tax_scaling(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    修复 Tax 5% 的缩放异常：尝试 *1, /10, /100, /1000 选择最贴近期望税额的方案
    Fix tax scaling issues by trying factors and picking the best match to expected tax.

    期望税额：如果 Total 含税且税率约 5%，则 expected_tax = Total / 21
    Expected tax: for ~5% tax with Total = 1.05*subtotal => tax = Total/21
    """
    df = df.copy()  # 复制 / Copy
    stats = {"tax_scaled_rows": 0}  # 统计信息 / Stats container

    if ("Tax 5%" not in df.columns) or ("Total" not in df.columns):  # 必要列不存在 / Required columns missing
        return df, stats  # 直接返回 / Return as-is

    tax = pd.to_numeric(df["Tax 5%"], errors="coerce")  # 税转数值 / Coerce tax to numeric
    total = pd.to_numeric(df["Total"], errors="coerce")  # 总额转数值 / Coerce total to numeric
    expected_tax = total / 21.0  # 期望税额 / Expected tax

    # 候选缩放因子 / Candidate scaling factors
    factors = np.array([1.0, 0.1, 0.01, 0.001])  # 原值、除10、除100、除1000 / Keep, /10, /100, /1000

    # 计算每个因子的相对误差 / Compute relative error for each factor
    err_matrix = []  # 存储误差矩阵 / Store errors
    for f in factors:  # 遍历因子 / Loop factors
        cand = tax * f  # 候选修复税 / Candidate fixed tax
        err = (cand - expected_tax).abs() / (expected_tax.abs() + 1e-9)  # 相对误差 / Relative error
        err_matrix.append(err.to_numpy())  # 转 numpy 存起来 / Save as numpy

    err_matrix = np.vstack(err_matrix)  # 形状: [num_factors, n_rows] / Shape stacking
    best_idx = np.nanargmin(err_matrix, axis=0)  # 每行选误差最小因子 / Pick best factor per row
    best_factor = factors[best_idx]  # 最佳因子 / Best factor
    tax_fixed = tax * best_factor  # 修复后的税 / Fixed tax

    # 统计哪些行被缩放了 / Count scaled rows
    scaled_mask = (best_factor != 1.0) & tax.notna() & expected_tax.notna()  # 非1因子且不缺失 / Non-1 factors
    stats["tax_scaled_rows"] = int(scaled_mask.sum())  # 记录数量 / Save count

    df["Tax_fixed"] = tax_fixed  # 写入修复税列 / Store fixed tax
    df["Subtotal_fixed"] = total - df["Tax_fixed"]  # 小计 = 总额 - 税 / Subtotal = Total - Tax

    return df, stats  # 返回 / Return


def drop_invalid_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    删除明显无效数据：负值、零值、关键字段缺失、极端不一致的账务关系
    Drop invalid rows: negatives/zeros, missing critical fields, and inconsistent accounting relations.
    """
    df = df.copy()  # 复制 / Copy
    stats = {}  # 统计 / Stats

    # 关键字段列表（存在就用） / Critical numeric fields (use if present)
    critical = [c for c in ["Unit price", "Quantity", "Total"] if c in df.columns]  # 关键列 / Critical cols
    before = len(df)  # 清洗前行数 / Row count before

    # 删除关键字段缺失 / Drop rows with missing critical fields
    if critical:  # 若有关键列 / If critical columns exist
        df = df.dropna(subset=critical)  # 删除缺失行 / Drop missing critical
    stats["dropped_missing_critical"] = int(before - len(df))  # 记录数量 / Record count
    before = len(df)  # 更新 before / Update before

    # 删除非正值：Unit price<=0, Quantity<=0, Total<=0 / Drop non-positive values
    for c in ["Unit price", "Quantity", "Total"]:  # 遍历 / Loop fields
        if c in df.columns:  # 若存在 / If exists
            df = df[df[c].notna()]  # 再确保不缺失 / Ensure not NaN
            df = df[df[c] > 0]  # 过滤 >0 / Keep > 0

    stats["dropped_nonpositive"] = int(before - len(df))  # 记录删除行数 / Count drops
    before = len(df)  # 更新 / Update

    # 若有 Tax_fixed 与 Subtotal_fixed，检查一致性 / If we have fixed tax/subtotal, validate consistency
    if ("Tax_fixed" in df.columns) and ("Subtotal_fixed" in df.columns) and ("Total" in df.columns):  # 列存在 / Columns exist
        # total 应该接近 subtotal + tax / total should be close to subtotal + tax
        recon = df["Subtotal_fixed"] + df["Tax_fixed"]  # 重构总额 / Reconstructed total
        rel_err = (recon - df["Total"]).abs() / (df["Total"].abs() + 1e-9)  # 相对误差 / Relative error
        df = df[rel_err <= 0.02]  # 允许 2% 误差 / Allow 2% tolerance
        stats["dropped_inconsistent_total"] = int(before - len(df))  # 记录 / Record
        before = len(df)  # 更新 / Update
    else:
        stats["dropped_inconsistent_total"] = 0  # 没有相关列则为0 / Set 0 if not available

    return df, stats  # 返回 / Return


def deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    去重策略：
    - 如果有 Invoice ID：按 Invoice ID 去重（保留第一条）
    - 否则：整行去重
    Dedup strategy:
    - If Invoice ID exists: dedupe by Invoice ID (keep first)
    - Else: dedupe full rows
    """
    df = df.copy()  # 复制 / Copy
    stats = {"dropped_duplicates": 0}  # 统计 / Stats
    before = len(df)  # 去重前 / Before

    if "Invoice ID" in df.columns:  # 若有发票ID / If invoice id exists
        df = df.drop_duplicates(subset=["Invoice ID"], keep="first")  # 按发票去重 / Dedupe by Invoice ID
    else:  # 否则 / Else
        df = df.drop_duplicates(keep="first")  # 整行去重 / Dedupe full row

    stats["dropped_duplicates"] = int(before - len(df))  # 记录数量 / Record count
    return df, stats  # 返回 / Return


def cap_outliers_iqr(df: pd.DataFrame, cols: list, k: float = 3.0) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    温和处理异常值：用 IQR 上界截断（不删除），避免极端值扭曲分析
    Gentle outlier handling: cap at upper bound using IQR (no dropping), to reduce distortion.

    k=3.0 比较宽松 / k=3.0 is fairly lenient
    """
    df = df.copy()  # 复制 / Copy
    stats = {}  # 统计 / Stats

    for c in cols:  # 遍历列 / Loop columns
        if c not in df.columns:  # 若列不存在 / If missing
            continue  # 跳过 / Skip
        s = df[c].dropna()  # 去缺失 / Drop NaNs
        if len(s) < 10:  # 样本太少不处理 / Too few samples
            stats[f"capped_{c}"] = 0  # 记录0 / Record 0
            continue  # 继续 / Continue

        q1 = s.quantile(0.25)  # 第1四分位 / Q1
        q3 = s.quantile(0.75)  # 第3四分位 / Q3
        iqr = q3 - q1  # 四分位距 / IQR
        upper = q3 + k * iqr  # 上界 / Upper bound

        before_cap = df[c].copy()  # 保存截断前 / Save before capping
        df[c] = np.where(df[c] > upper, upper, df[c])  # 大于上界则截断 / Cap above upper
        stats[f"capped_{c}"] = int((before_cap > upper).sum())  # 统计被截断数量 / Count capped rows

    return df, stats  # 返回 / Return


def write_report(report_path: str, sections: list) -> None:
    """写清洗报告到文本文件 / Write cleaning report to a text file."""
    with open(report_path, "w", encoding="utf-8") as f:  # 打开文件 / Open file
        for title, content in sections:  # 遍历段落 / Iterate sections
            f.write(f"=== {title} ===\n")  # 写标题 / Write header
            f.write(content.strip() + "\n\n")  # 写内容 / Write content


def main() -> None:
    # 解析参数 / Parse args
    parser = argparse.ArgumentParser()  # 创建解析器 / Create parser
    parser.add_argument("--in", dest="infile", required=True, help="输入CSV路径 / Input CSV path")  # 输入 / Input
    parser.add_argument("--out", dest="outfile", required=True, help="输出清洗后CSV路径 / Output cleaned CSV path")  # 输出 / Output
    parser.add_argument("--report", dest="report", default="cleaning_report.txt", help="清洗报告路径 / Report path")  # 报告 / Report
    args = parser.parse_args()  # 解析 / Parse

    infile = args.infile  # 输入路径 / Input path
    outfile = args.outfile  # 输出路径 / Output path
    report_path = args.report  # 报告路径 / Report path

    if not os.path.exists(infile):  # 检查输入是否存在 / Check input exists
        print(f"[ERROR] Input file not found: {infile}")  # 打印错误 / Print error
        sys.exit(1)  # 退出 / Exit

    # 读取数据 / Load data
    df = pd.read_csv(infile)  # 读CSV / Read CSV
    n0 = len(df)  # 原始行数 / Original rows
    c0 = len(df.columns)  # 原始列数 / Original cols

    # 基础预处理 / Basic preprocessing
    df = normalize_columns(df)  # 标准化列名 / Normalize columns
    df = trim_text_columns(df)  # 清理文本空格 / Trim text columns
    df = parse_datetime_if_possible(df)  # 解析日期时间 / Parse datetime if possible

    # 数值转换 / Numeric coercion
    df = coerce_numeric(df, ["Unit price", "Quantity", "Tax 5%", "Total", "cogs", "gross income", "Rating"])  # 常见列 / Common cols

    # 修复税缩放 / Fix tax scaling
    df, tax_stats = fix_tax_scaling(df)  # 修复并拿统计 / Fix and get stats

    # 去重 / Deduplicate
    df, dedup_stats = deduplicate(df)  # 去重并记录 / Deduplicate and record

    # 删除无效行 / Drop invalid rows
    df, invalid_stats = drop_invalid_rows(df)  # 删除并记录 / Drop and record

    # 温和处理异常值（只截断，不删除） / Gentle outlier capping (cap, don't drop)
    df, cap_stats = cap_outliers_iqr(df, cols=[c for c in ["Unit price", "Quantity", "Total"] if c in df.columns], k=3.0)  # 截断 / Cap

    # 输出目录确保存在 / Ensure output directory exists
    out_dir = os.path.dirname(outfile)  # 输出文件夹 / Output folder
    if out_dir.strip() != "":  # 如果有目录部分 / If has directory part
        os.makedirs(out_dir, exist_ok=True)  # 创建目录 / Make dirs

    # 保存清洗后数据 / Save cleaned data
    df.to_csv(outfile, index=False)  # 写出CSV / Write CSV

    # 汇总报告内容 / Build report content
    n1 = len(df)  # 清洗后行数 / Rows after cleaning
    c1 = len(df.columns)  # 清洗后列数 / Cols after cleaning

    summary = (  # 汇总文本 / Summary text
        f"Input rows/cols: {n0} / {c0}\n"
        f"Output rows/cols: {n1} / {c1}\n"
        f"Rows removed: {n0 - n1}\n"
    )

    steps = []  # 报告段落列表 / Report sections list
    steps.append(("Summary / 总览", summary))  # 总览 / Summary
    steps.append(("Tax scaling fix / 税缩放修复", f"Rows with scaled tax fixed: {tax_stats.get('tax_scaled_rows', 0)}"))  # 税修复 / Tax
    steps.append(("Dedup / 去重", f"Dropped duplicates: {dedup_stats.get('dropped_duplicates', 0)}"))  # 去重 / Dedup
    steps.append(("Invalid rows / 无效行删除", "\n".join([f"{k}: {v}" for k, v in invalid_stats.items()])))  # 无效行 / Invalid
    steps.append(("Outlier capping / 异常值截断", "\n".join([f"{k}: {v}" for k, v in cap_stats.items()])))  # 截断 / Cap

    # 写报告 / Write report
    write_report(report_path, steps)  # 输出报告 / Write report

    # 终端提示 / Terminal message
    print("[DONE] Cleaning finished.")  # 完成提示 / Done
    print(f"Cleaned CSV saved to: {outfile}")  # 输出路径 / Output path
    print(f"Report saved to: {report_path}")  # 报告路径 / Report path


if __name__ == "__main__":  # 脚本入口 / Script entry
    main()  # 执行主函数 / Run main
