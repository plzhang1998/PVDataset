"""用于计算区域指标、边界 F 分数并生成可视化结果的二值分割评估工具。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_mask(path: Path, is_label: bool = False) -> np.ndarray:
    """读取图像第一通道，并将前景值为 1 或 255 的数据转换为二值掩膜。"""
    with Image.open(path) as image:
        array = np.asarray(image).copy()
    if array.ndim == 3:
        array = array[..., 0]
    array = np.nan_to_num(array)
    if is_label:
        return array > 0
    values = set(np.unique(array).tolist())
    if not values.issubset({0, 1, 255}):
        raise ValueError(f"预测掩膜 {path.name} 只能包含 0/1 或 0/255，实际值为 {sorted(values)}")
    array[array == 255] = 1
    return array == 1


def region_counts(label: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    """统计真阳性、假阳性、假阴性和真阴性的像素数。"""
    return {
        "tp": int(np.sum(label & pred)),
        "fp": int(np.sum(~label & pred)),
        "fn": int(np.sum(label & ~pred)),
        "tn": int(np.sum(~label & ~pred)),
    }


def region_metrics(c: dict[str, int]) -> dict[str, float]:
    """根据混淆矩阵计数计算区域分割指标。"""
    tp, fp, fn, tn = (c[k] for k in ("tp", "fp", "fn", "tn"))
    pred_n, label_n = tp + fp, tp + fn
    precision = tp / pred_n if pred_n else float(label_n == 0)
    recall = tp / label_n if label_n else 1.0
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
    }


def boundary(mask: np.ndarray) -> np.ndarray:
    """使用八邻域提取二值掩膜的单像素内边界。"""
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    eroded = np.ones_like(mask)
    for y in range(3):
        for x in range(3):
            eroded &= padded[y : y + h, x : x + w]
    return mask & ~eroded


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """按给定的方形邻域半径膨胀二值掩膜。"""
    if radius == 0:
        return mask
    h, w = mask.shape
    padded = np.pad(mask, radius, constant_values=False)
    result = np.zeros_like(mask)
    size = 2 * radius + 1
    for y in range(size):
        for x in range(size):
            result |= padded[y : y + h, x : x + w]
    return result


def boundary_counts(label: np.ndarray, pred: np.ndarray, radius: int) -> dict[str, int]:
    """统计给定容差下的边界匹配数和边界带重叠量。"""
    label_b, pred_b = boundary(label), boundary(pred)
    label_band, pred_band = dilate(label_b, radius), dilate(pred_b, radius)
    return {
        "pred_match": int(np.sum(pred_b & label_band)),
        "pred_total": int(np.sum(pred_b)),
        "label_match": int(np.sum(label_b & pred_band)),
        "label_total": int(np.sum(label_b)),
        "intersection": int(np.sum(label_band & pred_band)),
        "union": int(np.sum(label_band | pred_band)),
    }


def boundary_metrics(c: dict[str, int]) -> dict[str, float]:
    """根据边界计数计算 Boundary IoU、精确率、召回率和 F1。"""
    pn, ln = c["pred_total"], c["label_total"]
    precision = c["pred_match"] / pn if pn else float(ln == 0)
    recall = c["label_match"] / ln if ln else 1.0
    return {
        "iou": c["intersection"] / c["union"] if c["union"] else 1.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def save_visualization(label: np.ndarray, pred: np.ndarray, output: Path) -> None:
    """按原始尺寸保存 TP 为蓝色、FP 为红色、FN 为绿色的可视化图像。"""
    visualization = np.zeros((*label.shape, 3), dtype=np.uint8)
    visualization[label & pred] = (0, 255, 0)
    visualization[~label & pred] = (255, 0, 0)
    visualization[label & ~pred] = (0,0, 255)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visualization).save(output)


def find_pairs(pred_dir: Path, label_dir: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    """按照不含扩展名的文件名匹配预测图像和标签图像。"""
    preds = sorted(p for p in pred_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    labels = {p.stem: p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    return [(p, labels[p.stem]) for p in preds if p.stem in labels], [p.name for p in preds if p.stem not in labels]


def add_counts(total: dict[str, int], current: dict[str, int]) -> None:
    """将当前指标计数累加到总计数中。"""
    for key, value in current.items():
        total[key] += value


def empty_boundary_counts() -> dict[str, int]:
    """创建一组值全部为零的边界统计计数。"""
    return {
        k: 0
        for k in (
            "pred_match",
            "pred_total",
            "label_match",
            "label_total",
            "intersection",
            "union",
        )
    }


def save_statistics_txt(counts: dict[str, int], metrics: dict[str, float], output: Path) -> None:
    """将原始计数和评估指标写入文本文件。"""
    lines = [*(f"{name}: {value}" for name, value in counts.items()), ""]
    lines.extend(f"{name} = {value:.6f}" for name, value in metrics.items())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def eval_seg(args) -> None:
    """解析命令行参数、评估全部掩膜对并写出报告。"""
    if any(r < 0 for r in args.tolerances):
        parser.error("tolerances must be >= 0")
    pairs, missing = find_pairs(args.pred_dir, args.label_dir)
    if not pairs:
        parser.error("no prediction/label pairs found (file stems must match)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    region_total = {k: 0 for k in ("tp", "fp", "fn", "tn")}
    boundary_total = {r: empty_boundary_counts() for r in args.tolerances} if args.eval_boundary else {}
    boundary_all_total = empty_boundary_counts() if args.eval_boundary else None
    statistics_dir = args.out_dir / "statistics"
    global_dir = statistics_dir / "global"
    rows = []

    total_images = len(pairs)
    for index, (pred_path, label_path) in enumerate(pairs, start=1):
        progress = index / total_images * 100
        print(f"正在评估 [{index}/{total_images}] {progress:6.2f}%：{pred_path.name}", flush=True)
        pred = read_mask(pred_path)
        label = read_mask(label_path, is_label=True)
        if pred.shape != label.shape:
            raise ValueError(f"shape mismatch: {pred_path.name} {pred.shape} != {label_path.name} {label.shape}")

        counts = region_counts(label, pred)
        add_counts(region_total, counts)
        row = {"image": pred_path.name, **counts, **region_metrics(counts)}
        image_statistics_dir = statistics_dir / pred_path.stem
        if args.eval_boundary:
            image_boundary_all = empty_boundary_counts()
            for radius in args.tolerances:
                b_counts = boundary_counts(label, pred, radius)
                add_counts(boundary_total[radius], b_counts)
                add_counts(boundary_all_total, b_counts)
                add_counts(image_boundary_all, b_counts)
                for name, value in boundary_metrics(b_counts).items():
                    row[f"boundary_{name}@{radius}px"] = value
                save_statistics_txt(
                    b_counts,
                    boundary_metrics(b_counts),
                    image_statistics_dir / f"{pred_path.stem}_bd_{radius}.txt",
                )
            image_boundary_all_metrics = boundary_metrics(image_boundary_all)
            for name, value in image_boundary_all_metrics.items():
                row[f"boundary_{name}@all"] = value
            save_statistics_txt(
                image_boundary_all,
                image_boundary_all_metrics,
                image_statistics_dir / f"{pred_path.stem}_bd_all.txt",
            )
        rows.append(row)
        save_statistics_txt(
            counts,
            region_metrics(counts),
            image_statistics_dir / f"{pred_path.stem}_stat.txt",
        )
        if not args.no_vis:
            save_visualization(label, pred, args.out_dir / "visualizations" / f"{pred_path.stem}.png")

    total_row = {"image": "__TOTAL__", **region_total, **region_metrics(region_total)}
    boundary_summary = {}
    if args.eval_boundary:
        for radius, counts in boundary_total.items():
            metrics = boundary_metrics(counts)
            boundary_summary[f"{radius}px"] = metrics
            for name, value in metrics.items():
                total_row[f"boundary_{name}@{radius}px"] = value
        boundary_all_metrics = boundary_metrics(boundary_all_total)
        boundary_summary["overall"] = boundary_all_metrics
        for name, value in boundary_all_metrics.items():
            total_row[f"boundary_{name}@all"] = value
    rows.append(total_row)

    save_statistics_txt(
        region_total,
        region_metrics(region_total),
        global_dir / "global_statistics.txt",
    )
    if args.eval_boundary:
        save_statistics_txt(
            boundary_all_total,
            boundary_all_metrics,
            global_dir / "global_boundary_statistics_all.txt",
        )

    global_dir.mkdir(parents=True, exist_ok=True)
    with (global_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "evaluated_images": len(pairs),
        "unmatched_predictions": missing,
        "region": {**region_total, **region_metrics(region_total)},
    }
    if args.eval_boundary:
        summary["boundary"] = boundary_summary
    (global_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"评估完成：共处理 {total_images} 张图像。")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate binary segmentation masks with matching file stems.")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--eval-boundary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否计算并输出 Boundary 评估指标",
    )
    parser.add_argument("--tolerances", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()
    eval_seg(args)

