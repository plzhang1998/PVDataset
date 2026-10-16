# Copyright (c) OpenMMLab. All rights reserved.
"""
逐个子测试集推理 + 合并评估脚本

解决问题：当测试集非常大（如26万张图像）时，一次性推理会导致内存溢出。
本脚本将ConcatDataset中的每个子数据集单独推理，推理完一个子集后
将结果存盘并释放内存，最后合并所有子集结果统一评估。

用法：
    python tools/test_per_subset.py \
        configs/mask2former/mask2former_r50_8xb2-lsj-50e_pv_panel.py \
        work_dirs/xxx/best_coco_segm_mAP_epoch_10.pth \
        --out-dir ./test_results

    # 只跑指定子集（子集名称，即电站编号）
    python tools/test_per_subset.py CONFIG CHECKPOINT \
        --out-dir ./test_results --subsets 013 022

    # 跳过已完成的子集（断点续跑）
    python tools/test_per_subset.py CONFIG CHECKPOINT \
        --out-dir ./test_results --skip-existing

    # 仅合并评估（跳过推理，要求已有pkl结果文件）
    python tools/test_per_subset.py CONFIG CHECKPOINT \
        --out-dir ./test_results --eval-only
"""

import argparse
import gc
import os
import os.path as osp
import pickle
import sys
import time

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.logging import MMLogger
from mmengine.registry import DefaultScope
from mmengine.runner import load_checkpoint
from torch.utils.data import DataLoader

from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmdet.registry import DATASETS  # CocoDataset注册在mmdet.registry中
from mmdet.registry import MODELS
from mmdet.structures.mask import encode_mask_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='逐子集推理并合并评估（节省内存）')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型权重文件路径')
    parser.add_argument(
        '--out-dir',
        default='./test_results_per_subset',
        help='保存每个子集推理结果的目录（默认：./test_results_per_subset）')
    parser.add_argument(
        '--subsets',
        nargs='+',
        default=None,
        help='只推理指定的子集（如 --subsets 013 022），默认推理全部')
    parser.add_argument(
        '--batch-size', type=int, default=8,
        help='每个子集推理时的batch_size（默认8，内存紧张可设更小）')
    parser.add_argument(
        '--num-workers', type=int, default=8,
        help='dataloader的num_workers（默认8）')
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='跳过已推理完成的子集（根据pkl文件判断，用于断点续跑）')
    parser.add_argument(
        '--eval-only', action='store_true',
        help='仅合并评估（跳过推理，要求--out-dir中已有各子集的pkl文件）')
    parser.add_argument(
        '--eval-per-subset', action='store_true',
        help='在合并评估之外，也单独评估每个子集的精度')
    parser.add_argument(
        '--work-dir', default=None,
        help='工作目录（用于保存日志，默认与--out-dir相同）')
    return parser.parse_args()


def build_single_dataset(cfg, dataset_cfg, pipeline_cfg):
    """根据配置构建单个CocoDataset（不经过ConcatDataset）。"""
    ds_cfg = dict(dataset_cfg)
    ds_cfg['pipeline'] = list(pipeline_cfg)
    # 确保 test_mode=True
    ds_cfg['test_mode'] = True
    dataset = DATASETS.build(ds_cfg)
    return dataset


def build_dataloader(dataset, batch_size, num_workers):
    """构建单数据集的DataLoader，使用pseudo_collate适配mmdet格式。"""
    from torch.utils.data import SequentialSampler
    sampler = SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=pseudo_collate,  # mmdet要求使用pseudo_collate
        pin_memory=False,
        persistent_workers=False,
    )
    return loader


@torch.no_grad()
def run_inference(model, dataloader, dataset_name, logger):
    """
    对一个dataloader跑推理，返回CocoMetric格式的results列表。
    每条result = (gt_dict, pred_dict)
    """
    model.eval()
    results = []
    total = len(dataloader.dataset)
    logger.info(f'[{dataset_name}] 开始推理，共 {total} 张图像...')

    processed = 0
    t0 = time.time()

    for batch_idx, data_batch in enumerate(dataloader):
        # mmdet的DataPreprocessor在model.forward内部处理
        # 直接调用 model.test_step
        data_samples = model.test_step(data_batch)

        for data_sample in data_samples:
            result = dict()
            pred = data_sample.pred_instances

            result['img_id'] = data_sample.img_id
            result['bboxes'] = pred.bboxes.cpu().numpy()
            result['scores'] = pred.scores.cpu().numpy()
            result['labels'] = pred.labels.cpu().numpy()

            # 编码mask为RLE格式
            if 'masks' in pred:
                masks_np = pred.masks.cpu().numpy() if isinstance(
                    pred.masks, torch.Tensor) else pred.masks
                result['masks'] = encode_mask_results(masks_np)

            if 'mask_scores' in pred:
                result['mask_scores'] = pred.mask_scores.cpu().numpy()

            # gt信息（用于无ann_file时的评估，有ann_file时可为空）
            gt = dict()
            gt['width'] = data_sample.ori_shape[1]
            gt['height'] = data_sample.ori_shape[0]
            gt['img_id'] = data_sample.img_id
            # 若有gt_instances则保存，否则留空
            if hasattr(data_sample, 'gt_instances'):
                gt_anns = []
                gt_inst = data_sample.gt_instances
                # gt_inst.masks 可能是 BitmapMasks、Tensor 或 numpy array
                gt_masks_np = None
                if 'masks' in gt_inst:
                    if isinstance(gt_inst.masks, torch.Tensor):
                        gt_masks_np = gt_inst.masks.cpu().numpy()
                    elif hasattr(gt_inst.masks, 'masks'):
                        # BitmapMasks 对象，内部 .masks 为 numpy array (N, H, W)
                        gt_masks_np = gt_inst.masks.masks
                    else:
                        gt_masks_np = np.asarray(gt_inst.masks)

                for i in range(len(gt_inst.labels)):
                    ann = dict(
                        bbox_label=int(gt_inst.labels[i]),
                        bbox=gt_inst.bboxes[i].cpu().numpy().tolist())
                    if gt_masks_np is not None:
                        ann['mask'] = encode_mask_results([gt_masks_np[i]])[0]
                    gt_anns.append(ann)
                gt['anns'] = gt_anns
            else:
                gt['anns'] = []

            results.append((gt, result))

        processed += len(data_samples)
        if (batch_idx + 1) % 50 == 0 or processed >= total:
            elapsed = time.time() - t0
            speed = processed / elapsed if elapsed > 0 else 0
            logger.info(
                f'[{dataset_name}] {processed}/{total} '
                f'({processed/total*100:.1f}%) '
                f'{speed:.1f} img/s')

    logger.info(f'[{dataset_name}] 推理完成，耗时 {time.time()-t0:.1f}s')
    return results


def evaluate_results(results, ann_file, metrics, logger, tag=''):
    """用CocoMetric评估一组results。"""
    coco_metric = CocoMetric(
        ann_file=ann_file,
        metric=metrics,
        classwise=False,
    )
    # CocoMetric需要dataset_meta，手动设置
    coco_metric.dataset_meta = dict(classes=('pv',))
    # 直接调用compute_metrics
    eval_results = coco_metric.compute_metrics(list(results))

    logger.info(f'[{tag}] 评估结果：')
    for k, v in eval_results.items():
        logger.info(f'  {k}: {v:.4f}')
    return eval_results


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 初始化logger
    log_file = osp.join(args.out_dir, 'test_per_subset.log')
    logger = MMLogger.get_instance(
        'test_per_subset',
        log_file=log_file,
        log_level='INFO')
    logger.info(f'配置文件: {args.config}')
    logger.info(f'模型权重: {args.checkpoint}')
    logger.info(f'输出目录: {args.out_dir}')

    # ==================== 加载配置 ====================
    cfg = Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = args.out_dir

    # 设置default_scope，确保mmdet的transforms能正确找到
    default_scope = cfg.get('default_scope', 'mmdet')
    DefaultScope.get_instance('test_per_subset', scope_name=default_scope)
    logger.info(f'default_scope: {default_scope}')

    # ==================== 解析子数据集 ====================
    # 从配置中提取ConcatDataset的各个子数据集
    val_dataloader_cfg = cfg.val_dataloader
    concat_cfg = val_dataloader_cfg.dataset
    assert concat_cfg.get('type') == 'ConcatDataset', \
        'val_dataloader.dataset必须是ConcatDataset类型'

    sub_datasets = concat_cfg.datasets  # list of dataset dicts

    # 获取test_pipeline（从第一个子数据集中提取）
    test_pipeline = sub_datasets[0].get('pipeline', cfg.get('test_pipeline', []))

    # 为每个子数据集生成一个可读名称
    def get_dataset_name(ds_cfg):
        ann_file = ds_cfg.get('ann_file', '')
        # 从路径中提取电站编号，如 test-crop/013/instances.json -> 013
        parts = ann_file.replace('\\', '/').split('/')
        for p in parts:
            if p.isdigit():
                return p
        return osp.splitext(osp.basename(ann_file))[0]

    dataset_names = [get_dataset_name(ds) for ds in sub_datasets]
    logger.info(f'检测到 {len(sub_datasets)} 个子数据集: {dataset_names}')

    # 筛选要跑的子集
    if args.subsets:
        selected_indices = [
            i for i, name in enumerate(dataset_names)
            if name in args.subsets
        ]
        if not selected_indices:
            logger.error(f'指定的子集 {args.subsets} 未找到！')
            logger.error(f'可用子集: {dataset_names}')
            sys.exit(1)
        logger.info(f'将只推理指定子集: {[dataset_names[i] for i in selected_indices]}')
    else:
        selected_indices = list(range(len(sub_datasets)))

    # ==================== 构建模型 ====================
    if not args.eval_only:
        logger.info('正在构建模型（不构建DataLoader，节省内存）...')
        # 直接构建模型，不通过Runner（避免构建大型ConcatDataset）
        model = MODELS.build(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        model = model.cuda()
        model.eval()
        logger.info('模型构建完成，已加载权重')

    # ==================== 评估设置 ====================
    val_evaluator_cfg = cfg.val_evaluator
    ann_file = val_evaluator_cfg.get('ann_file', None)
    metrics = val_evaluator_cfg.get('metric', ['bbox', 'segm'])

    # 为每个子数据集准备单独的ann_file（如果有）
    data_root = cfg.get('data_root',
                        sub_datasets[0].get('data_root', ''))

    def get_subset_ann_file(ds_cfg):
        """获取子数据集自己的ann_file路径。"""
        ds_root = ds_cfg.get('data_root', data_root)
        ds_ann = ds_cfg.get('ann_file', '')
        if osp.isabs(ds_ann):
            return ds_ann
        return osp.join(ds_root, ds_ann)

    # ==================== 逐子集推理 ====================
    all_pkl_files = {}

    for idx in selected_indices:
        ds_name = dataset_names[idx]
        ds_cfg = sub_datasets[idx]
        pkl_path = osp.join(args.out_dir, f'results_{ds_name}.pkl')
        all_pkl_files[ds_name] = pkl_path

        if args.eval_only:
            if not osp.exists(pkl_path):
                logger.warning(f'[{ds_name}] 结果文件不存在: {pkl_path}，跳过')
                del all_pkl_files[ds_name]
            else:
                logger.info(f'[{ds_name}] 找到已有结果文件: {pkl_path}')
            continue

        if args.skip_existing and osp.exists(pkl_path):
            logger.info(f'[{ds_name}] 结果文件已存在，跳过: {pkl_path}')
            continue

        logger.info(f'\n{"="*60}')
        logger.info(f'正在推理子数据集 [{ds_name}] ({idx+1}/{len(selected_indices)})')
        logger.info(f'{"="*60}')

        # 构建该子数据集和dataloader
        ds_pipeline = ds_cfg.get('pipeline', test_pipeline)
        dataset = build_single_dataset(cfg, ds_cfg, ds_pipeline)
        loader = build_dataloader(dataset, args.batch_size, args.num_workers)

        logger.info(f'[{ds_name}] 数据集大小: {len(dataset)}')

        # 推理
        results = run_inference(model, loader, ds_name, logger)

        # 保存结果到pkl
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f'[{ds_name}] 结果已保存: {pkl_path} ({len(results)}条)')

        # 单独评估（可选）
        if args.eval_per_subset:
            subset_ann = get_subset_ann_file(ds_cfg)
            if osp.exists(subset_ann):
                logger.info(f'[{ds_name}] 单独评估...')
                try:
                    eval_res = evaluate_results(
                        results, subset_ann, metrics, logger, tag=ds_name)
                except Exception as e:
                    logger.warning(f'[{ds_name}] 单独评估失败: {e}')
            else:
                logger.warning(f'[{ds_name}] 未找到子集ann_file: {subset_ann}')

        # 释放内存
        del dataset, loader, results
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f'[{ds_name}] 内存已释放')

    # ==================== 合并所有子集结果并评估 ====================
    logger.info(f'\n{"="*60}')
    logger.info('合并所有子集结果，进行统一评估')
    logger.info(f'{"="*60}')

    all_results = []
    for ds_name, pkl_path in all_pkl_files.items():
        if not osp.exists(pkl_path):
            logger.warning(f'结果文件不存在，跳过: {pkl_path}')
            continue
        logger.info(f'加载: {pkl_path}')
        with open(pkl_path, 'rb') as f:
            subset_results = pickle.load(f)
        logger.info(f'  [{ds_name}] {len(subset_results)} 条记录')
        all_results.extend(subset_results)
        del subset_results
        gc.collect()

    if not all_results:
        logger.error('没有找到任何推理结果，无法评估！')
        sys.exit(1)

    logger.info(f'\n合并后共 {len(all_results)} 条预测记录')

    # 使用合并的ann_file评估
    if ann_file and osp.exists(ann_file):
        logger.info(f'使用标注文件: {ann_file}')
        logger.info(f'评估指标: {metrics}')
        eval_results = evaluate_results(
            all_results, ann_file, metrics, logger, tag='合并评估')
    else:
        logger.warning(
            f'未找到合并标注文件: {ann_file}，'
            '请在pv_concat_instance.py中设置val_evaluator的ann_file')
        logger.info('提示：也可以用各子集的ann_file分别评估（--eval-per-subset）')

    logger.info('\n全部完成！')


if __name__ == '__main__':
    main()
