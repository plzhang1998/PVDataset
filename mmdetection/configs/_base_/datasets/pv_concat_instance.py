# dataset settings for instance segmentation
# 使用ConcatDataset组合多个电站的COCO数据集

dataset_type = 'CocoDataset'
data_root = './data/'

# 类别信息
metainfo = dict(
    classes=('pv_panel', ),
    palette=[(255, 0, 0)]  # 红色
)

# ==================== 训练数据管道 ====================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# ==================== 测试数据管道 ====================
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    # 测试集不应在 pipeline 中加载 gt annotations（会导致 OOM 且无必要），CocoMetric 会直接读取 JSON 评测
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# ==================== 训练集子数据集 ====================
# 训练集：12个电站（train-crop目录）

train_dataset_009 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train-crop/009/instances.json',
    data_prefix=dict(img='train-crop/009/images'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    metainfo=metainfo)

train_dataset_010 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train-crop/010/instances.json',
    data_prefix=dict(img='train-crop/010/images'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    metainfo=metainfo)

train_dataset_012 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train-crop/012/instances.json',
    data_prefix=dict(img='train-crop/012/images'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    metainfo=metainfo)

train_dataset_016 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train-crop/016/instances.json',
    data_prefix=dict(img='train-crop/016/images'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    metainfo=metainfo)

train_dataset_017 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train-crop/017/instances.json',
    data_prefix=dict(img='train-crop/017/images'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    metainfo=metainfo)

train_dataset_018 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train-crop/018/instances.json',
    data_prefix=dict(img='train-crop/018/images'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    metainfo=metainfo)

train_dataset_019 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train-crop/019/instances.json',
    data_prefix=dict(img='train-crop/019/images'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    metainfo=metainfo)

train_dataset_020 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train-crop/020/instances.json',
    data_prefix=dict(img='train-crop/020/images'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    metainfo=metainfo)

train_dataset_021 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train-crop/021/instances.json',
    data_prefix=dict(img='train-crop/021/images'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    metainfo=metainfo)


train_dataset_029 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train-crop/029/instances.json',
    data_prefix=dict(img='train-crop/029/images'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    metainfo=metainfo)

train_dataset_033 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train-crop/033/instances.json',
    data_prefix=dict(img='train-crop/033/images'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    metainfo=metainfo)

train_dataset_034 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train-crop/034/instances.json',
    data_prefix=dict(img='train-crop/034/images'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    metainfo=metainfo)

# ==================== 测试集子数据集 ====================
# 测试集：6个电站（test-crop目录）

test_dataset_013 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='test-crop/013/instances.json',
    data_prefix=dict(img='test-crop/013/images'),
    pipeline=test_pipeline,
    test_mode=True,
    metainfo=metainfo)

# test_dataset_022 = dict(
#     type=dataset_type,
#     data_root=data_root,
#     ann_file='test-crop/NingBo/022/instances.json',
#     data_prefix=dict(img='test-crop/NingBo/022/images'),
#     pipeline=test_pipeline,
#     test_mode=True,
#     metainfo=metainfo)

# test_dataset_024 = dict(
#     type=dataset_type,
#     data_root=data_root,
#     ann_file='test-crop/NingBo/024/instances.json',
#     data_prefix=dict(img='test-crop/NingBo/024/images'),
#     pipeline=test_pipeline,
#     test_mode=True,
#     metainfo=metainfo)

# test_dataset_028 = dict(
#     type=dataset_type,
#     data_root=data_root,
#     ann_file='test-crop/028/instances.json',
#     data_prefix=dict(img='test-crop/028/images'),
#     pipeline=test_pipeline,
#     test_mode=True,
#     metainfo=metainfo)

# test_dataset_030 = dict(
#     type=dataset_type,
#     data_root=data_root,
#     ann_file='test-crop/NingBo/030/instances.json',
#     data_prefix=dict(img='test-crop/NingBo/030/images'),
#     pipeline=test_pipeline,
#     test_mode=True,
#     metainfo=metainfo)

# test_dataset_031 = dict(
#     type=dataset_type,
#     data_root=data_root,
#     ann_file='test-crop/031/instances.json',
#     data_prefix=dict(img='test-crop/031/images'),
#     pipeline=test_pipeline,
#     test_mode=True,
#     metainfo=metainfo)

# test_shenzhen = dict(
#     type=dataset_type,
#     data_root=data_root,
#     ann_file='test-crop/ShenZhen/031/instances.json',
#     data_prefix=dict(img='test/ShenZhen/031/images'),
#     pipeline=test_pipeline,
#     test_mode=True,
#     metainfo=metainfo)



# ==================== DataLoader配置 ====================

# 训练集DataLoader（使用ConcatDataset组合所有训练电站）
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            train_dataset_009,
            train_dataset_010,
            train_dataset_012,
            train_dataset_016,
            train_dataset_017,
            train_dataset_018,
            train_dataset_019,
            train_dataset_020,
            train_dataset_021,
            train_dataset_029,
            train_dataset_033,
            train_dataset_034
        ]))

# 验证集DataLoader（使用ConcatDataset组合所有测试电站）
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            test_dataset_013,
            # test_dataset_022,
            # test_dataset_024,
            # test_dataset_028,
            # test_dataset_030,
            # test_dataset_031  # validation uses station 013 only
            # test_shenzhen
        ]))


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + "/test-crop/013/instances.json",
    metric=['bbox', 'segm'],
    format_only=False)

test_dataloader = val_dataloader
test_evaluator = val_evaluator
