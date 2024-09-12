_base_ = ['schedules/schedule_1x.py', 'default_runtime.py']


# model settings
model = dict(
    type='SingleStageDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[200, 200, 200],
        std=[20.0, 20.0, 20.0],
        bgr_to_rgb=False),
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=128,
        start_level=1,
        num_outs=3),
    bbox_head=dict(
        type='AnchorFreeHeadTest',
        num_classes=3,
        in_channels=128,
        stacked_convs=2,
        feat_channels=64,
        strides=[8, 16, 32],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_object_score=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
        ),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
)

# learning rate
base_lr = 0.001
param_scheduler = [
    # dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,  # 按迭代更新学习率
         begin=0,
         end=100),  # 预热前 100 次迭代
    # dict(
    #     type='MultiStepLR',
    #     begin=0,
    #     end=30,
    #     by_epoch=True,
    #     milestones=[15, 20, 25], 
    #     gamma=0.5)
    dict(type='CosineAnnealingLR', 
         by_epoch=True, 
         T_max=30, 
         convert_to_iter_based=True)
]

# optimizer
optim_wrapper = dict(
    # optimizer=dict(lr=base_lr),
    optimizer=dict(_delete_=True, type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))


# dataset settings
dataset_type = 'BaseDetDatasetTest'
data_root = 'data/simple/'
batch_size = 16
num_worker = 2
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_worker,
    persistent_workers=num_worker > 0,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='train2/images', ann='train2/annotations'),
        ann_file='train2/annotations',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_worker,
    persistent_workers=num_worker > 0,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='test2/images', ann='test2/annotations'),
        ann_file='test2/annotations',
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='TestMetric',
    iou_thrs=[0.5, 0.75])
test_evaluator = val_evaluator


train_cfg = dict(max_epochs=30)
# test_cfg = None
# val_cfg = None

default_hooks = dict(
    logger=dict(interval=1),
    visualization=dict(score_thr=0.3),
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1  # only keep latest 3 checkpoints
    ))

# resume = "auto"

auto_scale_lr = dict(enable=False, base_batch_size=16)