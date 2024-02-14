_base_="base.py"


classes = ('pos',)
dataset_type="CocoDataset"
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(

        sup=dict(
            type=dataset_type,
            classes=classes,
            ann_file="data/coco/annotations/instances_train2017.json",
            img_prefix="data/coco/train2017/",

        ),
        unsup=dict(
            type=dataset_type,
            classes=classes,
            ann_file="data/coco/annotations/instances_unlabeled2017.json",
            img_prefix="data/coco/unlabeled2017/",

        ),
    ),

    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/coco/annotations/instances_test2017.json',
        img_prefix='data/coco/test2017/'),

    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=0.6,
    )
)

lr_config = dict(step=[120000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000*3)#max_iters=180000 * 4

