_base_ = [
    'bbox_level_multi_task_vfnet_wholebody_crop.py'  # reuse same model/settings
]

# -----------------------------------------------------------------------------
# Dataset paths
# -----------------------------------------------------------------------------

dataset_type = 'CustomCocoDataset'
data_root = '/workspace/dataset/mackerel/'

# ann files (created by csv_to_coco_attr + split script)
train_ann = 'splits/train.json'
val_ann   = 'splits/val.json'
test_ann  = 'splits/test.json'

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img='')
    ))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img='')
    ))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=test_ann,
        data_prefix=dict(img='')
    ))

# -----------------------------------------------------------------------------
# Model tweaks: 3 classes (id 2,3,4)
# -----------------------------------------------------------------------------
model = dict(
    bbox_head=dict(num_classes=3)
)

# -----------------------------------------------------------------------------
# Checkpoint & work dir
# -----------------------------------------------------------------------------
load_from = None   # train from scratch (backbone pretrained already specified)
resume = False
work_dir = './work_dirs/vfnet_mackerel' 