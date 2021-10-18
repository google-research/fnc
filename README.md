# Boosting Contrastive Self-Supervised Learning with False Negative Cancellation

This repository contains pretrained models and code base for the paper ["Boosting Contrastive Self-Supervised Learning with False Negative Cancellation"](https://arxiv.org/abs/2011.11765).

## Pre-trained models

|      |  Semi-supervised (1%) |   Semi-supervised (10%) |   Linear eval |   Supervised |
|--------:|--------:|------:|--------:|-------------:|--------------:|---------------:|-----------------:|--------------:|
|      Top-1 |  63.7     |   71.1    |  74.4   |   76.5      |     
|      Top-5 |  85.3     |   90.2    |  91.8   |    93.3     |   


* Pretrained FNC model (with linear eval head): [gs://fnc_checkpoints/ResNet-50/pretrained_linear_eval](https://console.cloud.google.com/storage/browser/fnc_checkpoints/ResNet-50/pretrained_linear_eval)
* Fine-tuned FNC model on 1% of labels: [gs://fnc_checkpoints/ResNet-50/finetuned_semi1pt](https://console.cloud.google.com/storage/browser/fnc_checkpoints/ResNet-50/finetuned_semi1pt)
* Fine-tuned FNC models on 10% of labels: [gs://fnc_checkpoints/ResNet-50/finetuned_semi10pt](https://console.cloud.google.com/storage/browser/fnc_checkpoints/ResNet-50/finetuned_semi10pt)

## Environment setup

Our models are trained with TPUs. It is recommended to run distributed training with TPUs when using our code for pretraining.

Install dependencies:
```
pip install -r requirements.txt
```
Add [TPU official models](https://github.com/tensorflow/tpu/tree/master/models/official) to PYTHONPATH:
```
PYTHONPATH=/your_path/tpu/models/official:$PYTHONPATH
```

## Pretraining

To pretrain the model on ImageNet with Cloud TPUs, first check out the [Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist) for basic information on how to use Google Cloud TPUs.

Once you have created virtual machine with Cloud TPUs, and pre-downloaded the ImageNet data for [tensorflow_datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012), please set the following enviroment variables:

```
TPU_NAME=<tpu-name>
STORAGE_BUCKET=gs://<storage-bucket>
DATA_DIR=$STORAGE_BUCKET/<path-to-tensorflow-dataset>
MODEL_DIR=$STORAGE_BUCKET/<path-to-store-checkpoints>
```

The following command can be used to pretrain a ResNet-50 on ImageNet:

```
python run.py --train_mode=pretrain \
  --train_batch_size=4096 --train_epochs=100 --temperature=0.1 \
  --learning_rate=0.1 --learning_rate_scaling=sqrt --weight_decay=1e-4 \
  --dataset=imagenet2012 --image_size=224 --eval_split=validation \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0
```

## Finetuning the linear head (linear eval)

For fine-tuning a linear head on ImageNet using Cloud TPUs, first set the `CHKPT_DIR` to pretrained model dir and set a new `MODEL_DIR`, then use the following command:

```
python run.py --mode=train --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' \
  --global_bn=True --optimizer=momentum --learning_rate=0.005 --weight_decay=0 \
  --train_epochs=90 --train_batch_size=1024 --warmup_epochs=0 \
  --dataset=imagenet2012 --image_size=224 --eval_split=validation \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --checkpoint=$CHKPT_DIR \
  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0
```

## Semi-supervised learning and fine-tuning the whole network

You can access 1% and 10% ImageNet subsets used for semi-supervised learning via [tensorflow datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012_subset): simply set `dataset=imagenet2012_subset/1pct` and `dataset=imagenet2012_subset/10pct` in the command line for fine-tuning on these subsets.

You can also find image IDs of these subsets in `imagenet_subsets/`.

To fine-tune the whole network on ImageNet (1% of labels), refer to the following command:

```
python run.py --mode=train --train_mode=finetune \
  --fine_tune_after_block=-1 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)Momentum|head_supervised)' \
  --global_bn=True --optimizer=lars --learning_rate=0.005 \
  --learning_rate_scaling=sqrt --weight_decay=0 \
  --train_epochs=60 --train_batch_size=1024 --warmup_epochs=0 \
  --dataset=imagenet2012_subset/1pct --image_size=224 --eval_split=validation \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --checkpoint=$CHKPT_DIR \
  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0 \
  --ft_proj_selector=1
```

## Cite

[FNC paper](https://arxiv.org/abs/2011.11765):

```
@article{huynh2020fnc,
  title={Boosting Contrastive Self-Supervised Learning with False Negative Cancellation},
  author={Huynh, Tri and Kornblith, Simon and Walter, Matthew R. and Maire, Michael and Khademi, Maryam},
  journal={arXiv preprint arXiv:2011.11765},
  year={2020}
}
```
## Acknowledgement
The code base is developed based on [SimCLR repository](https://github.com/google-research/simclr).

## Disclaimer
This is not an official Google product.
