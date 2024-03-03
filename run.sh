#!/bin/bash

# All Pretrain
python multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu --tasks sst
python multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu --tasks quora
python multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu --tasks sts
python multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu --tasks cfimdb

# All Pretrain with SMART
python multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu --tasks sst --smart
python multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu --tasks quora --smart
python multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu --tasks sts --smart
python multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu --tasks cfimdb --smart

# All finetune
python multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --tasks sst
python multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --tasks quora
python multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --tasks sts
python multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --tasks cfimdb

# All finetune with SMART
python multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --tasks sst --smart 
python multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --tasks quora --smart 
python multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --tasks sts --smart 
python multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --tasks cfimdb --smart 

# Multitask pretrain
python multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu --tasks sst quora sts
# (Just classification task)
python multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu --tasks sst quora
# (Just pairwise task)
python multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu --tasks quora sts


# Multitask pretrain with SMART
python multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu --tasks sst quora sts --smart
# (Just classification task)
python multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu --tasks sst quora --smart
# (Just pairwise task)
python multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu --tasks quora sts --smart

# Multitask finetune
python multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --tasks sst quora sts
# (Just classification task)
python multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --tasks sst quora
# (Just pairwise task)
python multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --tasks quora sts

# Multitask finetune with SMART
python multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --tasks sst quora sts --smart
# (Just classification task)
python multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --tasks sst quora --smart
# (Just pairwise task)
python multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --tasks quora sts --smart
