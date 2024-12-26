#!/bin/bash
method='Entropy'
dul_m_in=20
dul_m_out=30
lamb=0.1
gamma=2
tau=2
dataset='cifar100'
aux='imagenet'

python finetune.py --method ${method} \
    --dul_m_in ${dul_m_in} --dul_m_out ${dul_m_out} --lamb ${lamb} --gamma ${gamma} \
    --tau ${tau} --aux ${aux} --dataset ${dataset}