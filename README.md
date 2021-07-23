## Pytorch implmentation of the paper --> [Unsupervised Cross-spectral Stereo Matching by Learning to Synthesize](https://arxiv.org/pdf/1903.01078.pdf)

## Steps to train on Pittsburgh dataset:
- Set up the config file in `./config` folder, adjust the hyperparameters as per your need
- Download the pittsburgh RGB-NIR stereo dataset specify the location in the config file (`basepath`)

## Qualitative Results
- **Available Soon**

## Quantitative Results
- **Available Soon**

## Trained Weights:
- **Available Soon**

## Training
- The model follows iterative optimization technique
- Step1--> Training the discriminator
- Step2--> Training Feature Extractors and Generators 
- Step3--> Training Stereo Matching Netowrk
- Step4--> Training on auxilary loss

- The network is trained for 10 epochs with only step1 and step2, thus `warmup: True, stereo: False, auxilary: False` in config file
- Change `warmup: False, stereo: True, auxilary: False` for step 3 for another 2-3 epochs
- Change `warmup: False, stereo: False, auxilary: True` for step 4, the global optimization stage


## Summary
- Summaries are created inside `./summary` folder
- to view loss logs, intermediate results run `tensorboard --logdir ./summary`
