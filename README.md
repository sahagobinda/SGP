# SGP
 
Official Code Repository for "Continual Learning with Scaled Gradient Projection", AAAI 2023 (accepted). [[Paper]](https://arxiv.org/abs/2302.01386)
 
## Abstract 
In neural networks, continual learning results in gradient interference among sequential tasks, leading to catastrophic forgetting of old tasks while learning new ones. This issue is addressed in recent methods by storing the important gradient spaces for old tasks and updating the model orthogonally during new tasks. However, such restrictive orthogonal gradient updates hamper the learning capability of the new tasks resulting in sub-optimal performance. To improve new learning while minimizing forgetting, in this paper we propose a Scaled Gradient Projection (SGP) method, where we combine the orthogonal gradient projections with scaled gradient steps along the important gradient spaces for the past tasks. The degree of gradient scaling along these spaces depends on the importance of the bases spanning them. We propose an efficient method for computing and accumulating importance of these bases using the singular value decomposition of the input representations for each task. We conduct extensive experiments ranging from continual image classification to reinforcement learning tasks and report better performance with less training overhead than the state-of-the-art approaches.  

## Authors
```
Gobinda Saha, Kaushik Roy
```

## Instructions  

A. Setup Environment and Packages: 

1. Create a conda environment and activate it:
```
conda create --name sgpenv python=3.9.13
```
2. Install pytorch and related packeages (following command installs pytorch 1.11.0, torchvision 0.12.0, cudatoolkit 11.3.1): 
```
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```
3. Install other required packages:
```
pip install -r requirements.txt
```
4. Install Gym Atari : 
```
conda install -c conda-forge gym-atari
```


B. After successful installations for Continual Image Classification go to 'ImageClassification' folder and run: 
```
source run_experiments.sh 
```
This command will run Split CIFAR100 experiment by default, CIFAR-100 Superclass can also be selected in "run_experiments.sh" file. 


C. For Continual Reinforcement learning tasks go to 'RL_Experiments' folder and run: 
```
source run_rl_agents.sh 0
```
This command will run SGP experiment by default, other methods (GPM,BLIP,EWC,FT) can be run by selecting options in "run_rl_agents.sh" file. 


## Citation
```
@article{https://doi.org/10.48550/arxiv.2302.01386,
  doi = {10.48550/ARXIV.2302.01386},  
  url = {https://arxiv.org/abs/2302.01386},  
  author = {Saha, Gobinda and Roy, Kaushik},  
  title = {Continual Learning with Scaled Gradient Projection},  
  publisher = {arXiv},  
  year = {2023},  
}

```
