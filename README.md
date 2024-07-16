# DG-PINNs
This repository contains the code for the paper:
- [Data-Guided Physics-Informed Neural Networks for Solving Inverse Problems in Partial Differential Equations](https://arxiv.org/abs/2407.10836)

In this work, we proposed a new PINN framework: data-guided physics-informed neural networks.

•	DG-PINNs: A novel two-phase framework for solving inverse problems in PDEs.\
•	Pre-training phase focuses on data loss; fine-tuning phase embeds physical laws.\
•	Improves efficiency and maintains accuracy compared to existing PINNs.

# Schematic of DG-PINNs
Here we show the schematic of DG-PINNs for solving inverse problems in PDEs.

<p align="center">
  <img src="DGPINN_diagram.png" width="800">
</p>

# Naming convention of code
Take the heat equation for example:\
DG_PINN_heat_equation_NTK_M1 -- the sensitivity analysis on $M_1$ \
DG_PINN_heat_equation_NTK_Nd -- the sensitivity analysis on $N_d$ \
DG_PINN_heat_equation_NTK_noise -- the study of the noise-robustness of DG-PINNs \
DG_PINN_vs_PINN_heat_equation_NTK -- the study of the efficiency of PINNs and DG-PINNs

# Data for NS equation
We use the same data in the original paper of PINN for the NS equation, which can be downloaded in this link https://github.com/maziarraissi/PINNs/blob/1b3e90e82c47d49aee290aa481550f5e9c582d9a/main/Data/cylinder_nektar_wake.mat
# Citation

@article{zhou2024dgpinn,\
  title={Data-Guided Physics-Informed Neural Networks for Solving Inverse Problems in Partial Differential Equations},\
  author={Wei Zhou, Y.F. Xu},\
  journal={arXiv preprint arXiv:2407.10836},\
  year={2024}
}


