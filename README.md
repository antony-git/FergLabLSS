# Ferg Lab LSS
Latent Space Simulators (LSS) are a pipeline of three deep learning neural networks that together are able to learn kinetic models and generate dynamic trajectories over a longer time scales than traditional molecular dynamics (MD) allows. As shown in the paper 'Molecular Latent Space Simulators' by Ferguson, Chen, and Sidky, the LSS can ultimately achieve long synthetic folding trajectories that accurately recreate atomic structure, thermodynamics, and kinetics at a far lower computational cost. 

In the original paper, this was applied to TRP cage miniprotein. This repository includes those files as well as the trajectory and notebooks required to train the LSS for chignolin, another miniprotein.

## Requirements
The LSS pipeline requires the following libraries:
- numpy
- scipy
- keras
- tensorflow
- scikit-learn
- nglview
- matplotlib
- pyemma
A full list of the packages and their versions that should be installed into the environmnent to run the codes in this repository can be found [here](https://ferglab.slack.com/files/UB5REGR7Z/F017TKT1XGV/version.txt?origin_team=TB421Q20Y&origin_channel=D016G6HUXBL). 
