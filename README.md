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

## Installation
If running Anaconda, all of the above packages should be installed into a virtual environment. Once the environment is created, these can usually be installed by running either `conda install 'name of library'` or `conda install -c conda-forge 'name of library'`. One could also directly download my virtual environment 'PythonGPU' which I have included in the repository and put it into the correct directory. I also created a virtual environment for the LSS on Midway, 'Antony'.

Next, this repository should be cloned and installed, either manually or by running the following.
```bash
$ git clone https://github.com/antony-git/FergLabLSS
$ pip install ./FergLabLSS
```
After activating the virtual environment, open Jupyter and the notebooks should be able to run.

## Training the LSS pipeline
For somebody not familiar with the LSS's components, the following is a basic high level overview: The MD trajectory is processed by an SRV (further explanation [here](https://github.com/hsidky/srv) and learns an encoding of molecular configurations into a latent space while discovering a certain number of slow modes. This latent trajectory is propagated through time to transition probabilities by a mixture density network (MDN) denoted 'propagator' in the codes. The third network of the LSS, a conditional Wasserstein GAN (cWGAN) decodes the latent space coordinates back into a molecular configuration. 

This notebook contains codes which can run and train all of these components on the MD trajectory of chignolin, included in the chignolin folder. Thus to train the LSS one must:
1. Import the MD coordinates and load in a certain number of frames in main analysis notebook (for chignolin it is in the chignolin folder and called chignolin_analysis.ipynb) (more is better but will be more computationally intensive)
2. Open the SRV training notebook (chignolin.ipynb). Load in desired number of frames, extract features, and then train the SRV and save the model and latent coordinates
3. Load latent coordinates into analysis notebook
4. Train propagator
5. Train molgen (most intensive, must use high number of epochs for accuracy; LSS paper used 100000)

Note: 
- relevant parameters of each neural network can and should be tweaked (i.e. learning rate, number of epochs, etc.)
- the LSS can be trained locally, but it would be very computationally intensive even at lower numbers of frames and epochs. I have included a python script I used to train the propagator and the molgen on Midway

## Verification and Analysis
The notebooks include the codes needed to verify various steps of the LSS. For example, the SRV's timescales can be verified by running kTICA and the MSM. The propagator get be verified with 'synth_trajs'. Once satisfied with this accuracy, one can move on to analyzing the accuracy of the resultant kinetic and thermodynamic predictions.
Sample outputs can be found on 'trp_cage_analysis.ipynb', or by running 'chignolin_analysis.ipynb'.  
