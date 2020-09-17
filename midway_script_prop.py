#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Midway Script

Created on Thu Aug 20 16:06:39 2020

@author: antonyawad
"""
# a test script for training the propagator on Midway
#%matplotlib inline 
import numpy as np 
import os, pickle, time
import pyemma as py 
import mdtraj as md 
import nglview as nv 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

import tensorflow as tf
import tensorflow.keras.backend as K
#import tensorflow.compat.v1.keras.backend as K
config = tf.ConfigProto()
#config = tf.compat.v1.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 1.0
# Create a session with the above options specified.
#K.tensorflow.compat.v1.keras.backend.set_session(tf.Session(config=config))

import tensorflow.compat

import sys
sys.path.append('/home/antonyawad/LSS_chig_files/hde')
from hde import hde, MolGen, Propagator

# generate traj
trajs_files = ['/home/antonyawad/LSS_chig_files/DESRES-Trajectory_CLN025-0-protein/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein/CLN025-0-protein-%03d.dcd' % item
              for item in range(54)]
trajs_all = [md.load(item, top='/home/antonyawad/LSS_chig_files/DESRES-Trajectory_CLN025-0-protein/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein/CLN025-0-protein.pdb')
             for item in trajs_files]
traj = md.join(trajs_all)
#/home/antonyawad/LSS_chig_files/DESRES-Trajectory_CLN025-0-protein/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein/CLN025-0-protein.pdb
def get_timescales(data, lag):
    if type(data) is list:
        temp_pair = (np.concatenate([item_traj[:-lag:lag] for item_traj in data]),
                     np.concatenate([item_traj[lag::lag] for item_traj in data]))
    else:
        temp_pair = (data[:-lag:lag], data[lag::lag])
    return -lag / np.log(np.corrcoef(temp_pair[0], temp_pair[1])[0][1])


trj_dir = '/home/antonyawad/LSS_chig_files/DESRES-Trajectory_CLN025-0-protein/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein'
trj_file = os.path.join(trj_dir, "CLN025-0-protein-000.dcd")
pdb_file = os.path.join(trj_dir, "CLN025-0-protein.pdb")

traj = md.load(trj_file, top=pdb_file)

traj.superpose(traj[0])
alpha = traj.top.select_atom_indices('alpha')
traj_ca = traj.atom_slice(alpha)
traj_ca.superpose(traj_ca[0])
xyz = traj_ca.xyz.reshape(-1, len(alpha)*3)
scaler = MinMaxScaler(feature_range=(-1,1))
y_train = scaler.fit_transform(xyz)

filename = '/home/antonyawad/LSS_chig_files/srv-master/paper_notebooks/hde_model2_chig.pkl'
srv = pickle.load(open(filename, 'rb'))
srv._encoder.summary()

Z = np.load('/home/antonyawad/LSS_chig_files/srv-master/paper_notebooks/hde_coords_chig.npy')
print(Z)
srv_scaler = MinMaxScaler(feature_range=(-1,1))
x_train = srv_scaler.fit_transform(Z[:,[0,1,2]])
print(x_train)

# propagator
n_mix = 25
lag = 100

prop_scaler = MinMaxScaler(feature_range=(0,1))
traj_prop = x_train

traj_prop_scaled = prop_scaler.fit_transform(traj_prop)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)
]
prop = Propagator(
    traj_prop_scaled.shape[1], 
    n_components=n_mix, 
    lag_time=lag, 
    batch_size=200000, 
    learning_rate=0.0005, 
    n_epochs=20000,
    callbacks=callbacks,
    hidden_size=100,
    activation='relu'
)
from hde.propagator import get_mixture_loss_func
prop.model.compile(loss=get_mixture_loss_func(prop.input_dim, prop.n_components), optimizer=tf.keras.optimizers.Adam(lr=0.0001))
prop.model.compile(loss=get_mixture_loss_func(prop.input_dim, prop.n_components), optimizer=tf.keras.optimizers.Adam(lr=0.001))
prop.fit([traj_prop_scaled, traj_prop_scaled[::-1]])

with open('/home/antonyawad/LSS_chig_files/lss_files/chig_prop_weights.pkl', 'wb') as f:
     pickle.dump(prop.model.get_weights(), f)

with open('/home/antonyawad/LSS_chig_files/lss_files/chig_prop_weights.pkl', 'rb') as f:
    prop.model.set_weights(pickle.load(f))
    