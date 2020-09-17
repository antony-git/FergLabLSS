
"""
Midway Script

Created on Thu Aug 20 16:06:39 2020
This script will train the last two parts of the LSS pipeline (the propagator and the molgen) for the protein chignolin
@author: antonyawad
"""
#%matplotlib inline 
# set up
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

# import and generate traj
trajs_files = ['/home/antonyawad/LSS_chig_files/DESRES-Trajectory_CLN025-0-protein/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein/CLN025-0-protein-%03d.dcd' % item
              for item in range(5)]
trajs_all = [md.load(item, top='/home/antonyawad/LSS_chig_files/DESRES-Trajectory_CLN025-0-protein/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein/CLN025-0-protein.pdb')
             for item in trajs_files]
traj = md.join(trajs_all)

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

# initialize and train propagator
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
    learning_rate=0.00025, 
    n_epochs=20000,
    callbacks=callbacks,
    hidden_size=100,
    activation='relu'
)
from hde.propagator import get_mixture_loss_func
prop.model.compile(loss=get_mixture_loss_func(prop.input_dim, prop.n_components), optimizer=tf.keras.optimizers.Adam(lr=0.001))
prop.fit([traj_prop_scaled, traj_prop_scaled[::-1]])

with open('/home/antonyawad/LSS_chig_files/lss_files/chig_prop_weights.pkl', 'wb') as f:
    pickle.dump(prop.model.get_weights(), f) # saving resultant weights and biases
    
with open('/home/antonyawad/LSS_chig_files/lss_files/chig_prop_weights.pkl', 'rb') as f:
    prop.model.set_weights(pickle.load(f))
    
# initialize and train generator (going from SRV space back to regular space)
molgen = MolGen(
    latent_dim=x_train.shape[1],
    output_dim=y_train.shape[1],
    batch_size=20000,
    noise_dim=50,
    n_epochs=40000,
    hidden_layer_depth=2,
    hidden_size=200,
    n_discriminator=5
)
# train on SRV coordinates, then on molecular outputs
molgen.fit(x_train, y_train)
molgen.generator.save('/home/antonyawad/LSS_chig_files/lss_files/generator_chig3.h5')
molgen.discriminator.save('/home/antonyawad/LSS_chig_files/lss_files/discriminator_chig3.h5')

with open('/home/antonyawad/LSS_chig_files/lss_files/chignolin_molgen.pkl', 'wb') as f:
    pickle.dump(molgen, f) # saving the model

from hde.molgen import swish
# saving resultant generator and discriminator
molgen.generator = tf.keras.models.load_model('/home/antonyawad/LSS_chig_files/lss_files/generator_chig3.h5', custom_objects={'swish': swish})
molgen.discriminator = tf.keras.models.load_model('/home/antonyawad/LSS_chig_files/lss_files/discriminator_chig3.h5', custom_objects={'swish': swish})