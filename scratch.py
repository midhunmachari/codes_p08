# Standard library imports
import itertools


from deepclimate.tensorflow.train import ModelTraining, Pix2Pix, SRGAN
from deepclimate.tensorflow.utils import (
    load_inputs_target_pairs, 
    take_paired_data_subset_by_bounds
)
from deepclimate.tensorflow.models import UNET, Attention_UNET, PatchDiscriminator
from deepclimate.tensorflow.losses import weighted_mae, BernoulliGammaLoss  

# ai4klim utility imports
import deepclimate.tensorflow.utils as utils

# Print the available attributes and methods in the 'utils' module
print("Available methods in 'utils':", dir(utils))


#%%

from ai4klima.tensorflow.models import UNET, PatchDiscriminator

u = UNET(
    input_shape = (160,360,7), # Edit here
    layer_N=[64, 96, 128, 160],
    input_stack_num=2, 
    pool=True, 
    activation='relu',
    n_classes = 1,
    dropout_rate=0,
    isgammaloss=False,
    )
u.summary()
# %%
