"""
author: Midhun Murukesh

Experiment description: P08A_DeepDown_CMIP6
"""
import itertools

from ai4klima.tensorflow.train import CustomModelTrainer
from ai4klima.tensorflow.utils import load_inputs_target_pairs, take_paired_data_subset_by_bounds

from utils import configure_model, load_pretrained_model_path

######### EDIT BELOW #########
activation = 'prelu'
ups_method = 'convtranspose'
add_input_noise = False
input_noise_stddev = 0.1
reducelr_on_plateau = False
######### EDIT ABOVE #########

def RunExperiment(prefix, data_path, save_path, model_path, refd_path, epochs, models_dict, losses_dict, lr_dict, bs_dict):

    DATA_PATH, SAVE_PATH, MODEL_PATH, REFD_PATH = data_path, save_path, model_path, refd_path

    # Print the variables for debugging
    print(f"INFO: DATA_PATH: {DATA_PATH}")
    print(f"INFO: SAVE_PATH: {SAVE_PATH}")
    print(f"INFO: REFD_PATH: {REFD_PATH}")

    # P08A.Q03.MPI-ESM1-2-HR Ready to launch
    inputs_dict = {
        'era5': {
            'prec' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_PREC_DAY_1979_2023_RBIL_LOG.npy',

            'z850' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_ZG850_DAY_1979_2023_RBIL_STD.npy',
            'z700' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_ZG700_DAY_1979_2023_RBIL_STD.npy',
            'z500' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_ZG500_DAY_1979_2023_RBIL_STD.npy',

            'q850' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_HUS850_DAY_1979_2023_RBIL_STD.npy',
            'q700' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_HUS700_DAY_1979_2023_RBIL_STD.npy',
            'q500' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_HUS500_DAY_1979_2023_RBIL_STD.npy',

            't850' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_TA850_DAY_1979_2023_RBIL_STD.npy',
            't700' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_TA700_DAY_1979_2023_RBIL_STD.npy',
            't500' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_TA500_DAY_1979_2023_RBIL_STD.npy',

            'u850' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_UA850_DAY_1979_2023_RBIL_STD.npy',
            'u700' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_UA700_DAY_1979_2023_RBIL_STD.npy',
            'u500' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_UA500_DAY_1979_2023_RBIL_STD.npy',

            'v850' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_VA850_DAY_1979_2023_RBIL_STD.npy',
            'v700' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_VA700_DAY_1979_2023_RBIL_STD.npy',
            'v500' : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_VA500_DAY_1979_2023_RBIL_STD.npy',

            'msl'  : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_PSL_DAY_1979_2023_RBIL_STD.npy',
            't2m'  : f'{DATA_PATH}/IND32M_HRMIP_MPI-ESM1-2-HR_100_TAS_DAY_1979_2023_RBIL_STD.npy',
            } 
        }

    
    static_dict = {
        'elev': {
            'gtop' : f'{DATA_PATH}/IND32M_GTOP_010_ELEV_DAY_1979_2023_LOGN.npy',
            }
        }
    
    target_dict = {
        'mswx': {
            'prec': f'{DATA_PATH}/IND32M_MSWX_010_PREC_DAY_1979_2023_LOG.npy'
            },
        } 

    ################################
    # START THE EXPERIMENT ITERATOR
    ################################
    for (
        (loss_id, loss_fn), 
        (lr_id, gen_lr), 
        (bs_id, bs), 
        (i_id, inputs_channels), 
        (t_id, target_channels), 
        (s_id, static_channels), 
        (model_id, model_name)
    ) in itertools.product(
        losses_dict.items(), 
        lr_dict.items(), 
        bs_dict.items(), 
        inputs_dict.items(), 
        target_dict.items(), 
        static_dict.items(), 
        models_dict.items()
    ):

        print('\nRunning in Exp. loop next ...')
        print('#'*100)
        print(f"""
            m_id: {model_id}, gen_opt: {model_name}
            i_id: {i_id}, inputs_channels: {inputs_channels}
            t_id: {t_id}, target_channels: {target_channels}
            s_id: {s_id}, target_channels: {static_channels}
            l_id: {loss_id}, target_channels: {loss_fn}
            r_id: {lr_id}, gen_lr: {gen_lr}
            b_id: {bs_id}, bs: {bs}
            """)
        
        exp_prefix = f"{prefix}_{model_id}_{loss_id}_{i_id}_{t_id}_{lr_id}_{bs_id}"
        print(f'\nInitiate experiment: {exp_prefix}')
        
        ############################# INITIALIZE MODEL TRAINERS #############################
        try:

            """
            Train UNET variants -> Deterministic Modelling
            """
            mt = CustomModelTrainer(prefix = exp_prefix, save_path=SAVE_PATH)

            # Generate test data
            X, y, S = load_inputs_target_pairs(inputs_channels, target_channels, static_channels)
            X_test, _, S_test = take_paired_data_subset_by_bounds(X, y, S, bounds=(11323, None)) # 2010 JAN 01 : 2023 DEC 31 -> Edit here
            mt.generate_data_and_build_netcdf(
                [X_test, S_test],
                model_path   = load_pretrained_model_path(model_name, MODEL_PATH),
                refd_path    = REFD_PATH, 
                batch_size   = 8, 
                save_raw_npy = False, # Edit here
                build_netcdf = True, # Edit here
                varname      = 'prec', 
                start_date   = "2010-01-01",  # Edit here
                end_date     = "2023-12-31",  # Edit here
                tag          = None,
                )
                
        except Exception as e:
            print(f"An error occurred with run_id: {exp_prefix}: {e}")
        
    print("\n########################################## END OF SCRIPT ##########################################")

