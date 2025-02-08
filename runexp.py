"""
author: Midhun Murukesh

Experiment description: P05A_DeepEnsemble_Globals
"""
import itertools

from ai4klima.tensorflow.train import ModelTraining
from ai4klima.tensorflow.utils import load_inputs_target_pairs, take_paired_data_subset_by_bounds

from utils import configure_model

######### EDIT BELOW #########
activation = 'prelu'
ups_method = 'convtranspose'
add_input_noise = False
input_noise_stddev = 0.1
######### EDIT ABOVE #########

def RunExperiment(prefix, model_id, data_path, save_path, refd_path, epochs, models_dict, losses_dict, lr_dict, bs_dict, keras_train=False):

    DATA_PATH, SAVE_PATH, REFD_PATH = data_path, save_path, refd_path

    # Print the variables for debugging
    print(f"INFO: DATA_PATH: {DATA_PATH}")
    print(f"INFO: SAVE_PATH: {SAVE_PATH}")
    print(f"INFO: REFD_PATH: {REFD_PATH}")

    inputs_dict = {
        'i01p': {
            'prec' : f'{DATA_PATH}/IND32M_ERA5_100_PREC_DAY_1979_2023_RCON_LOG.npy',

            'z850' : f'{DATA_PATH}/IND32M_ERA5_100_Z850_DAY_1979_2023_RBIL_STD.npy',
            'z700' : f'{DATA_PATH}/IND32M_ERA5_100_Z700_DAY_1979_2023_RBIL_STD.npy',
            'z500' : f'{DATA_PATH}/IND32M_ERA5_100_Z500_DAY_1979_2023_RBIL_STD.npy',

            'q850' : f'{DATA_PATH}/IND32M_ERA5_100_Q850_DAY_1979_2023_RBIL_STD.npy',
            'q700' : f'{DATA_PATH}/IND32M_ERA5_100_Q700_DAY_1979_2023_RBIL_STD.npy',
            'q500' : f'{DATA_PATH}/IND32M_ERA5_100_Q500_DAY_1979_2023_RBIL_STD.npy',

            't850' : f'{DATA_PATH}/IND32M_ERA5_100_T850_DAY_1979_2023_RBIL_STD.npy',
            't700' : f'{DATA_PATH}/IND32M_ERA5_100_T700_DAY_1979_2023_RBIL_STD.npy',
            't500' : f'{DATA_PATH}/IND32M_ERA5_100_T500_DAY_1979_2023_RBIL_STD.npy',

            'u850' : f'{DATA_PATH}/IND32M_ERA5_100_U850_DAY_1979_2023_RBIL_STD.npy',
            'u700' : f'{DATA_PATH}/IND32M_ERA5_100_U700_DAY_1979_2023_RBIL_STD.npy',
            'u500' : f'{DATA_PATH}/IND32M_ERA5_100_U500_DAY_1979_2023_RBIL_STD.npy',

            'v850' : f'{DATA_PATH}/IND32M_ERA5_100_V850_DAY_1979_2023_RBIL_STD.npy',
            'v700' : f'{DATA_PATH}/IND32M_ERA5_100_V700_DAY_1979_2023_RBIL_STD.npy',
            'v500' : f'{DATA_PATH}/IND32M_ERA5_100_V500_DAY_1979_2023_RBIL_STD.npy',

            'msl'  : f'{DATA_PATH}/IND32M_ERA5_100_MSL_DAY_1979_2023_RBIL_STD.npy',
            't2m'  : f'{DATA_PATH}/IND32M_ERA5_100_T2M_DAY_1979_2023_RBIL_STD.npy',
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
    for (model_id, model_name), (bs_id, bs), (lr_id, gen_lr), (loss_id, loss_fn), (i_id, inputs_channels), (t_id, target_channels), (s_id, static_channels) in itertools.product(
        models_dict.items(), bs_dict.items(), lr_dict.items(), losses_dict.items(), inputs_dict.items(), target_dict.items(), static_dict.items()):
        
        # model_name = models_dict[model_id]

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

        # Load the dataset for model training and validation
        X, y, S = load_inputs_target_pairs(inputs_channels, target_channels, static_channels)

        # Assuming X and y have the same number of samples
        X_train, y_train, S_train = take_paired_data_subset_by_bounds(X, y, S, bounds=(None, 11323))  # 1979 JAN 01 : 2009 DEC 31 -> Edit here
        X_val, y_val, S_val       = take_paired_data_subset_by_bounds(X, y, S, bounds=(11323, 13149)) # 2005 JAN 01 : 2009 DEC 31 -> Edit here

        # Print details of the data
        print(f"X_train shape: {X_train.shape}, S_train shape: {S_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, S_val shape: {S_val.shape}, y_val shape: {y_val.shape}")

        exp_prefix = f"{prefix}_{model_id}_{loss_id}_{i_id}_{t_id}_{lr_id}_{bs_id}"
        print(f'\nInitiate experiment: {exp_prefix}')

        ############################# INITIALIZE MODEL TRAINERS #############################
        try:

            """
            Train UNET variants -> Deterministic Modelling
            """
            gen_arch = configure_model(
                model_id, 
                input_shape = X_train.shape[1:],
                target_shape = y_train.shape[1:],
                input_shape_hr = S_train.shape[1:],
                activation = activation,
                ups_method = ups_method,
                add_input_noise = add_input_noise,
                input_noise_stddev = input_noise_stddev,
                )

            mt = ModelTraining(
                prefix = exp_prefix, 
                save_path = SAVE_PATH,
                generator = gen_arch,
                loss_fn = loss_fn,
                lr_init = gen_lr,
                log_tensorboard = True,
                enable_function = True,
                )
            
            # if keras_train:
            #     mt.train_by_fit(
            #         train_data = (X_train, y_train), 
            #         val_data = (X_val, y_val), 
            #         epochs = epochs,  # Edit here
            #         batch_size = bs, 
            #         monitor="val_loss",
            #         mode = "min",
            #         min_lr = 1e-10,
            #         save_ckpt = True,
            #         ckpt_interval = 1,
            #         save_ckpt_best = True,
            #         reducelr_on_plateau = True,
            #         reducelr_factor = 0.1,
            #         reducelr_patience = 7,
            #         early_stopping=False,
            #         early_stopping_patience = 18,
            #     )
            
            # else:
            #     mt.train(
            #         train_data = (X_train, S_train, y_train), 
            #         val_data = (X_val, S_val, y_val), 
            #         epochs = epochs,  # Edit here
            #         batch_size = bs, 
            #         monitor="val_loss",
            #         mode = "min",
            #         min_lr = 1e-10,
            #         save_ckpt = True,
            #         ckpt_interval = 1,
            #         save_ckpt_best = True,
            #         reducelr_on_plateau = True,
            #         reducelr_factor = 0.1,
            #         reducelr_patience = 12,
            #         early_stopping=False,
            #         early_stopping_patience = 18,
            #     )

            mt.plot_training_curves()
        
            X_test, _, S_test = take_paired_data_subset_by_bounds(X, y, bounds=(11323, None)) # 2010 JAN 01 : 2023 DEC 31 -> Edit here
            mt.generate_data_and_builf_netcdf(
                [X_test, S_test],
                model_path = None,
                refd_path=REFD_PATH, 
                batch_size = 32, 
                save_raw_npy=True, # Edit here
                build_netcdf=True, # Edit here
                varname = 'prec', 
                start_date = "2010-01-01",  # Edit here
                end_date   = "2023-12-31",  # Edit here
                tag=None,
                )
                
        except Exception as e:
            print(f"An error occurred with run_id: {exp_prefix}: {e}")
        
    print("\n########################################## END OF SCRIPT ##########################################")

