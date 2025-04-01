from ai4klima.tensorflow.models import MegaUNet, SRCNN, FSRCNN, EDRN, SRDRN

def configure_model(
        model_id, 
        input_shape,
        target_shape,
        input_shape_hr,
        activation = 'prelu',
        ups_method = 'convtranspose',
        add_input_noise = False,
        input_noise_stddev = 0.1,      
        ):
    
    """
    MegaUNet(
        input_shape,
        target_shape,
        input_shape_2=None,
        n_classes=1,
        output_activation='linear',

        # Convolution-related settings
        convblock_opt='conv',
        layer_N=[64, 96, 128, 160],
        kernel_size=(3, 3),
        stack_num=1,
        activation='relu',
        initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
        regularizer=tf.keras.regularizers.l2(0.01),
        batchnorm_on=True,

        # Pooling and upsampling settings
        pool_opt='maxpool',
        pool_size=2,
        ups_method='bilinear',

        # Attention mechanism
        attention_on=False,
        attention_control=None,

        # Noise-related settings
        add_input_noise=False,
        input_noise_stddev=0.1,
        add_latent_noise=False,
        latent_noise_stddev=0.1,

        # Recurrent connections
        reccur_iter=2,

        # Dropout settings
        dropout_rate=0,

        # Final layer settings
        last_conv_filters=16,
        isgammaloss=False
        )
    
    """
    
    if model_id == 'unet': # U-Net
        return MegaUNet(
            input_shape = input_shape,
            target_shape = target_shape,
            input_shape_2 = input_shape_hr,
            convblock_opt = 'conv',
            layer_N = [64, 96, 128, 160],
            stack_num = 2,
            activation = activation,
            ups_method = ups_method,
            attention_on = False,
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            last_conv_filters = 16,
            )
    
    elif model_id == 'attention_unet': 
        return MegaUNet(
            input_shape = input_shape,
            target_shape = target_shape,
            input_shape_2 = input_shape_hr,
            convblock_opt = 'conv',
            layer_N = [64, 96, 128, 160],
            stack_num = 2,
            activation = activation,
            ups_method = ups_method,
            attention_on = True,
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            last_conv_filters = 16,
            ) 

    elif model_id == 'recurrent_unet':
        return MegaUNet(
            input_shape = input_shape,
            target_shape = target_shape,
            input_shape_2 = input_shape_hr,
            convblock_opt = 'recurrent_conv',
            layer_N = [64, 96, 128, 160],
            stack_num = 2,
            reccur_iter= 2,
            activation = activation,
            ups_method = ups_method,
            attention_on = False,
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            last_conv_filters = 16,
            ) 
    
    elif model_id == 'residual_unet':
        return MegaUNet(
            input_shape = input_shape,
            target_shape = target_shape,
            input_shape_2 = input_shape_hr,
            convblock_opt = 'residual_conv',
            layer_N = [64, 96, 128, 160],
            stack_num = 2,
            activation = activation,
            ups_method = ups_method,
            attention_on = False,
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            last_conv_filters = 16,
            ) 
    
    elif model_id == 'recurrent_residual_attention_unet': 
        return MegaUNet(
            input_shape = input_shape,
            target_shape = target_shape,
            input_shape_2 = input_shape_hr,
            convblock_opt = 'recurrent_residual_conv',
            layer_N = [64, 96, 128, 160],
            stack_num = 1, # Edit here
            reccur_iter= 2,
            activation = activation,
            ups_method = ups_method,
            attention_on = False,
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            last_conv_filters = 16,
            ) 

    elif model_id == 'srcnn':
        return SRCNN(
            input_shape = input_shape,
            target_shape = target_shape,
            input_shape_2 = input_shape_hr,
            n_classes = 1,
            output_activation='linear',
            last_kernel_size = 5,
            last_kernel_stride = 1,
            activation = activation,
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            )

    elif model_id == 'fsrcnn':
        return FSRCNN(
            input_shape = input_shape,
            target_shape = target_shape,
            ups_blocks_factors = (5, 2),
            input_shape_2 = input_shape_hr,
            output_activation='linear',
            n_classes = 1,
            last_kernel_size = 3, 
            last_kernel_stride = 1, 
            n = 128,
            d = 64, 
            s = 32,
            m = 4,
            activation = activation,
            ups_method = ups_method, 
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            )

    elif model_id == 'edrn':
        return EDRN(
            input_shape = input_shape,
            target_shape = target_shape,
            ups_blocks_factors = (5, 2),
            input_shape_2 = input_shape_hr,
            output_activation='linear',
            n_filters = 64,
            n_res_blocks = 16, 
            n_ups_filters = 128,
            n_classes = 1,
            last_kernel_size = 9,
            last_kernel_stride = 1,
            activation = activation,
            ups_method = ups_method, 
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            )

    elif model_id == 'srdrn':
        return SRDRN(
            input_shape = input_shape,
            target_shape = target_shape,
            ups_blocks_factors = (5, 2),
            input_shape_2 = input_shape_hr,
            output_activation='linear',
            n_filters = 64,
            n_res_blocks = 16, 
            n_ups_filters = 128,
            n_classes = 1,
            last_kernel_size = 9,
            last_kernel_stride = 1,
            activation = activation,
            ups_method = ups_method, 
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            )

    else:
        raise ValueError(
            f"Invalid model_id: {model_id}"
        )

#%% Test below

# model_names = ['srcnn', 'fsrcnn', 'edrn', 
#                 'srdrn', 'unet', 'attention_unet', 
#                 'recurrent_unet', 'residual_unet', 'recurrent_residual_attention_unet',]

# for model_name in model_names:
#     m = configure_model(
#         model_id = model_name, 
#         input_shape = (32, 32, 18),
#         target_shape = (320, 320, 1),
#         input_shape_hr = (320, 320, 1), 
#         activation = 'prelu',
#         ups_method = 'convtranspose',
#         add_input_noise = True,
#         input_noise_stddev = 0.1,    
#         )

#     m.summary()
#     print(m.name)
#     m.save(f"/home/midhunm/f{model_name}_noisy.keras")

#%%

from tensorflow.keras.models import load_model

def load_keras_model(model_path, custom_objects=None):
    """
    Load a Keras model from the specified path with optional custom objects.

    Parameters:
    - model_path (str): Path to the saved Keras model (HDF5 or SavedModel format).
    - custom_objects (dict, optional): Dictionary of custom objects used in the model.

    Returns:
    - model: Loaded Keras model.
    """
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return None

def load_pretrained_model(model_id, model_path):
    
    if model_id == 'unet': # U-Net
        print(f"\t[INFO] Loading ... 'UNET' model")
        return load_keras_model(f"{model_path}/p08a_q01_u01cnn_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras")
    
    elif model_id == 'attention_unet':
        print(f"\t[INFO] Loading ... 'ATTENTION-UNET' model")
        return load_keras_model(f"{model_path}/p08a_q01_u02att_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras")

    elif model_id == 'recurrent_unet':
        print(f"\t[INFO] Loading ... 'RECURRENT-UNET' model")
        return load_keras_model(f"{model_path}/p08a_q01_u03rec_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras")
    
    elif model_id == 'residual_unet':
        print(f"\t[INFO] Loading ... 'RESIDUAL-UNET' model")
        return load_keras_model(f"{model_path}/p08a_q01_u04res_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras")
    
    elif model_id == 'recurrent_residual_attention_unet':
        print(f"\t[INFO] Loading ... 'RECURRENT-RESIDUAL-UNET' model")
        return load_keras_model(f"{model_path}/p08a_q01_u04rra_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras")

    elif model_id == 'srcnn':
        print(f"\t[INFO] Loading ... 'SRCNN' model")
        return load_keras_model(f"{model_path}/p08a_q01_b01src_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras")

    elif model_id == 'fsrcnn':
        print(f"\t[INFO] Loading ... 'FSRCNN' model")
        return load_keras_model(f"{model_path}/p08a_q01_b02fsr_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras")

    elif model_id == 'edrn':
        print(f"\t[INFO] Loading ... 'EDRN' model")
        return load_keras_model(f"{model_path}/p08a_q01_b03edr_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras")

    elif model_id == 'srdrn':
        print(f"\t[INFO] Loading ... 'SRDRN' model")
        return load_keras_model(f"{model_path}/p08a_q01_b04srd_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras")
    else:
        raise ValueError(
            f"Invalid model_id: {model_id}"
        )
