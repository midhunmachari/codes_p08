from ai4klima.tensorflow.models import MegaUNet, SRCNN, FSRCNN, EDRN, SRDRN
from ai4klima.tensorflow.losses import weighted_mae
from tensorflow.keras.models import load_model
import tensorflow as tf

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


def load_keras_model(model_path, custom_objects={"weighted_mae": weighted_mae}):
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
        return load_keras_model(f"{model_path}/p08a_q01_u05rra_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras")

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
    
def load_pretrained_model_path(model_id, model_path):

    if model_id == 'unet': # U-Net
        model_path = f"{model_path}/p08a_q01_u01cnn_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras"
        print(f"\t[INFO] Loading ... 'UNET' model from {model_path}")
        return model_path
    
    elif model_id == 'attention_unet':
        model_path = f"{model_path}/p08a_q01_u02att_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras"
        print(f"\t[INFO] Loading ... 'ATTENTION-UNET' model from {model_path}")
        return model_path

    elif model_id == 'recurrent_unet':
        model_path = f"{model_path}/p08a_q01_u03rec_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras"
        print(f"\t[INFO] Loading ... 'RECURRENT-UNET' model from {model_path}")
        return model_path
    
    elif model_id == 'residual_unet':
        model_path = f"{model_path}/p08a_q01_u04res_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras"
        print(f"\t[INFO] Loading ... 'RESIDUAL-UNET' model from {model_path}")
        return model_path
    
    elif model_id == 'recurrent_residual_attention_unet':
        model_path = f"{model_path}/p08a_q01_u04rra_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras"
        print(f"\t[INFO] Loading ... 'RECURRENT-RESIDUAL-UNET' model from {model_path}")
        return model_path

    elif model_id == 'srcnn':
        model_path = f"{model_path}/p08a_q01_b01src_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras"
        print(f"\t[INFO] Loading ... 'SRCNN' model from {model_path}")
        return model_path

    elif model_id == 'fsrcnn':
        model_path = f"{model_path}/p08a_q01_b02fsr_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras"
        print(f"\t[INFO] Loading ... 'FSRCNN' model from {model_path}")
        return model_path

    elif model_id == 'edrn':
        model_path = f"{model_path}/p08a_q01_b03edr_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras"
        print(f"\t[INFO] Loading ... 'EDRN' model from {model_path}")
        return model_path

    elif model_id == 'srdrn':
        model_path = f"{model_path}/p08a_q01_b04srd_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras"
        print(f"\t[INFO] Loading ... 'SRDRN' model from {model_path}")
        return model_path
    
    else:
        raise ValueError(
            f"Invalid model_id: {model_id}"
        )
    
def load_keras_model_finetuner(model_path, custom_objects=None, unfreeze_layers=None):
    """
    Load a Keras model and prepare it for fine-tuning.

    Parameters:
    - model_path (str): Path to the saved Keras model (HDF5 or SavedModel format).
    - custom_objects (dict, optional): Dictionary of custom objects used in the model.
    - unfreeze_layers (int or list, optional): Number of last layers to unfreeze, or list of layer names to unfreeze.
    - lr (float, optional): Learning rate for fine-tuning.

    Returns:
    - model: Loaded and modified Keras model ready for fine-tuning.
    """
    try:
        # Load the model
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded successfully from {model_path}")

        # Unfreeze layers for fine-tuning
        if unfreeze_layers:
            if isinstance(unfreeze_layers, int):
                for layer in model.layers[-unfreeze_layers:]:
                    layer.trainable = True
            elif isinstance(unfreeze_layers, list):
                for layer in model.layers:
                    if layer.name in unfreeze_layers:
                        layer.trainable = True
            else:
                raise ValueError("unfreeze_layers must be an int (number of layers) or a list of layer names.")
            
            print(f"Unfroze {unfreeze_layers} layers for fine-tuning.")

        return model

    except Exception as e:
        print(f"Failed to load and prepare model from {model_path}: {e}")
        return None

def load_pretrained_model_for_finetune_unfreeze_layers(model_id, model_path, unfreeze_layers=None):

    if model_id == 'unet': # U-Net
        print(f"\t[INFO] Loading ... 'UNET' model")
        return load_keras_model_finetuner(f"{model_path}/p08a_q01_u01cnn_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae}, unfreeze_layers=unfreeze_layers)
    
    elif model_id == 'attention_unet':
        print(f"\t[INFO] Loading ... 'ATTENTION-UNET' model")
        return load_keras_model_finetuner(f"{model_path}/p08a_q01_u02att_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae}, unfreeze_layers=unfreeze_layers)

    elif model_id == 'recurrent_unet':
        print(f"\t[INFO] Loading ... 'RECURRENT-UNET' model")
        return load_keras_model_finetuner(f"{model_path}/p08a_q01_u03rec_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae}, unfreeze_layers=unfreeze_layers)
    
    elif model_id == 'residual_unet':
        print(f"\t[INFO] Loading ... 'RESIDUAL-UNET' model")
        return load_keras_model_finetuner(f"{model_path}/p08a_q01_u04res_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae}, unfreeze_layers=unfreeze_layers)
    
    elif model_id == 'recurrent_residual_attention_unet':
        print(f"\t[INFO] Loading ... 'RECURRENT-RESIDUAL-UNET' model")
        return load_keras_model_finetuner(f"{model_path}/p08a_q01_u05rra_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae}, unfreeze_layers=unfreeze_layers)

    elif model_id == 'srcnn':
        print(f"\t[INFO] Loading ... 'SRCNN' model")
        return load_keras_model_finetuner(f"{model_path}/p08a_q01_b01src_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae}, unfreeze_layers=unfreeze_layers)

    elif model_id == 'fsrcnn':
        print(f"\t[INFO] Loading ... 'FSRCNN' model")
        return load_keras_model_finetuner(f"{model_path}/p08a_q01_b02fsr_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae}, unfreeze_layers=unfreeze_layers)

    elif model_id == 'edrn':
        print(f"\t[INFO] Loading ... 'EDRN' model")
        return load_keras_model_finetuner(f"{model_path}/p08a_q01_b03edr_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae}, unfreeze_layers=unfreeze_layers)

    elif model_id == 'srdrn':
        print(f"\t[INFO] Loading ... 'SRDRN' model")
        return load_keras_model_finetuner(f"{model_path}/p08a_q01_b04srd_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae}, unfreeze_layers=unfreeze_layers)
    else:
        raise ValueError(
            f"Invalid model_id: {model_id}"
        )


def residual_block(x, filters=[128, 64], kernel_size=3, bn=False):
    """
    Residual block with Conv2D, Batch Normalization, and PReLU activations.

    Parameters:
    - x: Input tensor
    - filters (int): Number of filters for Conv2D layers.
    - kernel_size (int): Kernel size for Conv2D layers.

    Returns:
    - x: Output tensor after applying the residual block.
    """
    shortcut = x  # Save input tensor for residual connection

    # First Conv2D → BatchNorm → PReLU
    x = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(kernel_size, kernel_size), padding="same")(x)
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)  # Batch Norm for stability
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    # Second Conv2D → BatchNorm → PReLU
    x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(kernel_size, kernel_size), padding="same")(x)
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)  # Batch Norm for stability
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    # Residual connection (skip connection)
    x = tf.keras.layers.Add()([x, shortcut])  

    return x


def load_keras_model_finetuner_with_newlayer(model_path, custom_objects=None, num_res_blocks=1, rb_kernel_size=3, last_kernel_size=1):
    """
    Load a Keras model, unfreeze specified layers, add a convolutional block on top, and prepare for fine-tuning.

    Parameters:
    - model_path (str): Path to the saved Keras model (HDF5 or SavedModel format).
    - custom_objects (dict, optional): Dictionary of custom objects used in the model.
    - unfreeze_layers (int or list, optional): Number of last layers to unfreeze, or list of layer names to unfreeze.
    - conv_filters (int, optional): Number of filters for Conv2D layers in the added block.
    - kernel_size (int, optional): Kernel size for Conv2D layers.
    - lr (float, optional): Learning rate for fine-tuning.

    Returns:
    - model: Modified Keras model ready for fine-tuning.
    """
    try:
        # Load the base model
        base_model = load_model(model_path, custom_objects=custom_objects)
        print(f"\t[INFO] Base Model loaded successfully from {model_path}")
        
        # Freeze the entire base model
        base_model.trainable = False
        for layer in base_model.layers:
            layer.trainable = False
        print("\t[INFO] Base Model is now frozen (non-trainable).")

        # Get base model's output as input for the new layers
        x0 = x = base_model.output

        for _ in range(num_res_blocks):
            x = residual_block(x, filters=[128, 128], kernel_size=rb_kernel_size, bn=False)
        if num_res_blocks > 1:
            x = tf.keras.layers.Add()([x, x0])  
        x = tf.keras.layers.Conv2D(filters = 1, kernel_size = (last_kernel_size, last_kernel_size), padding="same")(x)

        # Define the new model
        model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
        print(f"\t[INFO] Model is ready for fine-tuning with {num_res_blocks} residual block.")

        return model

    except Exception as e:
        print(f"Failed to load and prepare model from {model_path}: {e}")
        return None
    
def load_pretrained_model_for_finetune(model_id, model_path):

    if model_id == 'unet': # U-Net
        print(f"\t[INFO] Loading ... 'UNET' model")
        return load_keras_model_finetuner_with_newlayer(f"{model_path}/p08a_q01_u01cnn_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae})
    
    elif model_id == 'attention_unet':
        print(f"\t[INFO] Loading ... 'ATTENTION-UNET' model")
        return load_keras_model_finetuner_with_newlayer(f"{model_path}/p08a_q01_u02att_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae})

    elif model_id == 'recurrent_unet':
        print(f"\t[INFO] Loading ... 'RECURRENT-UNET' model")
        return load_keras_model_finetuner_with_newlayer(f"{model_path}/p08a_q01_u03rec_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae})
    
    elif model_id == 'residual_unet':
        print(f"\t[INFO] Loading ... 'RESIDUAL-UNET' model")
        return load_keras_model_finetuner_with_newlayer(f"{model_path}/p08a_q01_u04res_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae})
    
    elif model_id == 'recurrent_residual_attention_unet':
        print(f"\t[INFO] Loading ... 'RECURRENT-RESIDUAL-UNET' model")
        return load_keras_model_finetuner_with_newlayer(f"{model_path}/p08a_q01_u05rra_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae})

    elif model_id == 'srcnn':
        print(f"\t[INFO] Loading ... 'SRCNN' model")
        return load_keras_model_finetuner_with_newlayer(f"{model_path}/p08a_q01_b01src_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae})

    elif model_id == 'fsrcnn':
        print(f"\t[INFO] Loading ... 'FSRCNN' model")
        return load_keras_model_finetuner_with_newlayer(f"{model_path}/p08a_q01_b02fsr_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae})

    elif model_id == 'edrn':
        print(f"\t[INFO] Loading ... 'EDRN' model")
        return load_keras_model_finetuner_with_newlayer(f"{model_path}/p08a_q01_b03edr_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae})

    elif model_id == 'srdrn':
        print(f"\t[INFO] Loading ... 'SRDRN' model")
        return load_keras_model_finetuner_with_newlayer(f"{model_path}/p08a_q01_b04srd_wmae_era5_mswx_r7e4_b08_ckpt_best_gen.keras",
                                                        custom_objects={'weighted_mae': weighted_mae})
    else:
        raise ValueError(
            f"Invalid model_id: {model_id}"
        )