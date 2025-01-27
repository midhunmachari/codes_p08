from ai4klima.tensorflow.models import MegaUNet

def configure_model(model_id, 
                    inputs_shape, 
                    input_shape_hr, 
                    add_input_noise=False,
                    input_noise_stddev=0.1,
                    activation='prelu', 
                    batchnorm_on=True, 
                    ups_method='bilinear'
                    ):

    """
    Defaults: MegaUNet(
                input_shape,
                input_shape_hr=None,
                lr_ups_size=None,
                n_classes=1,
                output_activation='linear',

                # Convolution-related settings
                convblock_opt='recurrent_residual_conv',
                layer_N=[64, 96, 128, 160],
                kernel_size=(3, 3),
                stack_num=2,
                activation='relu',
                initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                regularizer=tf.keras.regularizers.l2(0.01),
                batchnorm_on=True,

                # Pooling and upsampling settings
                pool_opt='maxpool',
                pool_size=2,
                ups_method='bilinear',

                # Attention mechanism
                attention_on=True,
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
            inputs_shape = inputs_shape, 
            input_shape_hr = input_shape_hr, 
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            activation = activation, 
            batchnorm_on = batchnorm_on, 
            ups_method = ups_method,

            )
    
    elif model_id == 'aunet': # Attention-U-Net 
        return MegaUNet(
            inputs_shape = inputs_shape, 
            input_shape_hr = input_shape_hr, 
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            activation = activation, 
            batchnorm_on = batchnorm_on, 
            ups_method = ups_method,
            ) 

    elif model_id == 'rcunet': # Recurrent-U-Net
        return MegaUNet(
            inputs_shape = inputs_shape, 
            input_shape_hr = input_shape_hr, 
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            activation = activation, 
            batchnorm_on = batchnorm_on, 
            ups_method = ups_method, 
            ) 
    
    elif model_id == 'arcunet': # Attention-Recurrent-U-Net
        return MegaUNet(
            inputs_shape = inputs_shape, 
            input_shape_hr = input_shape_hr, 
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            activation = activation, 
            batchnorm_on = batchnorm_on, 
            ups_method = ups_method,
            ) 
    
    elif model_id == 'rsunet': # Residual-U-Net 
        return MegaUNet(
            inputs_shape = inputs_shape, 
            input_shape_hr = input_shape_hr, 
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            activation = activation, 
            batchnorm_on = batchnorm_on, 
            ups_method = ups_method,
            ) 
    
    elif model_id == 'arsunet': # Attention-Residual-U-Net
        return MegaUNet(
            inputs_shape = inputs_shape, 
            input_shape_hr = input_shape_hr, 
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            activation = activation, 
            batchnorm_on = batchnorm_on, 
            ups_method = ups_method,
            ) 
    elif model_id == 'r2unet': # Sigmoid Discriminator
        return MegaUNet(
            inputs_shape = inputs_shape, 
            input_shape_hr = input_shape_hr, 
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            activation = activation, 
            batchnorm_on = batchnorm_on, 
            ups_method = ups_method,
            )
    
    elif model_id == 'ar2unet': # Sigmoid Discriminator
        return MegaUNet(
            inputs_shape = inputs_shape, 
            input_shape_hr = input_shape_hr, 
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            activation = activation, 
            batchnorm_on = batchnorm_on, 
            ups_method = ups_method,
            ) 
    
    elif model_id == 'r2unet2': # Sigmoid Discriminator
        return MegaUNet(
            inputs_shape = inputs_shape, 
            input_shape_hr = input_shape_hr, 
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            activation = activation, 
            batchnorm_on = batchnorm_on, 
            ups_method = ups_method,
            ) 
    elif model_id == 'ar2unet2': # Sigmoid Discriminator
        return MegaUNet(
            inputs_shape = inputs_shape, 
            input_shape_hr = input_shape_hr, 
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            activation = activation, 
            batchnorm_on = batchnorm_on, 
            ups_method = ups_method,
            ) 
    
    
    else:
        raise ValueError(
            f"Invalid model_id: {model_id}. Expected one of: "
            "['unet', 'aunet', 'rcunet', 'arcunet', 'rsunet', 'arsunet']."
        )

         