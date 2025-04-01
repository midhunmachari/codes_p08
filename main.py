import gc
import tensorflow as tf
import argparse
from ai4klima.tensorflow.losses import weighted_mae, MeanAbsoluteError
from runexp import RunExperiment

# Clear the session to release memory
tf.keras.backend.clear_session()

# List all GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Enable memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Enable memory growth
        print("Memory growth enabled for all GPUs.")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")
else:
    print("No GPUs found.")

# Force garbage collection to clear any leftover objects
gc.collect()

# Enable eager execution (if debugging is required)
tf.config.run_functions_eagerly(False)  # Set to False for better performance during training

# Check GPU availability
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow **IS** using the GPU")
else:
    print("TensorFlow **IS NOT** using the GPU")

########################################## EXECUTION SNIPPET BELOW #########################################

models_dict = {
    'b01src': 'srcnn',
    'b02fsr': 'fsrcnn',
    'b03edr': 'edrn',
    'b04srd': 'srdrn',
    'u01cnn': 'unet',
    'u02att': 'attention_unet',
    'u03rec': 'recurrent_unet',
    'u04res': 'residual_unet', 
    'u05rra': 'recurrent_residual_attention_unet',
    }

losses_dict = {
    'wmae': weighted_mae,
    # 'omae' : MeanAbsoluteError()
    }

lr_dict = {
    'r2e4' : 2e-4,
    }

bs_dict = {
    'b08' : 8,
    }


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add the arguments to the parser
    parser.add_argument('--pwd', type=str, required=True, help='Run/Job submission directory')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix for the experiment run')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train') 

    # Parse the command-line arguments
    args = parser.parse_args()

    #### EDIT BELOW ####
    REFD_PATH = "/nlsasfs/home/precipitation/midhunm/AI4KLIM/DATASET/DATA_IND32M/IND32M_010_GRID.nc"
    DATA_PATH = "/nlsasfs/home/precipitation/midhunm/AI4KLIM/DATASET/DATA_IND32M"
    MODEL_PATH = "/nlsasfs/home/precipitation/midhunm/AI4KLIM/EXPMNTS/pretrained_weights"
    SAVE_PATH = f"{args.pwd}/.."
    
    #### EDIT ABOVE ####
    RunExperiment(
        prefix = args.prefix, 
        data_path = DATA_PATH, 
        save_path = SAVE_PATH, 
        model_path = MODEL_PATH,
        refd_path = REFD_PATH, 
        epochs = args.epochs,
        models_dict = models_dict,
        losses_dict = losses_dict, 
        lr_dict = lr_dict, 
        bs_dict = bs_dict,
    )