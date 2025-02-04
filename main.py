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
    'u01_cnn1': 'unet',
    'u02_att1': 'unet-att',
    'u03_rec1': 'unet-rec',
    'u04_res1': 'unet-res', 
    'u05_r2a1': 'unet-r2a',
    }

losses_dict = {
    'wmae': weighted_mae,
    'omae' : MeanAbsoluteError()
    }

lr_dict = {
    'r7e4' : 7e-4,
    'r1e4' : 1e-4,
    }

bs_dict = {
    'b32' : 32,
    }


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add the arguments to the parser
    parser.add_argument('--pwd', type=str, required=True, help='Run/Job submission directory')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix for the experiment run')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train') 
    parser.add_argument('--model_id', type=str, required=True, help='one of Model ID: u01 ... u10')

    # Parse the command-line arguments
    args = parser.parse_args()

    #### EDIT BELOW ####
    REFD_PATH = "/nlsasfs/home/precipitation/midhunm/AI4KLIM/DATASET/DATA_IND32M/IND32M_010_GRID.nc"
    DATA_PATH = "/nlsasfs/home/precipitation/midhunm/AI4KLIM/DATASET/DATA_IND32M"
    SAVE_PATH = f"{args.pwd}/.."
    
    #### EDIT ABOVE ####
    RunExperiment(
        prefix = args.prefix, 
        model_id = args.model_id,
        data_path = DATA_PATH, 
        save_path = SAVE_PATH, 
        refd_path = REFD_PATH, 
        epochs = args.epochs,
        models_dict = models_dict,
        losses_dict = losses_dict, 
        lr_dict = lr_dict, 
        bs_dict = bs_dict
    )