
import argparse

# copy pasted sorry :)
def get_processing_args() -> argparse.Namespace:
    """Get the arguments needed for the processing script."""
    # This is the standard way to define command line arguments in python,
    # without using config files which you are welcome to do!
    # Here the argparse module can pick up command line arguments and return them
    # So if you type in `python train.py --num_mlp_layers 3` it will save the value 3
    # to `args.num_mlp_layers`.
    # This allows you to change the model without changing the code!
    # Each possible argument must be defined here.
    # Feel free to add more arguments as you see fit.

    # First we have to create a parser object
    parser = argparse.ArgumentParser()

    # Define the important paths for the project
    parser.add_argument(
        "--data_path",  # How we access the argument when calling `python train.py ...`
        type=str,  # We must also define the type of argument, here it is a string
        default="/home/yannh/Documents/uni/phd/classes/pai/costum_project/dataset/raw_standardized/J20_1MS.csv",  # The default value so you dont have to type it in every time
        help="Where is saved the raw dataset",  # A helpfull message
    )
    
    parser.add_argument(
        "--processed_path",
        type=str,
        default=None,
        help="Where to save the processed dataset",
    )

    # Arguments for the network
    parser.add_argument(
        "--labels_type",
        type=str,
        default=None,
        help="The type of labels",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The seed for the random split",
    )
   
    # This now collects all arguments
    args = parser.parse_args()

    # Now we return the arguments
    return args


def get_training_args() -> argparse.Namespace:
    """Get the arguments needed for the training script."""
    parser = argparse.ArgumentParser()

    # Define the important paths for the project
    parser.add_argument(
        "--processed_path",  # How we access the argument when calling `python train.py ...`
        type=str,  # We must also define the type of argument, here it is a string
        default="/home/yannh/Documents/uni/phd/classes/pai/costum_project/dataset/raw_standardized/J20_1MS.csv",  # The default value so you dont have to type it in every time
        help="Where is saved the raw dataset",  # A helpfull message
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The model to use",
    )
    
    parser.add_argument(
        "--hidden_dim",
        type=str,
        default="64",
        help="The model hidden dimensions",
    )
    
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="MSE",
        help="The loss to use",
    )
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="The optimizer to use",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The random seed",
    )

    # Arguments for the network
    parser.add_argument(
        "--learning_rate", "--lr",
        type=float,
        default=1e-4,
        help="The learning rate of the model",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="The number of epoch to train the model",
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=2/3,
        help="The training/(validation+test) ratio",
    )
    
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=2/3,
        help="The validation/test ratio",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size",
    )
   
   
    # This now collects all arguments

    args = parser.parse_args()
    args.hidden_dim = [int(h) for h in args.hidden_dim.split(",")]

    # Now we return the arguments
    return args


def get_testing_args() -> argparse.Namespace:
    """Get the arguments needed for the training script."""
    parser = argparse.ArgumentParser()

    # Define the important paths for the project
    parser.add_argument(
        "--processed_path",  # How we access the argument when calling `python train.py ...`
        type=str,  # We must also define the type of argument, here it is a string
        default="/home/yannh/Documents/uni/phd/classes/pai/costum_project/dataset/raw_standardized/J20_1MS.csv",  # The default value so you dont have to type it in every time
        help="Where is saved the raw dataset",  # A helpfull message
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The model to use",
    )
    
    parser.add_argument(
        "--hidden_dim",
        type=str,
        default="64",
        help="The model hidden dimensions",
    )
    
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="MSE",
        help="The loss to use",
    )
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="The optimizer to use",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The random seed",
    )

    # Arguments for the network
    parser.add_argument(
        "--learning_rate", "--lr",
        type=float,
        default=1e-4,
        help="The learning rate of the model",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="The number of epoch to train the model",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size",
    )
   
   
    # This now collects all arguments
    args = parser.parse_args()
    args.hidden_dim = [int(h) for h in args.hidden_dim.split(",")]

    # Now we return the arguments
    return args