'''
Advanced Machine Learning, 2024
Homework 1

Building predictors for brain-machine interfaces

Author: Andrew H. Fagg
Modified by: Alan Lee
Modified by: Brandon Michaud
'''
import numpy as np
from tensorflow import keras
import wandb
import socket

import pickle
import argparse
import os
import sys
from tensorflow.keras.utils import plot_model

from deep_networks import *
from symbiotic_metrics import *
from job_control import *


# Location for libraries (you will likely just use './')
tf_tools = "../../../../tf_tools/"

sys.path.append(tf_tools + "metrics")
sys.path.append(tf_tools + "networks")
sys.path.append(tf_tools + "experiment_control")


def extract_data(bmi, args):
    '''
    Translate BMI data structure from the file into a data set for training/evaluating a single model
    
    :param bmi: Dictionary containing the full BMI data set, as loaded from the pickle file.
    :param args: Argparse object, which contains key information, including Nfolds, 
            predict_dim, output_type, rotation
            
    :return: Numpy arrays in standard TF format for training set input/output, 
            validation set input/output and testing set input/output; and a
            dictionary containing the lists of folds that have been chosen
    '''
    # Number of folds in the data set
    ins = bmi['MI']
    Nfolds = len(ins)
    
    times = bmi['time']
    
    # Check that argument matches actual number of folds
    assert (Nfolds == args.Nfolds), "Nfolds must match folds in data set"
    
    # Pull out the data to be predicted
    outs = bmi[args.output_type]
    
    # Check that predict_dim is valid
    assert (args.predict_dim is None or (0 <= args.predict_dim < outs[0].shape[1]))
    
    # Rotation and number of folds to use for training
    r = args.rotation
    Ntraining = args.Ntraining
    
    # Compute which folds belong in which set
    folds_training = (np.array(range(Ntraining)) + r) % Nfolds
    folds_validation = (np.array([Nfolds-2]) + r) % Nfolds
    folds_testing = (np.array([Nfolds-1]) + r) % Nfolds
    
    # Log these choices
    folds = {'folds_training': folds_training, 'folds_validation': folds_validation, 'folds_testing': folds_testing}
    
    # Combine the folds into training/val/test data sets (pairs of input/output numpy arrays)
    ins_training = np.concatenate([ins[i] for i in folds_training], axis=0)
    outs_training = np.concatenate([outs[i] for i in folds_training], axis=0)
    time_training = np.concatenate([times[i] for i in folds_training], axis=0)
        
    ins_validation = np.concatenate([ins[i] for i in folds_validation], axis=0)
    outs_validation = np.concatenate([outs[i] for i in folds_validation], axis=0)
    time_validation = np.concatenate([times[i] for i in folds_validation], axis=0)
        
    ins_testing = np.concatenate([ins[i] for i in folds_testing], axis=0)
    outs_testing = np.concatenate([outs[i] for i in folds_testing], axis=0)
    time_testing = np.concatenate([times[i] for i in folds_testing], axis=0)
    
    # If a particular output dimension is specified, then extract it from the outputs
    if args.predict_dim is not None:
        outs_training = outs_training[:, [args.predict_dim]]
        outs_validation = outs_validation[:, [args.predict_dim]]
        outs_testing = outs_testing[:, [args.predict_dim]]
    
    return (ins_training, outs_training, time_training, ins_validation, outs_validation, time_validation, ins_testing,
            outs_testing, time_testing, folds)


def exp_type_to_hyperparameters(args):
    '''
    Translate the exp_type into a hyperparameter set

    :param args: ArgumentParser
    :return: Hyperparameter set (in dictionary form)
    '''
    if args.exp_type == 'bmi':
        p = {
            'rotation': range(20),
            'Ntraining': [1, 2, 3, 4, 5, 9, 13, 18]
        }
    else: 
        assert False, "Bad exp_type"

    return p


def augment_args(args):
    '''
    Use the jobiterator to override the command-line arguments based on the experiment index. 

    @return A string representing the selection of parameters to be used in the file name
    '''
    index = args.exp_index
    if index is None:
        # UPDATE
        return "Ntraining_%d_rotation_%d" % (args.Ntraining, args.rotation)
    
    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())
    
    # Check bounds
    assert (0 <= args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp_index, args)


def generate_fname(args, params_str):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.
    
    Expand this as needed
    '''
    # Hidden unit configuration
    hidden_str = '_'.join(str(x) for x in args.hidden)
    
    # Dimension being predicted
    if args.predict_dim is None:
        predict_str = args.output_type
    else:
        predict_str = '%s_%d' % (args.output_type, args.predict_dim)

    if args.L1_regularization is not None:
        Lx_str = '_L1_%f' % args.L1_regularization
        
    elif args.L2_regularization is not None:
        Lx_str = '_L2_%f' % args.L2_regularization
    else:
        Lx_str = ''
        
    # Put it all together, including #of training folds and the experiment rotation
    return "%s/%s_%s_%s%s_hidden_%s_%s" % (args.results_path, args.exp_type, args.label, predict_str, Lx_str,
                                           hidden_str, params_str)


def execute_exp(args=None):
    '''
    Perform the training and evaluation for a single model
    
    @args Argparse arguments
    '''
    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        # In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])
        
    # Modify the args if an exp_index is specified
    params_str = augment_args(args)
    
    print("Params:", params_str)
    
    # Compute output file name base
    fbase = generate_fname(args, params_str)
    
    print("File name base:", fbase)

    # Output pickle file name
    fname_out = "%s_results.pkl" % fbase

    # Check if this file exists
    if os.path.exists(fname_out):
        # File exists: abort the run
        print("File already exists")
        return None
    
    # Load the data
    bmi = None
    with open(args.dataset, "rb") as fp:
        bmi = pickle.load(fp)

    assert bmi is not None, "Unable to load data"

    # Extract the data sets.  This process uses rotation and Ntraining (among other exp args)
    (ins_training, outs_training, time_training, ins_validation, outs_validation, time_validation, ins_testing,
     outs_testing, time_testing, folds) = extract_data(bmi, args)

    # Is this a test run?
    if args.nogo:
        # Don't execute the experiment
        print("Test run only")
        return None
    
    # Start wandb
    run = wandb.init(project=args.project,
                     name=params_str,  # TODO
                     notes=fbase,
                     config=vars(args))
    # Log hostname
    wandb.log({'hostname': socket.gethostname()})

    # Metrics
    fvaf = FractionOfVarianceAccountedForSingle(outs_training.shape[1])
    rmse = tf.keras.metrics.RootMeanSquaredError()

    # Build the model
    model = deep_network_basic(ins_training.shape[1], args.hidden, outs_training.shape[1], activation=args.activation_hidden,
                               activation_output=args.activation_out, lrate=args.lrate, metrics=[fvaf, rmse])
    
    # Report if verbosity is turned on
    if args.verbose >= 1:
        print(model.summary())

    if args.render:
        fname = '%s_model_plot.png' % fbase
        plot_model(model, to_file=fname, show_shapes=True, show_layer_names=True)
        wandb.log({'model architecture': wandb.Image(fname)})
    
    # Callbacks
    cbs = []
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=args.min_delta,
                                                      patience=args.patience, verbose=args.verbose, mode='min')
    cbs.append(early_stopping_cb)

    # WandB callback
    wandb_metrics_cb = wandb.keras.WandbMetricsLogger()
    cbs.append(wandb_metrics_cb)
    
    # Learn
    history = model.fit(x=ins_training, y=outs_training,
                        epochs=args.epochs,
                        verbose=args.verbose >= 2,
                        validation_data=(ins_validation, outs_validation), 
                        callbacks=cbs)
        
    # Generate log data
    results = {}
    results['args'] = args

    # Task 1 data
    results['predict_testing'] = model.predict(ins_testing)
    results['actual_testing'] = outs_testing
    results['time_testing'] = time_testing

    # Task 2 data
    results['predict_training_fvaf'] = model.evaluate(ins_training, outs_training)[1]
    results['predict_validation_fvaf'] = model.evaluate(ins_validation, outs_validation)[1]
    results['predict_testing_fvaf'] = model.evaluate(ins_testing, outs_testing)[1]
    
    # Save results
    results['fname_base'] = fbase
    with open(fname_out, "wb") as fp:
        pickle.dump(results, fp)
    
    # Save the model (can't be included in the pickle file)
    if args.save:
        model.save("%s_model" % fbase)

    # Close WandB
    wandb.finish()
        
    return model


def create_parser():
    '''
    You will only use some of the arguments for HW1
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='BMI Learner', fromfile_prefix_chars='@')

    # Problem definition
    parser.add_argument('--dataset', type=str, default='/home/fagg/datasets/bmi/bmi_dataset.pkl', help='Data set file')
    parser.add_argument('--output_type', type=str, default='torque', help='Type to predict')
    parser.add_argument('--predict_dim', type=int, default=None, help="Dimension of the output to predict")
    parser.add_argument('--Nfolds', type=int, default=20, help='Maximum number of folds')

    # Network details
    parser.add_argument('--activation_out', type=str, default='sigmoid', help='Activation for output layer')
    parser.add_argument('--activation_hidden', type=str, default='sigmoid', help='Activation for hidden layers')
    parser.add_argument('--hidden', nargs='+', type=int, default=[10, 5], help='Number of hidden units per layer (sequence of ints)')

    # Experiment details
    parser.add_argument('--rotation', type=int, default=0, help='Cross-validation rotation')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--Ntraining', type=int, default=2, help='Number of training folds')

    # Meta experiment details
    parser.add_argument('--exp_type', type=str, default='bmi', help='High level name for this set of experiments; selects the specific Cartesian product')
    parser.add_argument('--exp_index', type=int, default=None, help='Experiment index for Cartesian experiment')
    parser.add_argument('--label', type=str, default='', help='Label used for fnames and WandB')

    # Training parameters
    parser.add_argument('--lrate', type=float, default=0.001, help="Learning rate")

    # Don't use these for HW 1
    parser.add_argument('--dropout', type=float, default=None, help="Dropout rate")
    parser.add_argument('--L1_regularization', '--l1', type=float, default=None, help="L1 regularization factor (only active if no L2)")
    parser.add_argument('--L2_regularization', '--l2', type=float, default=None, help="L2 regularization factor")

    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.001, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")

    # Computer config
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--cpus_per_task', type=int, default=None, help='Number of threads to use')

    # Results
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    
    parser.add_argument('--save', action='store_true', help='Save model')
    parser.add_argument('--render', action='store_true', help='Render the model')

    # Execution control
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--check', action='store_true', help='Check results for completeness')

    # WandB
    parser.add_argument('--project', type=str, default='hw1', help='WandB project name')
    
    return parser


def check_args(args):
    '''
    Check that key arguments are within appropriate bounds.
    Failing an assert causes a hard failure with meaningful output
    '''
    assert (0 <= args.rotation < args.Nfolds), "Rotation must be between 0 and Nfolds"
    assert (1 <= args.Ntraining <= (args.Nfolds - 2)), "Ntraining must be between 1 and Nfolds-2"
    assert (0.0 < args.lrate < 1), "Lrate must be between 0 and 1"


def check_completeness(args):
    '''
    Check the completeness of a Cartesian product run.

    All other args should be the same as if you executed your batch, however, the '--check' flag has been set

    Prints a report of the missing runs, including both the exp_index and the name of the missing results file

    :param args: ArgumentParser
    '''
    # Get the corresponding hyperparameters
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)

    print("Total jobs: %d" % ji.get_njobs())

    print("MISSING RUNS:")

    indices = []
    # Iterate over all possible jobs
    for i in range(ji.get_njobs()):
        params_str = ji.set_attributes_by_index(i, args)
        # Compute output file name base
        fbase = generate_fname(args, params_str)
    
        # Output pickle file name
        fname_out = "%s_results.pkl"%(fbase)

        if not os.path.exists(fname_out):
            # Results file does not exist: report it
            print("%3d\t%s" % (i, fname_out))
            indices.append(i)

    # Give the list of indices that can be inserted into the --array line of the batch file
    print("Missing indices (%d): %s" % (len(indices), ','.join(str(x) for x in indices)))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    # Turn off GPUs?
    if not args.gpu or "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        tf.config.set_visible_devices([], 'GPU')
        print('NO VISIBLE DEVICES!!!!')

    # GPU check
    visible_devices = tf.config.get_visible_devices('GPU')
    n_visible_devices = len(visible_devices)

    print('GPUS:', visible_devices)
    if n_visible_devices > 0:
        for device in visible_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print('We have %d GPUs\n' % n_visible_devices)
    else:
        print('NO GPU')

    # Set number of threads, if it is specified
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)

    # Which job to do?
    if args.check:
        # Just look at which results files have NOT been created yet
        check_completeness(args)
    else:
        # Do the work
        execute_exp(args)
