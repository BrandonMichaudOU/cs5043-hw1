import matplotlib.pyplot as plt
import numpy as np
import pickle
import wandb
import socket


def make_plot():
    rotations = range(20)
    Ntraining = [1, 2, 3, 4, 5, 9, 13, 18]
    fvafs_training = np.empty((len(rotations), len(Ntraining)))
    fvafs_validation = np.empty((len(rotations), len(Ntraining)))
    fvafs_testing = np.empty((len(rotations), len(Ntraining)))

    for r in rotations:
        for n in Ntraining:
            with open(f'bmi__ddtheta_1_hidden_100_10_Ntraining_{n}_rotation_{r}_results.pkl', "rb") as fp:
                results = pickle.load(fp)
                fvafs_training[r][n] = results['predict_training_fvaf']
                fvafs_validation[r][n] = results['predict_validation_fvaf']
                fvafs_testing[r][n] = results['predict_testing_fvaf']

    avg_fvafs_training = np.average(fvafs_training, axis=0)
    avg_fvafs_validation = np.average(fvafs_validation, axis=0)
    avg_fvafs_testing = np.average(fvafs_testing, axis=0)

    # Create line plot
    fig = plt.figure()
    plt.plot(Ntraining, avg_fvafs_training, label='training')
    plt.plot(Ntraining, avg_fvafs_validation, label='validation')
    plt.plot(Ntraining, avg_fvafs_testing, label='testing')
    plt.ylabel('FVAF')
    plt.xlabel('Training Folds')
    plt.title('FVAF vs Training Folds')
    plt.legend()
    fig.savefig('task2.png')


if __name__ == '__main""':
    make_plot()
