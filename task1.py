import matplotlib.pyplot as plt
import pickle
import wandb
import socket


def make_plot():
    with open("results/bmi__torque_1_hidden_100_10_Ntraining_18_rotation_10_results.pkl", "rb") as fp:
        obj = pickle.load(fp)

    # Create histogram
    fig = plt.figure()
    plt.plot(obj['time_testing'], obj['predict_testing'], label='predicted')
    plt.plot(obj['time_testing'], obj['actual_testing'], label='actual')
    plt.xlim([1310, 1317])
    plt.ylabel('Elbow Acceleration')
    plt.xlabel('Time')
    plt.title('Elbow Acceleration vs Time')
    plt.legend()
    fig.savefig('task1.png')

    # Initialize WandB
    wandb.init(project='hw1', name='task1_figure')

    # Log histogram to WandB
    wandb.log({'hostname': socket.gethostname()})
    wandb.log({'figure': fig})

    # Close WandB
    wandb.finish()


if __name__ == '__main__':
    make_plot()
