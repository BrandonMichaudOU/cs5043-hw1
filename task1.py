import matplotlib.pyplot as plt
import pickle


def make_plot():
    '''
    Function to make plot of elbow acceleration vs time
    Displays both actual and predicted
    '''
    # Open results for task 1
    with open("results/bmi__ddtheta_1_hidden_100_10_Ntraining_18_rotation_10_results.pkl", "rb") as fp:
        obj = pickle.load(fp)

    # Create line plot
    fig = plt.figure()
    plt.plot(obj['time_testing'], obj['predict_testing'], label='predicted')
    plt.plot(obj['time_testing'], obj['actual_testing'], label='actual')
    plt.xlim([1310, 1317])
    plt.ylabel('Elbow Acceleration')
    plt.xlabel('Time')
    plt.title('Elbow Acceleration vs Time')
    plt.legend()
    fig.savefig('task1.png')


if __name__ == '__main__':
    make_plot()
