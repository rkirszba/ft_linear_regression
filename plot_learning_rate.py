import sys
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    try:
        f = open('learning_info.pkl', 'rb')
        info = pickle.load(f)
        plt.plot(info['costs'])
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.show()

    except Exception as e:
        print('Something went wrong: {}'.format(e))
        sys.exit(1)