import sys
import pickle
import numpy as np
from ft_linear_regression import FTLinearRegression

if __name__ == '__main__':

    while True:
        user_input = input('Please give the mileage of the car\n')
        try:
            mileage = float(user_input)
            if mileage < 0:
                raise ValueError()
            break
        except ValueError:
            print('\'{}\' is not a correct input'.format(user_input))
            continue
        except KeyboardInterrupt:
            sys.exit(0)

    try:
        f = open('my_model.pkl', 'rb')
        model = pickle.load(f)
        X = np.array([[mileage]])
        estimated_price = np.squeeze(model.predict(X))
        print('The estimated price for a car with mileage {} is:\n{}'.format(mileage, estimated_price))
    except:
        model = FTLinearRegression()
        X = np.array([[mileage]])
        estimated_price = np.squeeze(model.predict(X))
        print('The estimated price for a car with mileage {} is:\n{}'.format(mileage, estimated_price))