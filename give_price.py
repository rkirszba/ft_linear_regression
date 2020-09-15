import sys
import pickle
from utils import estimate_price

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
        f = open('thetas.pkl', 'rb')
        thetas = pickle.load(f)
        theta0 = thetas[0]
        theta1 = thetas[1]
    except:
        theta0 = 0
        theta1 = 0
    
    estimated_price = estimate_price(theta0, theta1, mileage)
    print('The estimated price for a car with mileage {} is:\n{}'.format(mileage, estimated_price))