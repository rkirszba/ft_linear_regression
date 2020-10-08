# ft_linear_regression
📚 42 project ft_linear_regression

## Description
This program trains a model, using a linear regression, in order to be able to predict the price of a car thanks to its mileage.
The purpose of this project is to code from scratch the functions allowing to perform the linear regression.
In addition, some plot features have been set up in order to visualize correctly the data.


## Installing packages
At the root of the project, you can launch one of the following commands:

```bash
pip install -r requirements.txt
```

or

```bash
conda install --file requirements.txt
```

## Use

### Train the model

```bash
python train.py
```

This program trains a model thanks to the data written in data.csv file.
Then it pickles the model in a my_model.pkl file.

### Predict

```bash
python estimate_price.py
```

This script allows you to predict a car price thanks to its mileage you give it.

### Plot learning curves

```bash
python plot_learning_curve.py
```

This script allows you to visualize how the gradient descent worked.

### Train a model with Scikit Learn and then compare with my model

```bash
python sklearn_train.py
```

This script trains a model with Scikit Learn.

```bash
python plot_data_and_models.py
```

You can now visualize the data, your predicting line and Scikit Learn one

```bash
python compare_metrics
```

You also can compare metrics between the two models





