# ft_linear_regression
42 project ft_linear_regression

## Description
This program trains a model, using a linear regression, in order to be able to predict the price of car thanks to its mileage.
The purpose of this project is to code from scratch the functions allowing to perform the linear regression.
In addition, some plot features have been set up in order to visualize correctly the data.

## Installing Python 3

This process allows to install python on a mac machine on which you are not root. It is taken from 42AI (https://github.com/42-AI) process.

1. Copy paste the following code into your shell rc file (for, example: `~/.zshrc`).

```bash
function set_conda {
    HOME=$(echo ~)
    INSTALL_PATH="/INSTALL/PATH"
    MINICONDA_PATH=$INSTALL_PATH"/miniconda3/bin"
    PYTHON_PATH=$(which python)
    SCRIPT="Miniconda3-latest-MacOSX-x86_64.sh"
    DL_LINK="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"

    if echo $PYTHON_PATH | grep -q $INSTALL_PATH; then
	    echo "good python version :)"
    else
	cd
	if [ ! -f $SCRIPT ]; then
		curl -LO $DL_LINK
    	fi
    	if [ ! -d $MINICONDA_PATH ]; then
	    	sh $SCRIPT -b -p $INSTALL_PATH"/miniconda3"
	fi
	clear
	echo "Which python:"
	which python
	if grep -q "^export PATH=$MINICONDA_PATH" ~/.zshrc
	then
		echo "export already in .zshrc";
	else
		echo "adding export to .zshrc ...";
		echo "export PATH=$MINICONDA_PATH:\$PATH" >> ~/.zshrc
	fi
	source ~/.zshrc
    fi
}
```

2. Source your `.zshrc` with the following command:

```bash
source ~/.zshrc
```

3. Use the function `set_conda`:

```bash
set_conda
```

When the installation is done rerun the `set_conda` function.


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





