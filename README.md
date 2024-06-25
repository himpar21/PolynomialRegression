# Polynomial Regression with Python

This repository contains code to perform polynomial regression on a dataset, plot the results, and evaluate the model's accuracy.

## Dataset

The dataset should be a CSV file named `data.csv` with two columns:
- `Temperature`: The feature(s)
- `Sales`: The target variable

## Requirements

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

You can install the required packages using:

```sh
pip install pandas numpy matplotlib scikit-learn
```

## Usage
Clone the repository and navigate to the project directory:

```sh

git clone https://github.com/himpar21/PolynomialRegression.git
cd PolynomialRegression
```

Place your data.csv file in the project directory.

Run the script polynomialregression.py:


```sh
python polynomialregression.py
```


## Results
The script will plot the polynomial regression curve and the training and testing data points. It will also print out the training and testing RMSE (Root Mean Squared Error) and R-squared values to evaluate the model's accuracy.
```sh
Training RMSE: 3.09
Testing RMSE: 3.62
R-squared (Training): 0.94
R-squared (Testing): 0.88
```
![jij](https://github.com/himpar21/PolynomialRegression/assets/95409033/fe76e300-f504-4f05-99b7-0bfa37bd2be6)


## License
This project is licensed under the MIT License - see the LICENSE file for details.
