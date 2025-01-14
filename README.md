# Linear-Regression

Linear Regression and Visualization
With Jupyter Notebook

This Jupyter Notebook demonstrates how to perform linear regression and visualize the results using matplotlib and scikit-learn. It includes examples of plotting linear and quadratic functions, fitting a linear regression model, and evaluating its performance.

Table of Contents
Prerequisites

Getting Started

Running the Code

Code Explanation

Results

License

Prerequisites
Before running the code, ensure you have the following installed:

Python 3.x

Required Python libraries:

bash
Copy
pip install numpy matplotlib scikit-learn
Jupyter Notebook (to run the .ipynb file).

Getting Started
Launch Jupyter Notebook
Start Jupyter Notebook:

bash
Copy
jupyter notebook
Open the .ipynb file from the Jupyter Notebook interface.

Running the Code
Open the .ipynb file in Jupyter Notebook.

Run each cell sequentially to execute the code.

Code Explanation
1. Import Libraries
python
Copy
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statistics
Libraries used for numerical operations, visualization, and regression modeling.

2. Plot Linear Functions
python
Copy
x = np.array([0, 1, 2, 3, 4, 5])
y1 = 2 * x + 2
y2 = 2 * x

plt.scatter(x, y1, color="black")
plt.scatter(x, y2, color="black")
plt.plot(x, y1, color="red", label="y=2x+2", linewidth=3)
plt.plot(x, y2, color="green", label="y=2x", linewidth=3)
plt.xlabel('x-independent')
plt.ylabel('y-dependent')
plt.text(2, 2, 'a is line of slope\n b is intercept',
         rotation=0,
         horizontalalignment='center',
         verticalalignment='top',
         multialignment='center'
        )
plt.legend()
plt.show()
Plot two linear functions (y = 2x + 2 and y = 2x) and add annotations.

3. Plot Quadratic Function
python
Copy
y3 = x * x
plt.scatter(x, y3, color="black")
plt.plot(x, y3, color="red", label="y=x^2", linewidth=3)
plt.xlabel('x-independent')
plt.ylabel('y-dependent')
plt.legend()
plt.show()
Plot a quadratic function (y = x^2).

4. Scatter Plot for Data Points
python
Copy
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 2, 3, 5, 4])

plt.scatter(x, y, color="red")
plt.xlabel('independent variable')
plt.ylabel('target (dependent)')
plt.show()
Create a scatter plot for given data points.

5. Linear Regression Model
python
Copy
x = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
y = np.array([3, 2, 3, 5, 4])

reg = linear_model.LinearRegression()
reg.fit(x, y)

y_predict = reg.predict(x)
print("Coefficient:", reg.coef_)
print("Intercept:", reg.intercept_)
Fit a linear regression model to the data and predict values.

6. Plot Regression Line
python
Copy
plt.scatter(x, y, color="red")
plt.xlabel('independent variable')
plt.ylabel('target (dependent)')
plt.margins(0.1, 0.5)

y1 = 0.5 * x + 1.9
plt.scatter(x, y1, color="black")
plt.plot(x, y1, linewidth=1)

plt.show()
Plot the regression line along with the data points.

7. Evaluate Model
python
Copy
print("R² Score:", r2_score(y, y_predict))
print("Mean Squared Error:", mean_squared_error(y, y_predict))
mse = statistics.variance(y)
print("Variance of y:", mse)
Evaluate the model using R² score and mean squared error.

8. Visualize Multiple Regression Lines
python
Copy
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 2, 3, 5, 4])

plt.scatter(x, y, color="red")
plt.xlabel('independent variable')
plt.ylabel('target (dependent)')
plt.margins(0.1, 0.5)
plt.grid(True)

i = np.arange(0.1, 1, 0.1)
for value in i:
    y1 = value * x + 1.9
    plt.scatter(x, y1, color="black")
    plt.plot(x, y1, linewidth=1)
    plt.text(2, 8, 'y=ax(m=0.1:1:0.1)',
             rotation=0
            )
plt.show()
Visualize multiple regression lines with varying slopes.

Results
Regression Line: The best-fit line for the given data points.

R² Score: Indicates how well the model explains the variance in the target variable.

Mean Squared Error: Measures the average squared difference between predicted and actual values.

Visualizations: Plots of linear and quadratic functions, scatter plots, and regression lines.

License
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it as needed.

Support
If you encounter any issues or have questions, feel free to open an issue in this repository or contact me at minthukywe@gmail.com.

Enjoy exploring linear regression and visualization in Jupyter Notebook! 🚀
