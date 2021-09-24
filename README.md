# ml_assignment_1
Repository for the assignment of my 1st year MSc Machine Learning course. I tried to forecast flight delay using some machine learning models. Particularly *linear regression*, *polynomial regression* and *lasso regression*.

### Prequisites
- pandas==1.3.2
- numpy==1.20.3
- sklearn==0.24.2

Once you clone this repo, you can run `pip install -r requirements.txt` then navigate to the src folder and run *train.py*

Structure
```
├── data                     <- Data files directory
│   └── flight_delay.csv     <- Entire Dataset
│
├── notebooks                <- Notebooks for analysis and testing
│   └── ml_assignment.ipynb  <- Jupyter notebook used for experimentation, graphing, data exploration and analysis
│
└── src                      <- Main Python Code for this project.
    └── train.py             <- Script which trains Models and output results.
```

The data for this project was gotten from an Innopolis University partner company which analyzes flights delays. The dataset includes predictors made up of the departure airport, scheduled departure time, destination airport and scheduled arrival time and the target which is the flight delay (measured in minutes). 

The dataset contains more than 67000 entries collected from sometime in Oct, 2015 – Aug, 2018.

We used several Machine Learning algorithms to try to estimate a function which infers the flight delay based on the predictors. The algorithms used are the Linear Regression, Polynomial Regression and the Lasso Regression (which is a regularization on the linear regressor). These algorithms are standard part of the `sklearn` library.

The dataset for this project had categorical features, particularly, all the predictors were non – numerical, so had to be encoded using the Label Encoding (also part of the `sklearn library`). The label encoding just assigns a code from 1 to N – 1 for the rows in the dataset. We merged the arrival and departure time to fit the encoder. This was to ensure similar values had same encoding.

Visualization of the data was done using the included Jupyter Notebook and it was discovered to have outliers, so we eliminated these outliers. We used the z-score method to detect these outliers then we eliminated them from the dataset. More than 8000 rows were removed.

The dataset was then split into test and training. We took all the data from 2015 – 2017 for training and then the data for 2018 as test data. 

We then used then applied Min Max Scaling on the training and test predictors. Scaling the data yielded slightly better results than when we didn’t scale it.

When the `train.py` is executed, it runs and outputs a comparison of our machine learning models according to the Mean Absolute Error and Mean Square Error.
