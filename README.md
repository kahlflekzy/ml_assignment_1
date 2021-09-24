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
