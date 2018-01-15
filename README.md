# EngineFailurePrediction
Prediction of Running Useful Life Left in Engine

Data
The data provided in this task is engine simulation data where a number of engines were run until failure and data from 21 sensors were recorded from each cycle of the engines.
The engines start at a different level of wear and as some fail sooner than others the engines have run different number of cycles.
The data is collected as a number of datasets each with its own engines (meaning engines nr1 in data set X and  nr1 in data set Y are not the same engine).

The data is provided in three files:
●	test.csv - test runs of the engines
●	train.csv -  train runs of the engines
●	RUL.csv - remaining useful life. This corresponds to test.csv and hold data on the number of cycles left in each engine before failure
### The features in the tables are as follows:
1)	Dataset name
2)	Unit number
3)	time, in cycles
4)	operational setting 1
5)	operational setting 2
6)	operational setting 3
7)	sensor measurement 1
8)	sensor measurement  2
9)	sensor measurement  21

### Task
The client is looking for a way to predict failures to minimize possible damages and thus costs. Assuming that they can monitor these sensors in real time they could roll the engine into maintenance early enough to avoid catastrophic failure.
It is up to you to devise a way to predict the number of cycles an engine has left in it.

## Solution Readme
Mathematical Essence of the problem is vector autoregression:
https://en.wikipedia.org/wiki/Vector_autoregression and its usage in prediction (regression).

### Interpretation Note
The problem provided is a regression problem since the the final output: the remaining useful life of an aircraft engine is on a continuous scale (considering the problem on a discrete scale would be a very high-dimensional classification problem which would be unfeasible).

### Working Pipeline:
![alt text](https://lh3.googleusercontent.com/PA34O6XX1nkQHbXMABVaFB44VYnuVZunZvBh6K71DfVrocsCw5nxUOqJBw3SD5ofVV5iug90oEjyGQ=s599)



## Base Principles of Solving the Task
When averaging estimators to come up with a unified one, smaller time window size means that the estimation of a statistic on that window has higher  variance (tune in to details in higher resolution), but smaller bias. As known from statistics, expected prediction error can be seen as an addition of (the bias squared, variance) -- MSE and irreducible error (noise variance) terms. All the prediction models aiming for good results should concomitantly optimize for both low bias and low variance.
Since a neural network is a generic graphical model, it can have either high or low bias or high or low variance. It is known that it is easy to build a more complex (which can be estimated by the amount of parameters, e.g. 2 hidden layer LSTM with 64 and 32 hidden nodes and 1 output node for the task has 36513 parameters) network, which would then help to avoid high bias in the predictions. There are also techniques to avoid overfitting (due to having a too complex model), e.g. increasing the batch size or amount of hidden layers, regularization techniques and dropout.


## Data Preprocessing and Feature Engineering
* `rul` variable is given as an integer. This would not stop an analyst from creating a regression model since floor and ceil functions can always be used on the prediction estimates.  Sensory 17 and 18 were also given as integers.

* Considered all the four datasets independently as different engines. Recoded the engine IDs of different datasets to be unique (refer to the function `recode_engines` that operates on `fulldata`.
* Since the task description holds that the in the training simulations the engines were ran until failure, a new  variable: Remaining Useful Life (RUL) was created for the training set by grouping the training data  by ID, then aggregating by the maximum cycle and equating this with the `rul`, thus rul for each engine id = max(cycle)-cycle.
* Check for missing values, found 0 missing values.
* Data was dynamically range-scaled, that is min-max normalization was performed.
* Next the time series features in the training and testing data were normalized, and then for training data, the unnormalized features were joined with the normalized time series features. There is a small information leak, since later test/cross-validation split is performed on the normalized test data, where some information leaked into the cross-validation data as well (it would be better to split the data before normalizing, but in this case, the cross-validation and test set errors were almost identical, so it didn't play a role).

## Correlation Analysis
Correlation Matrix visualization on the `rawtest` was performed.
![alt text](http://i64.tinypic.com/2jbtrgn.png)

## Time Series Feature Generation
Time Series features were generated using the function `prepare_features` both for the `train`,`test` and `cv-data` without using Keras internal padding mechanisms, but just by subsetting out only data relevant to the time (cycle) window.

## Modelling: Deep Learning on Sequential LSTM Deep Network
Hyperparameters optimized were the dropout probability and layer one units. Layer 2 unit count was set to 20 for computational limitations. Out of other parameters, the results were tested out both with window size 5 and 18. The best model was chosen according to the best cross-validation error.

## Bayesian Optimization

Bayesian Optimization was experimented with Hyperas and Hyperopt, but the routine experiments ran a little bit too slowly on the current system. Indicated for further use for optimization in parallel mode.



