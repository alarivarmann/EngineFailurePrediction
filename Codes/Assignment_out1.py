
# coding: utf-8

# ## Data Reading and Wrangling

# In[7]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing

import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard



datadir2 = 'C:/data_week1/MindTitan/data'
###############################################################3
def do_data_processing(datadir):
    '''utility functions'''
    def output_len(data):
        return(data.shape[0])
    
    def num(s):
        try:
            return int(s)
        except ValueError:
            return float(s)
    
   
    ########### READING DATA ####################33
    direct_train = datadir + '/train.csv'
    direct_test = datadir + '/test.csv'
    direct_rul = datadir + '/rul.csv'
    
    rawtrain = pd.read_csv(direct_train,engine='python')
    rawtest = pd.read_csv(direct_test,engine='python')
    test_left_cycles = pd.read_csv(direct_rul,engine='python') # join this with the test data by dataset_id  and unit_id
    
    #rawtrain.columns = [x.strip() for x in rawtrain.columns] 
    
    rawtest.rename(columns=lambda x: x.strip())
    raw_trainlen = output_len(rawtrain)
    raw_testlen = output_len(rawtest)
    train = rawtrain.copy()
    test = rawtest.copy()
    together = [rawtrain,rawtest]
    rawfull = pd.concat(together)
    raw_full_len = output_len(rawfull)
    
    trainlengths = [sum(train['dataset_id']=='FD00'+str(i)) for i in range(1,5)]
    sum(trainlengths) == raw_trainlen
    testlengths = [sum(test['dataset_id']=='FD00'+str(i)) for i in range(1,5)]
    sum(testlengths) == raw_testlen
    
    ''' MERGING TEST DATA '''
    set(test.unit_id) == set(test_left_cycles.unit_id) # checking if the amount of unique engines is the same, so that inner join is meaningful
    test_merged = pd.merge(test, test_left_cycles, how='left', on=['dataset_id','unit_id'], sort=False) # OK
    
    together = [train,test_merged]
    fulldata1 = pd.concat(together)
    
    ''' Relabelling dataset id and reordering columns'''
    
    fulldata1['dataset_id'] = fulldata1['dataset_id'].str.replace('FD00', '')
    fulldata1['dataset_id'] = fulldata1['dataset_id'].apply(num)
    fulldata1 = fulldata1.rename(columns={'unit_id':'id'})
    fulldata_cols = fulldata1.columns.tolist()
    fulldata_cols = fulldata_cols[-1:] + fulldata_cols[:-1]
    fulldata1 = fulldata1[fulldata_cols]
    ''' Before we do anything else, we need to recode the engine ids since we have 4 datasets with different engines'''
    ''' Coding the engines to be unique -- since the engines in the test and train are recoded 
    in the same fashion, one should be allowed to use engine id as a predictor'''
    
    def recode_engines(fulldata1): 
        traininfo = {}
        testinfo = {}
        train_ = fulldata1[:raw_trainlen]
        test_ = fulldata1[raw_trainlen:]
    
        for i in range(1,5):
            indicatorvar = train_.dataset_id == i
            train = train_[indicatorvar]
        
            indicatorvar2 = test_.dataset_id == i
            test = test_[indicatorvar2]
            if i > 1:
                train['id']=train['id'] + max(traininfo[i-1]['id']) # take the maximum engine number
                test['id']=test['id'] + max(traininfo[i-1]['id'])
            traininfo[i] = train
            testinfo[i] = test
            
            if i == 4:
                train_full = pd.concat(traininfo.values())
                test_full = pd.concat(testinfo.values())
        #if output_len(train_full)==output_len(train):
        return([train_full,test_full])
    #    else:
    #        return('some data missing')
            
    train_full,test_full = recode_engines(fulldata1)
    output_len(train_full)+output_len(test_full) == output_len(fulldata1) # check ok
    
    sum(train_full.isnull().sum()) == 0 # we see that there are missing values. We know it's rul of train data
    sum(test_full.isnull().sum()) == 0  # true
    
    train_full.isnull().sum() # 160359 missing values standing for train data rul. Next we generate this feature
    test_full.isnull().sum()
    
    ''' In this step we create the remaining useful life variable for the train data'''
    rul_frame = pd.DataFrame(train_full.groupby('id')['cycle'].max()).reset_index()
    rul_frame.columns = ['id', 'max']
    train_full = train_full.merge(rul_frame, on=['id'], how='left')
    train_full['rul'] = train_full['max'] - train_full['cycle']
    train_full.drop('max', axis=1, inplace=True)
    
    np.mean((train_full.groupby('id')).min()) # ok average of the min rul's per engine is 0
    np.mean((test_full.groupby('id')).min()) # in the test data, the average of min rul's per engine is 81.4
    
    '''Checking that the train and test data lengths add up to full data'''
    output_len(train_full) + output_len(test_full) == raw_full_len # ok
    
    ''' Now are from hence: ID REFERS TO UNIT ID (ENGINE ID)'''
    
    full_lengths= [sum(fulldata1['dataset_id']==i) for i in range(1,5)]
    sum(full_lengths) == raw_full_len  # data size OK
    
    ''' check if any missing values in the data, should be 0 now'''
    sum(train_full.isnull().sum()) == 0 # we see that there are missing values. We know it's rul of train data
    sum(test_full.isnull().sum()) == 0 # ok
    ''' 
    160359 ELEMENTS TRAIN  
    104897 ELEMENTS TEST, 
    265256 ELEMENTS TOTAL '''
    ############## TRAINING DATA MOD ################
    trainlen = output_len(train_full)
    testlen = output_len(test_full)
    trainlen
    testlen
    
    ################################################################################3
    
    frames = [train_full,test_full]
    '''combining train and test now should result in 0 NaNs since both have the rul variable'''
    fulldata = pd.concat(frames) 
    
    fulldata.isnull().sum() # 'Fine, fulldata has 0 Na
    
    print(fulldata.head())
    print('\n Data Types:')
    print(fulldata.dtypes)
    
    ' We see that sensors 17 and 18 are of integer type '
    
    ################################################## DATA NORMALIZATION ###################################3
    
    '''DATA NORMALIZATION -- we don't use test data when normalizing training data'''
    '''train_full.columns returns the index object with feature names for the training set'''
    
    feats_to_normalize = train_full.columns.difference(['id','cycle','rul','dataset_id']) 
    ''' exclude engine ID and remaining useful life columns from normalization because engine ID is just an identifier
    and remaining useful life should not be normalized to the max of the engines since each engine is different ?'''
    
    'Only sensor and setting columns are normalized'
    
    def normalize_data(data,feats_to_normalize):
        minmax_scaler = preprocessing.MinMaxScaler()
        normalized_data = pd.DataFrame(minmax_scaler.fit_transform(data[feats_to_normalize]), columns=feats_to_normalize, index=data.index)
        return(normalized_data)
    
    def merge_data(full,feats_to_normalize):
        normalized_train = normalize_data(full,feats_to_normalize)
        normalized_train.isnull().sum() # 'Fine, normalized_train has 0 Na
        unnormalized_features = full.columns.difference(feats_to_normalize)
        combined = full[unnormalized_features].join(normalized_train)
        if 'cycle_norm' in full.columns:
            combined.drop('cycle_norm',axis=1)
        combined = combined.reindex(columns = full.columns)
        return(combined)
        
    processed_train = merge_data(train_full,feats_to_normalize)
    processed_test = merge_data(test_full,feats_to_normalize)
    output_len(processed_train) + output_len(processed_test) == output_len(fulldata1) # check ok
    return(processed_train,processed_test)


# In[8]:

processed_train,processed_test = do_data_processing(datadir2)


# ## Data Exploration

# In[12]:

processed_train.head()


# In[10]:

processed_test.head()


# ## Plot Test Data Correlation Matrix

# In[22]:

size = 25
rawtest = pd.read_csv('./data/test.csv',engine='python')
rawtest_timeseries =  rawtest.iloc[:, [0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]] # index buggy?
grouped = rawtest_timeseries.groupby('id')

corrmatrix = rawtest_timeseries.corr()
fig, ax = plt.subplots(figsize=(size, size))
ax.matshow(corrmatrix)
plt.xticks(range(len(corrmatrix.columns)), corrmatrix.columns);
plt.yticks(range(len(corrmatrix.columns)), corrmatrix.columns);
fig.savefig("./timeseries_correlationmatrix.png")


# ## Time Series Features Preparation

# In[31]:

import numpy as np
from sklearn import cross_validation as cv

min_max_cycle_test= np.min(processed_test.groupby('id')['cycle'].max()) 

min_max_cycle_test= np.min(processed_test.groupby('id')['cycle'].max()) # NB! Smallest window in the testing set is 19, so cannot use longer windows than 18


test_feature_names  = list(filter(lambda s: (not s.startswith('dataset') and not s.startswith('lessthan') and not s.startswith('rul')), processed_test.columns))

def prepare_features(data,windowlength,features, iftest):
    def generate_features(data, windowlength, features):
        data_array = data[features].values
        data_len = data_array.shape[0]
        for start, end in zip(range(0, data_len-windowlength), range(windowlength, data_len)):
            yield data_array[start:end, :]
        return data_array
        
    def generate_labels(data, windowlength, label):
        data_array = data[label].values
        data_length = data_array.shape[0]
        return data_array[windowlength:data_length, :]
    
    listout = list()
    if iftest==1:
        
        initial_labels = data['rul']
        test_array = data[features]
        
        cv_array, red_test_array, cv_labels, red_test_labels = cv.train_test_split(test_array,initial_labels, test_size=0.5, random_state=1)
        
        cv_generator= (list(generate_features(data[data['id']==id], windowlength, features)) for id in data['id'].unique())

        test_generator= (list(generate_features(data[data['id']==id], windowlength, features)) for id in data['id'].unique())
        
        test_features = np.concatenate(list(test_generator)).astype(np.float32)
        cv_features = np.concatenate(list(cv_generator)).astype(np.float32)

        
        cv_labels = [generate_labels(data[data['id']==id], windowlength, ['rul']) for id in data['id'].unique()]
        cv_labels = np.concatenate(cv_labels).astype(np.float32)
        
        test_labels = [generate_labels(data[data['id']==id], windowlength, ['rul']) for id in data['id'].unique()]
        test_labels = np.concatenate(test_labels).astype(np.float32)
        listout = [cv_features,cv_labels,test_features,test_labels]
        
    else:
        train_generator =  (list(generate_features(data[data['id']==id], windowlength,                                                    features)) for id in data['id'].unique())
        train_features = np.concatenate(list(train_generator)).astype(np.float32)  
        train_labels = [generate_labels(data[data['id']==id], windowlength, ['rul']) for id in data['id'].unique()]
        train_labels = np.concatenate(train_labels).astype(np.float32)
        listout = [train_features,train_labels]
  
    return listout



# ## Saving the Data

# In[33]:

import pickle
test_feature_names  = list(filter(lambda s: (not s.startswith('dataset') and not s.startswith('lessthan') and not s.startswith('rul')), processed_test.columns))

for windowlength in [5,18]: 
    filename = 'datasets_final' + 'windowlen' + str(windowlength)
    outtrain = prepare_features(processed_train,windowlength,test_feature_names, 0)
    outtest = prepare_features(processed_test,windowlength,test_feature_names, 1)
    train_features,train_labels = outtrain[0],outtrain[1]
    cv_features,cv_labels,test_features,test_labels = outtest[0],outtest[1],outtest[2],outtest[3]
    with open(filename, 'wb') as f:
        pickle.dump([train_features,train_labels,test_features,test_labels,cv_features,cv_labels], f)


# ## Machine Learning Modelling

# In[35]:




#################################################3 MACHINE LEARNING ###########################################

import pickle
best_cv_mse = 10**20
modelidx=1

''' KICKSTART INTO TRAINING'''

units2 = 20
for windowlength in [5,18]:   
    for dropout in [0.22,0.5]:
        for units1 in [32, 16]:   
                filename = 'datasets_final' + 'windowlen' + str(windowlength)
                with open(filename, 'rb') as f:
                    [train_features,train_labels,test_features,test_labels,cv_features,cv_labels] = pickle.load(f)
            #                  idx = np.random.randint(100, size=2)
            #                  train_s = train_features[idx,:]
            #                  train_l_s = train_labels[idx,:]
                featurecount = 26
                model = Sequential()
                model.add(LSTM(
                         input_shape=(windowlength, featurecount),
                         units=units1, # number of hidden units in the 1st hidden layer -- use 32-multiple
                         return_sequences=True))
                model.add(Dropout(dropout))
                model.add(LSTM(
                          units=units2, # number of hidden units in the 2nd hidden layer
                          return_sequences=False))
                model.add(Dropout(dropout))
                 # ''' OPTIMIZE DROPOUT '''
                
                model.add(Dense(units=1)) # add dense output layer
                model.add(Activation("linear")) # '''******** CHOOSE ACTIVATION FUN ********
                model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['mse','mae'])
                
                print(model.summary())
                model_path = './model_window'+str(windowlength)+'hidden'+str(units1)+'_'+str(units2)+'dropout'+str(dropout)+'batch'+str(200)+'.h5'
                
                 
                history1 = model.fit(train_features, train_labels, epochs=30, batch_size=200, validation_split=0.05, verbose=2,
                          callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2, mode='min'),
                                       ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=2)]
                          )
                
                # list all data in history
                print(history1.history.keys())
                
                cv_scores = model.evaluate(cv_features,cv_labels, batch_size=200,verbose=1)
                cv_mse = cv_scores[1]
                if cv_mse < best_cv_mse:
                    best_model = model
                    best_cv_mse = cv_mse 
                modelidx +=1            

test_scores = best_model.evaluate(test_features,test_labels, batch_size=200,verbose=1)
print('\nBest model MSE: {}'.format(test_scores[1]))
print('\nBest model MAE: {}'.format(test_scores[2]))

test_predictions = best_model.predict(test_features)
test_predictions





# ## Bayesian Optimization using Hyperas

# The following routine seemed to run very slowly, so it was not practical to wait around that much  on a slow laptop. Parallel Training on Mongo workers is indicated. Further advice very welcome!

# In[ ]:

from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional



from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import sys
import numpy as np




def create_data(X,y,X_val,y_val):
        return X,y,X_val,y_val

X,y,X_val,y_val = create_data(X,y,X_val,y_val)

def create_model(X, X_val, y, y_val):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop




def get_space():
    return {
    'LSTM': choice('LSTM', [32, 64, 128]),
    'Dropout': uniform('Dropout', 0, 1),
    'optimizer': choice('optimizer', ['rmsprop', 'adam', 'sgd']),
    'batch_size': choice('batch_size', [64, 128]),
    'nb_epoch': choice('nb_epoch', [10, 20]),
    }


########################### VER 1 ##################################
space = {   'units1': hp.choice('units1', [32,64]),
            'units2': hp.choice('units2', [13,26]),

            'dropout1': hp.uniform('dropout1', .25,.75),
            'dropout2': hp.uniform('dropout2',  .25,.75),

            'batch_size' : hp.choice('batch_size', [32,64,128]),

            'epochs' :  5,
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            'activation': hp.choice('activation',['linear','relu'])
        }

#'choice': hp.choice('num_layers',
#                    [ {'layers':'two', },
#                    {'layers':'three',
#                    'units3': hp.choice('units3', [2]), 
#                    'dropout3': hp.uniform('dropout3', .25,.75)}
#                    ]),

def f_nn(params):   
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop

    print ('Params testing: ', params)
    model = Sequential()
    #model.add(Dense(output_dim=params['units1'], input_dim = X.shape[1])) 
    #model.add(Activation(params['activation']))
    model.add(LSTM(
         input_shape=(32, 29),
         units=params['units1'], 
         return_sequences=True))
    model.add(Dropout(params['dropout1']))
    model.add(LSTM(units=params['units2'],return_sequences=False))
    model.add(Dropout(params['dropout2']))
    model.add(Dense(units = 1))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer=params['optimizer'])
    model.fit(X, y, epochs=params['epochs'], batch_size=params['batch_size'], verbose = 2)
    score, acc = model.evaluate(X_val, y_val, show_accuracy=True, verbose=2)

    return {'loss': -acc, 'status': STATUS_OK}

# ,'model':model
trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=3, trials=trials)



########################################### REWRITING TO BE MORE CLEAN ###################################

def f_nn(params):   
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop

    print ('Params testing: ', params)
    model = Sequential()
    #model.add(Dense(output_dim=params['units1'], input_dim = X.shape[1])) 
    #model.add(Activation(params['activation']))
    model.add(LSTM(
         input_shape=(32, 29),
         units=params['units1'], 
         return_sequences=True, dropout = params['dropout1']))
    model.add(LSTM(units=params['units2'],return_sequences=False,dropout = params['dropout2']))
    model.add(Dense(units = 1,activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=params['optimizer'])

    model.fit(X, y, epochs=params['epochs'], batch_size=params['batch_size'], verbose = 2)
    score, acc = model.evaluate(X_val, y_val, show_accuracy=True, verbose=2)

    return {'loss': -acc, 'status': STATUS_OK, 'model':model}
###############################################
    
    
    
    ############################################### VER 2 ############################################33
    
space2 = {
            'dropout1': hp.uniform('dropout1', .25,.75),
        }
        
def f2_nn(params):   
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop

    print ('Params testing: ', params)
    model = Sequential()
    #model.add(Dense(output_dim=params['units1'], input_dim = X.shape[1])) 
    #model.add(Activation(params['activation']))
    model.add(LSTM(
         input_shape=(32, 29),
         units=20, 
         return_sequences=True, dropout = params['dropout1']))
    model.add(LSTM(units=10,return_sequences=False,dropout = 0.2))
    model.add(Dense(units = 1,activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    model.fit(X, y, epochs=5, batch_size=32, verbose = 2)
    score, acc = model.evaluate(X_val, y_val, show_accuracy=True, verbose=2)
    return {'loss': -acc, 'status': STATUS_OK, 'model':model}
    
best2 = fmin(f2_nn, space2, algo=tpe.suggest, max_evals=3, trials=trials)
# Info on why fmin : https://github.com/hyperopt/hyperopt/wiki/FMin


# In[37]:

test_predictions = best_model.predict(test_features,verbose=1,batch_size=1)
test_predictions

