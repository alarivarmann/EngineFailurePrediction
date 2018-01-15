
## Data Reading and Wrangling


```python
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

```


```python
processed_train,processed_test = do_data_processing(datadir2)
```

    C:\Users\BCI-EXPERT\Anaconda3\lib\site-packages\ipykernel\__main__.py:88: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    C:\Users\BCI-EXPERT\Anaconda3\lib\site-packages\ipykernel\__main__.py:89: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    

       id  cycle  dataset_id    rul  sensor 1  sensor 10  sensor 11  sensor 12  \
    0   1      1           1  191.0    518.67        1.3      47.47     521.66   
    1   1      2           1  190.0    518.67        1.3      47.49     522.28   
    2   1      3           1  189.0    518.67        1.3      47.27     522.42   
    3   1      4           1  188.0    518.67        1.3      47.13     522.86   
    4   1      5           1  187.0    518.67        1.3      47.28     522.19   
    
       sensor 13  sensor 14    ...      sensor 3  sensor 4  sensor 5  sensor 6  \
    0    2388.02    8138.62    ...       1589.70   1400.60     14.62     21.61   
    1    2388.07    8131.49    ...       1591.82   1403.14     14.62     21.61   
    2    2388.03    8133.23    ...       1587.99   1404.20     14.62     21.61   
    3    2388.08    8133.83    ...       1582.79   1401.87     14.62     21.61   
    4    2388.04    8133.80    ...       1582.85   1406.22     14.62     21.61   
    
       sensor 7  sensor 8  sensor 9  setting 1  setting 2  setting 3  
    0    554.36   2388.06   9046.19    -0.0007    -0.0004      100.0  
    1    553.75   2388.04   9044.07     0.0019    -0.0003      100.0  
    2    554.26   2388.08   9052.94    -0.0043     0.0003      100.0  
    3    554.45   2388.11   9049.48     0.0007     0.0000      100.0  
    4    554.00   2388.06   9055.15    -0.0019    -0.0002      100.0  
    
    [5 rows x 28 columns]
    
     Data Types:
    id              int64
    cycle           int64
    dataset_id      int64
    rul           float64
    sensor 1      float64
    sensor 10     float64
    sensor 11     float64
    sensor 12     float64
    sensor 13     float64
    sensor 14     float64
    sensor 15     float64
    sensor 16     float64
    sensor 17       int64
    sensor 18       int64
    sensor 19     float64
    sensor 2      float64
    sensor 20     float64
    sensor 21     float64
    sensor 3      float64
    sensor 4      float64
    sensor 5      float64
    sensor 6      float64
    sensor 7      float64
    sensor 8      float64
    sensor 9      float64
    setting 1     float64
    setting 2     float64
    setting 3     float64
    dtype: object
    

## Data Exploration


```python
processed_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cycle</th>
      <th>dataset_id</th>
      <th>rul</th>
      <th>sensor 1</th>
      <th>sensor 10</th>
      <th>sensor 11</th>
      <th>sensor 12</th>
      <th>sensor 13</th>
      <th>sensor 14</th>
      <th>...</th>
      <th>sensor 3</th>
      <th>sensor 4</th>
      <th>sensor 5</th>
      <th>sensor 6</th>
      <th>sensor 7</th>
      <th>sensor 8</th>
      <th>sensor 9</th>
      <th>setting 1</th>
      <th>setting 2</th>
      <th>setting 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>191</td>
      <td>1.0</td>
      <td>0.948718</td>
      <td>0.915132</td>
      <td>0.961313</td>
      <td>0.993194</td>
      <td>0.653748</td>
      <td>...</td>
      <td>0.927293</td>
      <td>0.902111</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.962153</td>
      <td>0.998776</td>
      <td>0.842550</td>
      <td>0.000190</td>
      <td>0.000237</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>190</td>
      <td>1.0</td>
      <td>0.948718</td>
      <td>0.916733</td>
      <td>0.962828</td>
      <td>0.993332</td>
      <td>0.637831</td>
      <td>...</td>
      <td>0.932957</td>
      <td>0.908192</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.960749</td>
      <td>0.998734</td>
      <td>0.840867</td>
      <td>0.000252</td>
      <td>0.000356</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>189</td>
      <td>1.0</td>
      <td>0.948718</td>
      <td>0.899119</td>
      <td>0.963170</td>
      <td>0.993222</td>
      <td>0.641715</td>
      <td>...</td>
      <td>0.922723</td>
      <td>0.910730</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.961923</td>
      <td>0.998818</td>
      <td>0.847906</td>
      <td>0.000105</td>
      <td>0.001068</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>188</td>
      <td>1.0</td>
      <td>0.948718</td>
      <td>0.887910</td>
      <td>0.964246</td>
      <td>0.993359</td>
      <td>0.643055</td>
      <td>...</td>
      <td>0.908829</td>
      <td>0.905152</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.962360</td>
      <td>0.998882</td>
      <td>0.845161</td>
      <td>0.000224</td>
      <td>0.000712</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>187</td>
      <td>1.0</td>
      <td>0.948718</td>
      <td>0.899920</td>
      <td>0.962608</td>
      <td>0.993249</td>
      <td>0.642988</td>
      <td>...</td>
      <td>0.908989</td>
      <td>0.915565</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.961324</td>
      <td>0.998776</td>
      <td>0.849660</td>
      <td>0.000162</td>
      <td>0.000475</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
processed_test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cycle</th>
      <th>dataset_id</th>
      <th>rul</th>
      <th>sensor 1</th>
      <th>sensor 10</th>
      <th>sensor 11</th>
      <th>sensor 12</th>
      <th>sensor 13</th>
      <th>sensor 14</th>
      <th>...</th>
      <th>sensor 3</th>
      <th>sensor 4</th>
      <th>sensor 5</th>
      <th>sensor 6</th>
      <th>sensor 7</th>
      <th>sensor 8</th>
      <th>sensor 9</th>
      <th>setting 1</th>
      <th>setting 2</th>
      <th>setting 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>57810</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>112.0</td>
      <td>1.0</td>
      <td>0.948718</td>
      <td>0.911837</td>
      <td>0.963560</td>
      <td>0.994672</td>
      <td>0.701119</td>
      <td>...</td>
      <td>0.933831</td>
      <td>0.912248</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.964743</td>
      <td>0.998923</td>
      <td>0.891455</td>
      <td>0.000262</td>
      <td>0.001068</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>57811</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>112.0</td>
      <td>1.0</td>
      <td>0.948718</td>
      <td>0.936327</td>
      <td>0.964637</td>
      <td>0.994755</td>
      <td>0.736893</td>
      <td>...</td>
      <td>0.942400</td>
      <td>0.905430</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.966937</td>
      <td>0.998860</td>
      <td>0.895018</td>
      <td>0.000143</td>
      <td>0.000356</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>57812</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>112.0</td>
      <td>1.0</td>
      <td>0.948718</td>
      <td>0.936327</td>
      <td>0.964172</td>
      <td>0.994672</td>
      <td>0.712688</td>
      <td>...</td>
      <td>0.938305</td>
      <td>0.919896</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.965228</td>
      <td>0.998944</td>
      <td>0.897148</td>
      <td>0.000214</td>
      <td>0.000831</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>57813</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>112.0</td>
      <td>1.0</td>
      <td>0.948718</td>
      <td>0.918367</td>
      <td>0.962727</td>
      <td>0.994727</td>
      <td>0.719807</td>
      <td>...</td>
      <td>0.930658</td>
      <td>0.932310</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.965136</td>
      <td>0.998902</td>
      <td>0.887363</td>
      <td>0.000307</td>
      <td>0.000712</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>57814</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>112.0</td>
      <td>1.0</td>
      <td>0.948718</td>
      <td>0.920816</td>
      <td>0.964613</td>
      <td>0.994672</td>
      <td>0.711264</td>
      <td>...</td>
      <td>0.938983</td>
      <td>0.921314</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.965344</td>
      <td>0.998860</td>
      <td>0.886742</td>
      <td>0.000240</td>
      <td>0.000712</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



## Plot Test Data Correlation Matrix


```python
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
```

## Time Series Features Preparation


```python
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
        train_generator =  (list(generate_features(data[data['id']==id], windowlength, \
                                                   features)) for id in data['id'].unique())
        train_features = np.concatenate(list(train_generator)).astype(np.float32)  
        train_labels = [generate_labels(data[data['id']==id], windowlength, ['rul']) for id in data['id'].unique()]
        train_labels = np.concatenate(train_labels).astype(np.float32)
        listout = [train_features,train_labels]
  
    return listout


```

## Saving the Data


```python
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
```


    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    <ipython-input-33-b2463d974f2c> in <module>()
          9     cv_features,cv_labels,test_features,test_labels = outtest[0],outtest[1],outtest[2],outtest[3]
         10     with open(filename, 'wb') as f:
    ---> 11         pickle.dump([train_features,train_labels,test_features,test_labels,cv_features,cv_labels], f)
    

    OSError: [Errno 28] No space left on device


## Machine Learning Modelling


```python



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




```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_3 (LSTM)                (None, 5, 32)             7552      
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 5, 32)             0         
    _________________________________________________________________
    lstm_4 (LSTM)                (None, 20)                4240      
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 20)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 21        
    _________________________________________________________________
    activation_2 (Activation)    (None, 1)                 0         
    =================================================================
    Total params: 11,813
    Trainable params: 11,813
    Non-trainable params: 0
    _________________________________________________________________
    None
    Train on 148973 samples, validate on 7841 samples
    Epoch 1/30
    Epoch 00001: val_loss improved from inf to 17460.30111, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 14s - loss: 18243.6865 - mean_squared_error: 18243.6865 - mean_absolute_error: 107.7474 - val_loss: 17460.3011 - val_mean_squared_error: 17460.3011 - val_mean_absolute_error: 106.0406
    Epoch 2/30
    Epoch 00002: val_loss improved from 17460.30111 to 14465.27907, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 11s - loss: 15119.9834 - mean_squared_error: 15119.9834 - mean_absolute_error: 95.0121 - val_loss: 14465.2791 - val_mean_squared_error: 14465.2791 - val_mean_absolute_error: 94.3814
    Epoch 3/30
    Epoch 00003: val_loss improved from 14465.27907 to 11958.90658, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 13s - loss: 12550.3028 - mean_squared_error: 12550.3028 - mean_absolute_error: 84.8045 - val_loss: 11958.9066 - val_mean_squared_error: 11958.9066 - val_mean_absolute_error: 84.7753
    Epoch 4/30
    Epoch 00004: val_loss improved from 11958.90658 to 9938.66998, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 12s - loss: 10469.5637 - mean_squared_error: 10469.5637 - mean_absolute_error: 76.7881 - val_loss: 9938.6700 - val_mean_squared_error: 9938.6700 - val_mean_absolute_error: 77.2142
    Epoch 5/30
    Epoch 00005: val_loss improved from 9938.66998 to 8406.66730, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 12s - loss: 8883.7065 - mean_squared_error: 8883.7065 - mean_absolute_error: 70.9533 - val_loss: 8406.6673 - val_mean_squared_error: 8406.6673 - val_mean_absolute_error: 71.7004
    Epoch 6/30
    Epoch 00006: val_loss improved from 8406.66730 to 7365.02157, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 11s - loss: 7783.6757 - mean_squared_error: 7783.6757 - mean_absolute_error: 67.3664 - val_loss: 7365.0216 - val_mean_squared_error: 7365.0216 - val_mean_absolute_error: 68.2328
    Epoch 7/30
    Epoch 00007: val_loss improved from 7365.02157 to 6820.86082, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 11s - loss: 7183.9095 - mean_squared_error: 7183.9095 - mean_absolute_error: 65.9879 - val_loss: 6820.8608 - val_mean_squared_error: 6820.8608 - val_mean_absolute_error: 66.7580
    Epoch 8/30
    Epoch 00008: val_loss improved from 6820.86082 to 6693.30261, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 12s - loss: 7002.0525 - mean_squared_error: 7002.0525 - mean_absolute_error: 66.0440 - val_loss: 6693.3026 - val_mean_squared_error: 6693.3026 - val_mean_absolute_error: 66.5881
    Epoch 9/30
    Epoch 00009: val_loss improved from 6693.30261 to 6683.24704, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 11s - loss: 6989.6743 - mean_squared_error: 6989.6743 - mean_absolute_error: 66.2718 - val_loss: 6683.2470 - val_mean_squared_error: 6683.2470 - val_mean_absolute_error: 66.5879
    Epoch 10/30
    Epoch 00010: val_loss improved from 6683.24704 to 5294.87814, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 13s - loss: 6422.0455 - mean_squared_error: 6422.0455 - mean_absolute_error: 61.6181 - val_loss: 5294.8781 - val_mean_squared_error: 5294.8781 - val_mean_absolute_error: 56.6062
    Epoch 11/30
    Epoch 00011: val_loss improved from 5294.87814 to 4587.70812, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 12s - loss: 5179.2983 - mean_squared_error: 5179.2983 - mean_absolute_error: 52.3440 - val_loss: 4587.7081 - val_mean_squared_error: 4587.7081 - val_mean_absolute_error: 52.7850
    Epoch 12/30
    Epoch 00012: val_loss improved from 4587.70812 to 4127.23232, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 12s - loss: 4758.4983 - mean_squared_error: 4758.4983 - mean_absolute_error: 50.3775 - val_loss: 4127.2323 - val_mean_squared_error: 4127.2323 - val_mean_absolute_error: 50.6670
    Epoch 13/30
    Epoch 00013: val_loss improved from 4127.23232 to 3797.20454, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 11s - loss: 4456.6145 - mean_squared_error: 4456.6145 - mean_absolute_error: 49.1481 - val_loss: 3797.2045 - val_mean_squared_error: 3797.2045 - val_mean_absolute_error: 49.0518
    Epoch 14/30
    Epoch 00014: val_loss did not improve
     - 11s - loss: 4296.2783 - mean_squared_error: 4296.2783 - mean_absolute_error: 48.6037 - val_loss: 3875.3147 - val_mean_squared_error: 3875.3147 - val_mean_absolute_error: 50.5620
    Epoch 15/30
    Epoch 00015: val_loss improved from 3797.20454 to 3558.56029, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 11s - loss: 4191.9394 - mean_squared_error: 4191.9394 - mean_absolute_error: 48.2457 - val_loss: 3558.5603 - val_mean_squared_error: 3558.5603 - val_mean_absolute_error: 48.3073
    Epoch 16/30
    Epoch 00016: val_loss improved from 3558.56029 to 3429.12304, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 12s - loss: 4148.1457 - mean_squared_error: 4148.1457 - mean_absolute_error: 48.1544 - val_loss: 3429.1230 - val_mean_squared_error: 3429.1230 - val_mean_absolute_error: 47.4970
    Epoch 17/30
    Epoch 00017: val_loss did not improve
     - 13s - loss: 4120.1283 - mean_squared_error: 4120.1283 - mean_absolute_error: 48.0475 - val_loss: 3537.5297 - val_mean_squared_error: 3537.5297 - val_mean_absolute_error: 48.5435
    Epoch 18/30
    Epoch 00018: val_loss improved from 3429.12304 to 3271.77879, saving model to ./model_window5hidden32_20dropout0.22batch200.h5
     - 12s - loss: 4089.2274 - mean_squared_error: 4089.2274 - mean_absolute_error: 47.9410 - val_loss: 3271.7788 - val_mean_squared_error: 3271.7788 - val_mean_absolute_error: 46.4184
    Epoch 19/30
    Epoch 00019: val_loss did not improve
     - 13s - loss: 4074.4856 - mean_squared_error: 4074.4856 - mean_absolute_error: 47.8921 - val_loss: 3868.4010 - val_mean_squared_error: 3868.4010 - val_mean_absolute_error: 51.8973
    Epoch 20/30
    Epoch 00020: val_loss did not improve
     - 12s - loss: 4055.7218 - mean_squared_error: 4055.7218 - mean_absolute_error: 47.7559 - val_loss: 3555.4776 - val_mean_squared_error: 3555.4776 - val_mean_absolute_error: 49.2776
    Epoch 21/30
    Epoch 00021: val_loss did not improve
     - 11s - loss: 4047.2257 - mean_squared_error: 4047.2257 - mean_absolute_error: 47.7298 - val_loss: 3511.2976 - val_mean_squared_error: 3511.2976 - val_mean_absolute_error: 49.1179
    Epoch 22/30
    Epoch 00022: val_loss did not improve
     - 11s - loss: 4030.6592 - mean_squared_error: 4030.6592 - mean_absolute_error: 47.5788 - val_loss: 3686.7367 - val_mean_squared_error: 3686.7367 - val_mean_absolute_error: 50.8124
    Epoch 23/30
    Epoch 00023: val_loss did not improve
     - 12s - loss: 4008.1302 - mean_squared_error: 4008.1302 - mean_absolute_error: 47.4770 - val_loss: 3568.1498 - val_mean_squared_error: 3568.1498 - val_mean_absolute_error: 49.8004
    Epoch 24/30
    Epoch 00024: val_loss did not improve
     - 11s - loss: 3999.7921 - mean_squared_error: 3999.7921 - mean_absolute_error: 47.3801 - val_loss: 3479.2709 - val_mean_squared_error: 3479.2709 - val_mean_absolute_error: 43.4856
    Epoch 25/30
    Epoch 00025: val_loss did not improve
     - 11s - loss: 3985.6833 - mean_squared_error: 3985.6833 - mean_absolute_error: 47.3081 - val_loss: 3667.8419 - val_mean_squared_error: 3667.8419 - val_mean_absolute_error: 50.7261
    Epoch 26/30
    Epoch 00026: val_loss did not improve
     - 11s - loss: 3975.1225 - mean_squared_error: 3975.1225 - mean_absolute_error: 47.2515 - val_loss: 3338.6143 - val_mean_squared_error: 3338.6143 - val_mean_absolute_error: 47.9253
    Epoch 27/30
    Epoch 00027: val_loss did not improve
     - 11s - loss: 3968.3731 - mean_squared_error: 3968.3731 - mean_absolute_error: 47.1488 - val_loss: 3775.1213 - val_mean_squared_error: 3775.1213 - val_mean_absolute_error: 51.6286
    Epoch 28/30
    Epoch 00028: val_loss did not improve
     - 12s - loss: 3962.3992 - mean_squared_error: 3962.3992 - mean_absolute_error: 47.1290 - val_loss: 3518.6160 - val_mean_squared_error: 3518.6160 - val_mean_absolute_error: 49.6193
    Epoch 00028: early stopping
    dict_keys(['val_mean_squared_error', 'mean_absolute_error', 'loss', 'val_loss', 'val_mean_absolute_error', 'mean_squared_error'])
    101362/101362 [==============================] - 3s 27us/step
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_5 (LSTM)                (None, 5, 16)             2752      
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 5, 16)             0         
    _________________________________________________________________
    lstm_6 (LSTM)                (None, 20)                2960      
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 20)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 21        
    _________________________________________________________________
    activation_3 (Activation)    (None, 1)                 0         
    =================================================================
    Total params: 5,733
    Trainable params: 5,733
    Non-trainable params: 0
    _________________________________________________________________
    None
    Train on 148973 samples, validate on 7841 samples
    Epoch 1/30
    Epoch 00001: val_loss improved from inf to 17310.67991, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 13s - loss: 18119.0642 - mean_squared_error: 18119.0642 - mean_absolute_error: 107.2514 - val_loss: 17310.6799 - val_mean_squared_error: 17310.6799 - val_mean_absolute_error: 105.4554
    Epoch 2/30
    Epoch 00002: val_loss improved from 17310.67991 to 14338.84317, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 12s - loss: 14997.6173 - mean_squared_error: 14997.6173 - mean_absolute_error: 94.5127 - val_loss: 14338.8432 - val_mean_squared_error: 14338.8432 - val_mean_absolute_error: 93.8927
    Epoch 3/30
    Epoch 00003: val_loss improved from 14338.84317 to 11854.56645, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 11s - loss: 12449.1164 - mean_squared_error: 12449.1164 - mean_absolute_error: 84.3861 - val_loss: 11854.5665 - val_mean_squared_error: 11854.5665 - val_mean_absolute_error: 84.3799
    Epoch 4/30
    Epoch 00004: val_loss improved from 11854.56645 to 9857.30704, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 10s - loss: 10381.9807 - mean_squared_error: 10381.9807 - mean_absolute_error: 76.4338 - val_loss: 9857.3070 - val_mean_squared_error: 9857.3070 - val_mean_absolute_error: 76.9150
    Epoch 5/30
    Epoch 00005: val_loss improved from 9857.30704 to 8349.01979, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 10s - loss: 8814.5700 - mean_squared_error: 8814.5700 - mean_absolute_error: 70.7400 - val_loss: 8349.0198 - val_mean_squared_error: 8349.0198 - val_mean_absolute_error: 71.5006
    Epoch 6/30
    Epoch 00006: val_loss improved from 8349.01979 to 7331.12764, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 10s - loss: 7750.9055 - mean_squared_error: 7750.9055 - mean_absolute_error: 67.2823 - val_loss: 7331.1276 - val_mean_squared_error: 7331.1276 - val_mean_absolute_error: 68.1293
    Epoch 7/30
    Epoch 00007: val_loss improved from 7331.12764 to 6807.14445, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 10s - loss: 7178.3005 - mean_squared_error: 7178.3005 - mean_absolute_error: 66.0085 - val_loss: 6807.1444 - val_mean_squared_error: 6807.1444 - val_mean_absolute_error: 66.7305
    Epoch 8/30
    Epoch 00008: val_loss improved from 6807.14445 to 6695.84985, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 10s - loss: 7004.4509 - mean_squared_error: 7004.4509 - mean_absolute_error: 66.1011 - val_loss: 6695.8498 - val_mean_squared_error: 6695.8498 - val_mean_absolute_error: 66.5885
    Epoch 9/30
    Epoch 00009: val_loss improved from 6695.84985 to 6683.28640, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 11s - loss: 6987.3465 - mean_squared_error: 6987.3465 - mean_absolute_error: 66.2784 - val_loss: 6683.2864 - val_mean_squared_error: 6683.2864 - val_mean_absolute_error: 66.5879
    Epoch 10/30
    Epoch 00010: val_loss improved from 6683.28640 to 6063.48299, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 11s - loss: 6876.7330 - mean_squared_error: 6876.7330 - mean_absolute_error: 65.3998 - val_loss: 6063.4830 - val_mean_squared_error: 6063.4830 - val_mean_absolute_error: 62.6513
    Epoch 11/30
    Epoch 00011: val_loss improved from 6063.48299 to 5234.55443, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 10s - loss: 5455.3883 - mean_squared_error: 5455.3883 - mean_absolute_error: 54.2914 - val_loss: 5234.5544 - val_mean_squared_error: 5234.5544 - val_mean_absolute_error: 58.0197
    Epoch 12/30
    Epoch 00012: val_loss improved from 5234.55443 to 4339.86166, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 10s - loss: 4883.5929 - mean_squared_error: 4883.5929 - mean_absolute_error: 51.1167 - val_loss: 4339.8617 - val_mean_squared_error: 4339.8617 - val_mean_absolute_error: 52.2343
    Epoch 13/30
    Epoch 00013: val_loss improved from 4339.86166 to 4021.74160, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 9s - loss: 4545.3259 - mean_squared_error: 4545.3259 - mean_absolute_error: 49.5668 - val_loss: 4021.7416 - val_mean_squared_error: 4021.7416 - val_mean_absolute_error: 50.7730
    Epoch 14/30
    Epoch 00014: val_loss improved from 4021.74160 to 3858.41947, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 10s - loss: 4340.8387 - mean_squared_error: 4340.8387 - mean_absolute_error: 48.7488 - val_loss: 3858.4195 - val_mean_squared_error: 3858.4195 - val_mean_absolute_error: 50.1825
    Epoch 15/30
    Epoch 00015: val_loss improved from 3858.41947 to 3689.35609, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 10s - loss: 4233.9159 - mean_squared_error: 4233.9159 - mean_absolute_error: 48.4288 - val_loss: 3689.3561 - val_mean_squared_error: 3689.3561 - val_mean_absolute_error: 49.0247
    Epoch 16/30
    Epoch 00016: val_loss improved from 3689.35609 to 3532.76918, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 10s - loss: 4170.1057 - mean_squared_error: 4170.1057 - mean_absolute_error: 48.2817 - val_loss: 3532.7692 - val_mean_squared_error: 3532.7692 - val_mean_absolute_error: 48.3232
    Epoch 17/30
    Epoch 00017: val_loss did not improve
     - 11s - loss: 4142.7160 - mean_squared_error: 4142.7160 - mean_absolute_error: 48.2362 - val_loss: 3717.0961 - val_mean_squared_error: 3717.0961 - val_mean_absolute_error: 50.1549
    Epoch 18/30
    Epoch 00018: val_loss improved from 3532.76918 to 3302.99318, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 10s - loss: 4114.4218 - mean_squared_error: 4114.4218 - mean_absolute_error: 48.1607 - val_loss: 3302.9932 - val_mean_squared_error: 3302.9932 - val_mean_absolute_error: 46.2680
    Epoch 19/30
    Epoch 00019: val_loss did not improve
     - 9s - loss: 4103.5430 - mean_squared_error: 4103.5430 - mean_absolute_error: 48.0979 - val_loss: 3504.7337 - val_mean_squared_error: 3504.7337 - val_mean_absolute_error: 48.5913
    Epoch 20/30
    Epoch 00020: val_loss improved from 3302.99318 to 3293.45098, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 10s - loss: 4080.8502 - mean_squared_error: 4080.8502 - mean_absolute_error: 48.0108 - val_loss: 3293.4510 - val_mean_squared_error: 3293.4510 - val_mean_absolute_error: 46.2851
    Epoch 21/30
    Epoch 00021: val_loss did not improve
     - 11s - loss: 4077.8651 - mean_squared_error: 4077.8651 - mean_absolute_error: 48.0267 - val_loss: 3387.2733 - val_mean_squared_error: 3387.2733 - val_mean_absolute_error: 47.5238
    Epoch 22/30
    Epoch 00022: val_loss did not improve
     - 10s - loss: 4061.0164 - mean_squared_error: 4061.0164 - mean_absolute_error: 47.9541 - val_loss: 3760.2049 - val_mean_squared_error: 3760.2049 - val_mean_absolute_error: 51.0497
    Epoch 23/30
    Epoch 00023: val_loss did not improve
     - 9s - loss: 4048.3199 - mean_squared_error: 4048.3199 - mean_absolute_error: 47.8463 - val_loss: 3402.6280 - val_mean_squared_error: 3402.6280 - val_mean_absolute_error: 48.0602
    Epoch 24/30
    Epoch 00024: val_loss improved from 3293.45098 to 3208.07822, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 11s - loss: 4040.2649 - mean_squared_error: 4040.2649 - mean_absolute_error: 47.7605 - val_loss: 3208.0782 - val_mean_squared_error: 3208.0782 - val_mean_absolute_error: 45.9213
    Epoch 25/30
    Epoch 00025: val_loss did not improve
     - 9s - loss: 4037.0065 - mean_squared_error: 4037.0065 - mean_absolute_error: 47.7739 - val_loss: 3620.6551 - val_mean_squared_error: 3620.6551 - val_mean_absolute_error: 49.7604
    Epoch 26/30
    Epoch 00026: val_loss did not improve
     - 10s - loss: 4020.2066 - mean_squared_error: 4020.2066 - mean_absolute_error: 47.6521 - val_loss: 3315.1013 - val_mean_squared_error: 3315.1013 - val_mean_absolute_error: 47.1795
    Epoch 27/30
    Epoch 00027: val_loss did not improve
     - 11s - loss: 4035.4189 - mean_squared_error: 4035.4189 - mean_absolute_error: 47.7063 - val_loss: 3535.4273 - val_mean_squared_error: 3535.4273 - val_mean_absolute_error: 49.5083
    Epoch 28/30
    Epoch 00028: val_loss did not improve
     - 11s - loss: 4005.1239 - mean_squared_error: 4005.1239 - mean_absolute_error: 47.5324 - val_loss: 3465.2887 - val_mean_squared_error: 3465.2887 - val_mean_absolute_error: 48.9044
    Epoch 29/30
    Epoch 00029: val_loss improved from 3208.07822 to 3148.58018, saving model to ./model_window5hidden16_20dropout0.22batch200.h5
     - 11s - loss: 4005.5608 - mean_squared_error: 4005.5608 - mean_absolute_error: 47.5320 - val_loss: 3148.5802 - val_mean_squared_error: 3148.5802 - val_mean_absolute_error: 45.8005
    Epoch 30/30
    Epoch 00030: val_loss did not improve
     - 11s - loss: 3983.5055 - mean_squared_error: 3983.5055 - mean_absolute_error: 47.3601 - val_loss: 3418.9356 - val_mean_squared_error: 3418.9356 - val_mean_absolute_error: 48.4840
    dict_keys(['val_mean_squared_error', 'mean_absolute_error', 'loss', 'val_loss', 'val_mean_absolute_error', 'mean_squared_error'])
    101362/101362 [==============================] - 3s 25us/step
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_7 (LSTM)                (None, 5, 32)             7552      
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 5, 32)             0         
    _________________________________________________________________
    lstm_8 (LSTM)                (None, 20)                4240      
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 20)                0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 21        
    _________________________________________________________________
    activation_4 (Activation)    (None, 1)                 0         
    =================================================================
    Total params: 11,813
    Trainable params: 11,813
    Non-trainable params: 0
    _________________________________________________________________
    None
    Train on 148973 samples, validate on 7841 samples
    Epoch 1/30
    Epoch 00001: val_loss improved from inf to 17635.31703, saving model to ./model_window5hidden32_20dropout0.5batch200.h5
     - 17s - loss: 18432.1852 - mean_squared_error: 18432.1852 - mean_absolute_error: 108.5427 - val_loss: 17635.3170 - val_mean_squared_error: 17635.3170 - val_mean_absolute_error: 106.7267
    Epoch 2/30
    Epoch 00002: val_loss improved from 17635.31703 to 14624.31375, saving model to ./model_window5hidden32_20dropout0.5batch200.h5
     - 13s - loss: 15302.4594 - mean_squared_error: 15302.4594 - mean_absolute_error: 95.7771 - val_loss: 14624.3137 - val_mean_squared_error: 14624.3137 - val_mean_absolute_error: 94.9963
    Epoch 3/30
    Epoch 00003: val_loss improved from 14624.31375 to 12098.01107, saving model to ./model_window5hidden32_20dropout0.5batch200.h5
     - 14s - loss: 12734.1419 - mean_squared_error: 12734.1419 - mean_absolute_error: 85.5456 - val_loss: 12098.0111 - val_mean_squared_error: 12098.0111 - val_mean_absolute_error: 85.3033
    Epoch 4/30
    Epoch 00004: val_loss improved from 12098.01107 to 10059.70777, saving model to ./model_window5hidden32_20dropout0.5batch200.h5
     - 14s - loss: 10689.7760 - mean_squared_error: 10689.7760 - mean_absolute_error: 77.6761 - val_loss: 10059.7078 - val_mean_squared_error: 10059.7078 - val_mean_absolute_error: 77.6603
    Epoch 5/30
    Epoch 00005: val_loss improved from 10059.70777 to 8509.11410, saving model to ./model_window5hidden32_20dropout0.5batch200.h5
     - 13s - loss: 9168.1176 - mean_squared_error: 9168.1176 - mean_absolute_error: 72.1459 - val_loss: 8509.1141 - val_mean_squared_error: 8509.1141 - val_mean_absolute_error: 72.0593
    Epoch 6/30
    Epoch 00006: val_loss improved from 8509.11410 to 7452.39651, saving model to ./model_window5hidden32_20dropout0.5batch200.h5
     - 14s - loss: 8132.8805 - mean_squared_error: 8132.8805 - mean_absolute_error: 68.7939 - val_loss: 7452.3965 - val_mean_squared_error: 7452.3965 - val_mean_absolute_error: 68.5057
    Epoch 7/30
    Epoch 00007: val_loss improved from 7452.39651 to 6194.72492, saving model to ./model_window5hidden32_20dropout0.5batch200.h5
     - 14s - loss: 7089.0752 - mean_squared_error: 7089.0752 - mean_absolute_error: 63.1224 - val_loss: 6194.7249 - val_mean_squared_error: 6194.7249 - val_mean_absolute_error: 61.7383
    Epoch 8/30
    Epoch 00008: val_loss improved from 6194.72492 to 5336.17228, saving model to ./model_window5hidden32_20dropout0.5batch200.h5
     - 12s - loss: 6116.0747 - mean_squared_error: 6116.0747 - mean_absolute_error: 57.2180 - val_loss: 5336.1723 - val_mean_squared_error: 5336.1723 - val_mean_absolute_error: 57.3654
    Epoch 9/30
    Epoch 00009: val_loss improved from 5336.17228 to 4838.08793, saving model to ./model_window5hidden32_20dropout0.5batch200.h5
     - 13s - loss: 5545.0578 - mean_squared_error: 5545.0578 - mean_absolute_error: 54.4196 - val_loss: 4838.0879 - val_mean_squared_error: 4838.0879 - val_mean_absolute_error: 55.3034
    Epoch 10/30
    Epoch 00010: val_loss improved from 4838.08793 to 4324.18363, saving model to ./model_window5hidden32_20dropout0.5batch200.h5
     - 13s - loss: 5176.9823 - mean_squared_error: 5176.9823 - mean_absolute_error: 52.8811 - val_loss: 4324.1836 - val_mean_squared_error: 4324.1836 - val_mean_absolute_error: 52.5189
    Epoch 11/30
    Epoch 00011: val_loss improved from 4324.18363 to 3977.09921, saving model to ./model_window5hidden32_20dropout0.5batch200.h5
     - 13s - loss: 4955.7832 - mean_squared_error: 4955.7832 - mean_absolute_error: 52.0162 - val_loss: 3977.0992 - val_mean_squared_error: 3977.0992 - val_mean_absolute_error: 50.0543
    Epoch 12/30
    Epoch 00012: val_loss improved from 3977.09921 to 3767.55455, saving model to ./model_window5hidden32_20dropout0.5batch200.h5
     - 13s - loss: 4812.3493 - mean_squared_error: 4812.3493 - mean_absolute_error: 51.4644 - val_loss: 3767.5545 - val_mean_squared_error: 3767.5545 - val_mean_absolute_error: 49.1643
    Epoch 13/30
    Epoch 00013: val_loss improved from 3767.55455 to 3467.40491, saving model to ./model_window5hidden32_20dropout0.5batch200.h5
     - 13s - loss: 4729.0726 - mean_squared_error: 4729.0726 - mean_absolute_error: 51.1602 - val_loss: 3467.4049 - val_mean_squared_error: 3467.4049 - val_mean_absolute_error: 46.8466
    Epoch 14/30
    Epoch 00014: val_loss did not improve
     - 12s - loss: 4694.2134 - mean_squared_error: 4694.2134 - mean_absolute_error: 51.0649 - val_loss: 3481.1414 - val_mean_squared_error: 3481.1414 - val_mean_absolute_error: 47.3835
    Epoch 15/30
    Epoch 00015: val_loss did not improve
     - 12s - loss: 4652.4308 - mean_squared_error: 4652.4308 - mean_absolute_error: 50.9261 - val_loss: 3498.9110 - val_mean_squared_error: 3498.9110 - val_mean_absolute_error: 47.6375
    Epoch 16/30
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-35-440add01784f> in <module>()
         43                 history1 = model.fit(train_features, train_labels, epochs=30, batch_size=200, validation_split=0.05, verbose=2,
         44                           callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2, mode='min'),
    ---> 45                                        ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=2)]
         46                           )
         47 
    

    C:\Users\BCI-EXPERT\Anaconda3\lib\site-packages\keras\models.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, **kwargs)
        958                               initial_epoch=initial_epoch,
        959                               steps_per_epoch=steps_per_epoch,
    --> 960                               validation_steps=validation_steps)
        961 
        962     def evaluate(self, x, y, batch_size=32, verbose=1,
    

    C:\Users\BCI-EXPERT\Anaconda3\lib\site-packages\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, **kwargs)
       1655                               initial_epoch=initial_epoch,
       1656                               steps_per_epoch=steps_per_epoch,
    -> 1657                               validation_steps=validation_steps)
       1658 
       1659     def evaluate(self, x=None, y=None,
    

    C:\Users\BCI-EXPERT\Anaconda3\lib\site-packages\keras\engine\training.py in _fit_loop(self, f, ins, out_labels, batch_size, epochs, verbose, callbacks, val_f, val_ins, shuffle, callback_metrics, initial_epoch, steps_per_epoch, validation_steps)
       1211                     batch_logs['size'] = len(batch_ids)
       1212                     callbacks.on_batch_begin(batch_index, batch_logs)
    -> 1213                     outs = f(ins_batch)
       1214                     if not isinstance(outs, list):
       1215                         outs = [outs]
    

    C:\Users\BCI-EXPERT\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py in __call__(self, inputs)
       2355         session = get_session()
       2356         updated = session.run(fetches=fetches, feed_dict=feed_dict,
    -> 2357                               **self.session_kwargs)
       2358         return updated[:len(self.outputs)]
       2359 
    

    C:\Users\BCI-EXPERT\Anaconda3\lib\site-packages\tensorflow\python\client\session.py in run(self, fetches, feed_dict, options, run_metadata)
        787     try:
        788       result = self._run(None, fetches, feed_dict, options_ptr,
    --> 789                          run_metadata_ptr)
        790       if run_metadata:
        791         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
    

    C:\Users\BCI-EXPERT\Anaconda3\lib\site-packages\tensorflow\python\client\session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
        995     if final_fetches or final_targets:
        996       results = self._do_run(handle, final_targets, final_fetches,
    --> 997                              feed_dict_string, options, run_metadata)
        998     else:
        999       results = []
    

    C:\Users\BCI-EXPERT\Anaconda3\lib\site-packages\tensorflow\python\client\session.py in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
       1130     if handle is None:
       1131       return self._do_call(_run_fn, self._session, feed_dict, fetch_list,
    -> 1132                            target_list, options, run_metadata)
       1133     else:
       1134       return self._do_call(_prun_fn, self._session, handle, feed_dict,
    

    C:\Users\BCI-EXPERT\Anaconda3\lib\site-packages\tensorflow\python\client\session.py in _do_call(self, fn, *args)
       1137   def _do_call(self, fn, *args):
       1138     try:
    -> 1139       return fn(*args)
       1140     except errors.OpError as e:
       1141       message = compat.as_text(e.message)
    

    C:\Users\BCI-EXPERT\Anaconda3\lib\site-packages\tensorflow\python\client\session.py in _run_fn(session, feed_dict, fetch_list, target_list, options, run_metadata)
       1119         return tf_session.TF_Run(session, options,
       1120                                  feed_dict, fetch_list, target_list,
    -> 1121                                  status, run_metadata)
       1122 
       1123     def _prun_fn(session, handle, feed_dict, fetch_list):
    

    KeyboardInterrupt: 


## Bayesian Optimization using Hyperas

The following routine seemed to run very slowly, so it was not practical to wait around that much  on a slow laptop. Parallel Training on Mongo workers is indicated. Further advice very welcome!


```python
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
```


```python
test_predictions = best_model.predict(test_features,verbose=1,batch_size=1)
test_predictions
```

    101362/101362 [==============================] - 188s 2ms/step
    




    array([[206.89023 ],
           [205.98921 ],
           [205.09381 ],
           ...,
           [ 75.5407  ],
           [ 80.448456],
           [ 68.66181 ]], dtype=float32)


