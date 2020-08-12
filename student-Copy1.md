### Final Project Submission

Please fill out:
* Student name: Cristina Townsend
* Student pace: part time 
* Scheduled project review date/time: Wednesday 08/12/2020 at 16:00
* Instructor name: James Irving
* Blog post URL:
* Presentation URL: https://docs.google.com/presentation/d/1V97mN3zDtDcKiJJ6MrYXO1FuYYEwQeSSoTMuaXWYeo4/edit?pli=1#slide=id.p


#### Business Case
In this notebook, we'll explore, and model King County House Sale dataset with a multivariate linear regression to predict the sale price of houses by answering the following questions:
- What are some of the important aspects when it comes to pricing your home for sale?
- Which things are not as important when trying to sell your home?
- How can we improve the potential sale price of your home?




## Obtaining the Data


```python
import pandas as pd
```


```python
# import data and ensure it loaded in properly
df = pd.read_csv('kc_house_data.csv')
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



### Column Names and descriptions for Kings County Data Set
* **id** - unique identified for a house
* **date** - Date house was sold
* **price** -  price prediction target
* **bedrooms** -  number of Bedrooms in the House
* **bathrooms** -  number of bathrooms
* **sqft_living** -  square footage of the home
* **sqft_lot** -  square footage of the lot
* **floors** -  total floors (levels) in house
* **waterfront** - House which has a view to a waterfront
* **view** - Has been viewed
* **condition** - How good the condition is ( Overall )
* **grade** - overall grade given to the housing unit, based on King County grading system
* **sqft_above** - square footage of house apart from basement
* **sqft_basement** - square footage of the basement
* **yr_built** - Built Year
* **yr_renovated** - Year when house was renovated
* **yrs_since_reno** - Number of years since renovated, if not renovated years since built
* **zipcode** - zip
* **subregion** - the subregion that the zipcode falls under
* **lat** - Latitude coordinate
* **long** - Longitude coordinate
* **sqft_living15** - The square footage of interior housing living space for the nearest 15 neighbors
* **sqft_lot15** - The square footage of the land lots of the nearest 15 neighbors

### Initial examination of data


```python
# See if there is anything that stands outs that we will need to deal with in
# scrubbing the data
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
    id               21597 non-null int64
    date             21597 non-null object
    price            21597 non-null float64
    bedrooms         21597 non-null int64
    bathrooms        21597 non-null float64
    sqft_living      21597 non-null int64
    sqft_lot         21597 non-null int64
    floors           21597 non-null float64
    waterfront       19221 non-null float64
    view             21534 non-null float64
    condition        21597 non-null int64
    grade            21597 non-null int64
    sqft_above       21597 non-null int64
    sqft_basement    21597 non-null object
    yr_built         21597 non-null int64
    yr_renovated     17755 non-null float64
    zipcode          21597 non-null int64
    lat              21597 non-null float64
    long             21597 non-null float64
    sqft_living15    21597 non-null int64
    sqft_lot15       21597 non-null int64
    dtypes: float64(8), int64(11), object(2)
    memory usage: 3.5+ MB



```python
# See if there are any unusual max/mins which would give us an idea if there 
# are place holder values
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2.159700e+04</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>19221.000000</td>
      <td>21534.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>17755.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>4.580474e+09</td>
      <td>5.402966e+05</td>
      <td>3.373200</td>
      <td>2.115826</td>
      <td>2080.321850</td>
      <td>1.509941e+04</td>
      <td>1.494096</td>
      <td>0.007596</td>
      <td>0.233863</td>
      <td>3.409825</td>
      <td>7.657915</td>
      <td>1788.596842</td>
      <td>1970.999676</td>
      <td>83.636778</td>
      <td>98077.951845</td>
      <td>47.560093</td>
      <td>-122.213982</td>
      <td>1986.620318</td>
      <td>12758.283512</td>
    </tr>
    <tr>
      <td>std</td>
      <td>2.876736e+09</td>
      <td>3.673681e+05</td>
      <td>0.926299</td>
      <td>0.768984</td>
      <td>918.106125</td>
      <td>4.141264e+04</td>
      <td>0.539683</td>
      <td>0.086825</td>
      <td>0.765686</td>
      <td>0.650546</td>
      <td>1.173200</td>
      <td>827.759761</td>
      <td>29.375234</td>
      <td>399.946414</td>
      <td>53.513072</td>
      <td>0.138552</td>
      <td>0.140724</td>
      <td>685.230472</td>
      <td>27274.441950</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000102e+06</td>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>370.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>2.123049e+09</td>
      <td>3.220000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1190.000000</td>
      <td>1951.000000</td>
      <td>0.000000</td>
      <td>98033.000000</td>
      <td>47.471100</td>
      <td>-122.328000</td>
      <td>1490.000000</td>
      <td>5100.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>3.904930e+09</td>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1910.000000</td>
      <td>7.618000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1560.000000</td>
      <td>1975.000000</td>
      <td>0.000000</td>
      <td>98065.000000</td>
      <td>47.571800</td>
      <td>-122.231000</td>
      <td>1840.000000</td>
      <td>7620.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>7.308900e+09</td>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.068500e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>2210.000000</td>
      <td>1997.000000</td>
      <td>0.000000</td>
      <td>98118.000000</td>
      <td>47.678000</td>
      <td>-122.125000</td>
      <td>2360.000000</td>
      <td>10083.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>9.900000e+09</td>
      <td>7.700000e+06</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>9410.000000</td>
      <td>2015.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.000000</td>
      <td>871200.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Initial Scrubbing

### Looking at individual columns

#### id


```python
# Drop, since will not add any useful information to the model
df.drop(columns=['id'],axis=1, inplace=True)
```

#### Date


```python
# Change to date-time
df['date'] = pd.to_datetime(df['date'])
df['date'].describe()
```




    count                   21597
    unique                    372
    top       2014-06-23 00:00:00
    freq                      142
    first     2014-05-02 00:00:00
    last      2015-05-27 00:00:00
    Name: date, dtype: object



#### Price


```python
# What we will be running the model against
df['price'].describe()
```




    count    2.159700e+04
    mean     5.402966e+05
    std      3.673681e+05
    min      7.800000e+04
    25%      3.220000e+05
    50%      4.500000e+05
    75%      6.450000e+05
    max      7.700000e+06
    Name: price, dtype: float64



#### Bedroom and Bathroom number


```python
# Looking at Bedrooms to make sure there aren't any hidden nulls or place holders
df['bedrooms'].describe()
```




    count    21597.000000
    mean         3.373200
    std          0.926299
    min          1.000000
    25%          3.000000
    50%          3.000000
    75%          4.000000
    max         33.000000
    Name: bedrooms, dtype: float64




```python
df['bedrooms'].unique()
```




    array([ 3,  2,  4,  5,  1,  6,  7,  8,  9, 11, 10, 33])




```python
df.drop(df[df['bedrooms'] == 33 ].index , inplace=True)
```


```python
# Looking at bathrooms to make sure there aren't any hidden nulls or place holders
df['bathrooms'].describe()
```




    count    21596.000000
    mean         2.115843
    std          0.768998
    min          0.500000
    25%          1.750000
    50%          2.250000
    75%          2.500000
    max          8.000000
    Name: bathrooms, dtype: float64




```python
df['bathrooms'].unique()
```




    array([1.  , 2.25, 3.  , 2.  , 4.5 , 1.5 , 2.5 , 1.75, 2.75, 3.25, 4.  ,
           3.5 , 0.75, 4.75, 5.  , 4.25, 3.75, 1.25, 5.25, 6.  , 0.5 , 5.5 ,
           6.75, 5.75, 8.  , 7.5 , 7.75, 6.25, 6.5 ])



#### sqft_living, sqft_lots, floors


```python
# Looking at sqft_living to make sure there aren't any hidden nulls or place holders
print(df['sqft_living'].describe())
print('Null values:', df['sqft_living'].isnull().sum())
```

    count    21596.000000
    mean      2080.343165
    std        918.122038
    min        370.000000
    25%       1430.000000
    50%       1910.000000
    75%       2550.000000
    max      13540.000000
    Name: sqft_living, dtype: float64
    Null values: 0



```python
# Looking at sqft_lot to make sure there aren't any hidden nulls or place holders
print(df['sqft_lot'].describe())
print('Null values:', df['sqft_lot'].isnull().sum())
```

    count    2.159600e+04
    mean     1.509983e+04
    std      4.141355e+04
    min      5.200000e+02
    25%      5.040000e+03
    50%      7.619000e+03
    75%      1.068550e+04
    max      1.651359e+06
    Name: sqft_lot, dtype: float64
    Null values: 0





```python
# Looking at sqft_lot to make sure there aren't any hidden nulls or place holders
print(df['floors'].describe())
print('Null values:', df['floors'].isnull().sum())
```

    count    21596.000000
    mean         1.494119
    std          0.539685
    min          1.000000
    25%          1.000000
    50%          1.500000
    75%          2.000000
    max          3.500000
    Name: floors, dtype: float64
    Null values: 0


#### Waterfront


```python
# Check for null values
df['waterfront'].unique()
```




    array([nan,  0.,  1.])




```python
# Assume that if it isn't reported, it does not have a waterfront view
df['waterfront'] = df['waterfront'].fillna(0)
```


```python
# check again to ensure no nulls
print('Null values:', df['waterfront'].isnull().sum())
```

    Null values: 0


#### View, Condition, and Grade


```python
# Check for null and place holders
print('Null values:', df['view'].isnull().sum())
df['view'].unique()
```

    Null values: 63





    array([ 0., nan,  3.,  4.,  2.,  1.])




```python
# Replace null with 0
df['view'] = df['view'].fillna(0)
```


```python
# check again to ensure no nulls
print('Null values:', df['view'].isnull().sum())
df['view'].unique()
```

    Null values: 0





    array([0., 3., 4., 2., 1.])




```python
# Check for null and place holders
print('Null values:', df['condition'].isnull().sum())
df['condition'].unique()
```

    Null values: 0





    array([3, 5, 4, 1, 2])




```python
# Check for null and place holders
print('Null values:', df['grade'].isnull().sum())
df['grade'].unique()
```

    Null values: 0





    array([ 7,  6,  8, 11,  9,  5, 10, 12,  4,  3, 13])



#### sqft_above and sqft_basement


```python
# Check for null and place holders
print('Null values:', df['sqft_above'].isnull().sum())
df['sqft_above'].describe()
```

    Null values: 0





    count    21596.000000
    mean      1788.631506
    std        827.763251
    min        370.000000
    25%       1190.000000
    50%       1560.000000
    75%       2210.000000
    max       9410.000000
    Name: sqft_above, dtype: float64




```python
# Check for nulls and place holders
print('Null values:', df['sqft_basement'].isnull().sum())
df['sqft_basement'].describe()
```

    Null values: 0





    count     21596
    unique      304
    top         0.0
    freq      12826
    Name: sqft_basement, dtype: object




```python
# There was an error when attempted to convert to float, so we know there is a place holder '?'
```


```python
# We know sqft_living and sqft_above do not have any null or place holders so if we 
# subtract them, we should get the sqft_basement for all properties
df['sqft_basement'] = df['sqft_living'] - df['sqft_above']
```


```python
# Check to ensure it worked
print('Null values:', df['sqft_basement'].isnull().sum())
df['sqft_basement'].describe()
```

    Null values: 0





    count    21596.000000
    mean       291.711660
    std        442.673703
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%        560.000000
    max       4820.000000
    Name: sqft_basement, dtype: float64



#### Year Built


```python
# Check for nulls and place holders
print('Null values:', df['yr_built'].isnull().sum())
df['yr_built'].describe()
```

    Null values: 0





    count    21596.000000
    mean      1971.000787
    std         29.375460
    min       1900.000000
    25%       1951.000000
    50%       1975.000000
    75%       1997.000000
    max       2015.000000
    Name: yr_built, dtype: float64



#### Year Renovated


```python
# Check for nulls and place holders
print('Null values:', df['yr_renovated'].isnull().sum())
df['yr_renovated'].describe()
```

    Null values: 3842





    count    17754.000000
    mean        83.641489
    std        399.957185
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max       2015.000000
    Name: yr_renovated, dtype: float64




```python
# Fill the null values with zeroes
df['yr_renovated'] = df['yr_renovated'].fillna(0)
```


```python
# Check for nulls again
print('Null values:', df['yr_renovated'].isnull().sum())
```

    Null values: 0



```python
# create a temp column to calculate years since renovation
df['temp'] = df['yr_renovated'].replace(to_replace=0, value=df['yr_built'])

```


```python
# Calculate years since the house had work done on it
df['yrs_since_reno'] = 2020 - df['temp']
df['yrs_since_reno']
```




    0        65.0
    1        29.0
    2        87.0
    3        55.0
    4        33.0
             ... 
    21592    11.0
    21593     6.0
    21594    11.0
    21595    16.0
    21596    12.0
    Name: yrs_since_reno, Length: 21596, dtype: float64




```python
# Drop temp
df.drop(columns=['temp'],axis=1, inplace=True)
```

#### Drop Lat/long



```python
df.drop(columns=['lat','long'],axis=1, inplace=True)
```


```python
df.columns
```




    Index(['date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
           'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
           'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15',
           'sqft_lot15', 'yrs_since_reno'],
          dtype='object')



#### Zipecodes


```python
subregion_dict = {'south_urban': [98001, 98002, 98003, 98023, 98030, 98031, 98032, 98042, 
                                 98055, 98056, 98058, 98092, 98148, 98166, 98168, 98178, 98188,
                                 98198], 
                 'east_urban' : [98004, 98005, 98006, 98007, 98008, 98009, 98027, 98029, 
                                 98033, 98034, 98039, 98040, 98052, 98053, 98059, 98074, 
                                98075, 98077],
                 'south_rural' : [98010, 98022, 98038],
                 'north' : [98011, 98028, 98072, 98155],
                 'east_rural' : [98014, 98019, 98024, 98045, 98065],
                 'vashon_island' : [98070], 
                 'seattle' : [98102, 98103, 98105, 98106, 98107, 98108, 98109, 98112, 98115,
                             98116, 98117, 98118, 98119, 98122, 98125, 98126, 98136, 98144, 98199], 
                 'north_and_seattle' : [98133, 98177], 
                 'south_and_seattle' : [98146]}
```


```python
def get_key(val): 
    for key, value in subregion_dict.items(): 
         if val in value: 
            return key 
```


```python
region_lst = []
for x in df['zipcode']:
    region_lst.append(get_key(x))

df['subregion'] = region_lst
df['subregion']    
```




    0              south_urban
    1                  seattle
    2                    north
    3                  seattle
    4               east_urban
                   ...        
    21592              seattle
    21593    south_and_seattle
    21594              seattle
    21595           east_urban
    21596              seattle
    Name: subregion, Length: 21596, dtype: object




```python
df.drop('zipcode', axis=1, inplace=True)
df3 = df.copy()
```

#### Convert columns into objects


```python
col_to_obj = ['view', 'condition']
```


```python
for x in col_to_obj:
    df[x] = df[x].astype(str)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21596 entries, 0 to 21596
    Data columns (total 19 columns):
    date              21596 non-null datetime64[ns]
    price             21596 non-null float64
    bedrooms          21596 non-null int64
    bathrooms         21596 non-null float64
    sqft_living       21596 non-null int64
    sqft_lot          21596 non-null int64
    floors            21596 non-null float64
    waterfront        21596 non-null float64
    view              21596 non-null object
    condition         21596 non-null object
    grade             21596 non-null int64
    sqft_above        21596 non-null int64
    sqft_basement     21596 non-null int64
    yr_built          21596 non-null int64
    yr_renovated      21596 non-null float64
    sqft_living15     21596 non-null int64
    sqft_lot15        21596 non-null int64
    yrs_since_reno    21596 non-null float64
    subregion         21596 non-null object
    dtypes: datetime64[ns](1), float64(6), int64(9), object(3)
    memory usage: 3.9+ MB


### One-Hot Encoding Categorical Columns


```python
feats = ['view','condition','grade','subregion']

df = pd.get_dummies(df, drop_first=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>...</th>
      <th>condition_4</th>
      <th>condition_5</th>
      <th>subregion_east_urban</th>
      <th>subregion_north</th>
      <th>subregion_north_and_seattle</th>
      <th>subregion_seattle</th>
      <th>subregion_south_and_seattle</th>
      <th>subregion_south_rural</th>
      <th>subregion_south_urban</th>
      <th>subregion_vashon_island</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2014-10-13</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>1180</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014-12-09</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>2170</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2015-02-25</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>770</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2014-12-09</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>1050</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2015-02-18</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>8</td>
      <td>1680</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



## Exploring the Data


```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import six
%matplotlib inline
```


```python
# Quick look at the details of the dataset
df.describe()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>...</th>
      <th>condition_4</th>
      <th>condition_5</th>
      <th>subregion_east_urban</th>
      <th>subregion_north</th>
      <th>subregion_north_and_seattle</th>
      <th>subregion_seattle</th>
      <th>subregion_south_and_seattle</th>
      <th>subregion_south_rural</th>
      <th>subregion_south_urban</th>
      <th>subregion_vashon_island</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2.159600e+04</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>2.159600e+04</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>...</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>5.402920e+05</td>
      <td>3.371828</td>
      <td>2.115843</td>
      <td>2080.343165</td>
      <td>1.509983e+04</td>
      <td>1.494119</td>
      <td>0.006761</td>
      <td>7.657946</td>
      <td>1788.631506</td>
      <td>291.711660</td>
      <td>...</td>
      <td>0.262873</td>
      <td>0.078718</td>
      <td>0.272828</td>
      <td>0.055427</td>
      <td>0.034636</td>
      <td>0.288572</td>
      <td>0.013336</td>
      <td>0.042693</td>
      <td>0.244397</td>
      <td>0.005418</td>
    </tr>
    <tr>
      <td>std</td>
      <td>3.673760e+05</td>
      <td>0.904114</td>
      <td>0.768998</td>
      <td>918.122038</td>
      <td>4.141355e+04</td>
      <td>0.539685</td>
      <td>0.081946</td>
      <td>1.173218</td>
      <td>827.763251</td>
      <td>442.673703</td>
      <td>...</td>
      <td>0.440204</td>
      <td>0.269305</td>
      <td>0.445424</td>
      <td>0.228817</td>
      <td>0.182860</td>
      <td>0.453109</td>
      <td>0.114711</td>
      <td>0.202169</td>
      <td>0.429739</td>
      <td>0.073407</td>
    </tr>
    <tr>
      <td>min</td>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>370.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>3.220000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>1190.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1910.000000</td>
      <td>7.619000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>1560.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.068550e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2210.000000</td>
      <td>560.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>7.700000e+06</td>
      <td>11.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>13.000000</td>
      <td>9410.000000</td>
      <td>4820.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>




```python
# Histogram of the dataset
df.hist(figsize = (20,18))
plt.savefig('Figures/df_hist.png', dpi=300, bbox_inches='tight');
```


![png](student-Copy1_files/student-Copy1_71_0.png)


### Joint plots
Looking at joint plots to see which variables


```python
import scipy.stats as stats
```


```python
sns.jointplot('bathrooms','price', data=df, kind='reg').annotate(stats.pearsonr)
plt.show();
```

    /Users/Crisitna/anaconda3/envs/learn-env/lib/python3.6/site-packages/seaborn/axisgrid.py:1847: UserWarning: JointGrid annotation is deprecated and will be removed in a future release.
      warnings.warn(UserWarning(msg))



![png](student-Copy1_files/student-Copy1_74_1.png)



```python
sns.jointplot('bedrooms','price', data=df, kind='reg').annotate(stats.pearsonr)
plt.show();
```


![png](student-Copy1_files/student-Copy1_75_0.png)



```python
sns.jointplot('floors','price', data=df, kind='reg').annotate(stats.pearsonr)
plt.show();
```


![png](student-Copy1_files/student-Copy1_76_0.png)



```python
sns.jointplot('grade','price', data=df, kind='reg').annotate(stats.pearsonr)
plt.show();
```


![png](student-Copy1_files/student-Copy1_77_0.png)



```python
sns.jointplot('sqft_above','price', data=df, kind='reg').annotate(stats.pearsonr)
plt.show();
```


![png](student-Copy1_files/student-Copy1_78_0.png)



```python
sns.jointplot('sqft_living','price', data=df, kind='reg').annotate(stats.pearsonr)
plt.show();
```


![png](student-Copy1_files/student-Copy1_79_0.png)



```python
sns.jointplot('sqft_lot','price', data=df, kind='reg').annotate(stats.pearsonr)
plt.show();
```


![png](student-Copy1_files/student-Copy1_80_0.png)



```python
sns.jointplot('sqft_living15','price', data=df, kind='reg').annotate(stats.pearsonr)
plt.show();
```


![png](student-Copy1_files/student-Copy1_81_0.png)



```python
sns.jointplot('sqft_lot15','price', data=df, kind='reg').annotate(stats.pearsonr)
plt.show();
```


![png](student-Copy1_files/student-Copy1_82_0.png)



```python
sns.jointplot('yr_built','price', data=df, kind='reg').annotate(stats.pearsonr)
plt.show();
```


![png](student-Copy1_files/student-Copy1_83_0.png)



```python
sns.jointplot('yr_renovated','price', data=df, kind='reg').annotate(stats.pearsonr)
plt.show();
```


![png](student-Copy1_files/student-Copy1_84_0.png)



```python
sns.jointplot('yrs_since_reno','price', data=df, kind='reg').annotate(stats.pearsonr)
plt.show();
```


![png](student-Copy1_files/student-Copy1_85_0.png)



```python
# Create a heatmap to see if there is danage of multicorrelation
feats = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
         'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
         'sqft_living15', 'sqft_lot15', 'yrs_since_reno']
corr = df[feats].corr()
corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>yrs_since_reno</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>bedrooms</td>
      <td>1.000000</td>
      <td>0.527870</td>
      <td>0.593178</td>
      <td>0.033602</td>
      <td>0.183707</td>
      <td>-0.002054</td>
      <td>0.366174</td>
      <td>0.492543</td>
      <td>0.309261</td>
      <td>0.160736</td>
      <td>0.018626</td>
      <td>0.404532</td>
      <td>0.031892</td>
      <td>-0.169793</td>
    </tr>
    <tr>
      <td>bathrooms</td>
      <td>0.527870</td>
      <td>1.000000</td>
      <td>0.755755</td>
      <td>0.088368</td>
      <td>0.502574</td>
      <td>0.063628</td>
      <td>0.665834</td>
      <td>0.686664</td>
      <td>0.283459</td>
      <td>0.507166</td>
      <td>0.047173</td>
      <td>0.569878</td>
      <td>0.088297</td>
      <td>-0.532382</td>
    </tr>
    <tr>
      <td>sqft_living</td>
      <td>0.593178</td>
      <td>0.755755</td>
      <td>1.000000</td>
      <td>0.173449</td>
      <td>0.353941</td>
      <td>0.104635</td>
      <td>0.762776</td>
      <td>0.876448</td>
      <td>0.435152</td>
      <td>0.318140</td>
      <td>0.051056</td>
      <td>0.756400</td>
      <td>0.184337</td>
      <td>-0.339134</td>
    </tr>
    <tr>
      <td>sqft_lot</td>
      <td>0.033602</td>
      <td>0.088368</td>
      <td>0.173449</td>
      <td>1.000000</td>
      <td>-0.004824</td>
      <td>0.021458</td>
      <td>0.114726</td>
      <td>0.184134</td>
      <td>0.015424</td>
      <td>0.052939</td>
      <td>0.004977</td>
      <td>0.144756</td>
      <td>0.718203</td>
      <td>-0.051862</td>
    </tr>
    <tr>
      <td>floors</td>
      <td>0.183707</td>
      <td>0.502574</td>
      <td>0.353941</td>
      <td>-0.004824</td>
      <td>1.000000</td>
      <td>0.020794</td>
      <td>0.458783</td>
      <td>0.523970</td>
      <td>-0.245694</td>
      <td>0.489175</td>
      <td>0.003785</td>
      <td>0.280072</td>
      <td>-0.010734</td>
      <td>-0.500702</td>
    </tr>
    <tr>
      <td>waterfront</td>
      <td>-0.002054</td>
      <td>0.063628</td>
      <td>0.104635</td>
      <td>0.021458</td>
      <td>0.020794</td>
      <td>1.000000</td>
      <td>0.082817</td>
      <td>0.071776</td>
      <td>0.082803</td>
      <td>-0.024491</td>
      <td>0.073938</td>
      <td>0.083822</td>
      <td>0.030657</td>
      <td>0.006895</td>
    </tr>
    <tr>
      <td>grade</td>
      <td>0.366174</td>
      <td>0.665834</td>
      <td>0.762776</td>
      <td>0.114726</td>
      <td>0.458783</td>
      <td>0.082817</td>
      <td>1.000000</td>
      <td>0.756069</td>
      <td>0.168240</td>
      <td>0.447854</td>
      <td>0.015618</td>
      <td>0.713863</td>
      <td>0.120974</td>
      <td>-0.459322</td>
    </tr>
    <tr>
      <td>sqft_above</td>
      <td>0.492543</td>
      <td>0.686664</td>
      <td>0.876448</td>
      <td>0.184134</td>
      <td>0.523970</td>
      <td>0.071776</td>
      <td>0.756069</td>
      <td>1.000000</td>
      <td>-0.052130</td>
      <td>0.424017</td>
      <td>0.020637</td>
      <td>0.731756</td>
      <td>0.195069</td>
      <td>-0.433412</td>
    </tr>
    <tr>
      <td>sqft_basement</td>
      <td>0.309261</td>
      <td>0.283459</td>
      <td>0.435152</td>
      <td>0.015424</td>
      <td>-0.245694</td>
      <td>0.082803</td>
      <td>0.168240</td>
      <td>-0.052130</td>
      <td>1.000000</td>
      <td>-0.133043</td>
      <td>0.067302</td>
      <td>0.200478</td>
      <td>0.017559</td>
      <td>0.107069</td>
    </tr>
    <tr>
      <td>yr_built</td>
      <td>0.160736</td>
      <td>0.507166</td>
      <td>0.318140</td>
      <td>0.052939</td>
      <td>0.489175</td>
      <td>-0.024491</td>
      <td>0.447854</td>
      <td>0.424017</td>
      <td>-0.133043</td>
      <td>1.000000</td>
      <td>-0.202565</td>
      <td>0.326353</td>
      <td>0.070767</td>
      <td>-0.926404</td>
    </tr>
    <tr>
      <td>yr_renovated</td>
      <td>0.018626</td>
      <td>0.047173</td>
      <td>0.051056</td>
      <td>0.004977</td>
      <td>0.003785</td>
      <td>0.073938</td>
      <td>0.015618</td>
      <td>0.020637</td>
      <td>0.067302</td>
      <td>-0.202565</td>
      <td>1.000000</td>
      <td>0.000675</td>
      <td>0.004283</td>
      <td>-0.150771</td>
    </tr>
    <tr>
      <td>sqft_living15</td>
      <td>0.404532</td>
      <td>0.569878</td>
      <td>0.756400</td>
      <td>0.144756</td>
      <td>0.280072</td>
      <td>0.083822</td>
      <td>0.713863</td>
      <td>0.731756</td>
      <td>0.200478</td>
      <td>0.326353</td>
      <td>0.000675</td>
      <td>1.000000</td>
      <td>0.183506</td>
      <td>-0.325502</td>
    </tr>
    <tr>
      <td>sqft_lot15</td>
      <td>0.031892</td>
      <td>0.088297</td>
      <td>0.184337</td>
      <td>0.718203</td>
      <td>-0.010734</td>
      <td>0.030657</td>
      <td>0.120974</td>
      <td>0.195069</td>
      <td>0.017559</td>
      <td>0.070767</td>
      <td>0.004283</td>
      <td>0.183506</td>
      <td>1.000000</td>
      <td>-0.069404</td>
    </tr>
    <tr>
      <td>yrs_since_reno</td>
      <td>-0.169793</td>
      <td>-0.532382</td>
      <td>-0.339134</td>
      <td>-0.051862</td>
      <td>-0.500702</td>
      <td>0.006895</td>
      <td>-0.459322</td>
      <td>-0.433412</td>
      <td>0.107069</td>
      <td>-0.926404</td>
      <td>-0.150771</td>
      <td>-0.325502</td>
      <td>-0.069404</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, center=0, annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5) ;
```


![png](student-Copy1_files/student-Copy1_87_0.png)


### Removing Outliers


```python
# Create boxplot to see what outliers there may be
sns.boxplot(x=df['price'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a25cf2a58>




![png](student-Copy1_files/student-Copy1_89_1.png)



```python
# Find the iqr
from scipy.stats import iqr
data = df['price']
iqr = iqr(data, axis=0)
```


```python
# define lower outliers
lower_outliers = (df['price'].median() - 1.5 * iqr)
lower_outliers
```




    -34500.0




```python
# define upper outliers
upper_outliers = (df['price'].median() + 1.5 * iqr)
upper_outliers
```




    934500.0




```python
# drop upper outliers, there aren't any lower outliers 
df.drop(df[ df['price'] >= upper_outliers ].index , inplace=True)
```


```python
# Take another look at the boxplot and describe
print(df['price'].describe())
sns.boxplot(x=df['price'])
```

    count     19758.000000
    mean     458327.726086
    std      185541.230425
    min       78000.000000
    25%      310000.000000
    50%      429900.000000
    75%      580000.000000
    max      934000.000000
    Name: price, dtype: float64





    <matplotlib.axes._subplots.AxesSubplot at 0x110ebcf28>




![png](student-Copy1_files/student-Copy1_94_2.png)



```python
df2 = df.copy()
```

## Initial Modeling the Data


```python
# Define the problem
outcome = 'price'
x_cols = list(df2.columns)
x_cols.remove(outcome)
x_cols.remove('date')
[x_cols.remove(col) for col in df2.columns if "view" in col]
x_cols
```




    ['bedrooms',
     'bathrooms',
     'sqft_living',
     'sqft_lot',
     'floors',
     'waterfront',
     'grade',
     'sqft_above',
     'sqft_basement',
     'yr_built',
     'yr_renovated',
     'sqft_living15',
     'sqft_lot15',
     'yrs_since_reno',
     'condition_2',
     'condition_3',
     'condition_4',
     'condition_5',
     'subregion_east_urban',
     'subregion_north',
     'subregion_north_and_seattle',
     'subregion_seattle',
     'subregion_south_and_seattle',
     'subregion_south_rural',
     'subregion_south_urban',
     'subregion_vashon_island']




```python
# Some brief preprocessing
df2.columns = [col.replace(' ', '_') for col in df2.columns]
for col in x_cols:
    df2[col] = (df2[col] - df2[col].mean())/df2[col].std()
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>...</th>
      <th>condition_4</th>
      <th>condition_5</th>
      <th>subregion_east_urban</th>
      <th>subregion_north</th>
      <th>subregion_north_and_seattle</th>
      <th>subregion_seattle</th>
      <th>subregion_south_and_seattle</th>
      <th>subregion_south_rural</th>
      <th>subregion_south_urban</th>
      <th>subregion_vashon_island</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2014-10-13</td>
      <td>221900.0</td>
      <td>-0.351980</td>
      <td>-1.476456</td>
      <td>-1.022612</td>
      <td>-0.223466</td>
      <td>-0.869532</td>
      <td>-0.045039</td>
      <td>-0.481355</td>
      <td>-0.711585</td>
      <td>...</td>
      <td>-0.601463</td>
      <td>-0.283820</td>
      <td>-0.570128</td>
      <td>-0.250652</td>
      <td>-0.192919</td>
      <td>-0.628195</td>
      <td>-0.119458</td>
      <td>-0.220734</td>
      <td>1.663177</td>
      <td>-0.075841</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014-12-09</td>
      <td>538000.0</td>
      <td>-0.351980</td>
      <td>0.322719</td>
      <td>0.862914</td>
      <td>-0.182376</td>
      <td>0.997071</td>
      <td>-0.045039</td>
      <td>-0.481355</td>
      <td>0.714416</td>
      <td>...</td>
      <td>-0.601463</td>
      <td>-0.283820</td>
      <td>-0.570128</td>
      <td>-0.250652</td>
      <td>-0.192919</td>
      <td>1.591781</td>
      <td>-0.119458</td>
      <td>-0.220734</td>
      <td>-0.601228</td>
      <td>-0.075841</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2015-02-25</td>
      <td>180000.0</td>
      <td>-1.488325</td>
      <td>-1.476456</td>
      <td>-1.578775</td>
      <td>-0.111193</td>
      <td>-0.869532</td>
      <td>-0.045039</td>
      <td>-1.493553</td>
      <td>-1.302151</td>
      <td>...</td>
      <td>-0.601463</td>
      <td>-0.283820</td>
      <td>-0.570128</td>
      <td>3.989397</td>
      <td>-0.192919</td>
      <td>-0.628195</td>
      <td>-0.119458</td>
      <td>-0.220734</td>
      <td>-0.601228</td>
      <td>-0.075841</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2014-12-09</td>
      <td>604000.0</td>
      <td>0.784364</td>
      <td>1.402224</td>
      <td>0.035453</td>
      <td>-0.240242</td>
      <td>-0.869532</td>
      <td>-0.045039</td>
      <td>-0.481355</td>
      <td>-0.898838</td>
      <td>...</td>
      <td>-0.601463</td>
      <td>3.523182</td>
      <td>-0.570128</td>
      <td>-0.250652</td>
      <td>-0.192919</td>
      <td>1.591781</td>
      <td>-0.119458</td>
      <td>-0.220734</td>
      <td>-0.601228</td>
      <td>-0.075841</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2015-02-18</td>
      <td>510000.0</td>
      <td>-0.351980</td>
      <td>-0.037116</td>
      <td>-0.344365</td>
      <td>-0.160748</td>
      <td>-0.869532</td>
      <td>-0.045039</td>
      <td>0.530843</td>
      <td>0.008617</td>
      <td>...</td>
      <td>-0.601463</td>
      <td>-0.283820</td>
      <td>1.753904</td>
      <td>-0.250652</td>
      <td>-0.192919</td>
      <td>-0.628195</td>
      <td>-0.119458</td>
      <td>-0.220734</td>
      <td>-0.601228</td>
      <td>-0.075841</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
from statsmodels.formula.api import ols
```


```python
# Fitting the actual model
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=df2).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.733</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.733</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   2168.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 12 Aug 2020</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>10:14:32</td>     <th>  Log-Likelihood:    </th> <td>-2.5467e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 19758</td>      <th>  AIC:               </th>  <td>5.094e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 19732</td>      <th>  BIC:               </th>  <td>5.096e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    25</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
               <td></td>                  <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                   <td> 4.583e+05</td> <td>  682.368</td> <td>  671.672</td> <td> 0.000</td> <td> 4.57e+05</td> <td>  4.6e+05</td>
</tr>
<tr>
  <th>bedrooms</th>                    <td>-5716.1901</td> <td>  899.939</td> <td>   -6.352</td> <td> 0.000</td> <td>-7480.147</td> <td>-3952.233</td>
</tr>
<tr>
  <th>bathrooms</th>                   <td> 1.109e+04</td> <td> 1185.120</td> <td>    9.356</td> <td> 0.000</td> <td> 8764.826</td> <td> 1.34e+04</td>
</tr>
<tr>
  <th>sqft_living</th>                 <td> 3.116e+04</td> <td>  770.674</td> <td>   40.430</td> <td> 0.000</td> <td> 2.96e+04</td> <td> 3.27e+04</td>
</tr>
<tr>
  <th>sqft_lot</th>                    <td> 9872.1679</td> <td>  973.962</td> <td>   10.136</td> <td> 0.000</td> <td> 7963.120</td> <td> 1.18e+04</td>
</tr>
<tr>
  <th>floors</th>                      <td> 3151.3391</td> <td> 1044.387</td> <td>    3.017</td> <td> 0.003</td> <td> 1104.254</td> <td> 5198.425</td>
</tr>
<tr>
  <th>waterfront</th>                  <td> 9574.0693</td> <td>  708.533</td> <td>   13.513</td> <td> 0.000</td> <td> 8185.285</td> <td>  1.1e+04</td>
</tr>
<tr>
  <th>grade</th>                       <td> 5.395e+04</td> <td> 1138.266</td> <td>   47.397</td> <td> 0.000</td> <td> 5.17e+04</td> <td> 5.62e+04</td>
</tr>
<tr>
  <th>sqft_above</th>                  <td> 2.718e+04</td> <td>  794.412</td> <td>   34.218</td> <td> 0.000</td> <td> 2.56e+04</td> <td> 2.87e+04</td>
</tr>
<tr>
  <th>sqft_basement</th>               <td> 1.023e+04</td> <td>  746.718</td> <td>   13.705</td> <td> 0.000</td> <td> 8770.106</td> <td> 1.17e+04</td>
</tr>
<tr>
  <th>yr_built</th>                    <td>-3.932e+04</td> <td> 4977.492</td> <td>   -7.900</td> <td> 0.000</td> <td>-4.91e+04</td> <td>-2.96e+04</td>
</tr>
<tr>
  <th>yr_renovated</th>                <td> 3070.6745</td> <td> 1759.992</td> <td>    1.745</td> <td> 0.081</td> <td> -379.057</td> <td> 6520.406</td>
</tr>
<tr>
  <th>sqft_living15</th>               <td> 3.037e+04</td> <td> 1125.053</td> <td>   26.994</td> <td> 0.000</td> <td> 2.82e+04</td> <td> 3.26e+04</td>
</tr>
<tr>
  <th>sqft_lot15</th>                  <td>-1297.5637</td> <td>  995.035</td> <td>   -1.304</td> <td> 0.192</td> <td>-3247.917</td> <td>  652.789</td>
</tr>
<tr>
  <th>yrs_since_reno</th>              <td>-5147.5838</td> <td> 4938.830</td> <td>   -1.042</td> <td> 0.297</td> <td>-1.48e+04</td> <td> 4532.938</td>
</tr>
<tr>
  <th>condition_2</th>                 <td> 2566.4168</td> <td> 1792.569</td> <td>    1.432</td> <td> 0.152</td> <td> -947.169</td> <td> 6080.002</td>
</tr>
<tr>
  <th>condition_3</th>                 <td> 3.274e+04</td> <td> 8702.084</td> <td>    3.762</td> <td> 0.000</td> <td> 1.57e+04</td> <td> 4.98e+04</td>
</tr>
<tr>
  <th>condition_4</th>                 <td> 3.977e+04</td> <td> 8060.409</td> <td>    4.934</td> <td> 0.000</td> <td>  2.4e+04</td> <td> 5.56e+04</td>
</tr>
<tr>
  <th>condition_5</th>                 <td> 3.082e+04</td> <td> 4825.810</td> <td>    6.387</td> <td> 0.000</td> <td> 2.14e+04</td> <td> 4.03e+04</td>
</tr>
<tr>
  <th>subregion_east_urban</th>        <td> 4.186e+04</td> <td> 1557.402</td> <td>   26.878</td> <td> 0.000</td> <td> 3.88e+04</td> <td> 4.49e+04</td>
</tr>
<tr>
  <th>subregion_north</th>             <td> 3388.2409</td> <td> 1030.123</td> <td>    3.289</td> <td> 0.001</td> <td> 1369.114</td> <td> 5407.368</td>
</tr>
<tr>
  <th>subregion_north_and_seattle</th> <td> 5723.0756</td> <td>  931.467</td> <td>    6.144</td> <td> 0.000</td> <td> 3897.322</td> <td> 7548.829</td>
</tr>
<tr>
  <th>subregion_seattle</th>           <td>  5.14e+04</td> <td> 1753.271</td> <td>   29.316</td> <td> 0.000</td> <td>  4.8e+04</td> <td> 5.48e+04</td>
</tr>
<tr>
  <th>subregion_south_and_seattle</th> <td>-2231.3115</td> <td>  792.615</td> <td>   -2.815</td> <td> 0.005</td> <td>-3784.904</td> <td> -677.719</td>
</tr>
<tr>
  <th>subregion_south_rural</th>       <td>-1.682e+04</td> <td>  953.500</td> <td>  -17.644</td> <td> 0.000</td> <td>-1.87e+04</td> <td> -1.5e+04</td>
</tr>
<tr>
  <th>subregion_south_urban</th>       <td>-4.416e+04</td> <td> 1585.313</td> <td>  -27.853</td> <td> 0.000</td> <td>-4.73e+04</td> <td> -4.1e+04</td>
</tr>
<tr>
  <th>subregion_vashon_island</th>     <td>  161.8216</td> <td>  755.445</td> <td>    0.214</td> <td> 0.830</td> <td>-1318.914</td> <td> 1642.557</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>994.511</td> <th>  Durbin-Watson:     </th> <td>   1.987</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1975.255</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.366</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.365</td>  <th>  Cond. No.          </th> <td>2.38e+15</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 1.97e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import scipy.stats as stats
# Run a qq Test
plt.style.use('ggplot')

resid = model.resid

fig = sm.graphics.qqplot(resid, dist=stats.norm, line='45', fit=True)
```


![png](student-Copy1_files/student-Copy1_101_0.png)


## Model 2


```python
# create a list of all p-values
p_values = model.pvalues
```


```python
# Create a list of p-values that weren't significant
remove_col = list(p_values[p_values > 0.05].index)
```


```python
# Check the column that will be removed due to p-values
remove_col
```




    ['yr_renovated',
     'sqft_lot15',
     'yrs_since_reno',
     'condition_2',
     'subregion_vashon_island']




```python
# Remove those columns for the seoncd model
x_cols2 = x_cols.copy()
[x_cols2.remove(col) for col in remove_col if col in x_cols2]
x_cols2
```




    ['bedrooms',
     'bathrooms',
     'sqft_living',
     'sqft_lot',
     'floors',
     'waterfront',
     'grade',
     'sqft_above',
     'sqft_basement',
     'yr_built',
     'sqft_living15',
     'condition_3',
     'condition_4',
     'condition_5',
     'subregion_east_urban',
     'subregion_north',
     'subregion_north_and_seattle',
     'subregion_seattle',
     'subregion_south_and_seattle',
     'subregion_south_rural',
     'subregion_south_urban']




```python
# Create a heatmap to see if there is danage of multicorrelation so examine
# heat map again to remove any with high correlation
feats = x_cols2.copy()
corr = df[feats].corr()
corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>...</th>
      <th>condition_3</th>
      <th>condition_4</th>
      <th>condition_5</th>
      <th>subregion_east_urban</th>
      <th>subregion_north</th>
      <th>subregion_north_and_seattle</th>
      <th>subregion_seattle</th>
      <th>subregion_south_and_seattle</th>
      <th>subregion_south_rural</th>
      <th>subregion_south_urban</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>bedrooms</td>
      <td>1.000000</td>
      <td>0.498361</td>
      <td>0.604197</td>
      <td>0.023417</td>
      <td>0.154123</td>
      <td>-0.033767</td>
      <td>0.315051</td>
      <td>0.475398</td>
      <td>0.288109</td>
      <td>0.167963</td>
      <td>...</td>
      <td>0.006296</td>
      <td>-0.002717</td>
      <td>0.014833</td>
      <td>0.154208</td>
      <td>0.028341</td>
      <td>-0.025545</td>
      <td>-0.187670</td>
      <td>-0.025935</td>
      <td>0.010015</td>
      <td>0.046373</td>
    </tr>
    <tr>
      <td>bathrooms</td>
      <td>0.498361</td>
      <td>1.000000</td>
      <td>0.706107</td>
      <td>0.055663</td>
      <td>0.501368</td>
      <td>-0.023147</td>
      <td>0.599546</td>
      <td>0.625924</td>
      <td>0.214750</td>
      <td>0.556765</td>
      <td>...</td>
      <td>0.203110</td>
      <td>-0.169902</td>
      <td>-0.049433</td>
      <td>0.201058</td>
      <td>0.005060</td>
      <td>-0.063854</td>
      <td>-0.165640</td>
      <td>-0.072936</td>
      <td>0.059471</td>
      <td>-0.040422</td>
    </tr>
    <tr>
      <td>sqft_living</td>
      <td>0.604197</td>
      <td>0.706107</td>
      <td>1.000000</td>
      <td>0.151885</td>
      <td>0.331289</td>
      <td>-0.008918</td>
      <td>0.681410</td>
      <td>0.845157</td>
      <td>0.375713</td>
      <td>0.358938</td>
      <td>...</td>
      <td>0.114319</td>
      <td>-0.080545</td>
      <td>-0.042889</td>
      <td>0.254451</td>
      <td>0.031522</td>
      <td>-0.049256</td>
      <td>-0.245074</td>
      <td>-0.057037</td>
      <td>0.036911</td>
      <td>-0.030457</td>
    </tr>
    <tr>
      <td>sqft_lot</td>
      <td>0.023417</td>
      <td>0.055663</td>
      <td>0.151885</td>
      <td>1.000000</td>
      <td>-0.021578</td>
      <td>0.013734</td>
      <td>0.083940</td>
      <td>0.156707</td>
      <td>0.007928</td>
      <td>0.037054</td>
      <td>...</td>
      <td>-0.023334</td>
      <td>0.020152</td>
      <td>-0.008010</td>
      <td>0.022186</td>
      <td>0.002105</td>
      <td>-0.031712</td>
      <td>-0.156364</td>
      <td>-0.017826</td>
      <td>0.148892</td>
      <td>-0.010833</td>
    </tr>
    <tr>
      <td>floors</td>
      <td>0.154123</td>
      <td>0.501368</td>
      <td>0.331289</td>
      <td>-0.021578</td>
      <td>1.000000</td>
      <td>-0.014993</td>
      <td>0.455392</td>
      <td>0.527080</td>
      <td>-0.303927</td>
      <td>0.519821</td>
      <td>...</td>
      <td>0.329535</td>
      <td>-0.262169</td>
      <td>-0.135483</td>
      <td>0.056449</td>
      <td>-0.049512</td>
      <td>-0.042807</td>
      <td>0.054324</td>
      <td>-0.057756</td>
      <td>0.057391</td>
      <td>-0.112697</td>
    </tr>
    <tr>
      <td>waterfront</td>
      <td>-0.033767</td>
      <td>-0.023147</td>
      <td>-0.008918</td>
      <td>0.013734</td>
      <td>-0.014993</td>
      <td>1.000000</td>
      <td>-0.019401</td>
      <td>-0.019738</td>
      <td>0.017804</td>
      <td>-0.037530</td>
      <td>...</td>
      <td>-0.018882</td>
      <td>0.018797</td>
      <td>0.000077</td>
      <td>-0.025679</td>
      <td>-0.011290</td>
      <td>-0.008689</td>
      <td>-0.025795</td>
      <td>0.013740</td>
      <td>-0.009942</td>
      <td>0.023916</td>
    </tr>
    <tr>
      <td>grade</td>
      <td>0.315051</td>
      <td>0.599546</td>
      <td>0.681410</td>
      <td>0.083940</td>
      <td>0.455392</td>
      <td>-0.019401</td>
      <td>1.000000</td>
      <td>0.688629</td>
      <td>0.060563</td>
      <td>0.507236</td>
      <td>...</td>
      <td>0.225655</td>
      <td>-0.149648</td>
      <td>-0.117901</td>
      <td>0.298139</td>
      <td>0.010764</td>
      <td>-0.037781</td>
      <td>-0.145105</td>
      <td>-0.087518</td>
      <td>-0.004030</td>
      <td>-0.110182</td>
    </tr>
    <tr>
      <td>sqft_above</td>
      <td>0.475398</td>
      <td>0.625924</td>
      <td>0.845157</td>
      <td>0.156707</td>
      <td>0.527080</td>
      <td>-0.019738</td>
      <td>0.688629</td>
      <td>1.000000</td>
      <td>-0.177822</td>
      <td>0.468185</td>
      <td>...</td>
      <td>0.213278</td>
      <td>-0.147814</td>
      <td>-0.113409</td>
      <td>0.274124</td>
      <td>0.016862</td>
      <td>-0.064258</td>
      <td>-0.324318</td>
      <td>-0.057447</td>
      <td>0.089519</td>
      <td>-0.004266</td>
    </tr>
    <tr>
      <td>sqft_basement</td>
      <td>0.288109</td>
      <td>0.214750</td>
      <td>0.375713</td>
      <td>0.007928</td>
      <td>-0.303927</td>
      <td>0.017804</td>
      <td>0.060563</td>
      <td>-0.177822</td>
      <td>1.000000</td>
      <td>-0.150914</td>
      <td>...</td>
      <td>-0.159312</td>
      <td>0.107991</td>
      <td>0.117667</td>
      <td>-0.006819</td>
      <td>0.028797</td>
      <td>0.020727</td>
      <td>0.111109</td>
      <td>-0.005406</td>
      <td>-0.087252</td>
      <td>-0.048677</td>
    </tr>
    <tr>
      <td>yr_built</td>
      <td>0.167963</td>
      <td>0.556765</td>
      <td>0.358938</td>
      <td>0.037054</td>
      <td>0.519821</td>
      <td>-0.037530</td>
      <td>0.507236</td>
      <td>0.468185</td>
      <td>-0.150914</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.382358</td>
      <td>-0.255982</td>
      <td>-0.232489</td>
      <td>0.226028</td>
      <td>0.018660</td>
      <td>-0.056320</td>
      <td>-0.387047</td>
      <td>-0.062510</td>
      <td>0.129867</td>
      <td>0.082654</td>
    </tr>
    <tr>
      <td>sqft_living15</td>
      <td>0.376356</td>
      <td>0.515309</td>
      <td>0.725607</td>
      <td>0.143845</td>
      <td>0.264520</td>
      <td>0.002575</td>
      <td>0.653267</td>
      <td>0.706366</td>
      <td>0.111180</td>
      <td>0.372161</td>
      <td>...</td>
      <td>0.141525</td>
      <td>-0.083807</td>
      <td>-0.097098</td>
      <td>0.359683</td>
      <td>0.049488</td>
      <td>-0.064381</td>
      <td>-0.330578</td>
      <td>-0.080248</td>
      <td>0.048752</td>
      <td>-0.051531</td>
    </tr>
    <tr>
      <td>condition_3</td>
      <td>0.006296</td>
      <td>0.203110</td>
      <td>0.114319</td>
      <td>-0.023334</td>
      <td>0.329535</td>
      <td>-0.018882</td>
      <td>0.225655</td>
      <td>0.213278</td>
      <td>-0.159312</td>
      <td>0.382358</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.819612</td>
      <td>-0.386761</td>
      <td>-0.003299</td>
      <td>0.024224</td>
      <td>-0.027854</td>
      <td>0.005194</td>
      <td>0.014694</td>
      <td>0.024870</td>
      <td>-0.057359</td>
    </tr>
    <tr>
      <td>condition_4</td>
      <td>-0.002717</td>
      <td>-0.169902</td>
      <td>-0.080545</td>
      <td>0.020152</td>
      <td>-0.262169</td>
      <td>0.018797</td>
      <td>-0.149648</td>
      <td>-0.147814</td>
      <td>0.107991</td>
      <td>-0.255982</td>
      <td>...</td>
      <td>-0.819612</td>
      <td>1.000000</td>
      <td>-0.170716</td>
      <td>0.033908</td>
      <td>-0.017150</td>
      <td>0.024423</td>
      <td>-0.054524</td>
      <td>-0.018344</td>
      <td>-0.022255</td>
      <td>0.067549</td>
    </tr>
    <tr>
      <td>condition_5</td>
      <td>0.014833</td>
      <td>-0.049433</td>
      <td>-0.042889</td>
      <td>-0.008010</td>
      <td>-0.135483</td>
      <td>0.000077</td>
      <td>-0.117901</td>
      <td>-0.113409</td>
      <td>0.117667</td>
      <td>-0.232489</td>
      <td>...</td>
      <td>-0.386761</td>
      <td>-0.170716</td>
      <td>1.000000</td>
      <td>-0.040910</td>
      <td>-0.010684</td>
      <td>0.016723</td>
      <td>0.076220</td>
      <td>-0.007731</td>
      <td>-0.000402</td>
      <td>-0.015315</td>
    </tr>
    <tr>
      <td>subregion_east_urban</td>
      <td>0.154208</td>
      <td>0.201058</td>
      <td>0.254451</td>
      <td>0.022186</td>
      <td>0.056449</td>
      <td>-0.025679</td>
      <td>0.298139</td>
      <td>0.274124</td>
      <td>-0.006819</td>
      <td>0.226028</td>
      <td>...</td>
      <td>-0.003299</td>
      <td>0.033908</td>
      <td>-0.040910</td>
      <td>1.000000</td>
      <td>-0.142911</td>
      <td>-0.109994</td>
      <td>-0.358170</td>
      <td>-0.068110</td>
      <td>-0.125853</td>
      <td>-0.342794</td>
    </tr>
    <tr>
      <td>subregion_north</td>
      <td>0.028341</td>
      <td>0.005060</td>
      <td>0.031522</td>
      <td>0.002105</td>
      <td>-0.049512</td>
      <td>-0.011290</td>
      <td>0.010764</td>
      <td>0.016862</td>
      <td>0.028797</td>
      <td>0.018660</td>
      <td>...</td>
      <td>0.024224</td>
      <td>-0.017150</td>
      <td>-0.010684</td>
      <td>-0.142911</td>
      <td>1.000000</td>
      <td>-0.048358</td>
      <td>-0.157466</td>
      <td>-0.029944</td>
      <td>-0.055330</td>
      <td>-0.150707</td>
    </tr>
    <tr>
      <td>subregion_north_and_seattle</td>
      <td>-0.025545</td>
      <td>-0.063854</td>
      <td>-0.049256</td>
      <td>-0.031712</td>
      <td>-0.042807</td>
      <td>-0.008689</td>
      <td>-0.037781</td>
      <td>-0.064258</td>
      <td>0.020727</td>
      <td>-0.056320</td>
      <td>...</td>
      <td>-0.027854</td>
      <td>0.024423</td>
      <td>0.016723</td>
      <td>-0.109994</td>
      <td>-0.048358</td>
      <td>1.000000</td>
      <td>-0.121197</td>
      <td>-0.023047</td>
      <td>-0.042586</td>
      <td>-0.115995</td>
    </tr>
    <tr>
      <td>subregion_seattle</td>
      <td>-0.187670</td>
      <td>-0.165640</td>
      <td>-0.245074</td>
      <td>-0.156364</td>
      <td>0.054324</td>
      <td>-0.025795</td>
      <td>-0.145105</td>
      <td>-0.324318</td>
      <td>0.111109</td>
      <td>-0.387047</td>
      <td>...</td>
      <td>0.005194</td>
      <td>-0.054524</td>
      <td>0.076220</td>
      <td>-0.358170</td>
      <td>-0.157466</td>
      <td>-0.121197</td>
      <td>1.000000</td>
      <td>-0.075047</td>
      <td>-0.138671</td>
      <td>-0.377708</td>
    </tr>
    <tr>
      <td>subregion_south_and_seattle</td>
      <td>-0.025935</td>
      <td>-0.072936</td>
      <td>-0.057037</td>
      <td>-0.017826</td>
      <td>-0.057756</td>
      <td>0.013740</td>
      <td>-0.087518</td>
      <td>-0.057447</td>
      <td>-0.005406</td>
      <td>-0.062510</td>
      <td>...</td>
      <td>0.014694</td>
      <td>-0.018344</td>
      <td>-0.007731</td>
      <td>-0.068110</td>
      <td>-0.029944</td>
      <td>-0.023047</td>
      <td>-0.075047</td>
      <td>1.000000</td>
      <td>-0.026370</td>
      <td>-0.071825</td>
    </tr>
    <tr>
      <td>subregion_south_rural</td>
      <td>0.010015</td>
      <td>0.059471</td>
      <td>0.036911</td>
      <td>0.148892</td>
      <td>0.057391</td>
      <td>-0.009942</td>
      <td>-0.004030</td>
      <td>0.089519</td>
      <td>-0.087252</td>
      <td>0.129867</td>
      <td>...</td>
      <td>0.024870</td>
      <td>-0.022255</td>
      <td>-0.000402</td>
      <td>-0.125853</td>
      <td>-0.055330</td>
      <td>-0.042586</td>
      <td>-0.138671</td>
      <td>-0.026370</td>
      <td>1.000000</td>
      <td>-0.132718</td>
    </tr>
    <tr>
      <td>subregion_south_urban</td>
      <td>0.046373</td>
      <td>-0.040422</td>
      <td>-0.030457</td>
      <td>-0.010833</td>
      <td>-0.112697</td>
      <td>0.023916</td>
      <td>-0.110182</td>
      <td>-0.004266</td>
      <td>-0.048677</td>
      <td>0.082654</td>
      <td>...</td>
      <td>-0.057359</td>
      <td>0.067549</td>
      <td>-0.015315</td>
      <td>-0.342794</td>
      <td>-0.150707</td>
      <td>-0.115995</td>
      <td>-0.377708</td>
      <td>-0.071825</td>
      <td>-0.132718</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>21 rows × 21 columns</p>
</div>




```python
# There is still some high correlation
fig, ax = plt.subplots(figsize=(14,14))
sns.heatmap(corr, center=0, annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5) ;
```


![png](student-Copy1_files/student-Copy1_108_0.png)



```python
# columns with greater than .65
remove_col2 = ['sqft_above', 'condition_4']
```


```python
# remove those columns
outcome = 'price'
[x_cols2.remove(col) for col in remove_col2 if col in x_cols2]
x_cols2
```




    ['bedrooms',
     'bathrooms',
     'sqft_living',
     'sqft_lot',
     'floors',
     'waterfront',
     'grade',
     'sqft_basement',
     'yr_built',
     'sqft_living15',
     'condition_3',
     'condition_5',
     'subregion_east_urban',
     'subregion_north',
     'subregion_north_and_seattle',
     'subregion_seattle',
     'subregion_south_and_seattle',
     'subregion_south_rural',
     'subregion_south_urban']




```python
# Run the second model
predictors = '+'.join(x_cols2)
formula = outcome + '~' + predictors
model2 = ols(formula=formula, data=df2).fit()
model2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.731</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.731</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   2827.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 12 Aug 2020</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>10:14:38</td>     <th>  Log-Likelihood:    </th> <td>-2.5474e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 19758</td>      <th>  AIC:               </th>  <td>5.095e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 19738</td>      <th>  BIC:               </th>  <td>5.097e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    19</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
               <td></td>                  <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                   <td> 4.583e+05</td> <td>  684.601</td> <td>  669.481</td> <td> 0.000</td> <td> 4.57e+05</td> <td>  4.6e+05</td>
</tr>
<tr>
  <th>bedrooms</th>                    <td>-5738.5282</td> <td>  900.734</td> <td>   -6.371</td> <td> 0.000</td> <td>-7504.043</td> <td>-3973.013</td>
</tr>
<tr>
  <th>bathrooms</th>                   <td> 1.218e+04</td> <td> 1178.223</td> <td>   10.336</td> <td> 0.000</td> <td> 9868.949</td> <td> 1.45e+04</td>
</tr>
<tr>
  <th>sqft_living</th>                 <td> 5.996e+04</td> <td> 1558.247</td> <td>   38.482</td> <td> 0.000</td> <td> 5.69e+04</td> <td>  6.3e+04</td>
</tr>
<tr>
  <th>sqft_lot</th>                    <td> 8739.4292</td> <td>  723.922</td> <td>   12.072</td> <td> 0.000</td> <td> 7320.481</td> <td> 1.02e+04</td>
</tr>
<tr>
  <th>floors</th>                      <td> 3427.6881</td> <td> 1044.891</td> <td>    3.280</td> <td> 0.001</td> <td> 1379.613</td> <td> 5475.763</td>
</tr>
<tr>
  <th>waterfront</th>                  <td> 9717.6119</td> <td>  689.586</td> <td>   14.092</td> <td> 0.000</td> <td> 8365.965</td> <td> 1.11e+04</td>
</tr>
<tr>
  <th>grade</th>                       <td> 5.501e+04</td> <td> 1136.436</td> <td>   48.409</td> <td> 0.000</td> <td> 5.28e+04</td> <td> 5.72e+04</td>
</tr>
<tr>
  <th>sqft_basement</th>               <td>-5193.5303</td> <td>  958.403</td> <td>   -5.419</td> <td> 0.000</td> <td>-7072.081</td> <td>-3314.979</td>
</tr>
<tr>
  <th>yr_built</th>                    <td>-3.631e+04</td> <td> 1097.214</td> <td>  -33.094</td> <td> 0.000</td> <td>-3.85e+04</td> <td>-3.42e+04</td>
</tr>
<tr>
  <th>sqft_living15</th>               <td> 2.949e+04</td> <td> 1123.165</td> <td>   26.253</td> <td> 0.000</td> <td> 2.73e+04</td> <td> 3.17e+04</td>
</tr>
<tr>
  <th>condition_3</th>                 <td>-8403.3182</td> <td>  812.387</td> <td>  -10.344</td> <td> 0.000</td> <td>-9995.666</td> <td>-6810.971</td>
</tr>
<tr>
  <th>condition_5</th>                 <td> 7559.1877</td> <td>  752.083</td> <td>   10.051</td> <td> 0.000</td> <td> 6085.041</td> <td> 9033.334</td>
</tr>
<tr>
  <th>subregion_east_urban</th>        <td> 4.209e+04</td> <td> 1495.128</td> <td>   28.154</td> <td> 0.000</td> <td> 3.92e+04</td> <td>  4.5e+04</td>
</tr>
<tr>
  <th>subregion_north</th>             <td> 3461.9402</td> <td> 1001.998</td> <td>    3.455</td> <td> 0.001</td> <td> 1497.940</td> <td> 5425.941</td>
</tr>
<tr>
  <th>subregion_north_and_seattle</th> <td> 5883.6175</td> <td>  910.898</td> <td>    6.459</td> <td> 0.000</td> <td> 4098.181</td> <td> 7669.053</td>
</tr>
<tr>
  <th>subregion_seattle</th>           <td> 5.103e+04</td> <td> 1681.976</td> <td>   30.341</td> <td> 0.000</td> <td> 4.77e+04</td> <td> 5.43e+04</td>
</tr>
<tr>
  <th>subregion_south_and_seattle</th> <td>-2351.3465</td> <td>  783.784</td> <td>   -3.000</td> <td> 0.003</td> <td>-3887.629</td> <td> -815.064</td>
</tr>
<tr>
  <th>subregion_south_rural</th>       <td>-1.646e+04</td> <td>  929.414</td> <td>  -17.707</td> <td> 0.000</td> <td>-1.83e+04</td> <td>-1.46e+04</td>
</tr>
<tr>
  <th>subregion_south_urban</th>       <td>-4.398e+04</td> <td> 1515.584</td> <td>  -29.019</td> <td> 0.000</td> <td> -4.7e+04</td> <td> -4.1e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>976.609</td> <th>  Durbin-Watson:     </th> <td>   1.986</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1984.028</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.353</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.383</td>  <th>  Cond. No.          </th> <td>    8.59</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The likely hood multicollinearity dropped.

### Train Test Split


```python
# Import model validation functions
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
```


```python
# Create a training and test set
df_train, df_test = train_test_split(df2,test_size=0.20,shuffle=True,
                                     random_state=420)
df_train.shape, df_test.shape
```




    ((15806, 32), (3952, 32))




```python
## Train Model using the training data 
model3 = smf.ols(formula=formula, data=df_train).fit()
```


```python
# Get Model Predictions and Calculate Training R2
r2_train = r2_score(df_train['price'],model3.predict(df_train))
print(f'Training Data R-Squared = {round(r2_train,3)}')

# Get Model Predictions and Calculate Training R2
r2_test = r2_score(df_test['price'],model3.predict(df_test))
print(f'Training Data R-Squared = {round(r2_test,3)}')

#Display model summary
# model3.summary()
```

    Training Data R-Squared = 0.732
    Training Data R-Squared = 0.728


## Interpretations and presentation figures

### Bedrooms and Bathrooms


```python
# Set context to look better on a presentation
sns.set_context("talk")
sns.set_style('dark')
```


```python
# Graph Price per Bedrooms
fig, axs = plt.subplots(figsize=(24,10), ncols=2)

bed_line = sns.regplot(x=df['bedrooms'], y=df['price'],data=df, ax=axs[0])
bed_line.set_title('Price per Bedrooms')

bed_box = sns.boxplot(x=df['bedrooms'], y=df['price'],data=df, showfliers=False, ax=axs[1])
bed_box.set_title('Price per Bedrooms')

# Save plot as png
plt.savefig('figures/bedrooms_subplots.png', transparent=True);
```


![png](student-Copy1_files/student-Copy1_121_0.png)



```python
fig, axs = plt.subplots(figsize=(24,10), ncols=2)

bath_line = sns.regplot(x=df['bathrooms'], y=df['price'],data=df, ax=axs[0])
bath_line.set_title('Price per Bathrooms')

bath_box = sns.boxplot(x=df['bathrooms'], y=df['price'],data=df, showfliers=False, ax=axs[1])
bath_box.set_title('Price per Bathrooms')

# Save plot as png
plt.savefig('figures/bathrooms_subplots.png', transparent=True);
```


![png](student-Copy1_files/student-Copy1_122_0.png)


#### Interpretation:
- In general, number of bathrooms and bedrooms relate to price of house.
- More bedrooms and bathrooms, the more the house goes for.

### Grade


```python
fig, axs = plt.subplots(figsize=(24,10), ncols=2)

grade_line = sns.regplot(x=df['grade'], y=df['price'],data=df, ax=axs[0])
grade_line.set_title('Price per Kings County Grade')

grade_box = sns.boxplot(x=df['grade'], y=df['price'],data=df, showfliers=False, ax=axs[1])
grade_box.set_title('Price per Kings County Grade')

# Save plot as png
plt.savefig('figures/grade_subplots.png', transparent=True);
```


![png](student-Copy1_files/student-Copy1_125_0.png)


#### Interpretation
- The higher the grade, the high the sales price.


```python
sqft_line = sns.regplot(x=df['sqft_living'], y=df['price'],data=df)
sqft_line.set_title('Price per Living Space (sqft)')
sqft_line.set_xlabel('Living Space (in sqft)')
sqft_line.set_ylabel('Price')


# Save plot as png
plt.savefig('figures/sqft_living_subplots.png', transparent=True);
```


![png](student-Copy1_files/student-Copy1_127_0.png)


### Subregion Boxplot


```python
# Create subregion list
subregion_list = []
for x in df.columns:
    if 'subregion' in x:
        subregion_list.append(x)
```


```python
subregion_list
```




    ['subregion_east_urban',
     'subregion_north',
     'subregion_north_and_seattle',
     'subregion_seattle',
     'subregion_south_and_seattle',
     'subregion_south_rural',
     'subregion_south_urban',
     'subregion_vashon_island']




```python
# Def to create a column with subregions for boxplot
def get_region(row):
    for c in subregion_list:
        if row[c]==1:
            return c
```


```python
df['region'] = df.apply(get_region, axis=1)
```


```python
# Create dict of subregions and price
region_dict = {}
for x in subregion_list:

    try:
        temp = df.groupby(x).get_group(True)['price']
        region_dict[x] = temp.reset_index()
    except:
        print(x)
# region_dict
```


```python
# Create Dictionary without a Dataframe in it
region_dict2 = {}
for x in subregion_list:
    region_dict2[x] = region_dict[x]['price']
# region_dict2['subregion_east_urban']
```


```python
# Create a DataFrame of subregions and price for a boxplot
df_region = pd.DataFrame.from_dict(region_dict2)
df_region.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subregion_east_urban</th>
      <th>subregion_north</th>
      <th>subregion_north_and_seattle</th>
      <th>subregion_seattle</th>
      <th>subregion_south_and_seattle</th>
      <th>subregion_south_rural</th>
      <th>subregion_south_urban</th>
      <th>subregion_vashon_island</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>510000.0</td>
      <td>180000.0</td>
      <td>385000.0</td>
      <td>538000.0</td>
      <td>229500.0</td>
      <td>323000.0</td>
      <td>221900.0</td>
      <td>369900.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>662500.0</td>
      <td>310000.0</td>
      <td>280000.0</td>
      <td>604000.0</td>
      <td>255000.0</td>
      <td>360000.0</td>
      <td>257500.0</td>
      <td>309600.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>400000.0</td>
      <td>452000.0</td>
      <td>410000.0</td>
      <td>468000.0</td>
      <td>445838.0</td>
      <td>720000.0</td>
      <td>291850.0</td>
      <td>517534.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>719000.0</td>
      <td>230000.0</td>
      <td>660000.0</td>
      <td>530000.0</td>
      <td>390000.0</td>
      <td>390000.0</td>
      <td>189000.0</td>
      <td>705000.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>580500.0</td>
      <td>527000.0</td>
      <td>420000.0</td>
      <td>650000.0</td>
      <td>232000.0</td>
      <td>360000.0</td>
      <td>230000.0</td>
      <td>290000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
region_box = sns.boxplot(data=df_region, showfliers=False)
region_box.set_xticklabels(subregion_list, rotation=45, ha='right')
region_box.set_title('Sale Price per Subregion (Without Outliers)')
region_box.set_xticklabels(df_region, rotation=45, ha='right')
region_box.set_ylabel('Sale Price')
region_box.set_xlabel('Subregions')
bottom, top = region_box.get_ylim()
region_box.set_ylim(bottom + 1.5, top - 0.5) 

# Save plot as png
plt.savefig('figures/subregion_boxwhisker.png', transparent=True);
```


![png](student-Copy1_files/student-Copy1_136_0.png)


### Things not important


```python
yr_built = sns.regplot(x=df['yr_built'], y=df['price'])
yr_built.set_title('Year Built')
yr_built.set_ylabel('Sales Price')
yr_built.set_xlabel('Year Built')

# Save plot as png
plt.savefig('figures/yr_built.png', transparent=True);
```


![png](student-Copy1_files/student-Copy1_138_0.png)


### Overall Recommendations:
- Year built has little relationship with price
- Number of bathroom and bedrooms have a greater relationship with price
- The living area square feet relates to sales price positively
- Subregion: where the house is located can effect the sale price
- King County Grade is a god predictor of the sales price. 
- If you want to increase sales price focus on increasing King County Grade(high end finishes, custom designs, high quality material used), increasing living square feet by adding bedrooms and/or bathrooms  


```python

```
