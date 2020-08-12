
## Business Case
In this notebook, we'll explore, and model King County House Sale dataset with a multivariate linear regression to predict the sale price of houses by answering the following questions:
- What are some of the important aspects when it comes to pricing your home for sale?
- Which things are not as important when trying to sell your home?
- How can we improve the potential sale price of your home?

## Methodology:

1. Obtaining and initial examination of data from King Country House Sale
2. Scrubbing the data to prepare it for modeling
3. Exploration of the data
4. Modeling the data
5. Reexamination of the data
6. A second model of the data
7. Interpretations and recommendations based on the models and examination of the data.


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

### Scrubbing of data

Now that we have an idea of the King County House Sales dataset we will be working with, we can start scrubbing the data to prepare it for exploration and modeling.

Here is a quick overview of what changes were made to the dataset during this step.

* **id** - Dropped, since it will not add anything to the final model.
* **date** - Converted to date-time
* **price** -  Examined for nulls of place holders
* **bedrooms** -  Examined for nulls and place holders
* **bathrooms** -  Examined for nulls and place holders
* **sqft_living** -  Examined for nulls and place holders
* **sqft_lots** -  Examined for nulls and place holders
* **floors** -  Examined for nulls and place holders
* **waterfront** - Examined for nulls and nulls replaced with 0
* **view** - Examined for nulls and nulls replaced with 0, converted into objects
* **condition** - Examined for nulls and place holders, converted to objects
* **grade** - Examined for nulls and place holders
* **sqft_above** - Examined for nulls and place holders
* **sqft_basement** - Place holders replaced with (sqft_living - sqft_above)
* **yr_built** -   Examined for nulls
* **yr_renovated** - Nulls and 0s replace with yr_built
* **yrs_since_reno** - Column created
* **zipcode** - This column was dropped in favor of the created subregion column
* **subregion** - Column created
* **lat** - Latitude coordinate
* **long** - Longitude coordinate
* **sqft_living15** - The square footage of interior housing living space for the nearest 15 neighbors
* **sqft_lot15** - The square footage of the land lots of the nearest 15 neighbors




### One-Hot Encoding Categorical Columns
Next object columns were one-hot encoded to use in modeling. This included 'view', 'condition', 'grade', 'subregion'.


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


### Histogram of Data

Here we can take a quick look at the quanitative data and get an idea of the normality of them.

```python
# Histogram of the dataset
df.hist(figsize = (20,18))
plt.savefig('Figures/df_hist.png', dpi=300, bbox_inches='tight');
```


![png](student-Copy1_files/student-Copy1_71_0.png)


### Joint plots
Looking at joint plots to get an idea of how the variables relate to sale price. 


### Heatmap
Next, we look at a heat map to help weed out any variable that could cause multicorrelarity in the model.


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



Here we see there are a few outliers we need to deal with. We can do this by dropping them from the dataframe.


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

Now that the data is all set we can run the first model.


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


Here we see that we still have a multicollinearity problem that will will need to deal with.



## Model 2

First, we are going to drop any variable that aren't significant due to their p-values

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


Now it is time to deal with the multicollinearity problem by looking at another heatmap and dropping columns that correlate with each other.


```python
fig, ax = plt.subplots(figsize=(14,14))
sns.heatmap(corr, center=0, annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5) ;
```


![png](student-Copy1_files/student-Copy1_108_0.png)

It looks like we can drop sqft_above and condition_4

```python
remove_col2 = ['sqft_above', 'condition_4']
```

Now we can run the second model.


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

 To test our second model

```python
# Import model validation functions
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
```


```python
# Create a training and test set
df_train, df_test = train_test_split(df2,test_size=0.20,shuffle=True,
                                     random_state=420)
```


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


![png](student-Copy1_files/student-Copy1_121_0.png)


![png](student-Copy1_files/student-Copy1_122_0.png)


- In general, number of bathrooms and bedrooms relate to price of house.
- More bedrooms and bathrooms, the more the house goes for.

### Living Space

![png](student-Copy1_files/student-Copy1_127_0.png)

- A larger living space can increase the sale price

### Grade


![png](student-Copy1_files/student-Copy1_125_0.png)


- The higher the grade, the high the sales price.


### Subregion Boxplot


![png](student-Copy1_files/student-Copy1_136_0.png)


- The old addage "Location, Location, Location" is supported with our finding with East Urban and Vashon Island generally selling for more. 

### Things not important

![png](student-Copy1_files/student-Copy1_138_0.png)

- The age of the house does not seem to be significantly related to price.


### Overall Recommendations:
- Year built has little relationship with price
- Number of bathroom and bedrooms have a greater relationship with price
- The living area square feet relates to sales price positively
- Subregion: where the house is located can effect the sale price
- King County Grade is a god predictor of the sales price. 
- If you want to increase sales price focus on increasing King County Grade(high end finishes, custom designs, high quality material used), increasing living square feet by adding bedrooms and/or bathrooms  


```python

```
