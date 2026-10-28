<H3>ENTER YOUR NAME: M AADHAVAN NAGARAJAN</H3>
<H3>ENTER YOUR REGISTER NO. 212225040001</H3>
<H3>EX. NO.1</H3>
<H3>28/10/2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import csv

d=pd.read_csv("FuelConsumption (2).csv")
print("FuelConsumption Content:")
print()
print(d.head())
print()
print("To check whether there is any null value:")
print(d.isnull().values.any())
print()
X=d.drop(['MAKE','MODEL','VEHICLECLASS','TRANSMISSION',
    'FUELTYPE', 'CO2EMISSIONS',
    'FUELCONSUMPTION_COMB_MPG'],axis=1)
y=d['CO2EMISSIONS']
print("After Dropping off unwanted content from the Dataset:")
print()
print(X.head())
print()
print(y.head())

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25)

scaler=StandardScaler()

xtrainscaler=scaler.fit_transform(xtrain)
xtestscaler=scaler.transform(xtest)

xtrainscaled_data = pd.DataFrame(xtrainscaler, columns=xtrain.columns)

print()
print("After Normalising and spliting of data from the Dataset:")
print()
print(xtrainscaled_data.head())
print()
print(ytrain.head())


```


## OUTPUT:
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/cd02543f-116a-464b-94a4-8346dd16fb39" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/90bf5f5d-c0f5-47a5-9b42-b47f1e872c2e" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


