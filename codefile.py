# Classification Problem
# importsimport numpy as npimport pandas as pdimport matplotlib.pyplot as pltimport seaborn as seimport warningsfrom sklearn.model_selection import train_test_splitfrom sklearn.preprocessing import LabelEncoderfrom sklearn.metrics import classification_report,plot_confusion_matriximport blobcity as bc
warnings.filterwarnings('ignore')

### Data Fetch
# Pandas is an open-source, BSD-licensed library providing high-performance,
# easy-to-use data manipulation and data analysis tools.
# Data Fetchfile='https://raw.githubusercontent.com/Thilakraj1998/Datasets_general/main/BreastCancer1.csv'df=pd.read_csv(file)df.head()

### Feature Selection
# It is the process of reducing the number of input variables when developing a predictive model.
# Used to reduce the number of input variables to reduce the computational cost of modelling and,
# in some cases,to improve the performance of the model.
# Selected Columnsfeatures=['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'radius_se', 'compactness_se', 'concavity_se', 'concave points_se', 'texture_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']target='diagnosis'
# X & YX=df[features]Y=df[target]

### Data Encoding
# Converting the string classes data in the datasets
# by encoding them to integer either using OneHotEncoding or LabelEncoding
# Handling Target Encodingdef EncodeY(Y):	actual_target=np.sort(pd.unique(Y), axis=-1, kind='mergesort')	Y=LabelEncoder().fit_transform(Y)	encoded_target=[xi for xi in range(len(actual_target))]	print('Encoded Target: {} to {}'.format(actual_target,encoded_target))	return YY=EncodeY(Y)

### Correlation Matrix
# In order to check the correlation between the features, we will plot a correlation matrix.
# It is effective in summarizing a large amount of data where the goal is to see patterns.

# Correlation Matrix
f,ax = plt.subplots(figsize=(18, 18))matrix = np.triu(X.corr())se.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, mask=matrix)plt.show()

### Train & Test
# The train-test split is a procedure for evaluating the performance of an algorithm.
# The procedure involves taking a dataset and dividing it into two subsets.
# The first subset is utilized to fit/train the model.
# The second subset is used for prediction.
# The main motive is to estimate the performance of the model on new data.
# Data split for training and testingX_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=123)
# Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers.
# These neural networks attempt to simulate the behavior of the human brain—albeit far from matching its ability—allowing it to “learn” from large amounts of data.
# While a neural network with a single layer can still make approximate predictions,
# additional hidden layers can help to optimize and refine for accuracy.

model = bc.load('PICKLE FILE PATH')#summarynn=model.modelnn.summary()
nn.fit(X_train,Y_train,epochs=10)
### Accuracy Metrics
# Performance metrics are a part of every machine learning pipeline. 
# They tell you if you're making progress, and put a number on it. All machine learning models,
# whether it's linear regression, or a SOTA technique like BERT, need a metric to judge performance.
y_pred=nn.predict(X_test)y_pred=np.round(y_pred)# Classification Reportprint(classification_report(Y_test,y_pred))
