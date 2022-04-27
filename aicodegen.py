# Classification Problem
# importsimport numpy as npimport pandas as pdimport matplotlib.pyplot as pltimport seaborn as seimport warningsfrom sklearn.model_selection import train_test_splitfrom sklearn.preprocessing import LabelEncoderfrom sklearn.metrics import classification_report,plot_confusion_matriximport blobcity as bc
warnings.filterwarnings('ignore')
# Data Fetchfile='https://raw.githubusercontent.com/Thilakraj1998/Datasets_general/main/BreastCancer1.csv'df=pd.read_csv(file)df.head()
# Selected Columnsfeatures=['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'radius_se', 'compactness_se', 'concavity_se', 'concave points_se', 'texture_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']target='diagnosis'
# X & YX=df[features]Y=df[target]
# Handling Target Encodingdef EncodeY(Y):	actual_target=np.sort(pd.unique(Y), axis=-1, kind='mergesort')	Y=LabelEncoder().fit_transform(Y)	encoded_target=[xi for xi in range(len(actual_target))]	print('Encoded Target: {} to {}'.format(actual_target,encoded_target))	return YY=EncodeY(Y)

# Correlation Matrix
f,ax = plt.subplots(figsize=(18, 18))matrix = np.triu(X.corr())se.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, mask=matrix)plt.show()
# Data split for training and testingX_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=123)

model = bc.load('PICKLE FILE PATH')#summarynn=model.modelnn.summary()
nn.fit(X_train,Y_train,epochs=10)y_pred=nn.predict(X_test)y_pred=np.round(y_pred)# Classification Reportprint(classification_report(Y_test,y_pred))
