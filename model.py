import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE



data = pd.read_csv("credit.csv")


#converting float to int
data["Age"] = data["Age"].astype(int)
data["Num_of_Loan"] = data["Num_of_Loan"].astype(int)
data["Num_Bank_Accounts"] =data["Num_Bank_Accounts"].astype(int)
data["Credit_History_Age"] = data["Credit_History_Age"].astype(int)
data["Num_Credit_Inquiries"] = data["Num_Credit_Inquiries"].astype(int)
data["Num_Credit_Card"] = data["Num_Credit_Card"].astype(int)
data["Interest_Rate"] = data["Interest_Rate"].astype(int)
data["Delay_from_due_date"] = data["Delay_from_due_date"].astype(int)


#drop irrelevent features
data1 = data.drop(columns=['ID','Customer_ID','Name','Month','Type_of_Loan','SSN','Amount_invested_monthly','Occupation','Credit_Utilization_Ratio','Total_EMI_per_month','Monthly_Inhand_Salary'])


#label encoding
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data1['Credit_Score'] = le.fit_transform(data1['Credit_Score'])
data1['Payment_of_Min_Amount'] = le.fit_transform(data1['Payment_of_Min_Amount'])
data1['Credit_Mix'] = le.fit_transform(data1['Credit_Mix'])
data1['Payment_Behaviour'] = le.fit_transform(data1['Payment_Behaviour'])




# Split the data into features (X) and target variable (y)

X = data1.drop("Credit_Score",axis=1)
Y = pd.DataFrame(data1["Credit_Score"])

from imblearn.over_sampling import SMOTE
sm = SMOTE()
X,Y = sm.fit_resample(X, Y)#over-sampling the data

from sklearn.preprocessing import MinMaxScaler#Perfrom MinMax Preprocessing
scaler = MinMaxScaler()

# Fit the scaler on the data and transform it
scaled_data = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split
#split the data into training, validation, and testing sets for model development.

Y= np.squeeze(Y)#squeeze() function is used to remove single-dimensional entries from the shape of an array.

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.20, random_state=42,)

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=1000,random_state=42)
model=rf_model.fit(X_train, Y_train)
rf_predictions= model.predict(X_test)
rf_Acc = accuracy_score(Y_test, rf_predictions)

pickle.dump(model,open('credit.pkl','wb'))

 
