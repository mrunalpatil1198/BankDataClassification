import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from django import forms

from sklearn import metrics


# Generic function for making a classification model and accessing performance:

def classification_model(model, data, predictors, outcome):
    # Fit the model:
    model.fit(data[predictors], data[outcome])

    # Make predictions on training set:
    predictions = model.predict(data[predictors])

    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

def input(gender,status,dependents,education,selfemp,income,cincome,loan_amount,loan_amount_term,credit_history,property_area):
    test_gender=gender
    test_married=status
    test_dependents=dependents
    test_education=education
    test_self_employed=selfemp
    test_ApplicantIncome=float(income)
    test_CoapplicantIncome=float(cincome)
    test_LoanAmount=float(loan_amount)
    test_LoanAmountTerm=float(loan_amount_term)
    test_creditHistory=int(credit_history)
    test_PropertyArea=property_area
    test_LoanID = "LP00888"
    test_Type = "Test"

    #return test_gender,test_married,test_dependents,test_education,test_self_employed,test_ApplicantIncome,test_CoapplicantIncome,test_LoanAmount,test_LoanAmountTerm,test_creditHistory,test_PropertyArea


    df = pd.read_csv(r'C:\Users\hp\Desktop\BE Project\train_u6lujuX_CVtuZ9i.csv')
    df = df.drop(['Loan_ID'], axis=1)
#test = pd.read_csv(r'C:\Users\Admin\Desktop\BE Project\Datasets\Loan Prediction\test_Y3wMUE5_7gLdaTN.csv')
    test=pd.DataFrame({"Gender":[test_gender], "Married":[test_married], "Dependants":[test_dependents], "Education":[test_education],
                   "Self_Employed":[test_self_employed], "ApplicantIncome":[test_ApplicantIncome], "CoapplicantIncome":[test_CoapplicantIncome],
                   "LoanAmount": [test_LoanAmount], "Loan_Amount_Term":[test_LoanAmountTerm], "Credit_History":[test_creditHistory],
                   "Property_Area": [test_PropertyArea]})
# Store total number of observation in training dataset
    df_length =len(df)

# Store total number of columns in testing data set
    test_col = len(test.columns)

    df['Self_Employed'].fillna('No', inplace=True)

# Add both ApplicantIncome and CoapplicantIncome to TotalIncome
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

# Perform log transformation of TotalIncome to make it closer to normal
    df['LoanAmount_log'] = np.log(df['LoanAmount'])

# Impute missing values for Gender
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

# Impute missing values for Married
    df['Married'].fillna(df['Married'].mode()[0],inplace=True)

# Impute missing values for Dependents
    df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

# Impute missing values for Credit_History
    df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

# Convert all non-numeric values to number
    cat=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']

    for var in cat:
        le = preprocessing.LabelEncoder()
        df[var]=le.fit_transform(df[var].astype('str'))
        print(df.dtypes)



#Combining both train and test dataset

#Create a flag for Train and Test Data set
    df['Type']='Train'
    test['Type']='Test'
    fullData = pd.concat([df,test], axis=0)

#Identify categorical and continuous variables
    #ID_col = ['Loan_ID']
    target_col = ["Loan_Status"]
    cat_cols = ['Credit_History','Dependents','Gender','Married','Education','Property_Area','Self_Employed']

#Imputing Missing values with mean for continuous variable
    fullData['LoanAmount'].fillna(fullData['LoanAmount'].mean(), inplace=True)
    fullData['LoanAmount_log'].fillna(fullData['LoanAmount_log'].mean(), inplace=True)
    fullData['Loan_Amount_Term'].fillna(fullData['Loan_Amount_Term'].mean(), inplace=True)
    fullData['ApplicantIncome'].fillna(fullData['ApplicantIncome'].mean(), inplace=True)
    fullData['CoapplicantIncome'].fillna(fullData['CoapplicantIncome'].mean(), inplace=True)

#Imputing Missing values with mode for categorical variables
    fullData['Gender'].fillna(fullData['Gender'].mode()[0], inplace=True)
    fullData['Married'].fillna(fullData['Married'].mode()[0], inplace=True)
    fullData['Dependents'].fillna(fullData['Dependents'].mode()[0], inplace=True)
    fullData['Loan_Amount_Term'].fillna(fullData['Loan_Amount_Term'].mode()[0], inplace=True)
    fullData['Credit_History'].fillna(fullData['Credit_History'].mode()[0], inplace=True)

#Create a new column as Total Income

    fullData['TotalIncome']=fullData['ApplicantIncome'] + fullData['CoapplicantIncome']

    fullData['TotalIncome_log'] = np.log(fullData['TotalIncome'])

#create label encoders for categorical features
    for var in cat_cols:
        number = LabelEncoder()
        fullData[var] = number.fit_transform(fullData[var].astype('str'))

    train_modified=fullData[fullData['Type']=='Train']
    test_modified=fullData[fullData['Type']=='Test']
    train_modified["Loan_Status"] = number.fit_transform(train_modified["Loan_Status"].astype('str'))

    print(train_modified.describe())
    for col in train_modified.columns:
        print(col)

    from sklearn.ensemble import RandomForestClassifier


#predictors_Logistic=['Credit_History','Education','Gender']

    predictors_Logistic=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area', 'TotalIncome']
    x_train = train_modified[list(predictors_Logistic)].values
#x_train = train_modified[:, [1, 11]]
    y_train = train_modified["Loan_Status"].values
    x_test=test_modified[list(predictors_Logistic)].values
#x_test=test_modified[:, [1, 11]].values

# Create Random Forest object
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_depth=8, random_state = 0)


# Train the model using the training sets
    classifier.fit(x_train, y_train)

#Predict Output
    predicted = classifier.predict(x_test)
    print("PREDICTED", predicted)

#Reverse encoding for predicted outcome
    predicted = number.inverse_transform(predicted)

#Store it to test dataset
    test_modified['Loan_Status'] = predicted

    outcome_var = 'Loan_Status'
#classification_model(classifier, df, predictors_Logistic, outcome_var)
    return predicted

