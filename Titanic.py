# Predict if survived or no 
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Titanic-Dataset.csv')

translator = LabelEncoder()

df = df.dropna()

model = GaussianNB()

newGender = translator.fit_transform(df['Sex'])

df['Gender'] = newGender

X = df[['Gender', 'Age', 'Pclass']]
Y = df['Survived']

X_learn, X_test, Y_learn, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=42)

model.fit(X_learn, Y_learn)

predict = model.predict(X_test)

prediction = model.predict_proba(X_test)

score = accuracy_score(Y_test, predict)

print(f'Accuracy of model is : {score * 100}')

def theBigQuestion(survive, percent):
    for results in zip(survive, percent):
        print(f'Prediction: {'Didnt Survive' if results[0] == 0 else 'Survived'}')
        print(f'Survival rate: {results[1][1] *100:.2f}%')
        print(f'Death rate: {results[1][0] *100:.2f}%')

print(model.classes_)

# Testing #1

print(' == Test One == ')

Gender = 0
Age = 90
Pclass = 1

Xnew = pd.DataFrame([[Gender, Age, Pclass]], columns=['Gender', 'Age', 'Pclass'])

predictOne = model.predict(Xnew)

XnewTest = model.predict_proba(Xnew)

theBigQuestion(predictOne, XnewTest)


# Testing #2 

print(' == Test Two == ')

Gender = 1
Age = 30
Pclass = 3

XsecondVar = pd.DataFrame([[Gender, Age, Pclass]], columns=['Gender', 'Age', 'Pclass'])

predictTwo = model.predict(XsecondVar)

XsecondTest = model.predict_proba(XsecondVar)

theBigQuestion(predictTwo, XsecondTest)