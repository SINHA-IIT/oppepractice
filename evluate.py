import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

def main():
    df = pd.read_csv('data/data_iris.csv')
    X = df[['sepal_length','sepal_width','petal_length','petal_width']]
    y = df['species']
    model = joblib.load('model/model.pkl')
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print("Accuracy:", acc)
    print(classification_report(y, preds))

if __name__ == "__main__":
    main()
