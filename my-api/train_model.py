from seaborn.utils import locator_to_legend_entries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
iris=load_iris()
x,y=iris.data,iris.target
print("|"*50)
print("TRAINING IRIS FLOWER PREDICTIONS")
print("|"*50)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(f' samples : {len(x)}')

print(f" FEATURES : {iris.feature_names}")
print(f" TARGETS : {iris.target_names}")
print(f" spitted data for training : {len(x_train)}")
print(f" spitted data for testing : {len(x_test)}")
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
predictions=model.predict(x_test)
accuracy=accuracy_score(y_test,predictions)
print(f' accuracy : {accuracy*100:.2f}%')
with open("iris.pkl", "wb") as file:
    pickle.dump(model,file)
print(f" model is saved")
loaded_model =pickle.load(open("iris.pkl", "rb"))
prediction=loaded_model.predict([[5.1,3.5,1.4,0.2]])
print(f" predicted features: {iris.feature_names[prediction[0]]}")
print(f" predicted species : {iris.target_names[prediction[0]]}")
