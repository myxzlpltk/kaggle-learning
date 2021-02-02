from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

test_path = './data/test.csv'
test_data = pd.read_csv(test_path)

train_path = './data/train.csv'
train_data = pd.read_csv(train_path)

y = train_data.Survived

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# data = []
# for n_estimators in range(50, 80, 1):
#     model = RandomForestClassifier(n_estimators=n_estimators, max_depth=5, random_state=1)
#     model.fit(train_X, train_y)
#     data.append((n_estimators, accuracy_score(model.predict(val_X), val_y)))
#
# plt.plot(list(map(lambda x : x[0], data)), list(map(lambda x : x[1], data)))
# plt.title('Decision Tree Regression')
# plt.xlabel('N_Estimators')
# plt.ylabel('Percentage')
# plt.show()

model = RandomForestClassifier(n_estimators=60, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({
    'PassengerId': test_data.PassengerId,
    'Survived': predictions
})

print(output.compare(pd.read_csv('./data/ground_truth.csv')))