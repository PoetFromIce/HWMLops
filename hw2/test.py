from sklearn.metrics import classification_report
import pickle

# Загрузка тестового датасета и модели
with open('X_test.pickle', 'rb') as f:
    X_test = pickle.load(f)

with open('y_test.pickle', 'rb') as f:
    y_test = pickle.load(f)

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

# Анализ качества модели
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
