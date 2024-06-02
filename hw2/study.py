from sklearn.ensemble import RandomForestClassifier
import pickle

# Загрузка тренировочного датасета
with open('X_train.pickle', 'rb') as f:
    X_train = pickle.load(f)

with open('y_train.pickle', 'rb') as f:
    y_train = pickle.load(f)

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Сохранение модели
with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)
