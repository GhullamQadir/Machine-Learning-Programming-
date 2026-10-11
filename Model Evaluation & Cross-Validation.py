from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, KFold,
    GridSearchCV, RandomizedSearchCV, learning_curve, validation_curve
)
from sklearn.metrics import make_scorer

# Cross-validation
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Simple CV
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Multiple metrics
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)
for metric in scoring:
    print(f"{metric}: {cv_results[f'test_{metric}'].mean():.4f}")

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}: Train={len(train_idx)}, Val={len(val_idx)}")

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)
print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)
print("Test Score:", grid.score(X_test, y_test))

# Randomized Search
from scipy.stats import randint
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': randint(2, 20)
}
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist, n_iter=50, cv=5, scoring='f1', n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)

# Learning Curves
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)
plt.plot(train_sizes, train_mean, 'o-', label='Training')
plt.plot(train_sizes, val_mean, 'o-', label='Validation')
plt.xlabel('Training Set Size'); plt.ylabel('Accuracy'); plt.legend(); plt.show()

# Validation Curves
param_range = [1, 3, 5, 7, 10, 15, 20]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(random_state=42), X, y,
    param_name='max_depth', param_range=param_range, cv=5, scoring='accuracy', n_jobs=-1
)
plt.plot(param_range, train_scores.mean(axis=1), 'o-', label='Training')
plt.plot(param_range, val_scores.mean(axis=1), 'o-', label='Validation')
plt.xlabel('max_depth'); plt.ylabel('Accuracy'); plt.legend(); plt.show()
