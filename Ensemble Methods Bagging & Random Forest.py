from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Random Forest Classifier
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',     # or 'log2', int, float
    bootstrap=True,
    oob_score=True,          # Out-of-bag score
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

print("Train Accuracy:", rf.score(X_train, y_train))
print("Test Accuracy:", rf.score(X_test, y_test))
print("OOB Score:", rf.oob_score_)

# Feature importance
importances = pd.Series(rf.feature_importances_, index=data.feature_names)
importances.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title("Top 10 Feature Importances"); plt.show()

# Bagging (manual)
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
bagging.fit(X_train, y_train)
print("Bagging Accuracy:", bagging.score(X_test, y_test))
