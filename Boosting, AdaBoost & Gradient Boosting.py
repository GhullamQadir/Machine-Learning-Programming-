from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor

# AdaBoost
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)
ada.fit(X_train, y_train)
print("AdaBoost Accuracy:", ada.score(X_test, y_test))

# Gradient Boosting (sklearn)
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)
print("Gradient Boosting Accuracy:", gb.score(X_test, y_test))

# Staged predictions (for finding optimal n_estimators)
test_scores = list(gb.staged_score(X_test, y_test))
best_n = np.argmax(test_scores) + 1
print(f"Optimal n_estimators: {best_n}")

# Regression
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gb_reg.fit(X_train, y_train)
