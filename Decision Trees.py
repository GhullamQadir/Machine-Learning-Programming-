from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.datasets import load_iris

# Classification Tree
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_clf = DecisionTreeClassifier(
    criterion='gini',        # or 'entropy', 'log_loss'
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
dt_clf.fit(X_train, y_train)

print("Train Accuracy:", dt_clf.score(X_train, y_train))
print("Test Accuracy:", dt_clf.score(X_test, y_test))

# Feature importance
for name, importance in zip(iris.feature_names, dt_clf.feature_importances_):
    print(f"{name}: {importance:.4f}")

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(dt_clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.show()

# Regression Tree
X_reg = np.sort(5 * np.random.rand(100, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.1, 100)
dt_reg = DecisionTreeRegressor(max_depth=5)
dt_reg.fit(X_reg, y_reg)

# Pruning (cost complexity)
path = dt_clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
trees = [DecisionTreeClassifier(random_state=42, ccp_alpha=ccp) for ccp in ccp_alphas]
scores = [tree.fit(X_train, y_train).score(X_test, y_test) for tree in trees]
best_alpha = ccp_alphas[np.argmax(scores)]
print(f"Best ccp_alpha: {best_alpha:.6f}")
