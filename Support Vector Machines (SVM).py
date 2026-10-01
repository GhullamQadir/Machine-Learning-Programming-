from sklearn.svm import SVC, SVR
from sklearn.datasets import make_moons, make_regression

# Classification with SVM
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)

# RBF Kernel SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)

# Polynomial Kernel
svm_poly = SVC(kernel='poly', degree=3, C=1.0)
svm_poly.fit(X_train, y_train)

print("Linear:", svm_linear.score(X_test, y_test))
print("RBF:", svm_rbf.score(X_test, y_test))
print("Poly:", svm_poly.score(X_test, y_test))

# Decision boundary visualization
def plot_decision_boundary(model, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.show()

plot_decision_boundary(svm_rbf, X, y)

# Regression (SVR)
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
svr = SVR(kernel='rbf', C=100, epsilon=0.1)
svr.fit(X_reg, y_reg)
print("SVR R²:", svr.score(X_reg, y_reg))
