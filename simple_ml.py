from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
	 [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
	 [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male',
	 'male', 'female', 'male', 'female', 'male']

Z = [[190, 70, 43]]

clf_tree = DecisionTreeClassifier()
clf_rf = RandomForestClassifier()
clf_nn = KNeighborsClassifier()

fit_tree = clf_tree.fit(X, Y)
fit_rf = clf_rf.fit(X, Y)
fit_nn = clf_nn.fit(X, Y)

pred_tree = fit_tree.predict(Z)
pred_rf = fit_rf.predict(Z)
pred_nn = fit_nn.predict(Z)

print("Decision Tree:", pred_tree)
print("Random Forest:", pred_rf)
print("Near Neighbors:", pred_nn)