# Layers definitions
from keras import backend as K
for l in range(len(model.layers)):
    print(l, model.layers[l])
    # feature extraction layer
getFeature = K.function([model.layers[0].input],
                        [model.layers[23].output])
train_filenames = getFeature([train_filenames])[0]
X_test = getFeature([X_test])[0]
train_labels = np.array(train_labels)
y_test = np.array(y_test)
print(train_filenames.shape, X_test.shape, train_labels.shape, y_test.shape)

#CONVOLUTION+RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators = 10,max_features=8)
random_forest.fit(train_filenames,train_labels)
y_testRF = random_forest.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(classification_report(y_test, y_testRF))
print("Accuracy: {0}".format(accuracy_score(y_test, y_testRF)))

#CONVOLUTION+NAIVE BAYES
from sklearn.naive_bayes import MultinomialNB
clf =MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
nbclf = clf.fit(train_filenames,train_labels)
y_testNB = nbclf.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(classification_report(y_test, y_testNB))
print("Accuracy: {0}".format(accuracy_score(y_test, y_testNB)))


#CONVOLUTION+SVM 
from sklearn import svm
classifier = svm.SVC(kernel='linear', C=0.8
classifier.fit(train_filenames,train_labels)
y_testSVM = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, y_testSVM))

print(classification_report(y_test, y_testSVM))
print("Accuracy: {0}".format(accuracy_score(y_test, y_testSVM)))
