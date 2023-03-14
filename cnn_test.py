import random
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

random.shuffle(test_data)

for sample in test_data[:10]:
    X_test = []
    y_test = []

for features,label in test_data:
    X_test.append(features)
    y_test.append(label)

y_test1 = np.array(y_test)

X_test= np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE,3)
model.evaluate(X_test, y_test1)
prediction = model.predict(X_test)

prediction= (prediction > 0.5)
prediction = prediction.astype(int)
confusion_matrix(y_test1, prediction)

print(confusion_matrix(y_test1, prediction))
print(classification_report(y_test1, prediction))
print("Accuracy: {0}".format(accuracy_score(y_test1, prediction)))


prediction = model.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test1, prediction)

auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='AUC = {:.3f}'.format(auc_keras))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi')
plt.legend(loc='best')
plt.show()
