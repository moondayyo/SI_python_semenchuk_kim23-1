import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data_metrics.csv')
df.head()

threshold = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()

def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

print('TP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))

def find_confusion_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def semenchuk_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_confusion_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

assert np.array_equal(
    semenchuk_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
    confusion_matrix(df.actual_label.values, df.predicted_RF.values),
    'novichkov_confusion_matrix() is not correct for RF')

assert np.array_equal(
    semenchuk_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
    confusion_matrix(df.actual_label.values, df.predicted_LR.values),
    'novichkov_confusion_matrix() is not correct for LR')


accuracy_score(df.actual_label.values, df.predicted_RF.values)

def semenchuk_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_confusion_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + FP + FN + TN)

assert semenchuk_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values, df.predicted_RF.values), 'novichkov_accuracy_score() failed on RF'
assert semenchuk_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(df.actual_label.values, df.predicted_LR.values), 'novichkov_accuracy_score() failed on LR'

print('Accuracy RF:', semenchuk_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print('Accuracy LR:', semenchuk_accuracy_score(df.actual_label.values, df.predicted_LR.values))


def semenchuk_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_confusion_matrix_values(y_true, y_pred)
    return TP / (TP + FN)

assert semenchuk_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values, df.predicted_RF.values), 'novichkov_recall_score() failed on RF'
assert semenchuk_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(df.actual_label.values, df.predicted_LR.values), 'novichkov_recall_score() failed on LR'

print('Recall RF:', semenchuk_recall_score(df.actual_label.values, df.predicted_RF.values))
print('Recall LR:', semenchuk_recall_score(df.actual_label.values, df.predicted_LR.values))


def semenchuk_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_confusion_matrix_values(y_true, y_pred)
    return TP / (TP + FP)

assert semenchuk_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(df.actual_label.values, df.predicted_RF.values), 'novichkov_precision_score() failed on RF'
assert semenchuk_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(df.actual_label.values, df.predicted_LR.values), 'novichkov_precision_score() failed on LR'

print('Precision RF:', semenchuk_precision_score(df.actual_label.values, df.predicted_RF.values))
print('Precision LR:', semenchuk_precision_score(df.actual_label.values, df.predicted_LR.values))


def semenchuk_f1_score(y_true, y_pred):
    precision = semenchuk_precision_score(y_true, y_pred)
    recall = semenchuk_recall_score(y_true, y_pred)
    return 2 * precision * recall / (precision + recall)

assert semenchuk_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values, df.predicted_RF.values), 'novichkov_f1_score() failed on RF'
assert semenchuk_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values, df.predicted_LR.values), 'novichkov_f1_score() failed on LR'

print('F1 RF:', semenchuk_f1_score(df.actual_label.values, df.predicted_RF.values))
print('F1 LR:', semenchuk_f1_score(df.actual_label.values, df.predicted_LR.values))


print('scores with threshold = 0.5')
print('Accuracy RF:', semenchuk_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print('Recall RF:', semenchuk_recall_score(df.actual_label.values, df.predicted_RF.values))
print('Precision RF:', semenchuk_precision_score(df.actual_label.values, df.predicted_RF.values))
print('F1 RF:', semenchuk_f1_score(df.actual_label.values, df.predicted_RF.values))
print('')
print('scores with threshold = 0.25')
print('Accuracy RF:', semenchuk_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values))
print('Recall RF:', semenchuk_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values))
print('Precision RF:', semenchuk_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values))
print('F1 RF:', semenchuk_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values))
print('')
print('scores with threshold = 0.1')
print('Accuracy RF:', semenchuk_accuracy_score(df.actual_label.values, (df.model_RF >= 0.1).astype('int').values))
print('Recall RF:', semenchuk_recall_score(df.actual_label.values, (df.model_RF >= 0.1).astype('int').values))
print('Precision RF:', semenchuk_precision_score(df.actual_label.values, (df.model_RF >= 0.1).astype('int').values))
print('F1 RF:', semenchuk_f1_score(df.actual_label.values, (df.model_RF >= 0.1).astype('int').values))


fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('\nAUC RF:', auc_RF)
print('AUC LR:', auc_LR)


plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC = %0.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC = %0.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()