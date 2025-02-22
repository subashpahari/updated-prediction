import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score,accuracy_score,roc_auc_score,average_precision_score,roc_curve,auc
from sklearn.feature_selection import RFECV
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import  cross_val_score,train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import joblib
df=pd.read_excel('appendicitis.xlsx')
columns=['AppendixDiameter','ReboundTenderness','CoughingPain','FreeFluids','MigratoryPain','BodyTemp','KetonesInUrine','Nausea',
         'WBCCount','NeutrophilPerc','CRPEntry','Peritonitis','DiagnosisByCriteria']

df=df[columns]

usg_col =['AppendixDiameter','FreeFluids']

df[usg_col]=df[usg_col].fillna(0)

mapping ={
    '+++' : 3,
    '++': 2,
    '+': 1,
    'no':0,
    'yes':1,
    'local':1,
    'generalised':2,
    'normal':0,
    'noAppendicitis':0,
    'appendicitis':1
}

for col in df.columns:
    df[col]=df[col].replace(mapping)

x = df.iloc[:, 0:-1]  
y = df.iloc[:, -1]


x_tempi, x_testi, y_temp, y_testi = train_test_split(x,y,test_size=0.2,random_state=17)
x_traini,x_vali,y_traini,y_vali = train_test_split(x_tempi,y_temp,test_size=0.25,random_state=17)

imputer_train = KNNImputer(n_neighbors=5)
imputer_val = KNNImputer(n_neighbors=5)
imputer_test = KNNImputer(n_neighbors=5)
imputer_temp=KNNImputer(n_neighbors=5)

x_temp=pd.DataFrame(imputer_temp.fit_transform(x_tempi),columns=x_tempi.columns,index=x_tempi.index)
x_train = pd.DataFrame(imputer_train.fit_transform(x_traini), columns=x_traini.columns, index=x_traini.index)
x_val = pd.DataFrame(imputer_val.fit_transform(x_vali), columns=x_vali.columns, index=x_vali.index)
x_test = pd.DataFrame(imputer_test.fit_transform(x_testi), columns=x_testi.columns, index=x_testi.index)

y_train=y_traini.squeeze()
y_val=y_vali.squeeze()
y_test=y_testi.squeeze()
y_temp=y_temp.squeeze()

columns_round = [ 'MigratoryPain','ReboundTenderness','CoughingPain','Peritonitis','KetonesInUrine','Nausea']

for col in columns_round:
    if col in x_temp.columns:
        x_temp[col]=x_temp[col].round().astype(int)
    if col in x_train.columns:
        x_train[col] = x_train[col].round().astype(int)
    if col in x_val.columns:
        x_val[col] = x_val[col].round().astype(int)
    if col in x_test.columns:
        x_test[col] = x_test[col].round().astype(int) 


rf_model=RandomForestClassifier(
    n_estimators=700,
    min_samples_split=12,
    min_samples_leaf=1,
    max_depth=35,
    random_state=17
)
scores = cross_val_score(rf_model, x_train, y_train, cv=10, scoring='accuracy', n_jobs=2)
print("Mean cross_val accuracy Score:", scores.mean())
print("Standard Deviation of cross_val accuracy Scores:", scores.std())

rf_model.fit(x_train,y_train)



# y_pred_prob=rf_model.predict_proba(x_val)[:,1]
# fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
# youden_index = tpr - fpr
# optimal_idx = np.argmax(youden_index)
# optimal_threshold = thresholds[optimal_idx]

# distances = np.sqrt((1 - tpr)**2 + fpr**2)
# optimal_idx = np.argmin(distances)
# optimal_threshold = thresholds[optimal_idx]

# print(f"Optimal Threshold: {optimal_threshold}")

threshold=0.7032966204148201

y_pred_prob=rf_model.predict_proba(x_val)[:,1]
y_thres=(y_pred_prob >= threshold).astype(int)
auc_score = roc_auc_score(y_val, y_pred_prob)
print("Val AUC:", auc_score)
val_conf_matrix = confusion_matrix(y_val, y_thres)
TNv, FPv, FNv, TPv = val_conf_matrix.ravel()
val_sensitivity = TPv / (TPv + FNv)
val_specificity = TNv / (TNv + FPv)
val_ppv = TPv/(TPv+FPv)
val_npv = TNv/(TNv+FNv)
print(f"Sensitivity: {val_sensitivity:.3f} ({val_sensitivity * 100:.2f}%)")
print(f"Specificity: {val_specificity:.3f} ({val_specificity * 100:.2f}%)")

print (f"Positive predictive value : {val_ppv:.3f}({val_ppv*100:.2f}%)")
print (f"Negative predictive value : {val_npv:.3f}({val_npv*100:.2f}%)")

print("Val Confusion Matrix:\n", val_conf_matrix)
val_acu=accuracy_score(y_val,y_thres)
print("Val accuracy score:",val_acu)

y_test_prob=rf_model.predict_proba(x_test)[:,1]
y_thres_test=(y_test_prob >= threshold).astype(int)
auc_score = roc_auc_score(y_test, y_test_prob)
print("Test AUC:", auc_score)
test_conf_matrix = confusion_matrix(y_test, y_thres_test)
print("Test Confusion Matrix:\n", test_conf_matrix)
TNt, FPt, FNt, TPt = test_conf_matrix.ravel()
test_sensitivity = TPt / (TPt + FNt)
test_specificity = TNt / (TNt + FPt)

test_ppv = TPt/(TPt+FPt)
test_npv = TNt/(TNt+FNt)

print(f"Sensitivity: {test_sensitivity:.3f} ({test_sensitivity * 100:.2f}%)")
print(f"Specificity: {test_specificity:.3f} ({test_specificity * 100:.2f}%)")

print (f"Positive predictive value : {test_ppv:.3f}({test_ppv*100:.2f}%)")
print (f"Negative predictive value : {test_npv:.3f}({test_npv*100:.2f}%)")

test_acu=accuracy_score(y_test,y_thres_test)
print("Test Accuracy score:",test_acu)
rf_model2 = rf_model.fit(x_temp,y_temp)
joblib.dump(rf_model2, 'rf_model.pkl')

y_test1_prob=rf_model.predict_proba(x_test)[:,1]
y_thres1_test=(y_test1_prob >= threshold).astype(int)
auc_score1 = roc_auc_score(y_test, y_test1_prob)
print("Test AUC:", auc_score1)
test1_conf_matrix = confusion_matrix(y_test, y_thres1_test)
print("Test Confusion Matrix:\n", test1_conf_matrix)
TNt1, FPt1, FNt1, TPt1 = test1_conf_matrix.ravel()

test_sensitivity1 = TPt1 / (TPt1 + FNt1)
test_specificity1 = TNt1 / (TNt1 + FPt1)

test_ppv1 = TPt1/(TPt1+FPt1)
test_npv1 = TNt1/(TNt1+FNt1)

print(f"Sensitivity: {test_sensitivity1:.3f} ({test_sensitivity1 * 100:.2f}%)")
print(f"Specificity: {test_specificity1:.3f} ({test_specificity1 * 100:.2f}%)")

print (f"Positive predictive value : {test_ppv1:.3f}({test_ppv1*100:.2f}%)")
print (f"Negative predictive value : {test_npv1:.3f}({test_npv1*100:.2f}%)")

test1_acu=accuracy_score(y_test,y_thres1_test)
print("Test Accuracy score:",test1_acu)