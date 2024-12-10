import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

# تحميل البيانات
df = pd.read_csv('/Users/shahaalfughom/Desktop/HeartDisease-Project/heart.csv')

# عرض المتغيرات
pd.set_option('display.max_rows', None)  # لعرض جميع الصفوف
pd.set_option('display.max_columns', None)  # لعرض جميع الأعمدة
print(df.head())

# تنظيف البيانات
df = df[df['Cholesterol'] > 0]
df = df[df['RestingBP'] > 0]
df = df[df['Oldpeak'] >= 0]

print(df.isnull().sum())


#الاحصاء الوصفي
pd.set_option("display.float", "{:.2f}".format)
print(df.describe())

# التحقق من قيم كل ميزة و تحديد نوع الميزه (فئوية او رقمية)
categorical_val = []
continous_val = []
for column in df.columns:
    print('==============================')
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)

print('==============================')
print(f"Categorical Features : {categorical_val}")
print(f"Continous Features : {continous_val}")

# ــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

# ــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

# تحويل المتغيرات الفئوية إلى متغيرات وهمية (One-Hot Encoding)
categorical_val.remove('HeartDisease')
dataset = pd.get_dummies(df, columns=categorical_val, drop_first=True)

from sklearn.preprocessing import StandardScaler

s_sc = StandardScaler()
col_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])

# تحويل قيم True و False إلى 0 و 1
dataset = dataset.replace({True: 1, False: 0})
print(dataset.head())

# تقسيم مجموعة البيانات إلى بيانات تدريب و اختبار
from sklearn.model_selection import train_test_split

X = dataset.drop('HeartDisease', axis=1)
y = dataset.HeartDisease
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X.dtypes)

# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

                              #Exploratory analysis
# Correlation:
df_corr = df[["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak", "HeartDisease"]]
# حساب مصفوفة الكورليشن
correlation_matrix = df_corr.corr()

# رسم خريطة الحرارة
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
plt.show()
#---------------------

# Violin Chart:
plt.figure(figsize=(12, 6))
sns.violinplot(x="Sex", y="MaxHR", hue="HeartDisease", data=df, split=True, palette="Blues")
plt.title("Violin Plot of Maximum Heart Rate by Gender and Heart Disease")
plt.xlabel("Gender (0 = Female, 1 = Male)")
plt.ylabel("Maximum Heart Rate")
plt.show()
#---------------------

# رسم مخطط تقدير كثافة النواة (KDE) بين العمر والكولسترول
plt.figure(figsize=(10, 6))
sns.kdeplot(x="Age", y="Cholesterol", data=df, cmap="Blues", fill=True)
plt.title("KDE Plot of Age and Cholesterol")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.show()
#---------------------

# رسم الكثافة لكل متغير رقمي بالنسبة لوجود أمراض القلب

# تحديد المتغيرات الرقمية
numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

plt.figure(figsize=(15, 10))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 2, i)
    sns.kdeplot(data=df, x=column, hue="HeartDisease", fill=True, common_norm=False, palette="muted")
    plt.title(f"Density Plot for {column} by HeartDisease")
    plt.xlabel(column)
    plt.ylabel("Density")

plt.tight_layout()
plt.show()
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ



#                                  الانحدار اللوجستي
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
#                                  التحقق من الشروط
# ---------------------------------------------------------------------------------------
#                               (No Multicollinearity):
# ---------------------------------------------------------------------------------------
print("اختبار شروط الانحدار اللوجستي")
print("اختبار شرط عدم وجود تعددية خطية")
from statsmodels.stats.outliers_influence import variance_inflation_factor

# حساب VIF لكل متغير
vif = pd.DataFrame()
vif["Feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
# عرض نتائج VIF
print(vif)
# رسم VIF باستخدام رسم الأعمدة
plt.figure(figsize=(10, 6))
sns.barplot(x="VIF", y="Feature", data=vif, palette="viridis")


# لإعادة تشكيل النص العربي
def reshape_arabic(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)


plt.title(reshape_arabic("مؤشر التضخم (VIF) لكل متغير مستقل"), fontsize=16)

# plt.title("Variance Inflation Factor (VIF) for each feature")
plt.xlabel("VIF")
plt.ylabel("Feature")
plt.show()
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
#  استدعاء دوال قياس الأداء
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# دالة طباعة تقرير أداء نموذج التصنيف
def print_score(clf, X_test, y_test):
    pred = clf.predict(X_test)
    clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
    print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
    print("_______________________________________________")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

# تدريب نموذج الانحدار اللوجستي للتنبؤ بأمراض القلب
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)
y_pred_lr = lr_clf.decision_function(X_test)

# طباعة تقرير أداء النموذج
print("Logistic Regression Results:")
print_score(lr_clf, X_test, y_test)

# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

# نموذج شجرة القرار
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
print("Decision Tree Results:")
print_score(dt_clf, X_test, y_test)

# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

# نموذج الغابة العشوائية
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
print("Random Forest Results:")
print_score(rf_clf, X_test, y_test)
# حساب دقة نموذج التدريب
training_accuracy = rf_clf.score(X_train, y_train)
print("Training Accuracy:", training_accuracy)

# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

# نموذج التعزيز الاشتقاقي
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train, y_train)
print("Gradient Boosting Results:")
print_score(gb_clf, X_test, y_test)

# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

#  نموذج SVM
from sklearn.svm import SVC

svm_clf = SVC(probability=True)
svm_clf.fit(X_train, y_train)
print("Support Vector Machines Results:")
print_score(svm_clf, X_test, y_test)

# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

#  نموذج K-NN
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
print("K-NN Results:")
print_score(knn_clf, X_test, y_test)

# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

#  نموذج Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
print("Naive Bayes Results:")
print_score(nb_clf, X_test, y_test)
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
#                                حساب قيمة ROC-AUC لكل نموذج
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

from sklearn.metrics import roc_curve, auc

models = [lr_clf, dt_clf, rf_clf, gb_clf, svm_clf, knn_clf, nb_clf]
model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "SVM", "K-NN",
               "Naive Bayes"]

plt.figure(figsize=(10, 8))

for model, name in zip(models, model_names):
    y_prob = model.predict_proba(X_test)[:, 1]  # الحصول على احتمالات الفئة الإيجابية
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# رسم الخط التوجيهي
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

sns.set(rc={'figure.figsize': (15, 8)})
xLabel_ar = arabic_reshaper.reshape("معدل الإيجابيات الخاطئة (FPR)")
ylabel_ar = arabic_reshaper.reshape("معدل الإيجابيات الحقيقية (TPR)")

xlabel_ar = get_display(xLabel_ar)
ylabel_ar = get_display(ylabel_ar)
plt.xlabel(xlabel_ar, fontsize=14)
plt.ylabel(ylabel_ar, fontsize=14)

plt.legend(loc='lower right')
plt.grid()
plt.show()
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

score_lr = lr_clf.score(X_test, y_test) * 100  # الانحدار اللوجستي
score_dt = dt_clf.score(X_test, y_test) * 100  # شجرة القرار
score_rf = rf_clf.score(X_test, y_test) * 100  # الغابة العشوائية
score_gb = gb_clf.score(X_test, y_test) * 100  # التعزيز التدرجي
score_svm = svm_clf.score(X_test, y_test) * 100  # SVM
score_knn = knn_clf.score(X_test, y_test) * 100  # K-NN
score_nb = nb_clf.score(X_test, y_test) * 100  # Naive Bayes

# قائمة لنتائج الدقة
scores = [score_lr, score_nb, score_svm, score_knn, score_dt, score_rf, score_gb]

# أسماء الخوارزميات
algorithms = ["Logistic Regression", "Naive Bayes", "SVM", "K-Nearest Neighbors",
              "Decision Tree", "Random Forest", "Gradient Boosting"]

# إعداد الشكل والرسم
sns.set(rc={'figure.figsize': (12, 8)})
xLabel_ar = arabic_reshaper.reshape("الخوارزميات")
ylabel_ar = arabic_reshaper.reshape("نسبة الدقة ٪؜")
xlabel_ar = get_display(xLabel_ar)
ylabel_ar = get_display(ylabel_ar)
plt.xlabel(xlabel_ar, fontsize=14)
plt.ylabel(ylabel_ar, fontsize=14)

# رسم المخطط الشريطي بتدرج اللون الأزرق
sns.barplot(x=algorithms, y=scores, palette='Blues')

# إضافة النسب المئوية فوق كل عمود
for i, score in enumerate(scores):
    plt.text(i, score + 0.5, f'{score:.2f}%', ha='center', fontsize=14)
plt.show()

# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

# إظهار نسبة دقة التدريب مع دقة الاختبار لكل نموذج
print("lr training score :",lr_clf.score(X_train,y_train))
print("lr testing score :",lr_clf.score(X_test,y_test))
print("dt training score :",dt_clf.score(X_train,y_train))
print("dt testing score :",dt_clf.score(X_test,y_test))
print("RF training score :",rf_clf.score(X_train,y_train))
print("RF testing score :",rf_clf.score(X_test,y_test))
print("gb training score :",gb_clf.score(X_train,y_train))
print("gb testing score :",gb_clf.score(X_test,y_test))
print("svm training score :",svm_clf.score(X_train,y_train))
print("svm testing score :",svm_clf.score(X_test,y_test))
print("k-nn training score :",knn_clf.score(X_train,y_train))
print("k-nn testing score :",knn_clf.score(X_test,y_test))
print("nb training score :",nb_clf.score(X_train,y_train))
print("nb testing score :",nb_clf.score(X_test,y_test))
