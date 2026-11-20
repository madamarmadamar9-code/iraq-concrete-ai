import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. قراءة قاعدة البيانات الهندسية التي أنشأناها
df = pd.read_csv('iraq_concrete_data.csv')

# تحديد المدخلات (أسعار المواد والعمالة والنقل) والمخرج المطلوب (السعر النهائي للفقرة)
X = df.drop('Final_Price_Per_m2', axis=1)
y = df['Final_Price_Per_m2']

# تقسيم البيانات: 80% للتدريب و 20% للاختبار الحقيقي
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("⏳ جاري تدريب نماذج الذكاء الاصطناعي على الدليل القياسي العراقي...\n")

# 2. بناء وتدريب النموذج الأول: Random Forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_preds) * 100
rf_mae = mean_absolute_error(y_test, rf_preds)

# 3. بناء وتدريب النموذج الثاني: XGBoost
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_preds) * 100
xgb_mae = mean_absolute_error(y_test, xgb_preds)

print(f"📊 نتائج نموذج Random Forest -> الدقة: {rf_r2:.2f}% | معدل الخطأ: {rf_mae:.0f} دينار")
print(f"📊 نتائج نموذج XGBoost       -> الدقة: {xgb_r2:.2f}% | معدل الخطأ: {xgb_mae:.0f} دينار\n")

# 4. اختيار النموذج الأدق وحفظه للاستخدام في التطبيق والوزارة
if xgb_r2 >= rf_r2:
    best_model = xgb_model
    best_name = "XGBoost"
    best_acc = xgb_r2
else:
    best_model = rf_model
    best_name = "Random Forest"
    best_acc = rf_r2

joblib.dump(best_model, 'concrete_ai_model.pkl')
print(f"✅ تم اعتماد وحفظ النموذج الأفضل ({best_name}) بدقة بلغت {best_acc:.2f}% في ملف: concrete_ai_model.pkl")