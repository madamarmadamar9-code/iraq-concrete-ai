import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. قراءة قاعدة البيانات الهندسية الشاملة التي أنشأناها في الملف الأول
df = pd.read_csv('iraq_concrete_data.csv')

# تحديد المدخلات (رمز الفقرة + أسعار المواد + أجور العمالة + هامش التحميلات) والمخرج (سعر الوحدة القياسي)
X = df.drop('Final_Unit_Price_IQD', axis=1)
y = df['Final_Unit_Price_IQD']

# تقسيم البيانات بشكل متوازن (Stratified) عبر جميع الفقرات الـ 9: 80% للتدريب و 20% للاختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['Item_Code']
)

print("⏳ جاري تدريب نماذج الذكاء الاصطناعي على جميع فقرات الدليل القياسي العراقي (9 فقرات)...\n")

# 2. بناء وتدريب النموذج الأول: Random Forest
rf_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_preds) * 100
rf_mae = mean_absolute_error(y_test, rf_preds)

# 3. بناء وتدريب النموذج الثاني: XGBoost
xgb_model = XGBRegressor(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_preds) * 100
xgb_mae = mean_absolute_error(y_test, xgb_preds)

print(f"📊 نتائج نموذج Random Forest -> الدقة الكلية: {rf_r2:.3f}% | معدل الخطأ: {rf_mae:,.0f} دينار")
print(f"📊 نتائج نموذج XGBoost       -> الدقة الكلية: {xgb_r2:.3f}% | معدل الخطأ: {xgb_mae:,.0f} دينار\n")

# 4. اختيار النموذج الأدق وتقييمه على كل فقرة إنشائية على حدة
if xgb_r2 >= rf_r2:
    best_model = xgb_model
    best_preds = xgb_preds
    best_name = "XGBoost"
    best_acc = xgb_r2
    best_mae = xgb_mae
else:
    best_model = rf_model
    best_preds = rf_preds
    best_name = "Random Forest"
    best_acc = rf_r2
    best_mae = rf_mae

item_names = {
    1: "الأسس - خرسانة غير مسلحة [1:3:6]",
    2: "الأسس المسلحة [1:2:4]",
    3: "الأعمدة والجسور والسقوف [1:1.5:3]",
    4: "أعمال البناء بالطابوق",
    5: "أعمال حديد التسليح",
    6: "أعمال الطرق والتبليط الأسفلتي",
    7: "أعمال الإنهاء واللياسة والأصباغ",
    8: "أعمال الأرضيات والسيراميك",
    9: "الأعمال الترابية والدفن بالسبيس"
}

print("📋 تقرير دقة التنبؤ لكل فقرة إنشائية على بيانات الاختبار:")
test_eval = X_test.copy()
test_eval['Actual'] = y_test
test_eval['Pred'] = best_preds
for code, name in item_names.items():
    sub = test_eval[test_eval['Item_Code'] == code]
    mape = np.mean(np.abs((sub['Actual'] - sub['Pred']) / sub['Actual'])) * 100
    print(f"   - الفقرة ({code}) {name}: دقة التطابق = {100 - mape:.2f}%")

# 5. حفظ حزمة النموذج المدرب والبيانات الوصفية للاستخدام المباشر في التطبيق
model_bundle = {
    "model": best_model,
    "model_name": best_name,
    "r2_score": round(best_acc, 2),
    "mae": round(best_mae, 0),
    "features": list(X.columns)
}
joblib.dump(model_bundle, 'concrete_ai_model.pkl')

print(f"\n✅ تم اعتماد وحفظ النموذج الأفضل ({best_name}) بدقة بلغت {best_acc:.2f}% في ملف: concrete_ai_model.pkl")
