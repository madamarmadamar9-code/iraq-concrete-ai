import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

print("⏳ جاري بناء محرك التنبؤ الزمني والمكاني من بيانات هيأة الإحصاء العراقية (2026 - 2030)...")

# 1. البيانات التاريخية الحقيقية المستخرجة من نشرات هيأة الإحصاء العراقية (كانون الأول 2025 - تشرين الأول 2026)
# Month_Index: 0 = Dec 2025, 1 = Jan 2026 ... 10 = Oct 2026
historical_data = pd.DataFrame({
    'Month_Index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Year':        [2025, 2026, 2026, 2026, 2026, 2026, 2026, 2026, 2026, 2026, 2026],
    'Month':       [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # الأسعار بالدينار العراقي (المصدر: هيأة الإحصاء ونظم المعلومات الجغرافية)
    'Cement_Ton':  [123000, 125000, 122000, 124000, 124000, 123000, 122000, 121000, 120000, 148000, 154000],
    'Sand_m3':     [18000,  17000,  17000,  16000,  16000,  16000,  16000,  16000,  16000,  17000,  18000],
    'Gravel_m3':   [22000,  22000,  20000,  20000,  20000,  20000,  20000,  20000,  21000,  21000,  22000],
    'Wood_m2':     [29000,  29000,  29000,  29000,  29000,  29000,  27000,  24000,  25000,  25000,  25000],
})

# حفظ البيانات التاريخية الرسمية كمرجع للجنة المناقشة
historical_data.to_csv('official_stats_2026.csv', index=False)

# 2. تدريب نماذج الذكاء الاصطناعي للتنبؤ الزمني (Time-Series Trend + Seasonality)
# نستخدم Ridge Regression مع ميزات موسمية (Sin/Cos) لمنع التضخم غير الواقعي وضمان استقرار التنبؤ حتى 2030
def create_time_features(df_in):
    X = pd.DataFrame()
    X['t'] = df_in['Month_Index']
    # إضافة التغير الموسمي السنوي (الشتاء vs الصيف)
    X['sin_month'] = np.sin(2 * np.pi * df_in['Month'] / 12.0)
    X['cos_month'] = np.cos(2 * np.pi * df_in['Month'] / 12.0)
    return X

X_hist = create_time_features(historical_data)

models_ts = {}
materials = ['Cement_Ton', 'Sand_m3', 'Gravel_m3', 'Wood_m2']

for mat in materials:
    # استخدام Ridge مع ضبط معامل التنظيم لالتقاط الاتجاه الواقعي دون قفزات عشوائية
    m = Ridge(alpha=15.0)
    m.fit(X_hist, historical_data[mat])
    models_ts[mat] = m

# 3. توليد التنبؤات المستقبلية من شهر 11/2026 إلى شهر 12/2030 (50 شهراً قادماً)
future_rows = []
idx = 11
for year in range(2026, 2031):
    start_m = 11 if year == 2026 else 1
    for month in range(start_m, 13):
        future_rows.append({'Month_Index': idx, 'Year': year, 'Month': month})
        idx += 1

future_df = pd.DataFrame(future_rows)
X_fut = create_time_features(future_df)

for mat in materials:
    preds = models_ts[mat].predict(X_fut)
    # ضمان عدم انخفاض الأسعار عن الحد المنطقي للسوق
    min_val = historical_data[mat].min() * 0.95
    future_df[mat] = np.round(np.maximum(preds, min_val), -2) # تقريب لأقرب 100 دينار

# دمج البيانات التاريخية (1-10/2026) مع التنبؤية (11/2026 - 12/2030) في جدول زمني واحد
full_timeline = pd.concat([historical_data[historical_data['Year'] == 2026], future_df], ignore_index=True)

# 4. قاعدة بيانات المحافظات العراقية الـ 18 (أجور العمال الرسمية من النشرات + معامل النقل للموقع)
governorates_data = {
    "بغداد":       {"skilled": 75000, "unskilled": 25000, "transport_factor": 1.15, "mix_transport": 26000},
    "المثنى":      {"skilled": 60000, "unskilled": 25000, "transport_factor": 1.18, "mix_transport": 24000},
    "البصرة":      {"skilled": 50000, "unskilled": 25000, "transport_factor": 1.25, "mix_transport": 28000},
    "ذي قار":      {"skilled": 60000, "unskilled": 25000, "transport_factor": 1.16, "mix_transport": 24000},
    "ميسان":       {"skilled": 65000, "unskilled": 25000, "transport_factor": 1.18, "mix_transport": 25000},
    "القادسية":    {"skilled": 75000, "unskilled": 25000, "transport_factor": 1.14, "mix_transport": 23000},
    "النجف":       {"skilled": 60000, "unskilled": 25000, "transport_factor": 1.12, "mix_transport": 23000},
    "كربلاء":      {"skilled": 75000, "unskilled": 25000, "transport_factor": 1.12, "mix_transport": 23000},
    "بابل":        {"skilled": 75000, "unskilled": 25000, "transport_factor": 1.12, "mix_transport": 23000},
    "واسط":        {"skilled": 45000, "unskilled": 25000, "transport_factor": 1.15, "mix_transport": 24000},
    "ديالى":       {"skilled": 75000, "unskilled": 25000, "transport_factor": 1.15, "mix_transport": 24000},
    "الأنبار":     {"skilled": 70000, "unskilled": 20000, "transport_factor": 1.20, "mix_transport": 25000},
    "صلاح الدين":  {"skilled": 40000, "unskilled": 25000, "transport_factor": 1.16, "mix_transport": 24000},
    "كركوك":       {"skilled": 50000, "unskilled": 25000, "transport_factor": 1.14, "mix_transport": 24000},
    "نينوى":       {"skilled": 65000, "unskilled": 25000, "transport_factor": 1.18, "mix_transport": 25000},
    "أربيل":       {"skilled": 70000, "unskilled": 25000, "transport_factor": 1.22, "mix_transport": 27000},
    "السليمانية":  {"skilled": 70000, "unskilled": 25000, "transport_factor": 1.22, "mix_transport": 27000},
    "دهوك":        {"skilled": 70000, "unskilled": 25000, "transport_factor": 1.24, "mix_transport": 28000},
}

# حفظ الجدول الزمني وبيانات المحافظات
full_timeline.to_csv('iraq_forecast_2026_2030.csv', index=False)
pd.DataFrame.from_dict(governorates_data, orient='index').to_csv('iraq_governorates_factors.csv')

print("✅ تم بناء محرك التنبؤ الزمني بنجاح!")
print("\n🔍 عينة من التنبؤ لشهر 11 و 12 لسنة 2026 وبداية 2027:")
print(full_timeline[full_timeline['Month_Index'].isin([10, 11, 12, 13, 24, 36, 48, 60])][['Year', 'Month', 'Cement_Ton', 'Sand_m3', 'Gravel_m3', 'Wood_m2']])