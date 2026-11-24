import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

print("⏳ جاري بناء محرك التنبؤ الزمني والمكاني من بيانات وزارة التخطيط العراقية (2026 - 2030)...")

# ==============================================================================
# 1. البيانات الرسمية المعتمدة من نشرات وزارة التخطيط العراقية لسنة 2026 (12 شهراً)
# ==============================================================================
official_2026 = pd.DataFrame({
    'Month_Index':      list(range(1, 13)),
    'Year':             [2026] * 12,
    'Month':            list(range(1, 13)),
    # الأسعار بالدينار العراقي للوحدة القياسية (طن / م3 / ألف طابوقة / م2 / لتر)
    'Cement_Ord_Ton':   [104000, 105000, 108000, 107000, 107000, 104000, 105000, 104000, 132000, 138000, 139000, 140000],
    'Cement_Res_Ton':   [125000, 122000, 124000, 124000, 123000, 122000, 121000, 120000, 148000, 154000, 155000, 156000],
    'Sand_m3':          [17000,  17000,  16000,  16000,  16000,  16000,  16000,  16000,  17000,  18000,  18000,  18000],
    'Gravel_m3':        [22000,  20000,  20000,  20000,  20000,  20000,  20000,  21000,  21000,  22000,  22000,  22000],
    'Brick_Ord_1000':   [159000, 157000, 160000, 163000, 165000, 167000, 164000, 161000, 184000, 203000, 205000, 206000],
    'Brick_Rep_1000':   [214000, 205000, 214000, 217000, 217000, 217000, 211000, 212000, 247000, 320000, 322000, 325000],
    'Thermostone_1000': [1833000,1825000,1808000,1789000,1795000,1810000,1820000,1830000,1878000,1967000,1975000,1980000],
    'Block_1000':       [630000, 633000, 635000, 631000, 627000, 627000, 630000, 627000, 663000, 675000, 678000, 680000],
    'Rebar_Ton':        [979000, 972000, 980000, 977000, 986000, 971000, 965000, 969000, 989000, 1050000,1055000,1060000],
    'Gypsum_Ton':       [96000,  98000,  95000,  92000,  99000,  99000,  97000,  98000,  103000, 106000, 107000, 108000],
    'Ceramic_m2':       [10000,  10000,  8000,   9000,   9000,   9000,   9000,   9000,   9000,   10000,  10000,  10000],
    'Marble_m2':        [32000,  32000,  32000,  34000,  34000,  34000,  35000,  35000,  35000,  55000,  55000,  56000],
    'Asphalt_Ton':      [45000,  45000,  46000,  46000,  46000,  46000,  47000,  47000,  48000,  50000,  50000,  51000],
    'Paint_Liter':      [3000,   3000,   4000,   4000,   4000,   4000,   4000,   4000,   4000,   4000,   4000,   4000],
    'Wood_m3':          [450000, 460000, 460000, 458000, 463000, 460000, 463000, 460000, 463000, 487000, 490000, 492000],
})

# حفظ البيانات التاريخية الرسمية لسنة 2026 كمرجع للجنة المناقشة
official_2026.to_csv('official_stats_2026.csv', index=False)

# ==============================================================================
# 2. تدريب نماذج الذكاء الاصطناعي للتنبؤ الزمني (Ridge Regression + Seasonality)
# ==============================================================================
def create_time_features(df_in):
    X = pd.DataFrame()
    X['t'] = df_in['Month_Index']
    X['sin_month'] = np.sin(2 * np.pi * df_in['Month'] / 12.0)
    X['cos_month'] = np.cos(2 * np.pi * df_in['Month'] / 12.0)
    return X

X_hist = create_time_features(official_2026)

materials_cols = [
    'Cement_Ord_Ton', 'Cement_Res_Ton', 'Sand_m3', 'Gravel_m3',
    'Brick_Ord_1000', 'Brick_Rep_1000', 'Thermostone_1000', 'Block_1000',
    'Rebar_Ton', 'Gypsum_Ton', 'Ceramic_m2', 'Marble_m2',
    'Asphalt_Ton', 'Paint_Liter', 'Wood_m3'
]

# توليد الأشهر المستقبلية من شهر 1/2027 إلى شهر 12/2030 (48 شهراً قادماً)
future_rows = []
idx = 13
for year in range(2027, 2031):
    for month in range(1, 13):
        future_rows.append({'Month_Index': idx, 'Year': year, 'Month': month})
        idx += 1

future_df = pd.DataFrame(future_rows)
X_fut = create_time_features(future_df)

# نقطة الربط بنهاية عام 2026 لضمان استمرارية السلسلة الزمنية دون هبوط مفاجئ
X_anchor = create_time_features(official_2026[official_2026['Month_Index'] == 12])

for mat in materials_cols:
    # معامل تنظيم مضبوط (alpha=650) لالتقاط الاتجاه الواقعي والتغير الموسمي دون تضخم مفرط
    m = Ridge(alpha=650.0)
    m.fit(X_hist, official_2026[mat])
    
    anchor_pred = m.predict(X_anchor)[0]
    actual_dec_2026 = official_2026.loc[official_2026['Month_Index'] == 12, mat].values[0]
    shift = actual_dec_2026 - anchor_pred
    
    raw_preds = m.predict(X_fut) + shift
    # تقريب لأقرب 100 دينار وضمان عدم الهبوط عن الحد الأدنى الرسمي
    future_df[mat] = np.round(np.maximum(raw_preds, actual_dec_2026 * 0.98), -2)

# دمج البيانات الرسمية (2026) مع التنبؤية (2027 - 2030) في جدول زمني موحد (60 شهراً)
full_timeline = pd.concat([official_2026, future_df], ignore_index=True)
full_timeline.to_csv('iraq_forecast_2026_2030.csv', index=False)

# ==============================================================================
# 3. توحيد قاعدة بيانات المحافظات العراقية الـ 18 (أجور العمال الرسمية + معامل النقل)
# ==============================================================================
governorates_data = [
    {"Governorate": "بغداد",       "Skilled": 75000, "Semi_Skilled": 35000, "Unskilled": 25000, "Engineer": 80000, "Admin": 25000, "Guard": 20000, "Transport_Factor": 1.00},
    {"Governorate": "البصرة",      "Skilled": 50000, "Semi_Skilled": 35000, "Unskilled": 25000, "Engineer": 80000, "Admin": 30000, "Guard": 20000, "Transport_Factor": 1.06},
    {"Governorate": "نينوى",       "Skilled": 65000, "Semi_Skilled": 35000, "Unskilled": 25000, "Engineer": 80000, "Admin": 25000, "Guard": 20000, "Transport_Factor": 1.04},
    {"Governorate": "أربيل",       "Skilled": 70000, "Semi_Skilled": 40000, "Unskilled": 25000, "Engineer": 80000, "Admin": 30000, "Guard": 20000, "Transport_Factor": 1.07},
    {"Governorate": "السليمانية",  "Skilled": 70000, "Semi_Skilled": 40000, "Unskilled": 25000, "Engineer": 80000, "Admin": 30000, "Guard": 20000, "Transport_Factor": 1.08},
    {"Governorate": "دهوك",        "Skilled": 65000, "Semi_Skilled": 35000, "Unskilled": 25000, "Engineer": 75000, "Admin": 25000, "Guard": 20000, "Transport_Factor": 1.08},
    {"Governorate": "كركوك",       "Skilled": 50000, "Semi_Skilled": 30000, "Unskilled": 25000, "Engineer": 50000, "Admin": 25000, "Guard": 15000, "Transport_Factor": 1.03},
    {"Governorate": "ديالى",       "Skilled": 75000, "Semi_Skilled": 30000, "Unskilled": 25000, "Engineer": 50000, "Admin": 20000, "Guard": 15000, "Transport_Factor": 1.01},
    {"Governorate": "الأنبار",     "Skilled": 70000, "Semi_Skilled": 30000, "Unskilled": 20000, "Engineer": 50000, "Admin": 25000, "Guard": 15000, "Transport_Factor": 1.03},
    {"Governorate": "بابل",        "Skilled": 75000, "Semi_Skilled": 35000, "Unskilled": 25000, "Engineer": 55000, "Admin": 20000, "Guard": 15000, "Transport_Factor": 0.99},
    {"Governorate": "كربلاء",      "Skilled": 75000, "Semi_Skilled": 35000, "Unskilled": 25000, "Engineer": 50000, "Admin": 25000, "Guard": 15000, "Transport_Factor": 1.00},
    {"Governorate": "النجف",       "Skilled": 60000, "Semi_Skilled": 35000, "Unskilled": 25000, "Engineer": 60000, "Admin": 25000, "Guard": 15000, "Transport_Factor": 1.00},
    {"Governorate": "واسط",        "Skilled": 45000, "Semi_Skilled": 35000, "Unskilled": 25000, "Engineer": 75000, "Admin": 25000, "Guard": 10000, "Transport_Factor": 1.01},
    {"Governorate": "صلاح الدين",  "Skilled": 40000, "Semi_Skilled": 30000, "Unskilled": 25000, "Engineer": 50000, "Admin": 20000, "Guard": 15000, "Transport_Factor": 1.02},
    {"Governorate": "القادسية",    "Skilled": 75000, "Semi_Skilled": 35000, "Unskilled": 25000, "Engineer": 40000, "Admin": 25000, "Guard": 15000, "Transport_Factor": 1.00},
    {"Governorate": "المثنى",      "Skilled": 60000, "Semi_Skilled": 35000, "Unskilled": 25000, "Engineer": 40000, "Admin": 20000, "Guard": 10000, "Transport_Factor": 1.01},
    {"Governorate": "ذي قار",      "Skilled": 60000, "Semi_Skilled": 30000, "Unskilled": 25000, "Engineer": 50000, "Admin": 30000, "Guard": 10000, "Transport_Factor": 1.02},
    {"Governorate": "ميسان",       "Skilled": 65000, "Semi_Skilled": 30000, "Unskilled": 25000, "Engineer": 40000, "Admin": 20000, "Guard": 10000, "Transport_Factor": 1.03},
]
pd.DataFrame(governorates_data).to_csv('iraq_governorates_factors.csv', index=False)

print("✅ تم بناء محرك التنبؤ الزمني والمكاني لجميع المواد الـ 15 والمحافظات الـ 18 بنجاح!")
print("\n🔍 عينة من الأسعار الرسمية (2026) والتنبؤية (2027 - 2030):")
print(full_timeline[full_timeline['Month'].isin([10])][['Year', 'Month', 'Cement_Res_Ton', 'Rebar_Ton', 'Brick_Ord_1000', 'Sand_m3', 'Gravel_m3']])
