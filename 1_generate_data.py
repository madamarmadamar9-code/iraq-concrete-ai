import pandas as pd
import numpy as np

# تثبيت العشوائية لضمان دقة النتائج وتكرارها
np.random.seed(42)
num_samples = 2000  # عدد الحالات الواقعية لتدريب الذكاء الاصطناعي

# 1. توليد أسعار السوق العراقية الواقعية (بالدينار العراقي)
# سعر طن السمنت المقاوم (يتراوح بين 105,000 إلى 145,000 دينار/طن)
cement_price_ton = np.random.uniform(105000, 145000, num_samples)

# سعر المتر المكعب من الرمل المغسول (يتراوح بين 20,000 إلى 40,000 دينار/م3)
sand_price_m3 = np.random.uniform(20000, 40000, num_samples)

# سعر المتر المكعب من الحصى المدرج (يتراوح بين 25,000 إلى 45,000 دينار/م3)
gravel_price_m3 = np.random.uniform(25000, 45000, num_samples)

# سعر المتر المربع من الخشب (يتراوح بين 12,000 إلى 22,000 دينار/م2)
wood_price_m2 = np.random.uniform(12000, 22000, num_samples)

# كلفة خلط ونقل وضخ المتر المكعب الواحد من الخرسانة (تتأثر بالمسافة والوقود)
concrete_transport_mix_m3 = np.random.uniform(18000, 35000, num_samples)

# كلفة تانكر الماء 8000 لتر (بين 15,000 إلى 30,000 دينار)
water_tanker_price = np.random.uniform(15000, 30000, num_samples)

# أجرة العامل الماهر اليومية (بين 35,000 إلى 60,000 دينار/يوم)
skilled_worker_wage = np.random.uniform(35000, 60000, num_samples)

# أجرة العامل العادي اليومية (بين 20,000 إلى 35,000 دينار/يوم)
unskilled_worker_wage = np.random.uniform(20000, 35000, num_samples)

# نسبة التحميلات الإدارية والأرباح حسب الدليل (بين 20% إلى 25%)
profit_margin = np.random.uniform(0.20, 0.25, num_samples)

# ---------------------------------------------------------
# 2. تطبيق معادلات الدليل القياسي العراقي (صفحة 25 - 32)
# ---------------------------------------------------------

# أ. كلفة المواد لـ 1 م3 خرسانة (نسبة خلط 1:3:6 المعتمدة تحت الأسس) + الضائعات
# السمنت: 200 كغم * 1.05 ضائعات = 210 كغم = 0.21 طن
cost_cement_m3 = 0.21 * cement_price_ton
# الرمل: 0.45 م3 * 1.08 ضائعات = 0.486 م3
cost_sand_m3 = 0.486 * sand_price_m3
# الحصى: 0.90 م3 * 1.08 ضائعات = 0.972 م3
cost_gravel_m3 = 0.972 * gravel_price_m3

# كلفة إنتاج 1 م3 خرسانة واصل ومطروح (مع 5% ضائعات للخرسانة حسب ص26)
total_concrete_m3 = (cost_cement_m3 + cost_sand_m3 + cost_gravel_m3 + concrete_transport_mix_m3) * 1.05

# ب. حساب كلفة فقرة (1-1) صب خرسانة غير مسلحة تحت الأسس سمك 10 سم (لكل 1 م2)
# 1. كلفة الخرسانة = كلفة م3 * 0.1 م (السمك)
item_concrete_cost = total_concrete_m3 * 0.1

# 2. قالب الخشب = 0.04 م2 لكل م2 صب (ويستخدم 12 مرة حسب ملاحظة الدليل ص32)
item_wood_cost = (0.04 * wood_price_m2) / 12.0

# 3. ماء الإسقاء لمدة 7 أيام (احتياج تقريبي 5 لتر/م2 باليوم * 7 أيام = 35 لتر)
cost_water_liter = water_tanker_price / 8000.0
item_water_cost = 35 * cost_water_liter

# مجموع المواد (أولاً)
first_materials_total = item_concrete_cost + item_wood_cost + item_water_cost

# ج. العمل (ثانياً): 1 عامل ماهر + 4 عمال + نقل 5 عمال + عدد ومستلزمات
workers_transport = 5 * 5000  # معدل نقل 5000 دينار للعامل
tools_cost = 10000            # عدد ومستلزمات عمل يومية
daily_labor_total = (1 * skilled_worker_wage) + (4 * unskilled_worker_wage) + workers_transport + tools_cost

# الإنتاجية القياسية = 300 م2 / يوم
productivity = 300.0
second_labor_per_m2 = daily_labor_total / productivity

# د. الكلفة الكلية النهائية للمتر المربع الواحد (مع التحميلات الإدارية والأرباح 20%-25%)
base_cost = first_materials_total + second_labor_per_m2
final_unit_price_m2 = base_cost * (1 + profit_margin)

# إضافة تباين سوقي طفيف واقعي (Noise 0.5%) لمحاكاة السوق الحقيقي بدقة
final_unit_price_m2 = final_unit_price_m2 * np.random.uniform(0.995, 1.005, num_samples)

# 3. حفظ البيانات في جدول منظم
df = pd.DataFrame({
    'Cement_Price_Ton': np.round(cement_price_ton, 0),
    'Sand_Price_m3': np.round(sand_price_m3, 0),
    'Gravel_Price_m3': np.round(gravel_price_m3, 0),
    'Wood_Price_m2': np.round(wood_price_m2, 0),
    'Concrete_Transport_Mix_m3': np.round(concrete_transport_mix_m3, 0),
    'Water_Tanker_Price': np.round(water_tanker_price, 0),
    'Skilled_Worker_Wage': np.round(skilled_worker_wage, 0),
    'Unskilled_Worker_Wage': np.round(unskilled_worker_wage, 0),
    'Profit_Margin': np.round(profit_margin, 3),
    'Final_Price_Per_m2': np.round(final_unit_price_m2, 0)
})

df.to_csv('iraq_concrete_data.csv', index=False)
print("✅ تم توليد قاعدة البيانات بنجاح! عدد السجلات:", len(df))
print(df.head())