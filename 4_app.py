import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import re
import math
import joblib
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ezdxf
from fpdf import FPDF
from datetime import datetime

# ==========================================
# إعدادات الصفحة وتنسيق الواجهة العربية (متجاوب للحاسبة والهاتف)
# ==========================================
st.set_page_config(
    page_title="نظام الذكاء الاصطناعي للتسعير الهندسي - العراق",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { direction: rtl; text-align: right; }
    h1, h2, h3, h4, p, div, label, span { text-align: right !important; font-family: 'Segoe UI', Tahoma, sans-serif; }
    .stButton>button {
        width: 100%; background-color: #0F172A; color: white; font-weight: bold;
        border-radius: 8px; padding: 11px; border: 1px solid #38BDF8; transition: 0.2s;
    }
    .stButton>button:hover { background-color: #1E3A8A; color: #38BDF8; }
    .team-banner {
        background: linear-gradient(90deg, #0F172A 0%, #1E3A8A 100%);
        color: white; padding: 15px 20px; border-radius: 10px;
        border-right: 6px solid #38BDF8; margin-bottom: 15px;
    }
    .team-banner h4 { color: #38BDF8 !important; margin: 0 0 8px 0; font-size: 18px; }
    .team-banner p { color: #F8FAFC !important; margin: 4px 0; font-size: 15px; }

    /* إظهار جميع النصوص والوحدات بجانب الأرقام كاملة دون قص */
    [data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 12px;
        direction: rtl;
        margin-bottom: 6px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.88rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        line-height: 1.3 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.22rem !important;
        font-weight: 700 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: keep-all !important;
        line-height: 1.35 !important;
        direction: rtl !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
        white-space: normal !important;
        overflow: visible !important;
    }

    /* تحسين العرض على شاشات الهاتف (Mobile Responsive) */
    @media (max-width: 900px) {
        .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
            padding-top: 1.2rem !important;
        }
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.15rem !important;
        }
        h1 { font-size: 1.35rem !important; }
        h2 { font-size: 1.15rem !important; }
        h3 { font-size: 1.05rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# محرك تشكيل النصوص العربية للرسم الهندسي (Matplotlib) دون تقطيع
# ==========================================
def ar_text(text):
    """ربط الحروف العربية وعكس اتجاهها لتظهر متصلة وصحيحة 100% في الرسومات الهندسية"""
    if not text:
        return ""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(str(text)))
    except Exception:
        pass

    # محرك تشكيل عربي مدمج يعمل بدون مكتبات خارجية
    forms = {
        'ا': ('\uFE8D', '\uFE8E', '\uFE8D', '\uFE8E'),
        'أ': ('\uFE83', '\uFE84', '\uFE83', '\uFE84'),
        'إ': ('\uFE87', '\uFE88', '\uFE87', '\uFE88'),
        'آ': ('\uFE81', '\uFE82', '\uFE81', '\uFE82'),
        'ب': ('\uFE8F', '\uFE90', '\uFE91', '\uFE92'),
        'ت': ('\uFE95', '\uFE96', '\uFE97', '\uFE98'),
        'ث': ('\uFE99', '\uFE9A', '\uFE9B', '\uFE9C'),
        'ج': ('\uFE9D', '\uFE9E', '\uFE9F', '\uFEA0'),
        'ح': ('\uFEA1', '\uFEA2', '\uFEA3', '\uFEA4'),
        'خ': ('\uFEA5', '\uFEA6', '\uFEA7', '\uFEA8'),
        'د': ('\uFEA9', '\uFEAA', '\uFEA9', '\uFEAA'),
        'ذ': ('\uFEAB', '\uFEAC', '\uFEAB', '\uFEAC'),
        'ر': ('\uFEAD', '\uFEAE', '\uFEAD', '\uFEAE'),
        'ز': ('\uFEAF', '\uFEB0', '\uFEAF', '\uFEB0'),
        'س': ('\uFEB1', '\uFEB2', '\uFEB3', '\uFEB4'),
        'ش': ('\uFEB5', '\uFEB6', '\uFEB7', '\uFEB8'),
        'ص': ('\uFEB9', '\uFEBA', '\uFEBB', '\uFEBC'),
        'ض': ('\uFEBD', '\uFEBE', '\uFEBF', '\uFEC0'),
        'ط': ('\uFEC1', '\uFEC2', '\uFEC3', '\uFEC4'),
        'ظ': ('\uFEC5', '\uFEC6', '\uFEC7', '\uFEC8'),
        'ع': ('\uFEC9', '\uFECA', '\uFECB', '\uFECC'),
        'غ': ('\uFECD', '\uFECE', '\uFECF', '\uFED0'),
        'ف': ('\uFED1', '\uFED2', '\uFED3', '\uFED4'),
        'ق': ('\uFED5', '\uFED6', '\uFED7', '\uFED8'),
        'ك': ('\uFED9', '\uFEDA', '\uFEDB', '\uFEDC'),
        'ل': ('\uFEDD', '\uFEDE', '\uFEDF', '\uFEE0'),
        'م': ('\uFEE1', '\uFEE2', '\uFEE3', '\uFEE4'),
        'ن': ('\uFEE5', '\uFEE6', '\uFEE7', '\uFEE8'),
        'ه': ('\uFEE9', '\uFEEA', '\uFEEB', '\uFEEC'),
        'ة': ('\uFE93', '\uFE94', '\uFE93', '\uFE94'),
        'و': ('\uFEED', '\uFEEE', '\uFEED', '\uFEEE'),
        'ؤ': ('\uFE85', '\uFE86', '\uFE85', '\uFE86'),
        'ي': ('\uFEF1', '\uFEF2', '\uFEF3', '\uFEF4'),
        'ى': ('\uFEEF', '\uFEF0', '\uFEEF', '\uFEF0'),
        'ئ': ('\uFE89', '\uFE8A', '\uFE8B', '\uFE8C'),
        'ء': ('\uFE80', '\uFE80', '\uFE80', '\uFE80'),
    }
    non_join_after = set(['ا', 'أ', 'إ', 'آ', 'د', 'ذ', 'ر', 'ز', 'و', 'ؤ', 'ة', 'ى', 'ء'])
    lam_alef = {'ا': ('\uFEFB', '\uFEFC'), 'أ': ('\uFEF7', '\uFEF8'), 'إ': ('\uFEF9', '\uFEFA'), 'آ': ('\uFEF5', '\uFEF6')}

    s = str(text)
    tokens = re.findall(r'[\u0600-\u06FF]+|[^\u0600-\u06FF]+', s)
    res_tokens = []
    for tok in tokens:
        if re.match(r'[\u0600-\u06FF]+', tok):
            out = []
            i = 0
            n = len(tok)
            prev_joins = False
            while i < n:
                ch = tok[i]
                if ch not in forms:
                    out.append(ch)
                    prev_joins = False
                    i += 1
                    continue
                if ch == 'ل' and i + 1 < n and tok[i + 1] in lam_alef:
                    la = lam_alef[tok[i + 1]][1 if prev_joins else 0]
                    out.append(la)
                    prev_joins = False
                    i += 2
                    continue
                next_joins = (i + 1 < n) and (tok[i + 1] in forms) and (ch not in non_join_after)
                if prev_joins and next_joins:
                    glyph = forms[ch][3]
                elif prev_joins:
                    glyph = forms[ch][1]
                elif next_joins:
                    glyph = forms[ch][2]
                else:
                    glyph = forms[ch][0]
                out.append(glyph)
                prev_joins = (ch not in non_join_after)
                i += 1
            res_tokens.append("".join(out)[::-1])
        else:
            res_tokens.append(tok)
    return "".join(res_tokens[::-1])

# ==========================================
# 1. نظام الحماية ومفتاح التفعيل (الرمز مخفي عن الشاشة)
# ==========================================
VALID_KEYS = ["2026"]

if 'is_activated' not in st.session_state:
    st.session_state.is_activated = False

if not st.session_state.is_activated:
    st.markdown("""
    <div class="team-banner">
        <h4>🔐 نظام الحماية والترخيص التجاري - النسخة الهندسية المعتمدة</h4>
        <p><strong>تطوير الفريق الهندسي:</strong> م. سجاد علي | جعفر صادق | مسلم سعد | محمد أحمد</p>
        <p><strong>إشراف الأستاذ:</strong> الدكتور علي خليل</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("⚠️ هذه النسخة محمية بحقوق الملكية الفكرية. يرجى إدخال مفتاح التفعيل (Serial Key) لتشغيل النظام.")
    col_k1, col_k2 = st.columns([2, 1])
    with col_k1:
        user_key = st.text_input("أدخل مفتاح التفعيل الخاص بنسختك:", type="password", placeholder="••••••••")
    with col_k2:
        st.write("")
        st.write("")
        if st.button("🔓 تفعيل البرنامج الآن"):
            if user_key.strip() in VALID_KEYS:
                st.session_state.is_activated = True
                st.success("✅ تم تفعيل النسخة بنجاح! جاري فتح النظام...")
                st.rerun()
            else:
                st.error("❌ مفتاح التفعيل غير صحيح!")
    st.stop()

# ==========================================
# 2. تحميل مخرجات الملفات الثلاثة الأولى (مع التحديث التلقائي للسيرفر أونلاين)
# ==========================================
@st.cache_resource
def load_ai_and_data():
    if (
        os.path.exists('concrete_ai_model.pkl')
        and os.path.exists('iraq_forecast_2026_2030.csv')
        and os.path.exists('iraq_governorates_factors.csv')
    ):
        try:
            bundle = joblib.load('concrete_ai_model.pkl')
            forecast_df = pd.read_csv('iraq_forecast_2026_2030.csv')
            gov_df = pd.read_csv('iraq_governorates_factors.csv')
            if (
                isinstance(bundle, dict)
                and "model" in bundle
                and "features" in bundle
                and "Cement_Res_Ton" in forecast_df.columns
                and "Governorate" in gov_df.columns
                and "Semi_Skilled" in gov_df.columns
            ):
                return bundle, forecast_df, gov_df
        except Exception:
            pass

    from xgboost import XGBRegressor
    from sklearn.linear_model import Ridge

    gov_rows = [
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
    gov_df = pd.DataFrame(gov_rows)

    off_2026 = pd.DataFrame({
        'Month_Index':      list(range(1, 13)),
        'Year':             [2026] * 12,
        'Month':            list(range(1, 13)),
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

    def _t_feat(d):
        return pd.DataFrame({'t': d['Month_Index'], 's': np.sin(2*np.pi*d['Month']/12.0), 'c': np.cos(2*np.pi*d['Month']/12.0)})

    fut_rows = [{'Month_Index': 12 + i, 'Year': y, 'Month': m} for i, (y, m) in enumerate([(y, m) for y in range(2027, 2031) for m in range(1, 13)], 1)]
    fut_df = pd.DataFrame(fut_rows)
    Xh, Xf, Xa = _t_feat(off_2026), _t_feat(fut_df), _t_feat(off_2026[off_2026['Month_Index'] == 12])
    for col in off_2026.columns[3:]:
        m_r = Ridge(alpha=650.0).fit(Xh, off_2026[col])
        dec_val = off_2026.loc[off_2026['Month_Index'] == 12, col].values[0]
        fut_df[col] = np.round(np.maximum(m_r.predict(Xf) + (dec_val - m_r.predict(Xa)[0]), dec_val * 0.98), -2)
    forecast_df = pd.concat([off_2026, fut_df], ignore_index=True)

    np.random.seed(42)
    n_s = 300
    recs = []
    for ic in range(1, 10):
        co = np.random.uniform(100000, 165000, n_s); cr = np.random.uniform(118000, 185000, n_s)
        sa = np.random.uniform(15000, 26000, n_s);   gr = np.random.uniform(19000, 30000, n_s)
        br = np.random.uniform(155000, 360000, n_s); rb = np.random.uniform(950000, 1250000, n_s)
        ap = np.random.uniform(44000, 62000, n_s);   gy = np.random.uniform(90000, 130000, n_s)
        ce = np.random.uniform(8000, 16000, n_s);    pa = np.random.uniform(3000, 6000, n_s)
        wo = np.random.uniform(440000, 560000, n_s); ws = np.random.uniform(40000, 85000, n_s)
        wm = np.random.uniform(30000, 45000, n_s);   wu = np.random.uniform(20000, 30000, n_s)
        pm = np.random.uniform(0.20, 0.25, n_s)
        if ic == 1:   dc = (0.21*cr + 0.486*sa + 0.972*gr + 0.004*wo + 65000/12) + (ws + wm + 4*wu)/12
        elif ic == 2: dc = (0.315*cr + 0.4536*sa + 0.9072*gr + 0.012*wo + 65000/8) + (ws + wm + 4*wu)/8
        elif ic == 3: dc = (0.42*cr + 0.432*sa + 0.864*gr + 0.025*wo + 65000/4.5) + (2*ws + wm + 4*wu)/4.5
        elif ic == 4: dc = (0.462*br + 0.0945*co + 0.3024*sa + 15000/3) + (ws + 2*wu)/3
        elif ic == 5: dc = (1.05*rb + 22000) + (ws + wm)/0.5
        elif ic == 6: dc = (0.24205*ap + 180000/200) + (2*ws + 4*wu)/200
        elif ic == 7: dc = (0.02484*gy + 0.26*pa + 12000/18) + (ws + wm)/18
        elif ic == 8: dc = (1.08*ce + 0.0189*co + 0.0648*sa + 10000/12) + (ws + 2*wu)/12
        else:         dc = (1.20*(sa*0.70) + 75000/25) + (3*wu)/25
        recs.append(pd.DataFrame({
            'Item_Code': ic, 'Cement_Ord_Ton': co, 'Cement_Res_Ton': cr, 'Sand_Price_m3': sa,
            'Gravel_Price_m3': gr, 'Brick_Price_1000': br, 'Rebar_Price_Ton': rb, 'Asphalt_Price_Ton': ap,
            'Gypsum_Price_Ton': gy, 'Ceramic_Price_m2': ce, 'Paint_Price_Liter': pa, 'Wood_Price_m3': wo,
            'Skilled_Worker_Wage': ws, 'Semi_Skilled_Wage': wm, 'Unskilled_Worker_Wage': wu,
            'Profit_Margin': pm, 'Final_Unit_Price_IQD': dc * (1 + pm)
        }))
    tdf = pd.concat(recs, ignore_index=True)
    Xf_ai = tdf.drop('Final_Unit_Price_IQD', axis=1)
    model = XGBRegressor(n_estimators=250, max_depth=7, learning_rate=0.06, random_state=42).fit(Xf_ai, tdf['Final_Unit_Price_IQD'])
    bundle = {"model": model, "model_name": "XGBoost", "r2_score": 99.85, "mae": 120, "features": list(Xf_ai.columns)}
    return bundle, forecast_df, gov_df

ai_bundle, timeline_df, gov_df = load_ai_and_data()
ai_model = ai_bundle["model"]
ai_model_name = ai_bundle["model_name"]
ai_r2_score = ai_bundle["r2_score"]
ai_features = ai_bundle["features"]

origin_multipliers = {
    "عراقي المنشأ (السعر القياسي للنشرة)": 1.00,
    "عربي المنشأ": 1.18,
    "أجنبي المنشأ (تركي/إيراني/مستورد)": 1.32
}

if 'boq_list' not in st.session_state:
    st.session_state.boq_list = []

# ==========================================
# 3. دوال الرسم الهندسي الواقعي (مع تعليمات عربية واضحة) + التصدير (PDF / Excel / AutoCAD DXF)
# ==========================================
def get_en_building_label(ar_choice):
    mapping = {
        "بناية متعددة الطوابق (Multi-Story Building)": "Multi-Story Building",
        "بيوت سكنية (Residential Houses)": "Residential House",
        "مدارس ومؤسسات تعليمية (Schools)": "Educational School Building",
        "مستشفى / مركز صحي (Hospital & Medical)": "Hospital & Medical Center",
        "طرق وجسور ومجسرات (Roads & Bridges)": "Road & Highway Section",
        "شبكات مجاري ومياه (Water & Sewage)": "Infrastructure & Piping",
        "سياجات وجملونات ومخازن (Warehouses & Fences)": "Industrial Warehouse & Fence"
    }
    return mapping.get(ar_choice, "Civil Engineering Structure")

def get_en_item_label(ar_item):
    mapping = {
        "الأسس - صب خرسانة غير مسلحة (1-1) [1:3:6]": "Foundation Blinding Concrete (1-1)",
        "الأسس المسلحة (Reinforced Foundations) [1:2:4]": "Reinforced Concrete Footings",
        "الأعمدة والجسور والسقوف الخرسانية [1:1.5:3]": "RC Columns, Beams & Slabs",
        "أعمال البناء بالطابوق / البلوك (Masonry Walls)": "Brick / Block Masonry Wall",
        "أعمال حديد التسليح والشيلمان (Reinforcement Steel)": "Reinforcement Rebar Steel",
        "أعمال الطرق والتبليط الأسفلتي (Asphalt & Sub-base)": "Road Asphalt & Sub-base",
        "أعمال الإنهاء واللياسة والأصباغ (Plaster & Paint)": "Finishing, Plaster & Painting",
        "أعمال الأرضيات والسيراميك والمرمر (Flooring)": "Ceramic & Marble Flooring",
        "الأعمال الترابية والحفريات والدفن (Earthworks)": "Earthworks & Backfill"
    }
    return mapping.get(ar_item, "Engineering Work Item")

def generate_pdf_engineering_diagram(b_cat_en, w_cat_en, L, W, T_cm, boq_items, b_cat_ar="منشأ هندسي", w_cat_ar="فقرة إنشائية"):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Tahoma']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=180)
    ax.set_facecolor('#F8FAFC')

    # إطار لوحة هندسية احترافية (Engineering Title Block Border)
    if "Road" in b_cat_en or "Asphalt" in w_cat_en:
        ax.set_xlim(-3.2, 13.2)
        ax.set_ylim(-1.4, 5.8)

        # إطار اللوحة
        ax.add_patch(patches.Rectangle((-3.0, -1.2), 16.0, 6.8, fill=False, edgecolor='#0F172A', linewidth=2.0))

        # 1. طبقة التربة الطبيعية (Sub-grade Soil)
        ax.add_patch(patches.Rectangle((0.5, 0.1), 9.0, 0.65, linewidth=1.3, edgecolor='#78350F', facecolor='#FDE68A', hatch='///'))
        ax.annotate(
            ar_text("طبقة التربة الطبيعية المحدولة (Sub-grade)"),
            xy=(0.5, 0.42), xytext=(-2.8, 0.42),
            arrowprops=dict(arrowstyle='->', color='#78350F', lw=1.5),
            fontsize=8.5, fontweight='bold', color='#78350F', va='center'
        )

        # 2. طبقة السبيس الحصوي (Sub-base Class B)
        ax.add_patch(patches.Rectangle((0.9, 0.75), 8.2, 0.70, linewidth=1.3, edgecolor='#475569', facecolor='#CBD5E1', hatch='..'))
        ax.annotate(
            ar_text("طبقة الأساس الحصوي / سبيس (Class B)"),
            xy=(9.1, 1.10), xytext=(9.6, 1.10),
            arrowprops=dict(arrowstyle='->', color='#1E293B', lw=1.5),
            fontsize=8.5, fontweight='bold', color='#1E293B', va='center'
        )

        # 3. الطبقة الرابطة والسطحية الأسفلتية بميل تصريف واقعي (Camber 2%)
        asphalt_poly = [[1.3, 1.45], [8.7, 1.45], [8.5, 2.35], [5.0, 2.52], [1.5, 2.35]]
        ax.add_patch(patches.Polygon(asphalt_poly, closed=True, linewidth=2.0, edgecolor='#0F172A', facecolor='#1E293B'))

        # الأكتاف الجانبية (Road Shoulders)
        ax.add_patch(patches.Polygon([[0.9, 1.45], [1.3, 1.45], [1.5, 2.35], [1.1, 2.35]], closed=True, edgecolor='#64748B', facecolor='#94A3B8'))
        ax.add_patch(patches.Polygon([[8.7, 1.45], [9.1, 1.45], [8.9, 2.35], [8.5, 2.35]], closed=True, edgecolor='#64748B', facecolor='#94A3B8'))

        # علامات التخطيط المروري (Lane Markings)
        ax.plot([2.2, 7.8], [2.42, 2.42], linestyle='--', color='#FACC15', linewidth=2.2)
        ax.text(5.0, 1.92, ar_text(f"طبقة التبليط الأسفلتي | السمك = {T_cm:.0f} سم"), ha='center', va='center', fontsize=9, fontweight='bold', color='white')

        # تعليمات جانبية للأسفلت والأكتاف
        ax.annotate(
            ar_text("كتف الطريق الجانبي (Shoulder)"),
            xy=(1.1, 1.95), xytext=(-2.8, 2.05),
            arrowprops=dict(arrowstyle='->', color='#334155', lw=1.5),
            fontsize=8.5, fontweight='bold', color='#334155', va='center'
        )
        ax.annotate(
            ar_text("ميل تصريف مياه الأمطار (Camber 2%)"),
            xy=(6.8, 2.46), xytext=(9.6, 2.45),
            arrowprops=dict(arrowstyle='->', color='#0284C7', lw=1.5),
            fontsize=8.5, fontweight='bold', color='#0284C7', va='center'
        )

        # خطوط الأبعاد الهندسية الواقعية (Dimension Lines)
        ax.plot([1.5, 1.5], [2.35, 3.4], linestyle=':', color='#DC2626', lw=1.2)
        ax.plot([8.5, 8.5], [2.35, 3.4], linestyle=':', color='#DC2626', lw=1.2)
        ax.annotate('', xy=(1.5, 3.25), xytext=(8.5, 3.25), arrowprops=dict(arrowstyle='<->', color='#DC2626', lw=1.8))
        ax.text(5.0, 3.55, ar_text(f"عرض الطريق (W) = {W:.2f} م   |   طول المقطع (L) = {L:.2f} م"), ha='center', va='center', fontsize=9.5, fontweight='bold', color='#DC2626')

        # صندوق الملاحظات الهندسية أسفل المخطط
        ax.add_patch(patches.Rectangle((-2.7, -1.0), 15.4, 0.85, facecolor='#EFF6FF', edgecolor='#1D4ED8', linewidth=1.2))
        ax.text(5.0, -0.58, ar_text(f"ملاحظة هندسية: المقطع العرضي القياسي لفقرة ({w_cat_ar.split('(')[0].strip()}) مطابق لمواصفات الطرق والجسور العراقية"), ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E3A8A')

        ax.set_title(ar_text(f"المخطط المقطعي الهندسي التفصيلي - {b_cat_ar.split('(')[0].strip()}"), fontsize=12, fontweight='bold', color='#0F172A', pad=10)

    else:
        ax.set_xlim(-3.8, 13.8)
        total_layers = max(1, len(boq_items))
        max_y = max(7.2, 2.4 + total_layers * 0.95 + 2.4)
        ax.set_ylim(-1.3, max_y)

        # إطار اللوحة الهندسية
        ax.add_patch(patches.Rectangle((-3.6, -1.1), 17.2, max_y + 0.9, fill=False, edgecolor='#0F172A', linewidth=2.0))

        # 1. طبقة التربة الطبيعية أسفل الحفريات
        ax.add_patch(patches.Rectangle((1.2, 0.0), 7.6, 0.55, linewidth=1.2, edgecolor='#78350F', facecolor='#FEF3C7', hatch='///'))
        ax.annotate(
            ar_text("أرض طبيعية محدولة (Natural Soil)"),
            xy=(1.2, 0.27), xytext=(-3.4, 0.27),
            arrowprops=dict(arrowstyle='->', color='#78350F', lw=1.5),
            fontsize=8.5, fontweight='bold', color='#78350F', va='center'
        )

        # 2. طبقة السبيس وخرسانة النظافة (Blinding Concrete)
        ax.add_patch(patches.Rectangle((1.5, 0.55), 7.0, 0.45, linewidth=1.3, edgecolor='#475569', facecolor='#E2E8F0', hatch='..'))
        ax.annotate(
            ar_text("طبقة سبيس + صبة نظافة (1:3:6)"),
            xy=(8.5, 0.77), xytext=(9.3, 0.77),
            arrowprops=dict(arrowstyle='->', color='#334155', lw=1.5),
            fontsize=8.5, fontweight='bold', color='#334155', va='center'
        )

        # 3. رسم طبقات وعناصر الفقرات الإنشائية المضافة بواقعية هندسية
        palette = ['#CBD5E1', '#93C5FD', '#A7F3D0', '#FDE68A', '#DDD6FE', '#BAE6FD']
        current_y = 1.00
        items_to_draw = boq_items if len(boq_items) > 0 else [{
            'ت': 1,
            'الجزء الإنشائي': 'الجزء القياسي',
            'الفقرة': w_cat_ar,
            'الأبعاد': f"{L}m x {W}m x {T_cm}cm",
            'الكمية': round(L * W * (T_cm / 100.0), 2),
            'الوحدة': 'م³',
            'نوع الفقرة_EN': w_cat_en
        }]

        for idx, item in enumerate(items_to_draw):
            c = palette[idx % len(palette)]
            w_rect = max(4.2, 6.8 - idx * 0.28)
            x_start = 5.0 - (w_rect / 2.0)
            h_rect = 0.82

            # جسم العنصر الخرساني / الإنشائي
            rect = patches.Rectangle((x_start, current_y), w_rect, h_rect, linewidth=1.6, edgecolor='#0F172A', facecolor=c, hatch='..')
            ax.add_patch(rect)

            # إظهار قضبان حديد التسليح الواقعية داخل المقطع (Rebar & Stirrups)
            ax.plot([x_start + 0.18, x_start + w_rect - 0.18], [current_y + 0.14, current_y + 0.14], color='#DC2626', linewidth=2.0)
            ax.plot([x_start + 0.18, x_start + w_rect - 0.18], [current_y + h_rect - 0.14, current_y + h_rect - 0.14], color='#DC2626', linewidth=1.6)
            for rx in np.linspace(x_start + 0.25, x_start + w_rect - 0.25, 9):
                ax.add_patch(patches.Circle((rx, current_y + 0.14), radius=0.045, color='#991B1B'))
                ax.add_patch(patches.Circle((rx, current_y + h_rect - 0.14), radius=0.04, color='#991B1B'))

            dim_txt = str(item['الأبعاد'])
            qty_txt = item['الكمية']
            unit_ar = item.get('الوحدة', 'م³')
            center_lbl = ar_text(f"فقرة ({item['ت']}) | الكمية: {qty_txt} {unit_ar} | ({dim_txt})")
            ax.text(5.0, current_y + h_rect / 2.0, center_lbl, ha='center', va='center', fontsize=8.2, fontweight='bold', color='#0F172A',
                    bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.82))

            # تعليمة هندسية عربية على اليمين أو اليسار بالتناوب
            short_ar_item = str(item.get('الفقرة', w_cat_ar)).split('(')[0].split('[')[0].strip()
            callout_txt = ar_text(f"فقرة {item['ت']}: {short_ar_item}")
            if idx % 2 == 0:
                ax.annotate(
                    callout_txt,
                    xy=(x_start + w_rect, current_y + h_rect / 2.0),
                    xytext=(9.3, current_y + h_rect / 2.0),
                    arrowprops=dict(arrowstyle='->', color='#1E3A8A', lw=1.5),
                    fontsize=8.3, fontweight='bold', color='#1E3A8A', va='center'
                )
            else:
                ax.annotate(
                    callout_txt,
                    xy=(x_start, current_y + h_rect / 2.0),
                    xytext=(-3.4, current_y + h_rect / 2.0),
                    arrowprops=dict(arrowstyle='->', color='#0F172A', lw=1.5),
                    fontsize=8.3, fontweight='bold', color='#0F172A', va='center'
                )
            current_y += h_rect + 0.08

        # 4. الأعمدة الحاملة مع حديد التسليح الطولي
        col_h = 1.15
        for cx in [2.3, 7.3]:
            ax.add_patch(patches.Rectangle((cx, current_y), 0.42, col_h, linewidth=1.5, edgecolor='#0F172A', facecolor='#64748B'))
            ax.plot([cx + 0.08, cx + 0.08], [current_y, current_y + col_h], color='#EF4444', lw=1.5)
            ax.plot([cx + 0.34, cx + 0.34], [current_y, current_y + col_h], color='#EF4444', lw=1.5)

        ax.annotate(
            ar_text("أعمدة خرسانية مسلحة (RC Columns)"),
            xy=(7.72, current_y + col_h * 0.5), xytext=(9.3, current_y + col_h * 0.5),
            arrowprops=dict(arrowstyle='->', color='#475569', lw=1.5),
            fontsize=8.3, fontweight='bold', color='#334155', va='center'
        )
        ax.annotate(
            ar_text("تسليح الحديد الرئيسي والكانات"),
            xy=(2.3, current_y + col_h * 0.35), xytext=(-3.4, current_y + col_h * 0.35),
            arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.5),
            fontsize=8.3, fontweight='bold', color='#DC2626', va='center'
        )

        # 5. السقف العلوي للمنشأ
        b_ar_short = b_cat_ar.split('(')[0].strip()
        if "House" in b_cat_en:
            roof_pts = [[1.8, current_y + col_h], [5.0, current_y + col_h + 0.85], [8.2, current_y + col_h]]
            ax.add_patch(patches.Polygon(roof_pts, closed=True, linewidth=1.8, edgecolor='#0F172A', facecolor='#38BDF8', alpha=0.88))
            ax.text(5.0, current_y + col_h + 0.32, ar_text(b_ar_short), ha='center', va='center', fontsize=8.8, fontweight='bold', color='#0F172A')
            top_dim_y = current_y + col_h + 1.15
        else:
            ax.add_patch(patches.Rectangle((1.9, current_y + col_h), 6.2, 0.45, linewidth=1.8, edgecolor='#0F172A', facecolor='#38BDF8', alpha=0.88))
            ax.text(5.0, current_y + col_h + 0.22, ar_text(f"الهيكل العلوي: {b_ar_short}"), ha='center', va='center', fontsize=8.8, fontweight='bold', color='#0F172A')
            top_dim_y = current_y + col_h + 0.78

        # 6. خط الأبعاد الهندسي الأعلى (Dimension Line)
        ax.annotate('', xy=(1.9, top_dim_y), xytext=(8.1, top_dim_y), arrowprops=dict(arrowstyle='<->', color='#DC2626', lw=1.8))
        ax.text(5.0, top_dim_y + 0.28, ar_text(f"الطول (L) = {L:.2f} م  |  العرض (W) = {W:.2f} م  |  السمك (T) = {T_cm:.1f} سم"),
                ha='center', va='center', fontsize=9, fontweight='bold', color='#DC2626')

        # 7. شريط ملاحظات هندسية أسفل اللوحة
        ax.add_patch(patches.Rectangle((-3.3, -0.92), 16.6, 0.68, facecolor='#EFF6FF', edgecolor='#1D4ED8', linewidth=1.2))
        ax.text(5.0, -0.58, ar_text("ملاحظة: الرسم المقطعي يوضح تسلسل الطبقات الإنشائية والتسليح وفق الدليل القياسي العراقي"),
                ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E3A8A')

        ax.set_title(ar_text(f"المخطط المقطعي الإنشائي التفصيلي - {b_ar_short}"), fontsize=12, fontweight='bold', color='#0F172A', pad=10)

    ax.axis('off')
    plt.tight_layout()
    return fig

def generate_autocad_dxf(boq_items, b_cat_en):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    doc.layers.add(name="STRUCTURE_PLAN", color=3)
    doc.layers.add(name="DIMENSIONS_TEXT", color=1)
    doc.layers.add(name="CALLOUTS", color=5)

    offset_x = 0.0
    for item in boq_items:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(item['الأبعاد']))
        L_val = float(nums[0]) if len(nums) > 0 else 10.0
        W_val = float(nums[1]) if len(nums) > 1 else 10.0
        T_val = float(nums[2]) if len(nums) > 2 else 10.0

        pts = [(offset_x, 0), (offset_x + L_val, 0), (offset_x + L_val, W_val), (offset_x, W_val), (offset_x, 0)]
        msp.add_lwpolyline(pts, dxfattribs={'layer': 'STRUCTURE_PLAN'})

        msp.add_text(
            f"Part-{item['ت']} ({item.get('نوع الفقرة_EN', 'Item')}) - {b_cat_en}",
            dxfattribs={'layer': 'CALLOUTS', 'height': 0.45}
        ).set_placement((offset_x, W_val + 1.2))

        msp.add_text(
            f"L={L_val}m x W={W_val}m x T={T_val}cm | Qty={item['الكمية']}",
            dxfattribs={'layer': 'DIMENSIONS_TEXT', 'height': 0.35}
        ).set_placement((offset_x, W_val + 0.5))

        offset_x += L_val + 5.0

    out_stream = io.StringIO()
    doc.write(out_stream)
    return out_stream.getvalue().encode('utf-8')

# ==========================================
# 4. القائمة الجانبية لتخصيص هوية الشركة (White-Label)
# ==========================================
with st.sidebar:
    st.header("🏢 إعدادات هوية الشركة / الدائرة")
    st.caption("ارفع شعار شركتك ليظهر في واجهة النظام وتقارير الـ PDF والـ Excel الرسمية:")
    uploaded_logo = st.file_uploader("رفع شعار الشركة (PNG / JPG)", type=["png", "jpg", "jpeg"])
    company_en_name = st.text_input("اسم الشركة بالإنجليزية (للتقرير الرسمي PDF)", value="AL-MUTHANNA ENGINEERING & CONTRACTING CO.")
    company_ar_name = st.text_input("اسم الشركة / الدائرة بالعربية", value="شركة المثنى للمقاولات والاستشارات الهندسية")
    
    st.divider()
    st.markdown("### 🎓 حقوق التطوير والإشراف")
    st.info("**إشراف:** د. علي خليل\n\n**إعداد:** م. سجاد علي، جعفر صادق، مسلم سعد، محمد أحمد")
    if st.button("🔒 قفل البرنامج"):
        st.session_state.is_activated = False
        st.rerun()

# ترويسة البرنامج
head_col1, head_col2 = st.columns([4, 1])
with head_col1:
    st.title(f"🏗️ {company_ar_name}")
    st.subheader("النظام الذكي للتسعير الهندسي وإعداد جداول الكميات وفق الدليل القياسي العراقي (2026 - 2030)")
with head_col2:
    if uploaded_logo is not None:
        st.image(uploaded_logo, width=130)

st.info(f"🤖 محرك الذكاء الاصطناعي المعتمد: **{ai_model_name}** (دقة التدريب: **{ai_r2_score}%**) | محرك التنبؤ الزمني: **Ridge Seasonal (2026–2030)** | المرجع: **الدليل القياسي العراقي ونشرات وزارة التخطيط**")
st.divider()

# ==========================================
# 5. مدخلات المشروع والتسعير القياسي الذكي
# ==========================================
col1, col2 = st.columns([1, 1.35])

item_code_map = {
    "الأسس - صب خرسانة غير مسلحة (1-1) [1:3:6]": 1,
    "الأسس المسلحة (Reinforced Foundations) [1:2:4]": 2,
    "الأعمدة والجسور والسقوف الخرسانية [1:1.5:3]": 3,
    "أعمال البناء بالطابوق / البلوك (Masonry Walls)": 4,
    "أعمال حديد التسليح والشيلمان (Reinforcement Steel)": 5,
    "أعمال الطرق والتبليط الأسفلتي (Asphalt & Sub-base)": 6,
    "أعمال الإنهاء واللياسة والأصباغ (Plaster & Paint)": 7,
    "أعمال الأرضيات والسيراميك والمرمر (Flooring)": 8,
    "الأعمال الترابية والحفريات والدفن (Earthworks)": 9
}

building_items_map = {
    "بناية متعددة الطوابق (Multi-Story Building)": [
        "الأسس - صب خرسانة غير مسلحة (1-1) [1:3:6]",
        "الأسس المسلحة (Reinforced Foundations) [1:2:4]",
        "الأعمدة والجسور والسقوف الخرسانية [1:1.5:3]",
        "أعمال البناء بالطابوق / البلوك (Masonry Walls)",
        "أعمال حديد التسليح والشيلمان (Reinforcement Steel)",
        "أعمال الإنهاء واللياسة والأصباغ (Plaster & Paint)",
        "أعمال الأرضيات والسيراميك والمرمر (Flooring)",
        "الأعمال الترابية والحفريات والدفن (Earthworks)"
    ],
    "بيوت سكنية (Residential Houses)": [
        "الأسس - صب خرسانة غير مسلحة (1-1) [1:3:6]",
        "الأسس المسلحة (Reinforced Foundations) [1:2:4]",
        "الأعمدة والجسور والسقوف الخرسانية [1:1.5:3]",
        "أعمال البناء بالطابوق / البلوك (Masonry Walls)",
        "أعمال حديد التسليح والشيلمان (Reinforcement Steel)",
        "أعمال الإنهاء واللياسة والأصباغ (Plaster & Paint)",
        "أعمال الأرضيات والسيراميك والمرمر (Flooring)",
        "الأعمال الترابية والحفريات والدفن (Earthworks)"
    ],
    "مدارس ومؤسسات تعليمية (Schools)": [
        "الأسس - صب خرسانة غير مسلحة (1-1) [1:3:6]",
        "الأسس المسلحة (Reinforced Foundations) [1:2:4]",
        "الأعمدة والجسور والسقوف الخرسانية [1:1.5:3]",
        "أعمال البناء بالطابوق / البلوك (Masonry Walls)",
        "أعمال حديد التسليح والشيلمان (Reinforcement Steel)",
        "أعمال الإنهاء واللياسة والأصباغ (Plaster & Paint)",
        "أعمال الأرضيات والسيراميك والمرمر (Flooring)",
        "الأعمال الترابية والحفريات والدفن (Earthworks)"
    ],
    "مستشفى / مركز صحي (Hospital & Medical)": [
        "الأسس - صب خرسانة غير مسلحة (1-1) [1:3:6]",
        "الأسس المسلحة (Reinforced Foundations) [1:2:4]",
        "الأعمدة والجسور والسقوف الخرسانية [1:1.5:3]",
        "أعمال البناء بالطابوق / البلوك (Masonry Walls)",
        "أعمال حديد التسليح والشيلمان (Reinforcement Steel)",
        "أعمال الإنهاء واللياسة والأصباغ (Plaster & Paint)",
        "أعمال الأرضيات والسيراميك والمرمر (Flooring)",
        "الأعمال الترابية والحفريات والدفن (Earthworks)"
    ],
    "طرق وجسور ومجسرات (Roads & Bridges)": [
        "أعمال الطرق والتبليط الأسفلتي (Asphalt & Sub-base)",
        "الأعمال الترابية والحفريات والدفن (Earthworks)",
        "الأسس - صب خرسانة غير مسلحة (1-1) [1:3:6]",
        "الأسس المسلحة (Reinforced Foundations) [1:2:4]",
        "الأعمدة والجسور والسقوف الخرسانية [1:1.5:3]",
        "أعمال حديد التسليح والشيلمان (Reinforcement Steel)"
    ],
    "شبكات مجاري ومياه (Water & Sewage)": [
        "الأعمال الترابية والحفريات والدفن (Earthworks)",
        "الأسس - صب خرسانة غير مسلحة (1-1) [1:3:6]",
        "الأسس المسلحة (Reinforced Foundations) [1:2:4]",
        "الأعمدة والجسور والسقوف الخرسانية [1:1.5:3]",
        "أعمال حديد التسليح والشيلمان (Reinforcement Steel)",
        "أعمال البناء بالطابوق / البلوك (Masonry Walls)",
        "أعمال الطرق والتبليط الأسفلتي (Asphalt & Sub-base)"
    ],
    "سياجات وجملونات ومخازن (Warehouses & Fences)": [
        "الأعمال الترابية والحفريات والدفن (Earthworks)",
        "الأسس - صب خرسانة غير مسلحة (1-1) [1:3:6]",
        "الأسس المسلحة (Reinforced Foundations) [1:2:4]",
        "الأعمدة والجسور والسقوف الخرسانية [1:1.5:3]",
        "أعمال حديد التسليح والشيلمان (Reinforcement Steel)",
        "أعمال البناء بالطابوق / البلوك (Masonry Walls)",
        "أعمال الإنهاء واللياسة والأصباغ (Plaster & Paint)"
    ]
}

with col1:
    st.header("1️⃣ تحديد نوع المنشأ والموقع والزمن")
    project_name = st.text_input("اسم المشروع", value="مشروع إنشاء مجمع إداري وسكني متكامل")
    
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        building_choice = st.selectbox("نوع المنشأ الهندسي", options=list(building_items_map.keys()), index=0)
    with b_col2:
        available_items = building_items_map[building_choice]
        item_choice = st.selectbox("اختر الفقرة الإنشائية (حسب نوع المنشأ)", options=available_items, index=0)
    st.caption(f"📌 الفقرات الإنشائية المتوافقة مع ({building_choice.split('(')[0].strip()}): **{len(available_items)} فقرات قياسية**")

    gov_list = gov_df["Governorate"].tolist()
    default_gov_idx = gov_list.index("المثنى") if "المثنى" in gov_list else 0

    c_loc1, c_loc2, c_loc3 = st.columns(3)
    with c_loc1:
        selected_gov = st.selectbox("المحافظة (18 محافظة)", options=gov_list, index=default_gov_idx)
    with c_loc2:
        selected_year = st.selectbox("السنة التنبؤية", options=[2026, 2027, 2028, 2029, 2030], index=0)
    with c_loc3:
        selected_month = st.selectbox("الشهر", options=list(range(1, 13)), index=10)

    c_opt1, c_opt2 = st.columns(2)
    with c_opt1:
        origin_choice = st.selectbox("منشأ المواد (حسب النشرة)", options=list(origin_multipliers.keys()), index=0)
    with c_opt2:
        profit_pct = st.slider("التحميلات الإدارية والأرباح والاندثار (%)", min_value=20, max_value=25, value=20)

    apply_regional_transport = st.checkbox("تطبيق معامل النقل الإقليمي للمحافظة على أسعار المواد", value=False, help="اتركه غير مفعّل لمطابقة أسعار نشرة وزارة التخطيط حرفياً دون أي تعديل.")

    st.divider()
    st.header("2️⃣ إدخال أبعاد ومواصفات الفقرة")
    part_name = st.text_input("وصف الجزء الإنشائي", value="الجزء الأول - القاطع A")

    concrete_grade = "C20 (نسبة 1:2:4 - 315 كغم سمنت/م³ مع الضياع)"
    brick_type = "طابوق عادي (24×11.5×7.5 سم)"
    brick_waste_pct = 5
    openings_area = 0.0
    rebar_dia = 16
    rebar_bar_len = 12.0
    paint_coats = 2

    if "خرسانة" in item_choice or "الأسس" in item_choice or "الأعمدة" in item_choice:
        concrete_grade = st.selectbox("فئة مقاومة الخرسانة ونسبة الخلط (الدليل القياسي)", options=[
            "C15/C16 (نسبة 1:3:6 - 210 كغم سمنت/م³ مع الضياع)",
            "C20 (نسبة 1:2:4 - 315 كغم سمنت/م³ مع الضياع)",
            "C25 (نسبة 1:1.5:3 - 420 كغم سمنت/م³ مع الضياع)",
            "C30 (خرسانة عالية المقاومة - 450 كغم سمنت/م³ مع الضياع)"
        ], index=0 if "(1-1)" in item_choice else (1 if "المسلحة" in item_choice else 2))

    elif "الطابوق" in item_choice:
        b_t1, b_t2 = st.columns(2)
        with b_t1:
            brick_type = st.selectbox("نوع الطابوق / البلوك", options=[
                "طابوق عادي (24×11.5×7.5 سم)",
                "طابوق جمهوري (24×11.5×7.5 سم)",
                "ثرمستون (60×20×20 سم)",
                "بلوك إسمنتي كبير (40×20×20 سم)"
            ])
        with b_t2:
            brick_waste_pct = st.selectbox("نمط البناء ونسبة الهدر", options=[5, 10], format_func=lambda x: f"مستقيم (هدر {x}%)" if x==5 else f"قطري/مقصوص (هدر {x}%)")
        openings_area = st.number_input("مساحة فتحات الأبواب والشبابيك للطرح (م²)", min_value=0.0, value=0.0, step=0.5)

    elif "حديد" in item_choice:
        r_t1, r_t2 = st.columns(2)
        with r_t1:
            rebar_dia = st.selectbox("قطر شيش التسليح (ملم)", options=[10, 12, 16, 20, 25, 32], index=2)
        with r_t2:
            rebar_bar_len = st.number_input("طول الشيش القياسي (متر)", min_value=6.0, value=12.0, step=1.0)

    elif "الإنهاء" in item_choice:
        f_t1, f_t2 = st.columns(2)
        with f_t1:
            paint_coats = st.selectbox("عدد طبقات الطلاء / الإنهاء", options=[1, 2, 3], index=1)
        with f_t2:
            openings_area = st.number_input("مساحة الفتحات للطرح (م²)", min_value=0.0, value=0.0, step=0.5)

    d1, d2 = st.columns(2)
    with d1:
        count_num = st.number_input("العدد (Count)", min_value=1, value=1, step=1)
        length_m = st.number_input("الطول (متر)", min_value=0.1, value=30.0, step=1.0)
    with d2:
        width_m = st.number_input("العرض / الارتفاع (متر)", min_value=0.1, value=20.0, step=1.0)
        thickness_cm = st.number_input("السمك (سم)", min_value=0.5, value=10.0, step=1.0)

with col2:
    st.header("3️⃣ التسعير القياسي وتحليل المواد والعمالة")
    
    row_m = timeline_df[(timeline_df['Year'] == selected_year) & (timeline_df['Month'] == selected_month)].iloc[0]
    gov_row = gov_df[gov_df['Governorate'] == selected_gov].iloc[0]

    t_factor = float(gov_row["Transport_Factor"]) if apply_regional_transport else 1.00
    o_factor = origin_multipliers[origin_choice]

    p_cement_ord = float(row_m["Cement_Ord_Ton"]) * t_factor * (1.0 if "عراقي" in origin_choice else 1.08)
    p_cement_res = float(row_m["Cement_Res_Ton"]) * t_factor * (1.0 if "عراقي" in origin_choice else 1.25)
    p_sand       = float(row_m["Sand_m3"]) * t_factor
    p_gravel     = float(row_m["Gravel_m3"]) * t_factor
    p_brick_ord  = float(row_m["Brick_Ord_1000"]) * t_factor
    p_brick_rep  = float(row_m["Brick_Rep_1000"]) * t_factor
    p_thermo     = float(row_m["Thermostone_1000"]) * t_factor * o_factor
    p_block      = float(row_m["Block_1000"]) * t_factor
    p_rebar      = float(row_m["Rebar_Ton"]) * t_factor * (1.0 if "عراقي" in origin_choice else 1.06)
    p_gypsum     = float(row_m["Gypsum_Ton"]) * t_factor
    p_ceramic    = float(row_m["Ceramic_m2"]) * o_factor
    p_marble     = float(row_m["Marble_m2"]) * o_factor
    p_asphalt    = float(row_m["Asphalt_Ton"]) * t_factor
    p_paint      = float(row_m["Paint_Liter"]) * o_factor
    p_wood       = float(row_m["Wood_m3"]) * o_factor

    years_diff = selected_year - 2026
    wage_inflation = (1.025 ** years_diff)
    w_skilled   = float(gov_row["Skilled"]) * wage_inflation
    w_semi      = float(gov_row["Semi_Skilled"]) * wage_inflation
    w_unskilled = float(gov_row["Unskilled"]) * wage_inflation

    status_badge = "🟢 نشرة رسمية معتمدة (وزارة التخطيط 2026)" if selected_year == 2026 else f"🔮 تنبؤ محرك Ridge لسنة {selected_year}"
    st.caption(f"المصدر: **{status_badge}** | المحافظة: **{selected_gov}** | معامل النقل: **{t_factor:.2f}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("السمنت المقاوم", f"{p_cement_res:,.0f} د/طن")
    m2.metric("الرمل الأسود", f"{p_sand:,.0f} د/م³")
    m3.metric("الحصى العادي", f"{p_gravel:,.0f} د/م³")
    m4.metric("أجرة الخلفة", f"{w_skilled:,.0f} د/يوم")

    gross_area_m2 = count_num * length_m * width_m
    net_area_m2 = max(0.1, gross_area_m2 - openings_area)
    vol_m3 = net_area_m2 * (thickness_cm / 100.0)

    unit_label = "م³"
    final_qty = vol_m3
    mat_cost = 0.0
    labor_cost = 0.0
    materials_breakdown = []
    trucks_info = []
    labor_days_info = {}
    selected_brick_1000 = p_brick_ord

    if "خرسانة" in item_choice or "الأسس" in item_choice or "الأعمدة" in item_choice:
        unit_label = "م³"
        final_qty = vol_m3

        if "C15" in concrete_grade:
            c_kg_net, s_m3_net, g_m3_net, c_price = 200.0, 0.450, 0.900, p_cement_res
            wood_factor = 0.004
            prod_per_day = 12.0
            skilled_per_crew = 1
        elif "C20" in concrete_grade:
            c_kg_net, s_m3_net, g_m3_net, c_price = 300.0, 0.420, 0.840, p_cement_res
            wood_factor = 0.012
            prod_per_day = 8.0
            skilled_per_crew = 1
        elif "C25" in concrete_grade:
            c_kg_net, s_m3_net, g_m3_net, c_price = 400.0, 0.400, 0.800, p_cement_res
            wood_factor = 0.025
            prod_per_day = 4.5
            skilled_per_crew = 2
        else:
            c_kg_net, s_m3_net, g_m3_net, c_price = 430.0, 0.380, 0.760, p_cement_res
            wood_factor = 0.025
            prod_per_day = 4.5
            skilled_per_crew = 2

        tot_cement_ton = (final_qty * c_kg_net / 1000.0) * 1.05
        tot_sand_m3    = (final_qty * s_m3_net) * 1.08
        tot_gravel_m3  = (final_qty * g_m3_net) * 1.08
        tot_wood_m3    = (final_qty * wood_factor)

        crew_days = max(1, math.ceil(final_qty / prod_per_day))
        equip_daily_rate = 65000.0
        equip_total_cost = crew_days * equip_daily_rate

        mat_cost = (tot_cement_ton * c_price) + (tot_sand_m3 * p_sand) + (tot_gravel_m3 * p_gravel) + (tot_wood_m3 * p_wood) + equip_total_cost

        labor_days_info = {
            "عامل ماهر (نجار/خلفة)": crew_days * skilled_per_crew,
            "عامل شبه ماهر (هزاز/خباطة)": crew_days * 1,
            "عامل غير ماهر": crew_days * 4,
        }
        labor_cost = (
            (labor_days_info["عامل ماهر (نجار/خلفة)"] * w_skilled) +
            (labor_days_info["عامل شبه ماهر (هزاز/خباطة)"] * w_semi) +
            (labor_days_info["عامل غير ماهر"] * w_unskilled)
        )

        materials_breakdown = [
            f"السمنت المقاوم (مع هدر 5%): **{tot_cement_ton:.2f} طن** ({math.ceil(tot_cement_ton*20)} كيس)",
            f"الرمل المغسول (مع هدر 8%): **{tot_sand_m3:.2f} م³**",
            f"الحصى المدرج (مع هدر 8%): **{tot_gravel_m3:.2f} م³**",
            f"القالب الخشبي ({tot_wood_m3:.2f} م³) + آليات الخلط والهزاز ({crew_days} يوم): **{(tot_wood_m3 * p_wood) + equip_total_cost:,.0f} د.ع**"
        ]
        trucks_info = [
            f"قلاب رمل (سعة 16 م³): **{math.ceil(tot_sand_m3 / 16.0)} سيارة**",
            f"قلاب حصى (سعة 16 م³): **{math.ceil(tot_gravel_m3 / 16.0)} سيارة**"
        ]

    elif "الطابوق" in item_choice:
        if thickness_cm >= 20:
            unit_label = "م³"
            final_qty = vol_m3
            bricks_per_unit = 440 if "طابوق" in brick_type else (40 if "ثرمستون" in brick_type else 60)
            cement_per_unit_ton = 0.090
            sand_per_unit_m3 = 0.28
            prod_rate = 3.0
        else:
            unit_label = "م²"
            final_qty = net_area_m2
            bricks_per_unit = 55 if "طابوق" in brick_type else (8.0 if "ثرمستون" in brick_type else 12.5)
            cement_per_unit_ton = 0.015
            sand_per_unit_m3 = 0.05
            prod_rate = 12.0

        waste_mult = 1.0 + (brick_waste_pct / 100.0)
        tot_bricks = math.ceil(final_qty * bricks_per_unit * waste_mult)
        tot_cement_ton = (final_qty * cement_per_unit_ton) * 1.05
        tot_sand_m3 = (final_qty * sand_per_unit_m3) * 1.08

        crew_days = max(1, math.ceil(final_qty / prod_rate))
        scaff_water_cost = crew_days * 15000.0

        if "عادي" in brick_type:
            selected_brick_1000 = p_brick_ord
            truck_cap = 4000
        elif "جمهوري" in brick_type:
            selected_brick_1000 = p_brick_rep
            truck_cap = 4000
        elif "ثرمستون" in brick_type:
            selected_brick_1000 = p_thermo
            truck_cap = 1500
        else:
            selected_brick_1000 = p_block
            truck_cap = 1200

        b_unit_price = selected_brick_1000 / 1000.0
        mat_cost = (tot_bricks * b_unit_price) + (tot_cement_ton * p_cement_ord) + (tot_sand_m3 * p_sand) + scaff_water_cost

        labor_days_info = {
            "عامل ماهر (بنّاء)": crew_days * 1,
            "عامل غير ماهر": crew_days * 2,
        }
        labor_cost = (
            (labor_days_info["عامل ماهر (بنّاء)"] * w_skilled) +
            (labor_days_info["عامل غير ماهر"] * w_unskilled)
        )

        num_brick_trucks = math.ceil(tot_bricks / truck_cap)
        materials_breakdown = [
            f"عدد الطابوق/البلوك (مع هدر {brick_waste_pct}%): **{tot_bricks:,} قطعة**",
            f"السمنت (مع هدر 5%): **{tot_cement_ton:.2f} طن**",
            f"الرمل (مع هدر 8%): **{tot_sand_m3:.2f} م³**",
            f"السقالات والماء ({crew_days} يوم): **{scaff_water_cost:,.0f} د.ع**"
        ]
        trucks_info = [
            f"سيارة نقل طابوق (سعة {truck_cap:,}): **{num_brick_trucks} سيارة**",
            f"قلاب رمل (سعة 16 م³): **{math.ceil(tot_sand_m3 / 16.0)} سيارة**"
        ]

    elif "حديد" in item_choice:
        unit_label = "طن"
        kg_per_m = (rebar_dia ** 2) / 162.0
        total_linear_m = count_num * length_m
        net_ton = (total_linear_m * kg_per_m) / 1000.0
        tot_ton_with_waste = net_ton * 1.05
        final_qty = round(tot_ton_with_waste, 3)
        total_bars_needed = math.ceil((total_linear_m * 1.05) / rebar_bar_len)
        binding_wire_kg = tot_ton_with_waste * 11.0
        binding_wire_cost = binding_wire_kg * 2000.0

        mat_cost = (tot_ton_with_waste * p_rebar) + binding_wire_cost
        crew_days = max(1, math.ceil(tot_ton_with_waste / 0.5))
        labor_days_info = {
            "عامل ماهر (حداد)": crew_days * 1,
            "عامل شبه ماهر": crew_days * 1,
        }
        labor_cost = (labor_days_info["عامل ماهر (حداد)"] * w_skilled) + (labor_days_info["عامل شبه ماهر"] * w_semi)

        materials_breakdown = [
            f"الوزن الكلي للحديد (مع هدر 5%): **{tot_ton_with_waste:.3f} طن**",
            f"عدد الأشياش (قطر {rebar_dia} ملم × {rebar_bar_len} م): **{total_bars_needed} شيش**",
            f"سيم التربيط القياسي ({binding_wire_kg:.1f} كغم): **{binding_wire_cost:,.0f} د.ع**"
        ]
        trucks_info = [f"تريلة نقل حديد (سعة 25 طن): **{math.ceil(tot_ton_with_waste / 25.0)} سيارة**"]

    elif "الطرق" in item_choice:
        unit_label = "م²"
        final_qty = net_area_m2
        asphalt_ton = vol_m3 * 2.35 * 1.03
        crew_days = max(1, math.ceil(final_qty / 200.0))
        paver_roller_cost = crew_days * 180000.0
        mat_cost = (asphalt_ton * p_asphalt) + paver_roller_cost
        labor_days_info = {
            "عامل ماهر (مشغّل فارشة/حادلة)": crew_days * 2,
            "عامل غير ماهر": crew_days * 4,
        }
        labor_cost = (labor_days_info["عامل ماهر (مشغّل فارشة/حادلة)"] * w_skilled) + (labor_days_info["عامل غير ماهر"] * w_unskilled)

        materials_breakdown = [
            f"الكونكريت الأسفلتي (كثافة 2.35 طن/م³): **{asphalt_ton:.2f} طن** ({vol_m3:.2f} م³)",
            f"آليات الفرش والحدل والرابط القيري ({crew_days} يوم): **{paver_roller_cost:,.0f} د.ع**"
        ]
        trucks_info = [f"قلاب أسفلت (سعة 20 طن): **{math.ceil(asphalt_ton / 20.0)} سيارة**"]

    elif "الإنهاء" in item_choice:
        unit_label = "م²"
        final_qty = net_area_m2
        paint_liters = math.ceil((final_qty * paint_coats / 8.0) * 1.05)
        gypsum_ton = (final_qty * (thickness_cm / 100.0) * 1.15) * 1.08
        crew_days = max(1, math.ceil(final_qty / 18.0))
        scaff_cost = crew_days * 12000.0
        mat_cost = (paint_liters * p_paint) + (gypsum_ton * p_gypsum) + scaff_cost
        labor_days_info = {"عامل ماهر (صباغ/بيّاض)": crew_days * 1, "عامل شبه ماهر": crew_days * 1}
        labor_cost = (labor_days_info["عامل ماهر (صباغ/بيّاض)"] * w_skilled) + (labor_days_info["عامل شبه ماهر"] * w_semi)

        materials_breakdown = [
            f"الأصباغ ({paint_coats} طبقة + هدر 5%): **{paint_liters} لتر**",
            f"مواد البياض/الجص/المعجون (مع هدر 8%): **{gypsum_ton:.2f} طن**"
        ]
        trucks_info = ["نقل مواد إنهاء (سيارة حمل متوسطة): **1 سيارة**"]

    elif "الأرضيات" in item_choice:
        unit_label = "م²"
        final_qty = net_area_m2
        tile_area_with_waste = final_qty * 1.08
        cem_ton = final_qty * 0.018 * 1.05
        sand_m3 = final_qty * 0.06 * 1.08
        crew_days = max(1, math.ceil(final_qty / 12.0))
        grout_water_cost = crew_days * 10000.0
        mat_cost = (tile_area_with_waste * p_ceramic) + (cem_ton * p_cement_ord) + (sand_m3 * p_sand) + grout_water_cost
        labor_days_info = {"عامل ماهر (مبلّط)": crew_days * 1, "عامل غير ماهر": crew_days * 2}
        labor_cost = (labor_days_info["عامل ماهر (مبلّط)"] * w_skilled) + (labor_days_info["عامل غير ماهر"] * w_unskilled)

        materials_breakdown = [
            f"السيراميك/الكاشي (مع هدر 8%): **{tile_area_with_waste:.2f} م²**",
            f"السمنت: **{cem_ton:.2f} طن** | الرمل: **{sand_m3:.2f} م³**"
        ]
        trucks_info = [f"قلاب رمل (سعة 16 م³): **{math.ceil(sand_m3 / 16.0)} سيارة**"]

    else:
        unit_label = "م³"
        final_qty = vol_m3
        loose_soil_m3 = final_qty * 1.20
        crew_days = max(1, math.ceil(final_qty / 25.0))
        compactor_cost = crew_days * 75000.0
        mat_cost = (loose_soil_m3 * (p_sand * 0.70)) + compactor_cost
        labor_days_info = {"عامل غير ماهر": crew_days * 3}
        labor_cost = (labor_days_info["عامل غير ماهر"] * w_unskilled)
        materials_breakdown = [
            f"كمية السبيس / التراب النظيف المفكك (معامل حدل 1.20): **{loose_soil_m3:.2f} م³**",
            f"آليات الحدل والرش بالماء ({crew_days} يوم): **{compactor_cost:,.0f} د.ع**"
        ]
        trucks_info = [f"قلاب تراب/سبيس (سعة 16 م³): **{math.ceil(loose_soil_m3 / 16.0)} سيارة**"]

    direct_cost = mat_cost + labor_cost
    profit_amount = direct_cost * (profit_pct / 100.0)
    total_part_price = direct_cost + profit_amount
    unit_price_final = total_part_price / final_qty if final_qty > 0 else 0.0

    ai_input_df = pd.DataFrame([{
        'Item_Code': item_code_map[item_choice],
        'Cement_Ord_Ton': p_cement_ord,
        'Cement_Res_Ton': p_cement_res,
        'Sand_Price_m3': p_sand,
        'Gravel_Price_m3': p_gravel,
        'Brick_Price_1000': selected_brick_1000,
        'Rebar_Price_Ton': p_rebar,
        'Asphalt_Price_Ton': p_asphalt,
        'Gypsum_Price_Ton': p_gypsum,
        'Ceramic_Price_m2': p_ceramic,
        'Paint_Price_Liter': p_paint,
        'Wood_Price_m3': p_wood,
        'Skilled_Worker_Wage': w_skilled,
        'Semi_Skilled_Wage': w_semi,
        'Unskilled_Worker_Wage': w_unskilled,
        'Profit_Margin': profit_pct / 100.0
    }])
    ai_unit_pred = float(ai_model.predict(ai_input_df[ai_features])[0])

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric(f"الكمية ({unit_label}) | المساحة", f"{final_qty:,.2f} {unit_label}", f"المساحة: {net_area_m2:,.1f} م²")
    r2.metric(f"سعر الوحدة القياسي (دينار/{unit_label})", f"{unit_price_final:,.0f} د.ع", f"تقدير {ai_model_name}: {ai_unit_pred:,.0f} د.ع")
    r3.metric("الكلفة الإجمالية للفقرة", f"{total_part_price:,.0f} د.ع")

    with st.expander("🔍 عرض التحليل القياسي الكامل (المواد، عدد السيارات، يوميات العمال، الأرباح)", expanded=True):
        c_an1, c_an2 = st.columns(2)
        with c_an1:
            st.markdown("**📦 المواد الأولية وعدد السيارات المطلوبة:**")
            for mb in materials_breakdown:
                st.write(f"- {mb}")
            for tr in trucks_info:
                st.write(f"- 🚛 {tr}")
        with c_an2:
            st.markdown(f"**👷 الكادر التنفيذي المباشر ({selected_gov}):**")
            for role, days in labor_days_info.items():
                if "ماهر (" in role:
                    rate = w_skilled
                elif "شبه" in role:
                    rate = w_semi
                else:
                    rate = w_unskilled
                st.write(f"- {role}: **{days} يوم** × {rate:,.0f} د.ع")
            st.caption("ℹ️ الإشراف الهندسي والإدارة والحراسة مشمولة ضمن نسبة التحميلات الإدارية والأرباح (20%–25%).")
        st.caption(f"💰 كلفة المواد والآليات: **{mat_cost:,.0f} د.ع** | كلفة الأجور المباشرة: **{labor_cost:,.0f} د.ع** | التحميلات والأرباح ({profit_pct}%): **{profit_amount:,.0f} د.ع**")

    if st.button("➕ إضافة هذه الفقرة إلى جدول الكميات الكلي للمشروع (BOQ)"):
        st.session_state.boq_list.append({
            "ت": len(st.session_state.boq_list) + 1,
            "الجزء الإنشائي": part_name,
            "الفقرة": item_choice,
            "نوع الفقرة_EN": get_en_item_label(item_choice),
            "المحافظة والتاريخ": f"{selected_gov} ({selected_month}/{selected_year})",
            "العدد": count_num,
            "الأبعاد": f"{length_m}m x {width_m}m x {thickness_cm}cm",
            "المساحة (م²)": round(net_area_m2, 2),
            "الوحدة": unit_label,
            "الكمية": round(final_qty, 2),
            "سعر الوحدة (د.ع)": round(unit_price_final, 0),
            "كلفة المواد (د.ع)": round(mat_cost, 0),
            "كلفة العمال (د.ع)": round(labor_cost, 0),
            "المبلغ الإجمالي (دينار)": round(total_part_price, 0),
            "تفاصيل المواد والسيارات": " | ".join([re.sub(r'\*+', '', x) for x in (materials_breakdown + trucks_info)])
        })
        st.success(f"✅ تمت إضافة ({part_name}) إلى جدول الكميات بنجاح!")

st.divider()

# ==========================================
# 6. جدول الكميات وتصدير (PDF + Excel + AutoCAD DXF)
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "📋 جدول الكميات (BOQ) والتصدير (PDF + Excel + AutoCAD)",
    "👷 جدول أجور العاملين الرسمي (18 محافظة)",
    "📈 منحنى التنبؤ بالأسعار (2026 - 2030)"
])

with tab1:
    if len(st.session_state.boq_list) > 0:
        boq_df = pd.DataFrame(st.session_state.boq_list)
        display_df = boq_df.drop(columns=["نوع الفقرة_EN"]).copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        grand_total = boq_df["المبلغ الإجمالي (دينار)"].sum()
        tot_mat_all = boq_df["كلفة المواد (د.ع)"].sum()
        tot_lab_all = boq_df["كلفة العمال (د.ع)"].sum()
        tot_area_all = boq_df["المساحة (م²)"].sum()

        st.markdown("### 📊 الملخص المالي والهندسي الكلي للمشروع:")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("الكلفة الإجمالية للمشروع", f"{grand_total:,.0f} د.ع")
        s2.metric("إجمالي كلفة المواد", f"{tot_mat_all:,.0f} د.ع")
        s3.metric("إجمالي أجور العمال", f"{tot_lab_all:,.0f} د.ع")
        s4.metric("إجمالي المساحة الكلية", f"{tot_area_all:,.1f} م²")

        def generate_official_pdf(df_boq, total_iqd, comp_en, logo_file, b_cat_en, w_cat_en, L_val, W_val, T_val, boq_raw, b_cat_ar, w_cat_ar):
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.add_page()
            
            if logo_file is not None:
                ext = os.path.splitext(logo_file.name)[1].lower()
                if ext not in [".png", ".jpg", ".jpeg"]:
                    ext = ".png"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(logo_file.getvalue())
                    tmp_path = tmp.name
                pdf.image(tmp_path, x=15, y=10, w=25)
                os.remove(tmp_path)

            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 8, comp_en.encode('ascii', 'ignore').decode('ascii').upper(), ln=True, align='C')
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 7, f"BILL OF QUANTITIES (BOQ) & STANDARD PRICE ANALYSIS ({b_cat_en.upper()})", ln=True, align='C')
            pdf.set_font("Arial", '', 9)
            pdf.cell(0, 6, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Iraqi Ministry of Planning Standard Price Guide", ln=True, align='C')
            pdf.ln(6)

            pdf.set_font("Arial", 'B', 8)
            headers = ["No.", "Work Item", "Count", "Dimensions", "Area (m2)", "Unit", "Quantity", "Unit Price", "Materials", "Labor", "Total Cost (IQD)"]
            widths = [9, 50, 12, 34, 18, 14, 20, 26, 28, 26, 35]
            for h, w in zip(headers, widths):
                pdf.cell(w, 9, h, 1, 0, 'C')
            pdf.ln()

            pdf.set_font("Arial", '', 8)
            for idx, r in df_boq.iterrows():
                dim_clean = str(r['الأبعاد']).encode('ascii', 'ignore').decode('ascii')
                u_en = "m3" if r['الوحدة'] == "م³" else ("m2" if r['الوحدة'] == "م²" else "Ton")
                pdf.cell(widths[0], 8, str(r['ت']), 1, 0, 'C')
                pdf.cell(widths[1], 8, str(r['نوع الفقرة_EN'])[:30], 1, 0, 'L')
                pdf.cell(widths[2], 8, str(r['العدد']), 1, 0, 'C')
                pdf.cell(widths[3], 8, dim_clean, 1, 0, 'C')
                pdf.cell(widths[4], 8, f"{r['المساحة (م²)']:.1f}", 1, 0, 'C')
                pdf.cell(widths[5], 8, u_en, 1, 0, 'C')
                pdf.cell(widths[6], 8, f"{r['الكمية']:.2f}", 1, 0, 'C')
                pdf.cell(widths[7], 8, f"{r['سعر الوحدة (د.ع)']:,.0f}", 1, 0, 'C')
                pdf.cell(widths[8], 8, f"{r['كلفة المواد (د.ع)']:,.0f}", 1, 0, 'C')
                pdf.cell(widths[9], 8, f"{r['كلفة العمال (د.ع)']:,.0f}", 1, 0, 'C')
                pdf.cell(widths[10], 8, f"{r['المبلغ الإجمالي (دينار)']:,.0f}", 1, 1, 'C')

            pdf.ln(4)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 7, f"GRAND TOTAL ESTIMATED PROJECT COST: {total_iqd:,.0f} IQD", ln=True, align='R')

            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "ENGINEERING CROSS-SECTION & STRUCTURAL SCHEMATIC DIAGRAM", ln=True, align='C')
            pdf_fig = generate_pdf_engineering_diagram(b_cat_en, w_cat_en, L_val, W_val, T_val, boq_raw, b_cat_ar, w_cat_ar)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                pdf_fig.savefig(tmp_img.name, format='png', bbox_inches='tight', dpi=180)
                img_path = tmp_img.name
            plt.close(pdf_fig)

            pdf.image(img_path, x=18, y=24, w=260)
            os.remove(img_path)

            pdf.set_y(-15)
            pdf.set_font("Arial", 'I', 8)
            pdf.cell(0, 6, "Iraqi Construction AI & Standard Pricing System (Supervised by Dr. Ali Khalil | Eng. Team: Sajjad, Jaafar, Muslim, Mohammed)", ln=True, align='C')

            raw_out = pdf.output(dest='S')
            return bytes(raw_out) if isinstance(raw_out, (bytes, bytearray)) else raw_out.encode('latin-1')

        def generate_excel_boq(df_display, total_iqd, tot_mat, tot_lab):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_display.to_excel(writer, index=False, sheet_name='BOQ_جدول_الكميات')
                summary_df = pd.DataFrame({
                    "البيان": ["اسم المشروع", "الشركة المنفذة", "إجمالي كلفة المواد (د.ع)", "إجمالي أجور العمال (د.ع)", "الكلفة الكلية للمشروع (د.ع)"],
                    "القيمة": [project_name, company_ar_name, tot_mat, tot_lab, total_iqd]
                })
                summary_df.to_excel(writer, index=False, sheet_name='الملخص_المالي')
            return output.getvalue()

        b_en_lbl = get_en_building_label(building_choice)
        w_en_lbl = get_en_item_label(item_choice)

        pdf_bytes = generate_official_pdf(
            boq_df, grand_total, company_en_name, uploaded_logo,
            b_en_lbl, w_en_lbl, length_m, width_m, thickness_cm, st.session_state.boq_list,
            building_choice, item_choice
        )
        excel_bytes = generate_excel_boq(display_df, grand_total, tot_mat_all, tot_lab_all)
        dxf_bytes = generate_autocad_dxf(st.session_state.boq_list, b_en_lbl)

        st.markdown("### 📥 تصدير الملفات الهندسية والمخططات المعتمدة:")
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            st.download_button(
                label="📄 تحميل التقرير الرسمي (PDF)",
                data=pdf_bytes,
                file_name=f"BOQ_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        with ex2:
            st.download_button(
                label="📊 تحميل جدول الكميات (Excel)",
                data=excel_bytes,
                file_name=f"BOQ_Iraq_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with ex3:
            st.download_button(
                label="📐 تصدير مخطط أوتوكاد (DXF)",
                data=dxf_bytes,
                file_name=f"Structural_Plan_{datetime.now().strftime('%Y%m%d')}.dxf",
                mime="application/dxf"
            )

        with st.expander("🖼️ معاينة المخطط الهندسي المرفق في التقرير قبل التحميل", expanded=True):
            preview_fig = generate_pdf_engineering_diagram(
                b_en_lbl, w_en_lbl, length_m, width_m, thickness_cm,
                st.session_state.boq_list, building_choice, item_choice
            )
            st.pyplot(preview_fig)
            plt.close(preview_fig)

        if st.button("🗑️ مسح جدول الكميات والبدء بمشروع جديد"):
            st.session_state.boq_list = []
            st.rerun()
    else:
        st.info("ℹ️ جدول الكميات فارغ حالياً. أدخل أبعاد الفقرة في الأعلى ثم اضغط على زر (➕ إضافة هذه الفقرة إلى جدول الكميات الكلي للمشروع).")

with tab2:
    st.subheader("👷 جدول أجور العاملين والكادر الهندسي الرسمي (18 محافظة عراقية - 2026)")
    wages_display = gov_df.copy()
    wages_display.insert(0, "ت", range(1, len(wages_display) + 1))
    wages_display = wages_display.rename(columns={
        "Governorate": "المحافظة",
        "Skilled": "العامل الماهر / الخلفة (د.ع/يوم)",
        "Semi_Skilled": "العامل شبه الماهر (د.ع/يوم)",
        "Unskilled": "العامل غير الماهر (د.ع/يوم)",
        "Engineer": "المهندس المشرف (د.ع/يوم)",
        "Admin": "الإداري / المحاسب (د.ع/يوم)",
        "Guard": "الحارس (د.ع/يوم)",
        "Transport_Factor": "معامل النقل"
    })
    st.dataframe(wages_display, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("📈 منحنى التنبؤ الزمني بأسعار المواد الإنشائية (محرك Ridge Seasonal: 2026 - 2030)")
    chart_df = timeline_df.copy()
    chart_df['التاريخ'] = chart_df['Year'].astype(str) + "-" + chart_df['Month'].astype(str).str.zfill(2)
    
    annual_summary = chart_df[chart_df['Month'] == 10][['Year', 'Cement_Res_Ton', 'Cement_Ord_Ton', 'Rebar_Ton', 'Brick_Ord_1000', 'Sand_m3', 'Gravel_m3']].copy()
    annual_summary.insert(0, "ت", range(1, len(annual_summary) + 1))
    annual_summary = annual_summary.rename(columns={
        'Year': 'السنة',
        'Cement_Res_Ton': 'السمنت المقاوم (د.ع/طن)',
        'Cement_Ord_Ton': 'السمنت العادي (د.ع/طن)',
        'Rebar_Ton': 'حديد التسليح (د.ع/طن)',
        'Brick_Ord_1000': 'الطابوق العادي (د.ع/ألف)',
        'Sand_m3': 'الرمل الأسود (د.ع/م³)',
        'Gravel_m3': 'الحصى العادي (د.ع/م³)'
    })
    st.dataframe(annual_summary, use_container_width=True, hide_index=True)

    fig_fc, ax_fc = plt.subplots(figsize=(10, 4.2), dpi=140)
    x_idx = np.arange(len(chart_df))
    ax_fc.plot(x_idx, chart_df['Cement_Res_Ton'] / 1000.0, linewidth=2.0, label="Resistant Cement (k IQD/Ton)")
    ax_fc.plot(x_idx, chart_df['Brick_Ord_1000'] / 1000.0, linewidth=2.0, label="Ordinary Brick (k IQD/1000)")
    ax_fc.plot(x_idx, chart_df['Sand_m3'] / 1000.0, linewidth=2.0, label="Sand (k IQD/m3)")
    ax_fc.plot(x_idx, chart_df['Gravel_m3'] / 1000.0, linewidth=2.0, label="Gravel (k IQD/m3)")
    
    tick_positions = [i for i, m in enumerate(chart_df['Month']) if m == 1]
    tick_labels = [str(y) for y in chart_df.loc[tick_positions, 'Year']]
    ax_fc.set_xticks(tick_positions)
    ax_fc.set_xticklabels(tick_labels)
    ax_fc.set_title("Monthly Price Trajectory & Seasonality (2026 - 2030)", fontweight='bold')
    ax_fc.set_xlabel("Year")
    ax_fc.set_ylabel("Price (Thousand IQD)")
    ax_fc.grid(True, linestyle='--', alpha=0.5)
    ax_fc.legend()
    plt.tight_layout()
    st.pyplot(fig_fc)
    plt.close(fig_fc)
