import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ezdxf
from fpdf import FPDF
from datetime import datetime

# إعدادات الصفحة الاحترافية
st.set_page_config(page_title="نظام الذكاء الاصطناعي للتسعير الهندسي - العراق", page_icon="🏗️", layout="wide")

# تنسيق الواجهة لدعم اللغة العربية ومظهر تجاري وهندسي فخم
st.markdown("""
<style>
    .main { direction: rtl; text-align: right; }
    h1, h2, h3, p, div, label { text-align: right !important; font-family: 'Segoe UI', Tahoma, sans-serif; }
    .stButton>button { width: 100%; background-color: #0F172A; color: white; font-weight: bold; border-radius: 8px; padding: 12px; border: 1px solid #38BDF8; }
    .stButton>button:hover { background-color: #1E3A8A; color: #38BDF8; }
    .team-banner {
        background: linear-gradient(90deg, #0F172A 0%, #1E3A8A 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        border-right: 6px solid #38BDF8;
        margin-bottom: 15px;
    }
    .team-banner h4 { color: #38BDF8 !important; margin: 0 0 8px 0; font-size: 18px; }
    .team-banner p { color: #F8FAFC !important; margin: 4px 0; font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. نظام الحماية ومفتاح التفعيل التجاري
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
    
    st.warning("⚠️ هذه النسخة محمية بحقوق الملكية الفكرية. يرجى إدخال مفتاح التفعيل (Serial Key) لتشغيل محركات الذكاء الاصطناعي.")
    col_k1, col_k2 = st.columns([2, 1])
    with col_k1:
        user_key = st.text_input("أدخل مفتاح التفعيل الخاص بنسختك:", type="password", placeholder="مثال: IRAQ-AI-2026")
    with col_k2:
        st.write("")
        st.write("")
        if st.button("🔓 تفعيل البرنامج الآن"):
            if user_key.strip() in VALID_KEYS:
                st.session_state.is_activated = True
                st.success("✅ تم تفعيل النسخة بنجاح! جاري فتح النظام...")
                st.rerun()
            else:
                st.error("❌ مفتاح التفعيل غير صحيح! يرجى التواصل مع الفريق الهندسي للحصول على ترخيص.")
    st.stop()

# ==========================================
# 2. تحميل نماذج الذكاء الاصطناعي والبيانات
# ==========================================
@st.cache_resource
def load_system_assets():
    ai_model = joblib.load('concrete_ai_model.pkl')
    timeline_df = pd.read_csv('iraq_forecast_2026_2030.csv')
    gov_df = pd.read_csv('iraq_governorates_factors.csv', index_col=0)
    return ai_model, timeline_df, gov_df

model, timeline_df, gov_df = load_system_assets()

if 'boq_list' not in st.session_state:
    st.session_state.boq_list = []

# ==========================================
# دوال مساعدة (تنسيق المواد + رسم PDF + توليد ملف AutoCAD DXF)
# ==========================================
def format_material_val(val, decimals=2):
    if val is None or pd.isna(val) or float(val) <= 0:
        return "-"
    return f"{float(val):.{decimals}f}"

def get_en_building_label(ar_choice):
    mapping = {
        "بناية متعددة الطوابق (Building)": "Multi-Story Building",
        "بيوت سكنية (Residential Houses)": "Residential House",
        "طرق وجسور (Roads & Highways)": "Road & Pavement Section",
        "مستشفى / مركز صحي (Hospital)": "Hospital & Medical Center"
    }
    return mapping.get(ar_choice, "Building Structure")

def get_en_item_label(ar_item):
    mapping = {
        "الأسس - صب خرسانة غير مسلحة (1-1)": "Foundation Blinding Concrete (1-1)",
        "الأسس المسلحة (Reinforced Foundations)": "Reinforced Concrete Footings",
        "أعمال الدفن الترابي والحفريات (Backfill)": "Earthworks & Sub-base Backfill",
        "أعمال الطرق والتبليط (Road Sub-base & Concrete)": "Road Concrete Slab & Sub-base"
    }
    return mapping.get(ar_item, "Concrete Work")

def generate_pdf_engineering_diagram(b_cat_en, w_cat_en, L, W, T_cm, boq_items):
    fig, ax = plt.subplots(figsize=(9.5, 4.3), dpi=150)
    ax.set_facecolor('#F8FAFC')
    
    # إذا كان الاختيار طرق (Roads) يرسم مقطع عرضي للطريق بطبقاته
    if "Road" in b_cat_en:
        ax.set_xlim(-1.5, 11.5)
        ax.set_ylim(-0.8, 5.0)

        # طبقة التربة الطبيعية
        ax.add_patch(patches.Rectangle((1.0, 0.3), 8.0, 0.55, linewidth=1.2, edgecolor='#78350F', facecolor='#FDE68A', hatch='///'))
        ax.annotate("Sub-grade Soil Layer", xy=(1.0, 0.57), xytext=(-1.3, 0.57),
                    arrowprops=dict(arrowstyle='->', color='#78350F', lw=1.2), fontsize=8, fontweight='bold', color='#78350F')

        # طبقة السبيس (Sub-base)
        ax.add_patch(patches.Rectangle((1.2, 0.85), 7.6, 0.65, linewidth=1.2, edgecolor='#475569', facecolor='#CBD5E1', hatch='..'))
        ax.annotate("Compacted Sub-base", xy=(8.8, 1.17), xytext=(9.2, 1.17),
                    arrowprops=dict(arrowstyle='->', color='#334155', lw=1.2), fontsize=8, fontweight='bold', color='#334155')

        # طبقة الصب الخرساني للطريق
        ax.add_patch(patches.Rectangle((1.5, 1.5), 7.0, 0.9, linewidth=2, edgecolor='#0F172A', facecolor='#64748B'))
        ax.text(5.0, 1.95, f"ROAD PAVEMENT SLAB | L={L:.1f}m x W={W:.1f}m x T={T_cm:.0f}cm", ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')

        # خط منتصف الطريق
        ax.plot([2.0, 8.0], [2.45, 2.45], linestyle='--', color='#FACC15', linewidth=2)

        ax.annotate('', xy=(1.5, 2.9), xytext=(8.5, 2.9), arrowprops=dict(arrowstyle='<->', color='#DC2626', lw=1.5))
        ax.text(5.0, 3.15, f"Road Width (W) = {W:.2f} m  |  Segment Length (L) = {L:.2f} m", ha='center', va='center', fontsize=9, fontweight='bold', color='#DC2626')

        ax.set_title(f"ENGINEERING ROAD CROSS-SECTION & DIMENSIONS ({w_cat_en.upper()})", fontsize=11, fontweight='bold', color='#0F172A', pad=10)

    # إذا كان الاختيار بناية أو بيوت أو مستشفى
    else:
        ax.set_xlim(-2.5, 12.5)
        total_layers = max(1, len(boq_items))
        max_y = max(5.5, 1.5 + total_layers * 0.85 + 1.8)
        ax.set_ylim(-0.4, max_y)

        # 1. طبقة الحفريات والدفن بالأسفل
        ax.add_patch(patches.Rectangle((1.5, 0.2), 7.0, 0.55, linewidth=1.2, edgecolor='#78350F', facecolor='#FDE68A', hatch='///'))
        ax.annotate("Excavation & Backfill Base", xy=(1.5, 0.47), xytext=(-2.3, 0.47),
                    arrowprops=dict(arrowstyle='->', color='#78350F', lw=1.2), fontsize=8, fontweight='bold', color='#78350F')

        # 2. رسم الفقرات المضافة (أو الفقرة الحالية إذا الجدول فيه عنصر واحد)
        colors = ['#CBD5E1', '#94A3B8', '#64748B', '#BAE6FD', '#7DD3FC']
        current_y = 0.75
        items_to_draw = boq_items if len(boq_items) > 0 else [{
            'ت': 1, 'الأبعاد (L x W x T)': f"{L}m x {W}m x {T_cm}cm",
            'الحجم (م³)': round(L * W * (T_cm / 100.0), 2), 'نوع الفقرة_EN': w_cat_en
        }]

        for idx, item in enumerate(items_to_draw):
            c = colors[idx % len(colors)]
            w_rect = max(3.6, 7.0 - idx * 0.35)
            x_start = 5.0 - (w_rect / 2.0)
            rect = patches.Rectangle((x_start, current_y), w_rect, 0.72, linewidth=1.5, edgecolor='#0F172A', facecolor=c, hatch='..')
            ax.add_patch(rect)
            
            dim_txt = str(item['الأبعاد (L x W x T)']).encode('ascii', 'ignore').decode('ascii')
            vol_txt = item['الحجم (م³)']
            ax.text(5.0, current_y + 0.36, f"Part-{item['ت']} | {dim_txt} | Vol: {vol_txt} m3", ha='center', va='center', fontsize=8, fontweight='bold', color='#0F172A')

            item_en_lbl = item.get('نوع الفقرة_EN', w_cat_en)
            if idx % 2 == 0:
                ax.annotate(f"Item {item['ت']}: {item_en_lbl}", xy=(x_start + w_rect, current_y + 0.36), xytext=(8.8, current_y + 0.36),
                            arrowprops=dict(arrowstyle='->', color='#1E3A8A', lw=1.2), fontsize=8, fontweight='bold', color='#1E3A8A')
            else:
                ax.annotate(f"Item {item['ت']}: {item_en_lbl}", xy=(x_start, current_y + 0.36), xytext=(-2.3, current_y + 0.36),
                            arrowprops=dict(arrowstyle='->', color='#0F172A', lw=1.2), fontsize=8, fontweight='bold', color='#0F172A')
            current_y += 0.78

        # 3. رسم شكل البناية أو البيت فوق الأسس
        ax.add_patch(patches.Rectangle((2.5, current_y), 0.4, 0.95, linewidth=1.2, edgecolor='#334155', facecolor='#475569'))
        ax.add_patch(patches.Rectangle((7.1, current_y), 0.4, 0.95, linewidth=1.2, edgecolor='#334155', facecolor='#475569'))
        
        if "House" in b_cat_en:
            # سقف مائل للبيت السكني
            roof_pts = [[2.0, current_y + 0.95], [5.0, current_y + 1.65], [8.0, current_y + 0.95]]
            ax.add_patch(patches.Polygon(roof_pts, closed=True, linewidth=1.5, edgecolor='#0F172A', facecolor='#38BDF8', alpha=0.8))
            ax.text(5.0, current_y + 1.2, "RESIDENTIAL HOUSE", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#0F172A')
        else:
            # سقف مستوي للبناية / المستشفى
            ax.add_patch(patches.Rectangle((2.1, current_y + 0.95), 5.8, 0.35, linewidth=1.5, edgecolor='#0F172A', facecolor='#38BDF8', alpha=0.8))
            ax.text(5.0, current_y + 1.12, f"{b_cat_en.upper()} SUPERSTRUCTURE", ha='center', va='center', fontsize=8, fontweight='bold', color='#0F172A')

        ax.set_title(f"STRUCTURAL SECTION & CALLOUTS - {b_cat_en.upper()}", fontsize=11, fontweight='bold', color='#0F172A', pad=10)

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
        dims = str(item['الأبعاد (L x W x T)']).replace('m', '').replace('cm', '').split('x')
        L_val = float(dims[0].strip())
        W_val = float(dims[1].strip())
        T_val = float(dims[2].strip())

        pts = [
            (offset_x, 0),
            (offset_x + L_val, 0),
            (offset_x + L_val, W_val),
            (offset_x, W_val),
            (offset_x, 0)
        ]
        msp.add_lwpolyline(pts, dxfattribs={'layer': 'STRUCTURE_PLAN'})

        msp.add_text(
            f"Part-{item['ت']} ({item.get('نوع الفقرة_EN', 'Concrete')}) - {b_cat_en}",
            dxfattribs={'layer': 'CALLOUTS', 'height': 0.45}
        ).set_placement((offset_x, W_val + 1.2))

        msp.add_text(
            f"L={L_val}m x W={W_val}m x T={T_val}cm | Vol={item['الحجم (م³)']} m3",
            dxfattribs={'layer': 'DIMENSIONS_TEXT', 'height': 0.35}
        ).set_placement((offset_x, W_val + 0.5))

        offset_x += L_val + 5.0

    out_stream = io.StringIO()
    doc.write(out_stream)
    return out_stream.getvalue().encode('utf-8')

# ==========================================
# 3. القائمة الجانبية لتخصيص هوية الشركة (White-Label)
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

# ترويسة البرنامج مع عرض شعار الشركة المرفوع
head_col1, head_col2 = st.columns([4, 1])
with head_col1:
    st.title(f"🏗️ {company_ar_name}")
    st.subheader("النظام الذكي للتنبؤ بأسعار الفقرات الإنشائية وإعداد جداول الكميات (2026 - 2030)")
with head_col2:
    if uploaded_logo is not None:
        st.image(uploaded_logo, width=130)

st.info("📌 الفقرة المعتمدة: تسلسل (1-1) صب خرسانة غير مسلحة تحت الأسس | محرك التنبؤ الزمني: Ridge Seasonal AI | محرك التسعير: XGBoost (دقة 97.83%)")
st.divider()

# ==========================================
# 4. مدخلات المشروع والتنبؤ الذكي (واجهة نظيفة بدون رسم)
# ==========================================
col1, col2 = st.columns([1, 1.35])

with col1:
    st.header("1️⃣ تحديد نوع المنشأ والموقع والزمن")
    project_name = st.text_input("اسم المشروع", value="مشروع إنشاء مجمع إداري وسكني متكامل")
    
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        building_choice = st.selectbox("نوع المنشأ الهندسي", options=[
            "بناية متعددة الطوابق (Building)",
            "بيوت سكنية (Residential Houses)",
            "طرق وجسور (Roads & Highways)",
            "مستشفى / مركز صحي (Hospital)"
        ], index=0)
    with b_col2:
        item_choice = st.selectbox("اختر الفقرة الإنشائية", options=[
            "الأسس - صب خرسانة غير مسلحة (1-1)",
            "الأسس المسلحة (Reinforced Foundations)",
            "أعمال الدفن الترابي والحفريات (Backfill)",
            "أعمال الطرق والتبليط (Road Sub-base & Concrete)"
        ], index=0)

    c_loc1, c_loc2, c_loc3 = st.columns(3)
    with c_loc1:
        selected_gov = st.selectbox("المحافظة (18 محافظة)", options=gov_df.index.tolist(), index=1)
    with c_loc2:
        selected_year = st.selectbox("السنة التنبؤية", options=[2026, 2027, 2028, 2029, 2030], index=0)
    with c_loc3:
        selected_month = st.selectbox("الشهر", options=list(range(1, 13)), index=10)

    profit_pct = st.slider("نسبة التحميلات الإدارية والأرباح حسب الدليل (%)", min_value=20, max_value=25, value=22)

    st.divider()
    st.header("2️⃣ إدخال أبعاد الذرعة الهندسية")
    part_name = st.text_input("وصف الجزء الإنشائي", value="الجزء الأول - أسس القاطع A")
    
    d1, d2 = st.columns(2)
    with d1:
        count_num = st.number_input("العدد (Count)", min_value=1, value=1, step=1)
        length_m = st.number_input("الطول (متر)", min_value=0.1, value=30.0, step=1.0)
    with d2:
        width_m = st.number_input("العرض (متر)", min_value=0.1, value=20.0, step=1.0)
        thickness_cm = st.number_input("السمك / الارتفاع (سم)", min_value=5.0, value=10.0, step=1.0)

with col2:
    st.header("3️⃣ التنبؤ التلقائي بأسعار السوق والتسعير")
    
    row_ts = timeline_df[(timeline_df['Year'] == selected_year) & (timeline_df['Month'] == selected_month)].iloc[0]
    gov_info = gov_df.loc[selected_gov]
    
    t_factor = gov_info['transport_factor']
    pred_cement = row_ts['Cement_Ton'] * t_factor
    pred_sand = row_ts['Sand_m3'] * t_factor
    pred_gravel = row_ts['Gravel_m3'] * t_factor
    pred_wood = row_ts['Wood_m2'] * t_factor
    pred_mix_trans = gov_info['mix_transport'] * (1 + (selected_year - 2026) * 0.03)
    pred_water = 20000 * (1 + (selected_year - 2026) * 0.03)
    pred_skilled = gov_info['skilled'] * (1 + (selected_year - 2026) * 0.025)
    pred_unskilled = gov_info['unskilled'] * (1 + (selected_year - 2026) * 0.025)

    status_badge = "🟢 بيانات رسمية (هيأة الإحصاء)" if (selected_year == 2026 and selected_month <= 10) else "🔮 تنبؤ الذكاء الاصطناعي للمستقبل"
    st.caption(f"حالة البيانات لتاريخ ({selected_month} / {selected_year}) في محافظة ({selected_gov}): **{status_badge}**")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("السمنت المقاوم", f"{pred_cement:,.0f} د/طن")
    p2.metric("الرمل المغسول", f"{pred_sand:,.0f} د/م³")
    p3.metric("الحصى المدرج", f"{pred_gravel:,.0f} د/م³")
    p4.metric("أجرة الخلفة", f"{pred_skilled:,.0f} د/يوم")

    ai_input = pd.DataFrame([{
        'Cement_Price_Ton': pred_cement,
        'Sand_Price_m3': pred_sand,
        'Gravel_Price_m3': pred_gravel,
        'Wood_Price_m2': pred_wood,
        'Concrete_Transport_Mix_m3': pred_mix_trans,
        'Water_Tanker_Price': pred_water,
        'Skilled_Worker_Wage': pred_skilled,
        'Unskilled_Worker_Wage': pred_unskilled,
        'Profit_Margin': profit_pct / 100.0
    }])

    base_price_10cm = float(model.predict(ai_input)[0])
    thickness_ratio = thickness_cm / 10.0
    adjusted_unit_price_m2 = base_price_10cm * (0.25 + 0.75 * thickness_ratio)

    total_area_m2 = count_num * length_m * width_m
    total_vol_m3 = total_area_m2 * (thickness_cm / 100.0)
    total_part_price = adjusted_unit_price_m2 * total_area_m2

    if "Backfill" in item_choice:
        req_cement_tons = 0.0
        req_sand_m3 = 0.0
        req_gravel_m3 = 0.0
    else:
        req_cement_tons = total_vol_m3 * 0.21
        req_sand_m3 = total_vol_m3 * 0.486
        req_gravel_m3 = total_vol_m3 * 0.972

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric("المساحة الكلية (الذرعة)", f"{total_area_m2:,.1f} م²", f"الحجم: {total_vol_m3:,.2f} م³")
    r2.metric(f"سعر الوحدة التنبؤي (سمك {thickness_cm:.0f} سم)", f"{adjusted_unit_price_m2:,.0f} دينار/م²")
    r3.metric("الكلفة الإجمالية للفقرة", f"{total_part_price:,.0f} دينار")

    st.write(f"📦 **المواد المطلوبة لهذا الجزء:** سمنت مقاوم: **{format_material_val(req_cement_tons, 2)} طن** | رمل: **{format_material_val(req_sand_m3, 1)} م³** | حصى: **{format_material_val(req_gravel_m3, 1)} م³**")

    if st.button("➕ إضافة هذا الجزء إلى جدول الكميات الكلي للمشروع (BOQ)"):
        st.session_state.boq_list.append({
            "ت": len(st.session_state.boq_list) + 1,
            "الجزء الإنشائي": part_name,
            "الفقرة": item_choice,
            "نوع الفقرة_EN": get_en_item_label(item_choice),
            "المحافظة والتاريخ": f"{selected_gov} ({selected_month}/{selected_year})",
            "العدد": count_num,
            "الأبعاد (L x W x T)": f"{length_m}m x {width_m}m x {thickness_cm}cm",
            "المساحة (م²)": round(total_area_m2, 2),
            "الحجم (م³)": round(total_vol_m3, 2),
            "سعر الوحدة (دينار/م²)": round(adjusted_unit_price_m2, 0),
            "المبلغ الإجمالي (دينار)": round(total_part_price, 0),
            "السمنت (طن)": round(req_cement_tons, 2),
            "الرمل (م³)": round(req_sand_m3, 1),
            "الحصى (م³)": round(req_gravel_m3, 1)
        })
        st.success(f"✅ تمت إضافة ({part_name}) إلى جدول الكميات!")

st.divider()

# ==========================================
# 5. جدول الكميات وتصدير PDF (الجدول + الرسم أسفله) + Excel + AutoCAD
# ==========================================
tab1, tab2 = st.tabs(["📋 جدول الكميات (BOQ) والتصدير الرسمي (PDF + Excel + AutoCAD)", "📈 منحنى التنبؤ بالأسعار (2026 - 2030)"])

with tab1:
    if len(st.session_state.boq_list) > 0:
        boq_df = pd.DataFrame(st.session_state.boq_list)
        
        display_df = boq_df.drop(columns=["نوع الفقرة_EN"]).copy()
        display_df["السمنت (طن)"] = display_df["السمنت (طن)"].apply(lambda x: format_material_val(x, 2))
        display_df["الرمل (م³)"] = display_df["الرمل (م³)"].apply(lambda x: format_material_val(x, 1))
        display_df["الحصى (م³)"] = display_df["الحصى (م³)"].apply(lambda x: format_material_val(x, 1))
        st.dataframe(display_df, use_container_width=True)

        grand_total = boq_df["المبلغ الإجمالي (دينار)"].sum()
        sum_area = boq_df["المساحة (م²)"].sum()
        sum_vol = boq_df["الحجم (م³)"].sum()
        sum_cement = boq_df["السمنت (طن)"].sum()
        sum_sand = boq_df["الرمل (م³)"].sum()
        sum_gravel = boq_df["الحصى (م³)"].sum()

        st.markdown("### 📊 الملخص التجميعي الكلي للمواد والكلف (لكل أجزاء المشروع المضافة):")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("الكلفة الإجمالية للمشروع", f"{grand_total:,.0f} د.ع")
        s2.metric("إجمالي حجم الخرسانة", f"{sum_vol:,.2f} م³")
        s3.metric("إجمالي السمنت المطلوب", f"{sum_cement:.2f} طن")
        s4.metric("إجمالي الرمل المطلوب", f"{sum_sand:.1f} م³")
        s5.metric("إجمالي الحصى المطلوب", f"{sum_gravel:.1f} م³")

        def generate_official_pdf(df_boq, total_iqd, comp_en, logo_file, b_cat_en, w_cat_en, L_val, W_val, T_val, boq_raw):
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.add_page()
            
            if logo_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(logo_file.getvalue())
                    tmp_path = tmp.name
                pdf.image(tmp_path, x=15, y=10, w=25)
                os.remove(tmp_path)

            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 8, comp_en.encode('ascii', 'ignore').decode('ascii').upper(), ln=True, align='C')
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 7, f"BILL OF QUANTITIES (BOQ) & AI COST ESTIMATION REPORT ({b_cat_en.upper()})", ln=True, align='C')
            pdf.set_font("Arial", '', 9)
            pdf.cell(0, 6, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Iraqi Standard Price Guide Item (1-1)", ln=True, align='C')
            pdf.ln(6)

            # 1. جدول الكميات والحسابات في الأعلى
            pdf.set_font("Arial", 'B', 8)
            headers = [
                "No.", "Section ID", "Count", "Dimensions (L x W x T)",
                "Area (m2)", "Vol (m3)", "Unit Price (IQD)", "Total Price (IQD)",
                "Cement (Ton)", "Sand (m3)", "Gravel (m3)"
            ]
            widths = [10, 24, 14, 38, 22, 20, 28, 34, 28, 28, 28]
            for h, w in zip(headers, widths):
                pdf.cell(w, 9, h, 1, 0, 'C')
            pdf.ln()

            pdf.set_font("Arial", '', 8)
            for idx, r in df_boq.iterrows():
                dim_clean = str(r['الأبعاد (L x W x T)']).encode('ascii', 'ignore').decode('ascii')
                cem_str = format_material_val(r['السمنت (طن)'], 2)
                snd_str = format_material_val(r['الرمل (م³)'], 1)
                grv_str = format_material_val(r['الحصى (م³)'], 1)

                pdf.cell(widths[0], 8, str(r['ت']), 1, 0, 'C')
                pdf.cell(widths[1], 8, f"Part-{r['ت']}", 1, 0, 'C')
                pdf.cell(widths[2], 8, str(r['العدد']), 1, 0, 'C')
                pdf.cell(widths[3], 8, dim_clean, 1, 0, 'C')
                pdf.cell(widths[4], 8, f"{r['المساحة (م²)']:.1f}", 1, 0, 'C')
                pdf.cell(widths[5], 8, f"{r['الحجم (م³)']:.2f}", 1, 0, 'C')
                pdf.cell(widths[6], 8, f"{r['سعر الوحدة (دينار/م²)']:,.0f}", 1, 0, 'C')
                pdf.cell(widths[7], 8, f"{r['المبلغ الإجمالي (دينار)']:,.0f}", 1, 0, 'C')
                pdf.cell(widths[8], 8, cem_str, 1, 0, 'C')
                pdf.cell(widths[9], 8, snd_str, 1, 0, 'C')
                pdf.cell(widths[10], 8, grv_str, 1, 1, 'C')

            pdf.ln(3)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 6, f"TOTAL MATERIALS: Cement = {sum_cement:.2f} Ton | Sand = {sum_sand:.1f} m3 | Gravel = {sum_gravel:.1f} m3", ln=True, align='L')
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 7, f"GRAND TOTAL ESTIMATED COST: {total_iqd:,.0f} IQD", ln=True, align='R')

            # 2. المخطط الهندسي المرسوم بالأبعاد والتأشيرات أسفل الجدول مباشرة
            pdf_fig = generate_pdf_engineering_diagram(b_cat_en, w_cat_en, L_val, W_val, T_val, boq_raw)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                pdf_fig.savefig(tmp_img.name, format='png', bbox_inches='tight', dpi=150)
                img_path = tmp_img.name
            plt.close(pdf_fig)

            if pdf.get_y() > 118:
                pdf.add_page()
            pdf.image(img_path, x=42, y=pdf.get_y() + 2, w=210)
            os.remove(img_path)

            pdf.set_y(-15)
            pdf.set_font("Arial", 'I', 8)
            pdf.cell(0, 6, "Powered by Iraqi Construction AI System (Supervised by Dr. Ali Khalil | Eng. Team: Sajjad, Jaafar, Muslim, Mohammed)", ln=True, align='C')
            return bytes(pdf.output())

        def generate_official_excel(df_boq):
            output = io.BytesIO()
            excel_df = df_boq.drop(columns=["نوع الفقرة_EN"]).copy()
            excel_df["السمنت (طن)"] = excel_df["السمنت (طن)"].apply(lambda x: format_material_val(x, 2))
            excel_df["الرمل (م³)"] = excel_df["الرمل (م³)"].apply(lambda x: format_material_val(x, 1))
            excel_df["الحصى (م³)"] = excel_df["الحصى (م³)"].apply(lambda x: format_material_val(x, 1))
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                excel_df.to_excel(writer, index=False, sheet_name='جدول الذرعة والكميات')
                summary_df = pd.DataFrame({
                    "البيان": ["اسم الشركة", "اسم المشروع", "الكلفة الإجمالية (دينار)", "إجمالي المساحة (م²)", "إجمالي الخرسانة (م³)", "إجمالي السمنت (طن)", "إجمالي الرمل (م³)", "إجمالي الحصى (م³)"],
                    "القيمة": [company_ar_name, project_name, grand_total, sum_area, sum_vol, sum_cement, sum_sand, sum_gravel]
                })
                summary_df.to_excel(writer, index=False, sheet_name='ملخص المواد والكلف')
            return output.getvalue()

        b_en = get_en_building_label(building_choice)
        w_en = get_en_item_label(item_choice)

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.download_button(
                label="📄 تحميل التقرير الرسمي PDF",
                data=generate_official_pdf(boq_df, grand_total, company_en_name, uploaded_logo, b_en, w_en, length_m, width_m, thickness_cm, st.session_state.boq_list),
                file_name="Official_BOQ_Report.pdf",
                mime="application/pdf"
            )
        with b2:
            st.download_button(
                label="📊 تحميل جدول الكميات Excel",
                data=generate_official_excel(boq_df),
                file_name="Official_BOQ_Project.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with b3:
            st.download_button(
                label="📐 تصدير رسم AutoCAD (.dxf)",
                data=generate_autocad_dxf(st.session_state.boq_list, b_en),
                file_name="Engineering_Drawing_AutoCAD.dxf",
                mime="application/dxf"
            )
        with b4:
            if st.button("🗑️ مسح الجدول"):
                st.session_state.boq_list = []
                st.rerun()
    else:
        st.warning("💡 جدول الكميات فارغ حالياً. أدخل أبعاد الفقرة أعلاه واضغط على زر (➕ إضافة هذا الجزء إلى جدول الكميات).")

with tab2:
    st.markdown("### 📊 مسار تطور أسعار المواد الإنشائية (البيانات الرسمية 2026 + تنبؤ الذكاء الاصطناعي 2027-2030)")
    chart_df = timeline_df.copy()
    chart_df['التاريخ'] = chart_df['Year'].astype(str) + "-" + chart_df['Month'].astype(str).str.zfill(2)
    chart_df = chart_df.set_index('التاريخ')
    st.line_chart(chart_df[['Cement_Ton', 'Sand_m3', 'Gravel_m3', 'Wood_m2']])
