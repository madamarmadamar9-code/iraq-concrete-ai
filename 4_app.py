import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os
import tempfile
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
VALID_KEYS =VALID_KEYS = ["2026"]

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
# 4. مدخلات المشروع والتنبؤ الذكي
# ==========================================
col1, col2 = st.columns([1, 1.35])

with col1:
    st.header("1️⃣ تحديد الموقع والزمن المستقبلي")
    project_name = st.text_input("اسم المشروع", value="مشروع إنشاء مجمع إداري وسكني متكامل")
    
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

    req_cement_tons = total_vol_m3 * 0.21
    req_sand_m3 = total_vol_m3 * 0.486
    req_gravel_m3 = total_vol_m3 * 0.972

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric("المساحة الكلية (الذرعة)", f"{total_area_m2:,.1f} م²", f"الحجم: {total_vol_m3:,.2f} م³")
    r2.metric(f"سعر الوحدة التنبؤي (سمك {thickness_cm:.0f} سم)", f"{adjusted_unit_price_m2:,.0f} دينار/م²")
    r3.metric("الكلفة الإجمالية للفقرة", f"{total_part_price:,.0f} دينار")

    st.write(f"📦 **المواد المطلوبة لهذا الجزء:** سمنت مقاوم: **{req_cement_tons:.2f} طن** | رمل: **{req_sand_m3:.1f} م³** | حصى: **{req_gravel_m3:.1f} م³**")

    if st.button("➕ إضافة هذا الجزء إلى جدول الكميات الكلي للمشروع (BOQ)"):
        st.session_state.boq_list.append({
            "ت": len(st.session_state.boq_list) + 1,
            "الجزء الإنشائي": part_name,
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
# 5. جدول الكميات وتصدير PDF + Excel + منحنى التنبؤ
# ==========================================
tab1, tab2 = st.tabs(["📋 جدول الكميات (BOQ) والتصدير الرسمي (PDF + Excel)", "📈 منحنى التنبؤ بالأسعار (2026 - 2030)"])

with tab1:
    if len(st.session_state.boq_list) > 0:
        boq_df = pd.DataFrame(st.session_state.boq_list)
        st.dataframe(boq_df, use_container_width=True)

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

        # دالة إنشاء PDF احترافي مع شعار الشركة المرفوع
        def generate_official_pdf(df_boq, total_iqd, comp_en, logo_file):
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.add_page()
            
            # إضافة شعار الشركة إذا قام المشتري برفعه
            if logo_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(logo_file.getvalue())
                    tmp_path = tmp.name
                pdf.image(tmp_path, x=15, y=10, w=25)
                os.remove(tmp_path)

            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 8, comp_en.encode('ascii', 'ignore').decode('ascii').upper(), ln=True, align='C')
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 7, "BILL OF QUANTITIES (BOQ) & AI COST ESTIMATION REPORT", ln=True, align='C')
            pdf.set_font("Arial", '', 9)
            pdf.cell(0, 6, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Iraqi Standard Price Guide Item (1-1)", ln=True, align='C')
            pdf.ln(8)

            pdf.set_font("Arial", 'B', 9)
            headers = ["No.", "Section ID", "Count", "Dimensions (L x W x T)", "Area (m2)", "Vol (m3)", "Unit Price (IQD)", "Total Price (IQD)", "Cement (Ton)"]
            widths = [12, 35, 18, 48, 28, 25, 38, 42, 30]
            for h, w in zip(headers, widths):
                pdf.cell(w, 10, h, 1, 0, 'C')
            pdf.ln()

            pdf.set_font("Arial", '', 9)
            for idx, r in df_boq.iterrows():
                dim_clean = str(r['الأبعاد (L x W x T)']).encode('ascii', 'ignore').decode('ascii')
                pdf.cell(widths[0], 9, str(r['ت']), 1, 0, 'C')
                pdf.cell(widths[1], 9, f"Part-{r['ت']}", 1, 0, 'C')
                pdf.cell(widths[2], 9, str(r['العدد']), 1, 0, 'C')
                pdf.cell(widths[3], 9, dim_clean, 1, 0, 'C')
                pdf.cell(widths[4], 9, f"{r['المساحة (م²)']:.1f}", 1, 0, 'C')
                pdf.cell(widths[5], 9, f"{r['الحجم (م³)']:.2f}", 1, 0, 'C')
                pdf.cell(widths[6], 9, f"{r['سعر الوحدة (دينار/م²)']:,.0f}", 1, 0, 'C')
                pdf.cell(widths[7], 9, f"{r['المبلغ الإجمالي (دينار)']:,.0f}", 1, 0, 'C')
                pdf.cell(widths[8], 9, f"{r['السمنت (طن)']:.2f}", 1, 1, 'C')

            pdf.ln(6)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, f"TOTAL MATERIALS: Cement = {sum_cement:.2f} Ton | Sand = {sum_sand:.1f} m3 | Gravel = {sum_gravel:.1f} m3", ln=True, align='L')
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, f"GRAND TOTAL ESTIMATED COST: {total_iqd:,.0f} IQD", ln=True, align='R')
            pdf.set_font("Arial", 'I', 8)
            pdf.cell(0, 6, "Powered by Iraqi Construction AI System (Supervised by Dr. Ali Khalil | Eng. Team: Sajjad, Jaafar, Muslim, Mohammed)", ln=True, align='C')
            return bytes(pdf.output())

        # دالة إنشاء ملف Excel رسمي (.xlsx)
        def generate_official_excel(df_boq):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_boq.to_excel(writer, index=False, sheet_name='جدول الذرعة والكميات')
                summary_df = pd.DataFrame({
                    "البيان": ["اسم الشركة", "اسم المشروع", "الكلفة الإجمالية (دينار)", "إجمالي المساحة (م²)", "إجمالي الخرسانة (م³)", "إجمالي السمنت (طن)", "إجمالي الرمل (م³)", "إجمالي الحصى (م³)"],
                    "القيمة": [company_ar_name, project_name, grand_total, sum_area, sum_vol, sum_cement, sum_sand, sum_gravel]
                })
                summary_df.to_excel(writer, index=False, sheet_name='ملخص المواد والكلف')
            return output.getvalue()

        b1, b2, b3 = st.columns(3)
        with b1:
            st.download_button(
                label="📄 تحميل التقرير الرسمي بصيغة PDF",
                data=generate_official_pdf(boq_df, grand_total, company_en_name, uploaded_logo),
                file_name="Official_BOQ_Report.pdf",
                mime="application/pdf"
            )
        with b2:
            st.download_button(
                label="📊 تحميل جدول الكميات بصيغة Excel",
                data=generate_official_excel(boq_df),
                file_name="Official_BOQ_Project.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with b3:
            if st.button("🗑️ مسح الجدول والبدء بمشروع جديد"):
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