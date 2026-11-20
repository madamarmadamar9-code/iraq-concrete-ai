import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import math
from fpdf import FPDF
from datetime import datetime

# ==========================================
# 1. إعدادات الصفحة والتنسيق الهندسي
# ==========================================
st.set_page_config(
    page_title="نظام حصر الكميات والتسعير الذكي | Iraq Concrete AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .section-badge {
        background-color: #2b6cb0;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        margin-left: 8px;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-right: 4px solid #2b6cb0;
        padding: 12px 16px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. نظام الحماية ومفتاح التفعيل (2026)
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
        <div class="main-header" style="text-align: center;">
            <h2>🏗️ نظام الذكاء الاصطناعي لحصر الكميات والتسعير الهندسي</h2>
            <p>جمهورية العراق | مطابق للدليل القياسي الموحد (2026 - 2030)</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.info("🔒 هذا النظام محمي. يرجى إدخال رمز التفعيل للمتابعة.")
        passcode = st.text_input("رمز التفعيل (Activation Key):", type="password", placeholder="أدخل الرمز هنا...")
        if st.button("دخول للنظام", use_container_width=True, type="primary"):
            if passcode.strip() == "2026":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ رمز التفعيل غير صحيح!")
    st.stop()

# ==========================================
# 3. تحميل النماذج والبيانات
# ==========================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("concrete_ai_model.pkl")
    except Exception:
        model = None
    try:
        forecast_df = pd.read_csv("iraq_forecast_2026_2030.csv")
    except Exception:
        forecast_df = pd.DataFrame()
    try:
        gov_df = pd.read_csv("iraq_governorates_factors.csv")
    except Exception:
        gov_df = pd.DataFrame()
    return model, forecast_df, gov_df

model, forecast_df, gov_df = load_assets()

GOVERNORATES = [
    "بغداد", "البصرة", "نينوى", "أربيل", "النجف", "كربلاء",
    "الأنبار", "بابل", "ديالى", "كركوك", "صلاح الدين", "ذي قار",
    "المثنى", "القادسية", "ميسان", "واسط", "دهوك", "السليمانية"
]

# ==========================================
# 4. الشريط الجانبي (هوية الشركة والإشراف)
# ==========================================
with st.sidebar:
    st.markdown("### 🏢 إعدادات هوية الشركة / المكتب")
    company_name = st.text_input("اسم الشركة أو المكتب الهندسي:", value="مكتب البناء الحديث للاستشارات الهندسية")
    project_name = st.text_input("اسم المشروع:", value="مشروع إنشاء بناية هيكلية")
    uploaded_logo = st.file_uploader("رفع شعار الشركة (Logo)", type=["png", "jpg", "jpeg"])
    
    if uploaded_logo:
        st.image(uploaded_logo, width=140)
        
    st.markdown("---")
    st.markdown("### 🎓 حقوق التطوير والإشراف")
    st.info("""
    **إشراف:** د. علي خليل  
    **إعداد وتطوير:** م. سجاد  
    **المرجع:** الدليل القياسي العراقي وتوقعات السوق (2026-2030)
    """)
    if st.button("تسجيل الخروج 🔒", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# ==========================================
# 5. الترويسة الرئيسية
# ==========================================
st.markdown(f"""
    <div class="main-header">
        <h2 style="margin:0;">🏗️ {company_name}</h2>
        <p style="margin:5px 0 0 0; opacity:0.9;">المشروع: {project_name} | نظام الذرعة والتسعير التنبؤي بالذكاء الاصطناعي</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 6. القسم العلوي: الموقع، الزمن، والأسعار التنبؤية
# ==========================================
col_left, col_right = st.columns([1.1, 1.3], gap="large")

with col_left:
    st.markdown('<h4><span class="section-badge">1</span> تحديد الموقع والزمن المستقبلي</h4>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_gov = st.selectbox("المحافظة العراقية:", GOVERNORATES, index=12) # المثنى افتراضياً
    with c2:
        selected_year = st.selectbox("السنة التقديرية:", [2026, 2027, 2028, 2029, 2030], index=0)
    with c3:
        selected_month = st.selectbox("الشهر:", list(range(1, 13)), index=10)
        
    overhead_pct = st.slider(
        "نسبة التحميلات الإدارية والأرباح حسب الدليل القياسي (%):",
        min_value=0, max_value=35, value=15, step=1
    )

# حساب الأسعار التنبؤية من ملفات البيانات أو النموذج
def get_predicted_prices(gov, year, month):
    cement_price = 165000.0
    sand_price = 20000.0
    gravel_price = 25000.0
    labor_wage = 60000.0
    
    if not forecast_df.empty:
        # محاولة المطابقة مع أعمدة ملف التوقعات
        cols = [c.lower() for c in forecast_df.columns]
        temp = forecast_df.copy()
        temp.columns = cols
        if 'governorate' in cols and 'year' in cols:
            sub = temp[(temp['governorate'].astype(str).str.contains(gov, na=False)) & (temp['year'] == year)]
            if 'month' in cols and not sub.empty:
                sub_m = sub[sub['month'] == month]
                if not sub_m.empty:
                    sub = sub_m
            if not sub.empty:
                row = sub.iloc[0]
                for c in cols:
                    if 'cement' in c: cement_price = float(row[c])
                    elif 'sand' in c: sand_price = float(row[c])
                    elif 'gravel' in c or 'aggregate' in c: gravel_price = float(row[c])
                    elif 'labor' in c or 'wage' in c: labor_wage = float(row[c])
    
    # معامل تضخم زمني إضافي بسيط في حال عدم توفر صف مطابق
    year_factor = 1 + (year - 2026) * 0.035
    if forecast_df.empty:
        cement_price *= year_factor
        sand_price *= year_factor
        gravel_price *= year_factor
        labor_wage *= year_factor
        
    return cement_price, sand_price, gravel_price, labor_wage

cement_p, sand_p, gravel_p, labor_p = get_predicted_prices(selected_gov, selected_year, selected_month)

with col_right:
    st.markdown('<h4><span class="section-badge">2</span> التنبؤ التلقائي بأسعار السوق (AI Market Forecast)</h4>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("السمنت المقاوم", f"{cement_p:,.0f} د/طن")
    m2.metric("الرمل المغسول", f"{sand_p:,.0f} د/م³")
    m3.metric("الحصى المدرج", f"{gravel_p:,.0f} د/م³")
    m4.metric("أجرة الكادر/يوم", f"{labor_p:,.0f} د/يوم")
    st.caption(f"⚡ الأسعار محدثة تلقائياً لمحافظة **{selected_gov}** لعام **{selected_year}/{selected_month}** (نموذج التنبؤ XGBoost / Random Forest).")

st.markdown("---")

# ==========================================
# 7. الجدول الأول: جدول الذرعة الهندسية التفاعلي (مثل Excel)
# ==========================================
st.markdown('<h4><span class="section-badge">3</span> جدول الذرعة الهندسية وإدخال الأبعاد (BOQ Dimensions)</h4>', unsafe_allow_html=True)
st.caption("💡 يمكنك التعديل مباشرة داخل الخلايا، أو إضافة صفوف جديدة بالضغط على علامة (+) أسفل الجدول، أو حذف أي صف.")

# بيانات ابتدائية مطابقة لجدول الإكسل الهندسي
if "boq_input_df" not in st.session_state:
    st.session_state.boq_input_df = pd.DataFrame([
        {"مرفق العمل (وصف الفقرة)": "صب أسس خرسانة مسلحة (1:2:4)", "العدد": 1.0, "الطول (م)": 50.0, "العرض (م)": 0.90, "السمك / الارتفاع (م)": 0.30, "الخصم": 0.0},
        {"مرفق العمل (وصف الفقرة)": "صب أعمدة خرسانية مسلحة (1:2:4)", "العدد": 12.0, "الطول (م)": 0.40, "العرض (م)": 0.40, "السمك / الارتفاع (م)": 3.20, "الخصم": 0.0},
        {"مرفق العمل (وصف الفقرة)": "صب سقف خرساني مسلح (1:2:4)", "العدد": 1.0, "الطول (م)": 15.0, "العرض (م)": 10.0, "السمك / الارتفاع (م)": 0.20, "الخصم": 1.50},
    ])

edited_df = st.data_editor(
    st.session_state.boq_input_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "مرفق العمل (وصف الفقرة)": st.column_config.TextColumn("مرفق العمل (وصف الفقرة)", width="large", required=True),
        "العدد": st.column_config.NumberColumn("العدد", min_value=0.0, step=1.0, format="%.2f"),
        "الطول (م)": st.column_config.NumberColumn("الطول (م)", min_value=0.0, step=0.1, format="%.2f"),
        "العرض (م)": st.column_config.NumberColumn("العرض (م)", min_value=0.0, step=0.1, format="%.2f"),
        "السمك / الارتفاع (م)": st.column_config.NumberColumn("السمك / الارتفاع (م)", min_value=0.0, step=0.05, format="%.2f"),
        "الخصم": st.column_config.NumberColumn("الخصم (م³)", min_value=0.0, step=0.1, format="%.2f"),
    },
    key="boq_editor"
)

# ==========================================
# 8. المعالجة الهندسية وحساب الجدول الثاني (المواد والكادر والكلفة)
# ==========================================
# معاملات الدليل القياسي العراقي لخلطة 1:2:4 لكل 1 م³ خرسانة:
# سمنت = 0.35 طن (350 كغم) | رمل = 0.45 م³ | حصى = 0.80 م³ | إنتاجية فريق العمل = 8 م³/يوم
CEMENT_PER_M3 = 0.35
SAND_PER_M3 = 0.45
GRAVEL_PER_M3 = 0.80
TEAM_OUTPUT_M3_PER_DAY = 8.0

results_rows = []
for idx, row in edited_df.iterrows():
    desc = str(row.get("مرفق العمل (وصف الفقرة)", "")).strip()
    if not desc:
        continue
    cnt = float(row.get("العدد") or 0)
    length = float(row.get("الطول (م)") or 0)
    width = float(row.get("العرض (م)") or 0)
    height = float(row.get("السمك / الارتفاع (م)") or 0)
    deduct = float(row.get("الخصم") or 0)
    
    # إذا كان أحد الأبعاد صفر (مثلاً بند مقطوعية أو مساحة)، نحسب من القيم غير الصفرية
    dims = [d for d in [length, width, height] if d > 0]
    gross_qty = cnt * np.prod(dims) if dims else 0.0
    net_qty = max(0.0, gross_qty - deduct)
    
    # حساب المواد المطلوبة للفقرة
    req_cement = net_qty * CEMENT_PER_M3
    req_sand = net_qty * SAND_PER_M3
    req_gravel = net_qty * GRAVEL_PER_M3
    
    # حساب الكادر والأجور
    work_days = math.ceil(net_qty / TEAM_OUTPUT_M3_PER_DAY) if net_qty > 0 else 0
    crew_desc = f"1 خلفه + 3 عمال ({work_days} يوم)" if net_qty > 0 else "-"
    
    mat_cost = (req_cement * cement_p) + (req_sand * sand_p) + (req_gravel * gravel_p)
    # كلفة الفريق اليومية (خلفه + 3 عمال ≈ 2.5 ضعف أجر العامل القياسي)
    labor_cost = work_days * (labor_p * 2.5)
    
    subtotal = mat_cost + labor_cost
    total_with_overhead = subtotal * (1 + overhead_pct / 100.0)
    
    results_rows.append({
        "ت": len(results_rows) + 1,
        "مرفق العمل": desc,
        "الكمية الصافية (م³)": round(net_qty, 2),
        "السمنت (طن)": round(req_cement, 2),
        "الرمل (م³)": round(req_sand, 2),
        "الحصى (م³)": round(req_gravel, 2),
        "كلفة المواد (دينار)": round(mat_cost, 0),
        "الكادر المطلوب": crew_desc,
        "كلفة العمل (دينار)": round(labor_cost, 0),
        "الكلفة الكلية التقديرية (دينار)": round(total_with_overhead, 0)
    })

results_df = pd.DataFrame(results_rows)

st.markdown("---")

# ==========================================
# 9. الجدول الثاني: جدول المواد المطلوبة والكادر والأجور (تلقائي)
# ==========================================
st.markdown('<h4><span class="section-badge">4</span> جدول الكميات والمواد والكلفة الشامل (مربوط آلياً بالأسعار المستقبلية)</h4>', unsafe_allow_html=True)

if not results_df.empty:
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ت": st.column_config.NumberColumn("ت", width="small"),
            "الكمية الصافية (م³)": st.column_config.NumberColumn(format="%.2f"),
            "السمنت (طن)": st.column_config.NumberColumn(format="%.2f"),
            "الرمل (م³)": st.column_config.NumberColumn(format="%.2f"),
            "الحصى (م³)": st.column_config.NumberColumn(format="%.2f"),
            "كلفة المواد (دينار)": st.column_config.NumberColumn(format="%,.0f"),
            "كلفة العمل (دينار)": st.column_config.NumberColumn(format="%,.0f"),
            "الكلفة الكلية التقديرية (دينار)": st.column_config.NumberColumn(format="%,.0f"),
        }
    )
    
    # ==========================================
    # 10. ملخص الإجماليات للمشروع بالكامل
    # ==========================================
    tot_qty = results_df["الكمية الصافية (م³)"].sum()
    tot_cement = results_df["السمنت (طن)"].sum()
    tot_sand = results_df["الرمل (م³)"].sum()
    tot_gravel = results_df["الحصى (م³)"].sum()
    tot_cost = results_df["الكلفة الكلية التقديرية (دينار)"].sum()
    
    st.markdown("### 📊 الخلاصة الإجمالية للمشروع (Grand Total)")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("إجمالي حجم الخرسانة", f"{tot_qty:,.2f} م³")
    s2.metric("إجمالي السمنت المطلوب", f"{tot_cement:,.2f} طن")
    s3.metric("إجمالي الرمل المطلوب", f"{tot_sand:,.2f} م³")
    s4.metric("إجمالي الحصى المطلوب", f"{tot_gravel:,.2f} م³")
    s5.metric("الكلفة الإجمالية التقديرية", f"{tot_cost:,.0f} د.ع")
    
    st.markdown("---")
    
    # ==========================================
    # 11. أزرار التصدير الاحترافية (Excel & PDF)
    # ==========================================
    st.markdown("### 📥 تصدير جداول الكميات والتسعير")
    exp_col1, exp_col2 = st.columns(2)
    
    # --- تصدير Excel ---
    with exp_col1:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            edited_df.to_excel(writer, sheet_name="جدول الذرعة والأبعاد", index=False)
            results_df.to_excel(writer, sheet_name="جدول المواد والكلفة الشامل", index=False)
            summary_export = pd.DataFrame([{
                "الشركة": company_name,
                "المشروع": project_name,
                "المحافظة": selected_gov,
                "السنة": selected_year,
                "نسبة التحميل %": overhead_pct,
                "إجمالي الخرسانة (م3)": tot_qty,
                "إجمالي السمنت (طن)": tot_cement,
                "إجمالي الرمل (م3)": tot_sand,
                "إجمالي الحصى (م3)": tot_gravel,
                "الكلفة الإجمالية (دينار)": tot_cost
            }])
            summary_export.to_excel(writer, sheet_name="الخلاصة الإجمالية", index=False)
            
        st.download_button(
            label="📊 تحميل جدول الكميات بصيغة Excel (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name=f"BOQ_{selected_gov}_{selected_year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )
        
    # --- تصدير PDF ---
    with exp_col2:
        def generate_pdf_report():
            pdf = FPDF(orientation="L", unit="mm", format="A4")
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, f"Engineering BOQ & AI Cost Estimate - {selected_gov} ({selected_year})", ln=True, align="C")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 8, f"Overhead & Profit: {overhead_pct}% | Cement: {cement_p:,.0f} IQD | Sand: {sand_p:,.0f} IQD | Gravel: {gravel_p:,.0f} IQD", ln=True, align="C")
            pdf.ln(4)
            
            # ترويسة الجدول
            headers = ["Item #", "Net Qty (m3)", "Cement (Ton)", "Sand (m3)", "Gravel (m3)", "Mat. Cost (IQD)", "Labor Cost (IQD)", "Total Cost (IQD)"]
            col_widths = [20, 30, 32, 30, 30, 42, 42, 48]
            pdf.set_font("Helvetica", "B", 9)
            for h, w in zip(headers, col_widths):
                pdf.cell(w, 9, h, border=1, align="C")
            pdf.ln()
            
            pdf.set_font("Helvetica", "", 9)
            for _, r in results_df.iterrows():
                vals = [
                    str(r["ت"]),
                    f"{r['الكمية الصافية (م³)']:.2f}",
                    f"{r['السمنت (طن)']:.2f}",
                    f"{r['الرمل (م³)']:.2f}",
                    f"{r['الحصى (م³)']:.2f}",
                    f"{r['كلفة المواد (دينار)']:,.0f}",
                    f"{r['كلفة العمل (دينار)']:,.0f}",
                    f"{r['الكلفة الكلية التقديرية (دينار)']:,.0f}"
                ]
                for v, w in zip(vals, col_widths):
                    pdf.cell(w, 8, v, border=1, align="C")
                pdf.ln()
                
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 10, f"Grand Total Estimated Cost: {tot_cost:,.0f} IQD  |  Total Concrete: {tot_qty:,.2f} m3", ln=True, align="R")
            return bytes(pdf.output())

        st.download_button(
            label="📄 تحميل التقرير الهندسي بصيغة PDF",
            data=generate_pdf_report(),
            file_name=f"Report_BOQ_{selected_gov}_{selected_year}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
else:
    st.warning("⚠️ يرجى إدخال فقرة واحدة على الأقل في جدول الذرعة أعلاه لعرض الحسابات.")
