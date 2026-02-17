import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool

# -----------------------------
# 1. Veri ve Model Yükleme
# -----------------------------
@st.cache_data
def load_data():
    # Excel dosyasını yüklüyoruz
    df = pd.read_excel("YuklenenDokumaDosya172.xlsx")
    return df

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("Dokuma_BirimSarfiyatModel.cbm")
    return model

df = load_data()
model = load_model()

# -----------------------------
# Streamlit Arayüzü
# -----------------------------
st.title("🧵 Birim Sarfiyat Tahmini (Dinamik Filtreli)")

st.markdown("""
Bu model **RMSE: 0.057** hata payı ile eğitilmiştir. 
Seçimlerinize göre seçenekler otomatik olarak güncellenecektir.
""")

# -----------------------------
# 2. Birbirine Bağlı Filtreler (Cascading)
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    # DEPARTMAN Filtresi
    dept_list = sorted(df['DEPARTMAN'].unique())
    selected_dept = st.selectbox("DEPARTMAN", dept_list)

    # MODEL_TURU (Departmana bağlı)
    m_turu_list = sorted(df[df['DEPARTMAN'] == selected_dept]['MODEL_TURU'].unique())
    selected_model_turu = st.selectbox("MODEL_TURU", m_turu_list)

    # MODEL_DETAYI (Departman ve Model Turuna bağlı)
    m_detay_list = sorted(df[
        (df['DEPARTMAN'] == selected_dept) & 
        (df['MODEL_TURU'] == selected_model_turu)
    ]['MODEL_DETAYI'].unique())
    selected_model_detay = st.selectbox("MODEL_DETAYI", m_detay_list)

with col2:
    # FIT (Önceki seçimlere bağlı)
    fit_list = sorted(df[
        (df['DEPARTMAN'] == selected_dept) & 
        (df['MODEL_TURU'] == selected_model_turu) &
        (df['MODEL_DETAYI'] == selected_model_detay)
    ]['FIT'].unique())
    selected_fit = st.selectbox("FIT", fit_list)

    # PASTAL_TURU
    pastal_turu_list = sorted(df['PASTAL_TURU'].unique())
    selected_pastal_turu = st.selectbox("PASTAL_TURU", pastal_turu_list)

    # ASORTI (Seçilen Fit ve Model Detayına göre süzülür)
    asorti_list = sorted(df[
        (df['MODEL_TURU'] == selected_model_turu) & 
        (df['FIT'] == selected_fit)
    ]['ASORTI'].unique())
    selected_asorti = st.selectbox("ASORTI", asorti_list)

st.divider()

# -----------------------------
# 3. Sayısal Girişler
# -----------------------------
c1, c2, c3 = st.columns(3)

with c1:
    kumas_eni = st.number_input("KUMAS_ENI", 90.0, 195.0, 145.0)
    asorti_sayisi = st.number_input("ASORTI_SAYISI", 1.0, 50.0, 10.0)

with c2:
    cekme_en = st.number_input("CEKME_EN", -13.0, 5.0, -4.0)
    parca_sayisi = st.number_input("PARCA_SAYISI", 1.0, 100.0, 19.0)

with c3:
    cekme_boy = st.number_input("CEKME_BOY", -22.0, 10.0, -3.0)
    pastal_detayi = st.selectbox("PASTAL_DETAYI", sorted(df['PASTAL_DETAYI'].unique()))

# -----------------------------
# 4. Tahminleme
# -----------------------------
if st.button("Hesapla ve Tahmin Et", type="primary"):
    # Modelin eğitildiği kolon sırasına ve isimlerine sadık kalarak input oluşturma
    inputs = {
        'DEPARTMAN': selected_dept,
        'MODEL_TURU': selected_model_turu,
        'MODEL_DETAYI': selected_model_detay,
        'FIT': selected_fit,
        'KUMAS_ENI': kumas_eni,
        'KUMAS_CEKME_DEGERI_EN': cekme_en,
        'KUMAS_CEKME_DEGERI_BOY': cekme_boy,
        'PASTAL_TURU': selected_pastal_turu,
        'PASTAL_DETAYI': pastal_detayi,
        'ASORTI': selected_asorti,
        'ASORTI_SAYISI': asorti_sayisi,
        'PARCA_SAYISI': parca_sayisi
    }

    X_new = pd.DataFrame([inputs])
    
    # Modelin beklediği kategorik özellikler 
    cat_features = [
        'DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT', 
        'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI'
    ]

    try:
        X_new_pool = Pool(X_new, cat_features=cat_features)
        prediction = model.predict(X_new_pool)[0]
        st.success(f"🔮 Tahmini Birim Sarfiyat: **{prediction:.4f}**")
    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu: {e}")
