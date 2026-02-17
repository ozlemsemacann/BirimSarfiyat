import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool

# Sayfa genişliği ayarı
st.set_page_config(layout="wide")

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
# Arayüz Başlığı
# -----------------------------
st.title("🧵 Dokuma Birim Sarfiyat Tahmini")
st.success("✅ Modeli önceden eğittik ve yükledik. Şimdi değerleri gir, tahmini al!")

# -----------------------------
# 2. Dinamik Filtreler ve Tasarım
# -----------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📋 Model Seçimi")
    
    # DEPARTMAN (Ana Filtre)
    dept_list = sorted(df['DEPARTMAN'].unique().tolist())
    selected_dept = st.selectbox("Departman", dept_list)

    # MODEL_TURU (Departmana bağlı)
    m_turu_list = sorted(df[df['DEPARTMAN'] == selected_dept]['MODEL_TURU'].unique().tolist())
    selected_model_turu = st.selectbox("Model_Turu", m_turu_list)

    # FIT (Departman ve Model Turuna bağlı)
    fit_list = sorted(df[
        (df['DEPARTMAN'] == selected_dept) & 
        (df['MODEL_TURU'] == selected_model_turu)
    ]['FIT'].unique().tolist())
    selected_fit = st.selectbox("Fit", fit_list)

with col_right:
    st.subheader("⚙️ Teknik Detaylar")
    
    # ASORTI (Filtrelere bağlı)
    asorti_list = sorted(df[
        (df['DEPARTMAN'] == selected_dept) &
        (df['MODEL_TURU'] == selected_model_turu) & 
        (df['FIT'] == selected_fit)
    ]['ASORTI'].unique().tolist())
    selected_asorti = st.selectbox("Asorti", asorti_list)

    # PASTAL_TURU
    pastal_turu_list = sorted(df['PASTAL_TURU'].unique().tolist())
    selected_pastal_turu = st.selectbox("Pastal_Turu", pastal_turu_list)

    # Sayısal Değerler
    c1, c2 = st.columns(2)
    with c1:
        kumas_eni = st.number_input("Kumas_Eni", value=145.0)
        asorti_sayisi = st.number_input("Toplam_Asorti", value=10.0)
    with c2:
        kumas_gramaji = st.number_input("Kumas_Gramaji", value=150.0)
        parca_sayisi = st.number_input("Parca_Sayisi", value=19.0)

# Modelin ihtiyacı olan ancak görselde görünmeyen diğer teknik detayları veriden alıyoruz
selected_row = df[
    (df['DEPARTMAN'] == selected_dept) & 
    (df['MODEL_TURU'] == selected_model_turu) & 
    (df['FIT'] == selected_fit) &
    (df['ASORTI'] == selected_asorti)
].iloc[0]

# -----------------------------
# 3. Hesaplama ve Tahmin
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("HESAPLA", use_container_width=True, type="primary"):
    
    # Sözlük yapısı hatasız şekilde oluşturuldu
    inputs = {
        'ASORTI_SAYISI': asorti_sayisi,
        'PARCA_SAYISI': parca_sayisi,
        'KUMAS_CEKME_DEGERI_BOY': selected_row.get('KUMAS_CEKME_DEGERI_BOY', -3.0),
        'KUMAS_CEKME_DEGERI_EN': selected_row.get('KUMAS_CEKME_DEGERI_EN', -4.0),
        'KUMAS_ENI': kumas_eni,
        'ASORTI': selected_asorti,
        'PASTAL_DETAYI': selected_row.get('PASTAL_DETAYI', 'YONSUZ'),
        'PASTAL_TURU': selected_pastal_turu,
        'MODEL_DETAYI': selected_row.get('MODEL_DETAYI', 'YOK'),
        'MODEL_TURU': selected_model_turu,
        'DEPARTMAN': selected_dept,
        'FIT': selected_fit
    }

    X_new = pd.DataFrame([inputs])
    
    # Modelin beklediği kategorik kolonlar
    cat_features = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT', 
                    'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']

    try:
        X_new_pool = Pool(X_new, cat_features=cat_features)
        prediction = model.predict(X_new_pool)[0]
        
        st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; border: 2px solid #ff4b4b;">
                <h2 style="color: #31333F;">🔮 Tahmini Birim Sarfiyat</h2>
                <h1 style="font-size: 60px; color: #ff4b4b;">{prediction:.4f}</h1>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Tahmin hatası: {e}")
