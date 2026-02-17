import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool

# Sayfa yapılandırması
st.set_page_config(layout="wide", page_title="Örme Birim Sarfiyat Tahmini")

# -----------------------------
# 1. Veri ve Model Yükleme
# -----------------------------
@st.cache_data
def load_data():
    # Excel dosyasını yüklüyoruz [cite: 1]
    df = pd.read_excel("YuklenenDokumaDosya172.xlsx")
    return df

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("Dokuma_BirimSarfiyatModel.cbm") # [cite: 1]
    return model

df = load_data()
model = load_model()

# -----------------------------
# Arayüz Başlığı (Görseldeki Format)
# -----------------------------
st.title("🧶 Örme Birim Sarfiyat Tahmini")
st.success("✅ Modeli önceden eğittik ve yükledik. Şimdi değerleri gir, tahmini al!")

# -----------------------------
# 2. Birbirine Bağlı Filtreler (Cascading)
# -----------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📋 Model Seçimi")
    
    # Departman Seçimi
    dept_list = sorted(df['DEPARTMAN'].unique().tolist())
    selected_dept = st.selectbox("Departman", dept_list)

    # Model Turu (Departmana bağlı)
    m_turu_list = sorted(df[df['DEPARTMAN'] == selected_dept]['MODEL_TURU'].unique().tolist())
    selected_model_turu = st.selectbox("Model_Turu", m_turu_list)

    # Fit (Departman ve Model Turuna bağlı)
    fit_list = sorted(df[
        (df['DEPARTMAN'] == selected_dept) & 
        (df['MODEL_TURU'] == selected_model_turu)
    ]['FIT'].unique().tolist())
    selected_fit = st.selectbox("Fit", fit_list)

with col_right:
    st.subheader("⚙️ Teknik Detaylar")
    
    # Asorti (Bağlı Filtre)
    asorti_list = sorted(df[
        (df['DEPARTMAN'] == selected_dept) &
        (df['MODEL_TURU'] == selected_model_turu) & 
        (df['FIT'] == selected_fit)
    ]['ASORTI'].unique().tolist())
    selected_asorti = st.selectbox("Asorti", asorti_list)

    # Pastal Turu
    pastal_turu_list = sorted(df['PASTAL_TURU'].unique().tolist())
    selected_pastal_turu = st.selectbox("Pastal_Turu", pastal_turu_list)

    # Sayısal Girişler (Görseldeki 2x2 düzeni)
    c1, c2 = st.columns(2)
    with c1:
        kumas_eni = st.number_input("Kumas_Eni", value=145.0)
        toplam_asorti = st.number_input("Toplam_Asorti", value=10.0)
    with c2:
        # Excel'deki gramaj veya diğer sayısal kriterler
        kumas_gramaji = st.number_input("Kumas_Gramaji", value=150.0)
        parca_sayisi = st.number_input("Parca_Sayisi", value=19.0)

# -----------------------------
# 3. Gizli Kriterlerin Otomatik Çekilmesi
# -----------------------------
# Kullanıcının seçmediği ama Excel'de olan (MODEL_DETAYI, CEKME vb.) 
# değerleri seçilen satırdan otomatik eşliyoruz.
matched_data = df[
    (df['DEPARTMAN'] == selected_dept) & 
    (df['MODEL_TURU'] == selected_model_turu) & 
    (df['FIT'] == selected_fit) &
    (df['ASORTI'] == selected_asorti)
].iloc[0]

# -----------------------------
# 4. Tahminleme (HESAPLA)
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("HESAPLA", use_container_width=True, type="primary"):
    
    # Modelin beklediği tüm 12 parametreyi eksiksiz tanımlıyoruz 
    input_dict = {
        'ASORTI_SAYISI': toplam_asorti,
        'PARCA_SAYISI': parca_sayisi,
        'KUMAS_CEKME_DEGERI_BOY': matched_data.get('KUMAS_CEKME_DEGERI_BOY', -3.0),
        'KUMAS_CEKME_DEGERI_EN': matched_data.get('KUMAS_CEKME_DEGERI_EN', -4.0),
        'KUMAS_ENI': kumas_eni,
        'ASORTI': selected_asorti,
        'PASTAL_DETAYI': matched_data.get('PASTAL_DETAYI', 'YONSUZ'),
        'PASTAL_TURU': selected_pastal_turu,
        'MODEL_DETAYI': matched_data.get('MODEL_DETAYI', 'YOK'),
        'MODEL_TURU': selected_model_turu,
        'DEPARTMAN': selected_dept,
        'FIT': selected_fit
    }

    X_new = pd.DataFrame([input_dict])
    
    # Kategorik özellikler listesi 
    cat_features = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT', 
                    'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']

    try:
        X_pool = Pool(X_new, cat_features=cat_features)
        prediction = model.predict(X_pool)[0]
        
        # Sonuç Kutusu (Görseldeki format)
        st.markdown(f"""
            <div style="text-align: center; padding: 25px; border: 2px solid #ff4b4b; border-radius: 10px; background-color: #ffffff;">
                <h2 style="color: #31333F; margin-bottom: 0;">🔮 Tahmini Birim Sarfiyat</h2>
                <h1 style="font-size: 72px; color: #ff4b4b; margin-top: 0;">{prediction:.4f}</h1>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Tahmin sırasında teknik bir hata oluştu: {e}")
