import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool

# Sayfa yapılandırması - Görseldeki geniş yerleşim için
st.set_page_config(layout="wide")

# -----------------------------
# 1. Veri ve Model Yükleme
# -----------------------------
@st.cache_data
def load_data():
    # [cite_start]Excel dosyasını yüklüyoruz [cite: 11]
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
# 2. Dinamik Filtre Mantığı ve Tasarım
# -----------------------------

# Görseldeki gibi iki ana bölüme ayırıyoruz
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📋 Model Seçimi")
    
    # DEPARTMAN Filtresi
    dept_list = sorted(df['DEPARTMAN'].unique().tolist())
    selected_dept = st.selectbox("Departman", dept_list)

    # [cite_start]MODEL_TURU (Seçilen Departmana bağlı) [cite: 11]
    m_turu_list = sorted(df[df['DEPARTMAN'] == selected_dept]['MODEL_TURU'].unique().tolist())
    selected_model_turu = st.selectbox("Model_Turu", m_turu_list)

    # FIT (Departman ve Model Turuna bağlı)
    fit_list = sorted(df[
        (df['DEPARTMAN'] == selected_dept) & 
        (df['MODEL_TURU'] == selected_model_turu)
    ]['FIT'].unique().tolist())
    selected_fit = st.selectbox("Fit", fit_list)
    
    # MODEL_DETAYI (Arka planda süzülür, modele gönderilir)
    m_detay_list = sorted(df[
        (df['DEPARTMAN'] == selected_dept) & 
        (df['MODEL_TURU'] == selected_model_turu) &
        (df['FIT'] == selected_fit)
    ]['MODEL_DETAYI'].unique().tolist())
    selected_model_detay = st.selectbox("Model Detayı", m_detay_list)

with col_right:
    st.subheader("⚙️ Teknik Detaylar")
    
    # [cite_start]ASORTI (Seçilen Fit ve Model Turuna bağlı) [cite: 11]
    asorti_list = sorted(df[
        (df['MODEL_TURU'] == selected_model_turu) & 
        (df['FIT'] == selected_fit)
    ]['ASORTI'].unique().tolist())
    selected_asorti = st.selectbox("Asorti", asorti_list)

    # PASTAL_TURU
    pastal_turu_list = sorted(df['PASTAL_TURU'].unique().tolist())
    selected_pastal_turu = st.selectbox("Pastal_Turu", pastal_turu_list)

    # Sayısal Değerler için yan yana kolonlar
    c1, c2 = st.columns(2)
    with c1:
        kumas_eni = st.number_input("Kumas_Eni", value=145.0)
        asorti_sayisi = st.number_input("Toplam_Asorti", value=10.0)
    with c2:
        # Dokuma modelinde çekme değerleri önemli olduğu için bunları ekledim
        cekme_en = st.number_input("Kumas_Cekme_En", value=-4.0)
        parca_sayisi = st.number_input("Parca_Sayisi", value=19.0)

# -----------------------------
# 3. Hesaplama Butonu ve Tahmin
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True) # Boşluk

# Görseldeki gibi geniş kırmızı buton efekti için
if st.button("HESAPLA", use_container_width=True, type="primary"):
    
    # [cite_start]Modelin beklediği kolon isimleri ve sırası [cite: 11]
    inputs = {
        'ASORTI_SAYISI': asorti_sayisi,
        'PARCA_SAYISI': parca_sayisi,
        'KUMAS_CEKME_DEGERI_BOY': -3.0, # Sabit veya input eklenebilir
        'KUMAS_CEKME_DEGERI_EN': cekme_en,
        'KUMAS_ENI': kumas_eni,
        'ASORTI': selected_asorti,
        [cite_start]'PASTAL_DETAYI': 'YONSUZ', # Varsayılan [cite: 11]
        'PASTAL_TURU': selected_pastal_turu,
        'MODEL_DETAYI': selected_model_detay,
        'MODEL_TURU': selected_model_turu,
        'DEPARTMAN': selected_dept,
        'FIT': selected_fit
    }

    X_new = pd.DataFrame([inputs])
    
    # [cite_start]Modelin kategorik kolonları [cite: 11]
    cat_features = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT', 
                    'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']

    try:
        X_new_pool = Pool(X_new, cat_features=cat_features)
        prediction = model.predict(X_new_pool)[0]
        
        st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                <h2 style="color: #ff4b4b;">🔮 Tahmini Birim Sarfiyat</h2>
                <h1 style="font-size: 50px;">{prediction:.4f}</h1>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Hata: {e}")
