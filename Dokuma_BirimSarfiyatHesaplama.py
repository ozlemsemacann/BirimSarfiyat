import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool

# Sayfa Genişliği ve Başlık
st.set_page_config(layout="wide", page_title="Birim Sarfiyat Tahmini")

# -----------------------------
# 1. Veri ve Model Yükleme
# -----------------------------
@st.cache_data
def load_data():
    # Dosya adının doğruluğundan emin olun
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
# 2. Görsel Arayüz Tasarımı
# -----------------------------
st.title("🧶 Örme Birim Sarfiyat Tahmini")
st.success("✅ Modeli önceden eğittik ve yükledik. Şimdi değerleri gir, tahmini al!")

# Ana sütunları oluştur (Görseldeki gibi 2 ana blok)
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📋 Model Seçimi")
    
    # 1. DEPARTMAN
    dept_list = sorted(df['DEPARTMAN'].unique().tolist())
    selected_dept = st.selectbox("Departman", dept_list)

    # 2. MODEL_TURU (Departmana bağlı)
    m_turu_list = sorted(df[df['DEPARTMAN'] == selected_dept]['MODEL_TURU'].unique().tolist())
    selected_model_turu = st.selectbox("Model_Turu", m_turu_list)

    # 3. FIT (Departman ve Model Turuna bağlı)
    fit_list = sorted(df[
        (df['DEPARTMAN'] == selected_dept) & 
        (df['MODEL_TURU'] == selected_model_turu)
    ]['FIT'].unique().tolist())
    selected_fit = st.selectbox("Fit", fit_list)

with col_right:
    st.subheader("⚙️ Teknik Detaylar")
    
    # 4. ASORTI (Departman, Model Turu ve Fit'e bağlı)
    # Filtrelerin daralması için tüm kademeleri ekledik
    asorti_list = sorted(df[
        (df['DEPARTMAN'] == selected_dept) &
        (df['MODEL_TURU'] == selected_model_turu) & 
        (df['FIT'] == selected_fit)
    ]['ASORTI'].unique().tolist())
    selected_asorti = st.selectbox("Asorti", asorti_list)

    # 5. PASTAL_TURU
    pastal_turu_list = sorted(df['PASTAL_TURU'].unique().tolist())
    selected_pastal_turu = st.selectbox("Pastal_Turu", pastal_turu_list)

    # Sayısal Girişler (2x2 düzen)
    c1, c2 = st.columns(2)
    with c1:
        kumas_eni = st.number_input("Kumas_Eni", value=145.0, step=1.0)
        toplam_asorti = st.number_input("Toplam_Asorti", value=10.0, step=1.0)
    with c2:
        kumas_gramaji = st.number_input("Kumas_Gramaji", value=150.0, step=1.0)
        parca_sayisi = st.number_input("Parca_Sayisi", value=19.0, step=1.0)

# -----------------------------
# 3. Excel'deki Diğer Gizli Kriterleri Eşleme
# -----------------------------
# Seçilen kombinasyona ait satırı bulup diğer kriterleri (Çekme, Detay vb.) çekiyoruz
filtered_df = df[
    (df['DEPARTMAN'] == selected_dept) & 
    (df['MODEL_TURU'] == selected_model_turu) & 
    (df['FIT'] == selected_fit) &
    (df['ASORTI'] == selected_asorti)
]

if not filtered_df.empty:
    matched_row = filtered_df.iloc[0]
else:
    # Eğer kombinasyon bulunamazsa (teorik olarak mümkün değil ama koruma amaçlı)
    matched_row = df.iloc[0]

# -----------------------------
# 4. Hesaplama ve Tahmin
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("HESAPLA", use_container_width=True, type="primary"):
    
    # Modelin tam olarak beklediği kolon seti ve sırası
    inputs = {
        'ASORTI_SAYISI': toplam_asorti, # Manuel giriş
        'PARCA_SAYISI': parca_sayisi,   # Manuel giriş
        'KUMAS_CEKME_DEGERI_BOY': matched_row.get('KUMAS_CEKME_DEGERI_BOY', -3.0), # Excel'den otomatik
        'KUMAS_CEKME_DEGERI_EN': matched_row.get('KUMAS_CEKME_DEGERI_EN', -4.0),   # Excel'den otomatik
        'KUMAS_ENI': kumas_eni,         # Manuel giriş
        'ASORTI': selected_asorti,      # Seçimden
        'PASTAL_DETAYI': matched_row.get('PASTAL_DETAYI', 'YONSUZ'),              # Excel'den otomatik
        'PASTAL_TURU': selected_pastal_turu,                                      # Seçimden
        'MODEL_DETAYI': matched_row.get('MODEL_DETAYI', 'YOK'),                   # Excel'den otomatik
        'MODEL_TURU': selected_model_turu,                                        # Seçimden
        'DEPARTMAN': selected_dept,                                               # Seçimden
        'FIT': selected_fit             # Seçimden
    }

    X_new = pd.DataFrame([inputs])
    
    # Model dosyasındaki kategorik özellikler
    cat_features = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT', 
                    'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']

    try:
        X_pool = Pool(X_new, cat_features=cat_features)
        prediction = model.predict(X_pool)[0]
        
        # Görseldeki Sonuç Kutusu Formatı
        st.markdown(f"""
            <div style="text-align: center; padding: 25px; border: 2px solid #ff4b4b; border-radius: 10px; background-color: #ffffff; margin-top: 20px;">
                <h2 style="color: #31333F; margin-bottom: 0;">🔮 Tahmini Birim Sarfiyat</h2>
                <h1 style="font-size: 80px; color: #ff4b4b; margin-top: 0;">{prediction:.4f}</h1>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Tahmin hatası: {e}")
