import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool
import os

# Sayfa Ayarı
st.set_page_config(page_title="Sarfiyat Tahmini", layout="wide")

# --------------------------------------------------------
# AYARLAR (GitHub'daki isimlerle BİREBİR aynı olmalı)
# --------------------------------------------------------
EXCEL_FILE_NAME = "_YuklenenDokumaDosya30.1.xlsx"
MODEL_FILE_NAME = "Dokuma_BirimSarfiyatModel.cbm"

# -----------------------------
# 1. VERİ YÜKLEME VE HATA AYIKLAMA
# -----------------------------
@st.cache_data
def load_data():
    # 1. Yöntem: Direkt dosya adını dene
    if os.path.exists(EXCEL_FILE_NAME):
        return pd.read_excel(EXCEL_FILE_NAME)
    
    # 2. Yöntem: Bulamazsa Debug Modunu Aç
    else:
        st.error(f"❌ KRİTİK HATA: '{EXCEL_FILE_NAME}' dosyası bulunamadı!")
        
        # Şu an hangi klasördeyiz?
        current_dir = os.getcwd()
        st.warning(f"📂 Çalışılan Klasör: {current_dir}")
        
        # Klasörde hangi dosyalar var? (Bunu ekrana basacağız)
        files = os.listdir(current_dir)
        st.info(f"📄 Bu klasördeki dosyalar: {files}")
        
        st.stop() # Programı durdur
        return pd.DataFrame()

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    if os.path.exists(MODEL_FILE_NAME):
        model.load_model(MODEL_FILE_NAME)
        return model
    else:
        st.error(f"❌ Model dosyası ({MODEL_FILE_NAME}) bulunamadı.")
        return None

# Yüklemeleri Başlat
df = load_data()
model = load_model()

# -----------------------------
# 2. ARAYÜZ (Filtreleme İşlemleri)
# -----------------------------
st.title("🧵 Akıllı Birim Sarfiyat Tahmini")

# Eğer veri boşsa veya okunmadıysa aşağıya geçme
if df is None or df.empty:
    st.stop()

inputs = {}
st.markdown("---")

col_main1, col_main2 = st.columns([1, 1])

with col_main1:
    st.subheader("📋 Model Seçimi")
    
    # 1. DEPARTMAN
    dept_list = sorted(df['DEPARTMAN'].astype(str).unique())
    secilen_dept = st.selectbox("DEPARTMAN", dept_list)
    inputs['DEPARTMAN'] = secilen_dept
    df_step1 = df[df['DEPARTMAN'] == secilen_dept]

    # 2. MODEL TÜRÜ
    tur_list = sorted(df_step1['MODEL_TURU'].astype(str).unique())
    secilen_tur = st.selectbox("MODEL_TURU", tur_list)
    inputs['MODEL_TURU'] = secilen_tur
    df_step2 = df_step1[df_step1['MODEL_TURU'] == secilen_tur]

    # 3. MODEL DETAYI
    detay_list = sorted(df_step2['MODEL_DETAYI'].astype(str).unique())
    secilen_detay = st.selectbox("MODEL_DETAYI", detay_list)
    inputs['MODEL_DETAYI'] = secilen_detay
    df_step3 = df_step2[df_step2['MODEL_DETAYI'] == secilen_detay]

    # 4. FIT
    fit_list = sorted(df_step3['FIT'].astype(str).unique())
    secilen_fit = st.selectbox("FIT", fit_list)
    inputs['FIT'] = secilen_fit

with col_main2:
    st.subheader("⚙️ Teknik Detaylar")
    
    # Diğer girişler
    inputs['PASTAL_TURU'] = st.selectbox("PASTAL_TURU", sorted(df['PASTAL_TURU'].astype(str).unique()))
    inputs['PASTAL_DETAYI'] = st.selectbox("PASTAL_DETAYI", sorted(df['PASTAL_DETAYI'].astype(str).unique()))

    # Asorti
    asorti_list = sorted(df_step2['ASORTI'].astype(str).unique())
    if not asorti_list: asorti_list = sorted(df['ASORTI'].astype(str).unique())
    inputs['ASORTI'] = st.selectbox("ASORTI", asorti_list)

    # Sayısal Girişler (Yan yana)
    c1, c2 = st.columns(2)
    inputs['KUMAS_ENI'] = c1.number_input("KUMAS_ENI", 90.0, 195.0, 146.0)
    inputs['KUMAS_CEKME_DEGERI_EN'] = c2.number_input("CEKME_EN", -13.0, 0.0, -1.5)
    
    c3, c4 = st.columns(2)
    inputs['KUMAS_CEKME_DEGERI_BOY'] = c3.number_input("CEKME_BOY", -22.0, 8.0, 1.5)
    inputs['ASORTI_SAYISI'] = c4.number_input("ASORTI_SAYISI", 5.0, 20.0, 10.0)

    inputs['PARCA_sAYISI'] = st.number_input("PARCA_SAYISI", 1.0, 30.0, 2.0)

# -----------------------------
# 3. HESAPLA
# -----------------------------
st.divider()
if st.button("HESAPLA", type="primary", use_container_width=True):
    if model:
        try:
            X_new = pd.DataFrame([inputs])
            # Sütun sıralaması eğitimdeki ile aynı olmalı
            cat_features = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT',
                            'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']
            
            X_new_pool = Pool(X_new, cat_features=cat_features)
            prediction = model.predict(X_new_pool)[0]
            
            st.success(f"🧵 Tahmini Birim Sarfiyat: **{prediction:.3f} mt**")
        except Exception as e:
            st.error(f"Hesaplama Hatası: {e}")
            st.warning("Veri setindeki sütun isimlerinin model ile uyumlu olduğundan emin olun.")
    else:
        st.error("Model yüklenemediği için hesaplama yapılamıyor.")
