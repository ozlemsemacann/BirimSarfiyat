import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool
import os

# -----------------------------
# 1. DOSYA YOLLARINI AYARLA (En Kritik Kısım)
# -----------------------------
# Bu kod, .py dosyasının olduğu klasörün tam adresini bulur.
current_dir = os.path.dirname(os.path.abspath(__file__))

# Dosya adlarını buraya tam olarak yazıyoruz (GitHub'daki adıyla aynı olmalı)
DATA_FILENAME = "_YuklenenDokumaDosya30.1.xlsx"
MODEL_FILENAME = "Dokuma_BirimSarfiyatModel.cbm"

# Tam dosya yollarını oluşturuyoruz
data_path = os.path.join(current_dir, DATA_FILENAME)
model_path = os.path.join(current_dir, MODEL_FILENAME)

# -----------------------------
# 2. VERİ VE MODEL YÜKLEME
# -----------------------------
@st.cache_data
def load_data():
    if not os.path.exists(data_path):
        st.error(f"❌ Veri dosyası bulunamadı! Aranan yol: {data_path}")
        return pd.DataFrame() # Boş dataframe dön
    
    # Excel dosyasını oku
    df = pd.read_excel(data_path)
    return df

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    if os.path.exists(model_path):
        model.load_model(model_path)
        return model
    else:
        st.error(f"❌ Model dosyası bulunamadı! Aranan yol: {model_path}")
        return None

# Yüklemeleri başlat
df = load_data()
model = load_model()

# -----------------------------
# 3. ARAYÜZ VE FİLTRELEME
# -----------------------------
st.title("🧵 Akıllı Birim Sarfiyat Tahmini")

# Eğer veri yüklenemediyse programı durdur
if df.empty:
    st.warning("Veri seti okunamadığı için işlem yapılamıyor.")
    st.stop()

inputs = {}
st.markdown("---")

# --- BASAMAKLI FİLTRELEME ---

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

# DİĞER GİRİŞLER
inputs['PASTAL_TURU'] = st.selectbox("PASTAL_TURU", sorted(df['PASTAL_TURU'].astype(str).unique()))
inputs['PASTAL_DETAYI'] = st.selectbox("PASTAL_DETAYI", sorted(df['PASTAL_DETAYI'].astype(str).unique()))

# Asorti
asorti_list = sorted(df_step2['ASORTI'].astype(str).unique())
if not asorti_list: asorti_list = sorted(df['ASORTI'].astype(str).unique())
inputs['ASORTI'] = st.selectbox("ASORTI", asorti_list)

# Sayısal Girişler
col1, col2 = st.columns(2)
with col1:
    inputs['KUMAS_ENI'] = st.number_input("KUMAS_ENI", 90.0, 195.0, 146.0)
    inputs['KUMAS_CEKME_DEGERI_EN'] = st.number_input("CEKME_EN", -13.0, 0.0, -1.5)
with col2:
    inputs['KUMAS_CEKME_DEGERI_BOY'] = st.number_input("CEKME_BOY", -22.0, 8.0, 1.5)
    inputs['ASORTI_SAYISI'] = st.number_input("ASORTI_SAYISI", 5.0, 20.0, 10.0)

inputs['PARCA_sAYISI'] = st.number_input("PARCA_SAYISI", 1.0, 30.0, 2.0)

# -----------------------------
# 4. TAHMİN
# -----------------------------
if st.button("HESAPLA", type="primary"):
    if model:
        X_new = pd.DataFrame([inputs])
        cat_features = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT',
                        'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']
        try:
            X_new_pool = Pool(X_new, cat_features=cat_features)
            prediction = model.predict(X_new_pool)[0]
            st.success(f"🔮 Tahmini Birim Sarfiyat: **{prediction:.3f} mt**")
        except Exception as e:
            st.error(f"Hata: {e}")
    else:
        st.error("Model dosyası yüklenemedi.")
