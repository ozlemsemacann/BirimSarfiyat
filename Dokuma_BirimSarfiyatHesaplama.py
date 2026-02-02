import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool
import os

# -----------------------------------------------------------------------------
# 1. AYARLAR VE OTOMATİK DOSYA BULMA
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Sarfiyat Tahmini", layout="wide")

# Dosya yollarını dinamik olarak bul
current_dir = os.path.dirname(os.path.abspath(__file__))

# --- DOSYA ADI BURADA GÜNCELLENDİ ---
EXCEL_NAME = "Yuklenecek.xlsx" 
MODEL_NAME = "Dokuma_BirimSarfiyatModel.cbm"

excel_path = os.path.join(current_dir, EXCEL_NAME)
model_path = os.path.join(current_dir, MODEL_NAME)

@st.cache_data
def load_data():
    if not os.path.exists(excel_path):
        st.error(f"❌ Excel dosyası bulunamadı! Aranan dosya adı: {EXCEL_NAME}")
        return None
    try:
        df = pd.read_excel(excel_path)
        return df
    except Exception as e:
        st.error(f"Excel okuma hatası: {e}")
        return None

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası bulunamadı! ({MODEL_NAME})")
        return None
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model

df = load_data()
model = load_model()

if df is None:
    st.stop()

# -----------------------------------------------------------------------------
# 2. TAM BAĞIMLI (CASCADING) FİLTRELEME ZİNCİRİ
# -----------------------------------------------------------------------------
st.title("🧵 Akıllı Birim Sarfiyat Tahmini")
st.success(f"✅ '{EXCEL_NAME}' dosyası başarıyla yüklendi.")

inputs = {}
st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Model Seçimi")

    # 1. DEPARTMAN
    dept_list = sorted(df['DEPARTMAN'].astype(str).unique())
    secilen_dept = st.selectbox("DEPARTMAN", dept_list)
    inputs['DEPARTMAN'] = secilen_dept
    
    # FİLTRE 1: Departmana göre daralt
    df_step1 = df[df['DEPARTMAN'] == secilen_dept]

    # 2. MODEL TÜRÜ
    tur_list = sorted(df_step1['MODEL_TURU'].astype(str).unique())
    secilen_tur = st.selectbox("MODEL_TURU", tur_list)
    inputs['MODEL_TURU'] = secilen_tur
    
    # FİLTRE 2: Türe göre daralt
    df_step2 = df_step1[df_step1['MODEL_TURU'] == secilen_tur]

    # 3. MODEL DETAYI
    detay_list = sorted(df_step2['MODEL_DETAYI'].astype(str).unique())
    secilen_detay = st.selectbox("MODEL_DETAYI", detay_list)
    inputs['MODEL_DETAYI'] = secilen_detay
    
    # FİLTRE 3: Detaya göre daralt
    df_step3 = df_step2[df_step2['MODEL_DETAYI'] == secilen_detay]

    # 4. FIT
    fit_list = sorted(df_step3['FIT'].astype(str).unique())
    secilen_fit = st.selectbox("FIT", fit_list)
    inputs['FIT'] = secilen_fit

    # FİLTRE 4: Fit'e göre daralt (ASORTİ İÇİN HAZIRLIK)
    df_step4 = df_step3[df_step3['FIT
