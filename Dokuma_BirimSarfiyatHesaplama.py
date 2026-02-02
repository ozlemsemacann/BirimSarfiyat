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

# DOSYA ADI (Senin istediğin gibi)
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
st.success(f"✅ Modeli önceden eğittik ve yükledik. Şimdi değerleri gir, tahmini al!")

inputs = {}
st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Model Seçimi")

    # 1. DEPARTMAN
    dept_list = sorted(df['DEPARTMAN'].astype(str).unique())
    secilen_dept = st.selectbox("DEPARTMAN", dept_list)
    inputs['DEPARTMAN'] = secilen_dept
    
    # FİLTRE 1
    df_step1 = df[df['DEPARTMAN'] == secilen_dept]

    # 2. MODEL TÜRÜ
    tur_list = sorted(df_step1['MODEL_TURU'].astype(str).unique())
    secilen_tur = st.selectbox("MODEL_TURU", tur_list)
    inputs['MODEL_TURU'] = secilen_tur
    
    # FİLTRE 2
    df_step2 = df_step1[df_step1['MODEL_TURU'] == secilen_tur]

    # 3. MODEL DETAYI
    detay_list = sorted(df_step2['MODEL_DETAYI'].astype(str).unique())
    secilen_detay = st.selectbox("MODEL_DETAYI", detay_list)
    inputs['MODEL_DETAYI'] = secilen_detay
    
    # FİLTRE 3
    df_step3 = df_step2[df_step2['MODEL_DETAYI'] == secilen_detay]

    # 4. FIT
    fit_list = sorted(df_step3['FIT'].astype(str).unique())
    secilen_fit = st.selectbox("FIT", fit_list)
    inputs['FIT'] = secilen_fit

    # FİLTRE 4 (Hata veren yer burasıydı, düzeltildi)
    df_step4 = df_step3[df_step3['FIT'] == secilen_fit]

with col_right:
    st.subheader("⚙️ Teknik Detaylar")

    # 5. ASORTI (BAĞLI FİLTRE)
    # Listeyi en son filtrelenen df_step4'ten çekiyoruz
    asorti_list = sorted(df_step4['ASORTI'].astype(str).unique())
    
    # Boş kalırsa önlem
    if not asorti_list:
        asorti_list = sorted(df['ASORTI'].astype(str).unique())
        
    inputs['ASORTI'] = st.selectbox("ASORTI", asorti_list)

    # Diğer Sabit Girişler
    inputs['PASTAL_TURU'] = st.selectbox("PASTAL_TURU", sorted(df['PASTAL_TURU'].astype(str).unique()))
    inputs['PASTAL_DETAYI'] = st.selectbox("PASTAL_DETAYI", sorted(df['PASTAL_DETAYI'].astype(str).unique()))

    # Sayısal Değerler
    c1, c2 = st.columns(2)
    inputs['KUMAS_ENI'] = c1.number_input("KUMAS_ENI", 90.0, 195.0, 152.0)
    inputs['KUMAS_CEKME_DEGERI_EN'] = c2.number_input("CEKME_EN", -13.0, 0.0, -3.0)
    
    c3, c4 = st.columns(2)
    inputs['KUMAS_CEKME_DEGERI_BOY'] = c3.number_input("CEKME_BOY", -22.0, 8.0, -3.0)
    inputs['ASORTI_SAYISI'] = c4.number_input("ASORTI_SAYISI", 5.0, 20.0, 10.0)

    # PARCA_SAYISI (Büyük harf)
    inputs['PARCA_SAYISI'] = st.number_input("PARCA_SAYISI", 1.0, 30.0, 18.0)

# -----------------------------------------------------------------------------
# 3. HESAPLAMA
# -----------------------------------------------------------------------------
st.divider()

if st.button("HESAPLA", type="primary", use_container_width=True):
    if model:
        try:
            X_new = pd.DataFrame([inputs])
            
            # Otomatik Sıralama
            beklenen_siralama = model.feature_names_
            X_new = X_new[beklenen_siralama]

            cat_features = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT',
                            'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']
            
            X_new_pool = Pool(X_new, cat_features=cat_features)
            prediction = model.predict(X_new_pool)[0]
            
            st.success(f"🧵 Tahmini Birim Sarfiyat: **{prediction:.3f} mt**")
            
        except KeyError as e:
            st.error(f"Sütun Hatası: {e}")
        except Exception as e:
            st.error(f"Hesaplama Hatası: {e}")
    else:
        st.error("Model yüklenemedi.")

