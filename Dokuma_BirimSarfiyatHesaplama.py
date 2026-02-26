import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool
import os

# -----------------------------------------------------------------------------
# 1. AYARLAR VE OTOMAT?K DOSYA BULMA
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Sarfiyat Tahmini", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))

# Dosya adlar?
EXCEL_NAME = "YuklenenDokumaDosya172.xlsx"
MODEL_NAME = "Dokuma_BirimSarfiyatModel.cbm"

excel_path = os.path.join(current_dir, EXCEL_NAME)
model_path = os.path.join(current_dir, MODEL_NAME)

@st.cache_data
def load_data():
    if not os.path.exists(excel_path):
        st.error(f"? Excel dosyas? bulunamad?! Aranan dosya ad?: {EXCEL_NAME}")
        return None
    try:
        df = pd.read_excel(excel_path)
        
        # --- VER? TEM?ZLEME (DATA CLEANING) ---
        # Buyuk/Kucuk harf ve bo?luk temizli?i
        text_columns = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT', 'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                
        return df
    except Exception as e:
        st.error(f"Excel okuma hatas?: {e}")
        return None

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"? Model dosyas? bulunamad?! ({MODEL_NAME})")
        return None
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model

df = load_data()
model = load_model()

if df is None:
    st.stop()

# -----------------------------------------------------------------------------
# 2. TAM BA?IMLI (CASCADING) F?LTRELEME Z?NC?R?
# -----------------------------------------------------------------------------
st.title("?? Ak?ll? Birim Sarfiyat Tahmini")
st.success(f"? '{EXCEL_NAME}' dosyas? ba?ar?yla yuklendi.")

inputs = {}
st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("?? Model Secimi")

    # 1. DEPARTMAN
    dept_list = sorted(df['DEPARTMAN'].unique())
    secilen_dept = st.selectbox("DEPARTMAN", dept_list)
    inputs['DEPARTMAN'] = secilen_dept
    
    # F?LTRE 1
    df_step1 = df[df['DEPARTMAN'] == secilen_dept]

    # 2. MODEL TURU
    tur_list = sorted(df_step1['MODEL_TURU'].unique())
    secilen_tur = st.selectbox("MODEL_TURU", tur_list)
    inputs['MODEL_TURU'] = secilen_tur
    
    # F?LTRE 2
    df_step2 = df_step1[df_step1['MODEL_TURU'] == secilen_tur]

    # 3. MODEL DETAYI
    detay_list = sorted(df_step2['MODEL_DETAYI'].unique())
    secilen_detay = st.selectbox("MODEL_DETAYI", detay_list)
    inputs['MODEL_DETAYI'] = secilen_detay
    
    # F?LTRE 3
    df_step3 = df_step2[df_step2['MODEL_DETAYI'] == secilen_detay]

    # 4. FIT
    fit_list = sorted(df_step3['FIT'].unique())
    secilen_fit = st.selectbox("FIT", fit_list)
    inputs['FIT'] = secilen_fit

    # F?LTRE 4
    df_step4 = df_step3[df_step3['FIT'] == secilen_fit]

with col_right:
    st.subheader("?? Teknik Detaylar")

    # 5. ASORTI
    asorti_list = sorted(df_step4['ASORTI'].unique())
    if not asorti_list:
        asorti_list = sorted(df['ASORTI'].unique())
    inputs['ASORTI'] = st.selectbox("ASORTI", asorti_list)

    # Di?er Sabit Giri?ler
    inputs['PASTAL_TURU'] = st.selectbox("PASTAL_TURU", sorted(df['PASTAL_TURU'].unique()))
    
    # --- BURASI DE???T? ---
    # reverse=True eklenerek Buyukten Kucu?e (veya Z-A) s?ralama yap?ld?
    inputs['PASTAL_DETAYI'] = st.selectbox("PASTAL_DETAYI", sorted(df['PASTAL_DETAYI'].unique(), reverse=True))

    # Say?sal De?erler
    c1, c2 = st.columns(2)
    inputs['KUMAS_ENI'] = c1.number_input("KUMAS_ENI", 90.0, 195.0, 152.0)
    inputs['KUMAS_CEKME_DEGERI_EN'] = c2.number_input("CEKME_EN", -13.0, 0.0, -3.0)
    
    c3, c4 = st.columns(2)
    inputs['KUMAS_CEKME_DEGERI_BOY'] = c3.number_input("CEKME_BOY", -22.0, 8.0, -3.0)
    inputs['ASORTI_SAYISI'] = c4.number_input("ASORTI_SAYISI", 5.0, 20.0, 10.0)

    # PARCA_SAYISI
    inputs['PARCA_SAYISI'] = st.number_input("PARCA_SAYISI", 1.0, 30.0, 18.0)

# -----------------------------------------------------------------------------
# 3. HESAPLAMA
# -----------------------------------------------------------------------------
st.divider()

if st.button("HESAPLA", type="primary", use_container_width=True):
    if model:
        try:
            X_new = pd.DataFrame([inputs])
            
            # Otomatik S?ralama
            beklenen_siralama = model.feature_names_
            X_new = X_new[beklenen_siralama]

            cat_features = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT',
                            'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']
            
            X_new_pool = Pool(X_new, cat_features=cat_features)
            prediction = model.predict(X_new_pool)[0]
            
            st.success(f"?? Tahmini Birim Sarfiyat: **{prediction:.3f} mt**")
            
        except KeyError as e:
            st.error(f"Sutun Hatas?: {e}")
        except Exception as e:
            st.error(f"Hesaplama Hatas?: {e}")
    else:
        st.error("Model yuklenemedi.")