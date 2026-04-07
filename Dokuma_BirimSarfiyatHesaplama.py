import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool
import os
from streamlit_gsheets import GSheetsConnection
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. AYARLAR VE BAĞLANTI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Sarfiyat Tahmini v3", layout="wide")

# Google Sheets URL (Sadeleştirilmiş URL)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1A2ayp13KH1EPJqKd7zCOJx9wtR3sxEf4cduLnOD0wCE/edit"

# Google Sheets Bağlantısı
conn = st.connection("gsheets", type=GSheetsConnection)

current_dir = os.path.dirname(os.path.abspath(__file__))
EXCEL_NAME = "YuklenenDokumaDosya262.xlsx"
MODEL_NAME = "Dokuma_BirimSarfiyatModel.cbm"

excel_path = os.path.join(current_dir, EXCEL_NAME)
model_path = os.path.join(current_dir, MODEL_NAME)

@st.cache_data
def load_data():
    if not os.path.exists(excel_path):
        return None
    try:
        df = pd.read_excel(excel_path)
        text_columns = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT', 'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        return df
    except:
        return None

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        return None
    try:
        model = CatBoostRegressor()
        model.load_model(model_path)
        return model
    except:
        return None

df = load_data()
model = load_model()

if df is None or model is None:
    st.error("❌ Kritik dosyalar (Excel veya Model) bulunamadı!")
    st.stop()

# -----------------------------------------------------------------------------
# 2. ARAYÜZ VE FİLTRELER
# -----------------------------------------------------------------------------
st.title("🎯 Akıllı Birim Sarfiyat Tahmini")

inputs = {}
st.markdown("---")
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📌 Model Seçimi")
    dept_list = sorted(df['DEPARTMAN'].unique())
    secilen_dept = st.selectbox("DEPARTMAN", dept_list)
    inputs['DEPARTMAN'] = secilen_dept
    
    df_s1 = df[df['DEPARTMAN'] == secilen_dept]
    inputs['MODEL_TURU'] = st.selectbox("MODEL_TURU", sorted(df_s1['MODEL_TURU'].unique()))
    
    df_s2 = df_s1[df_s1['MODEL_TURU'] == inputs['MODEL_TURU']]
    inputs['MODEL_DETAYI'] = st.selectbox("MODEL_DETAYI", sorted(df_s2['MODEL_DETAYI'].unique()))
    
    df_s3 = df_s2[df_s2['MODEL_DETAYI'] == inputs['MODEL_DETAYI']]
    inputs['FIT'] = st.selectbox("FIT", sorted(df_s3['FIT'].unique()))

with col_right:
    st.subheader("⚙️ Teknik Detaylar")
    inputs['ASORTI'] = st.selectbox("ASORTI", sorted(df['ASORTI'].unique()))
    inputs['PASTAL_TURU'] = st.selectbox("PASTAL_TURU", sorted(df['PASTAL_TURU'].unique()))
    inputs['PASTAL_DETAYI'] = st.selectbox("PASTAL_DETAYI", sorted(df['PASTAL_DETAYI'].unique(), reverse=True))

    c1, c2 = st.columns(2)
    inputs['KUMAS_ENI'] = c1.number_input("KUMAS_ENI", value=152.0)
    inputs['KUMAS_CEKME_DEGERI_EN'] = c2.number_input("CEKME_EN", value=-3.0)
    
    c3, c4 = st.columns(2)
    inputs['KUMAS_CEKME_DEGERI_BOY'] = c3.number_input("CEKME_BOY", value=-3.0)
    inputs['ASORTI_SAYISI'] = c4.number_input("ASORTI_SAYISI", value=10.0)
    inputs['PARCA_SAYISI'] = st.number_input("PARCA_SAYISI", value=18.0)

# -----------------------------------------------------------------------------
# 3. HESAPLAMA VE GÜVENLİ KAYIT
# -----------------------------------------------------------------------------
st.divider()

if st.button("HESAPLA VE E-TABLOYA KAYDET", type="primary", use_container_width=True):
    try:
        # 1. Tahmin Yap
        X_new = pd.DataFrame([inputs])
        # Modelin beklediği sütun sırasına sok
        X_new = X_new[model.feature_names_]
        
        cat_features = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT', 'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']
        pool = Pool(X_new, cat_features=cat_features)
        prediction = model.predict(pool)[0]
        
        st.success(f"📏 Tahmini Birim Sarfiyat: **{prediction:.4f} mt**")

        # 2. Kaydetme İşlemi (Hata Yakalamalı)
        with st.spinner("Veriler E-Tabloya gönderiliyor..."):
            try:
                # Mevcut verileri çek (Erişim hatası alırsak boş tablo oluşturur)
                try:
                    history_df = conn.read(spreadsheet=SHEET_URL, ttl=0)
                except:
                    history_df = pd.DataFrame()

                # Yeni satırı oluştur
                new_data = X_new.copy()
                new_data['Tahmin_Sarfiyat'] = round(prediction, 4)
                new_data['Kayit_Tarihi'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Mevcut tabloyla birleştir
                updated_df = pd.concat([history_df, new_data], ignore_index=True)

                # Google Sheets'e geri gönder
                conn.update(spreadsheet=SHEET_URL, data=updated_df)
                st.toast("Veri başarıyla kaydedildi! ✅")
            except Exception as write_error:
                st.error(f"Kayıt Hatası: {write_error}")
                st.info("Lütfen Google Sheet dosyanızın 'Herkes (Düzenleyici)' olarak paylaşıldığından emin olun.")

    except Exception as e:
        st.error(f"Hesaplama hatası: {e}")

# -----------------------------------------------------------------------------
# 4. GEÇMİŞ GÖRÜNTÜLEME
# -----------------------------------------------------------------------------
st.divider()
with st.expander("📂 Tahmin Arşivini Gör"):
    try:
        # ttl=0 önbelleği temizler, her açışta güncel veriyi çeker
        archived_data = conn.read(spreadsheet=SHEET_URL, ttl=0)
        st.dataframe(archived_data, use_container_width=True)
    except:
        st.warning("Arşiv şu an yüklenemiyor. Erişim yetkisi veya internet sorunu olabilir.")
