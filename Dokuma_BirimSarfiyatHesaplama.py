import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool

# -----------------------------
# 1. VERİ VE MODEL YÜKLEME
# -----------------------------
@st.cache_data
def load_data():
    # BURASI ÖNEMLİ: Kendi veri setinin adını buraya yazmalısın.
    # Bu dosya, modelin eğitildiği orijinal veridir.
    # Eğer dosyan CSV ise: pd.read_csv("dosya_adi.csv") kullan.
    try:
        df = pd.read_excel("_YuklenenDokumaDosya30.1.xlsx") # Dosya adını kendine göre düzelt
        return df
    except Exception as e:
        st.error(f"Veri seti yüklenemedi! Lütfen dosya adını kontrol et. Hata: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    try:
        model.load_model("Dokuma_BirimSarfiyatModel.cbm")
        return model
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return None

df = load_data()
model = load_model()

# -----------------------------
# 2. ARAYÜZ VE BASAMAKLI FİLTRELEME
# -----------------------------
st.title("🧵 Akıllı Birim Sarfiyat Tahmini")

if df.empty:
    st.warning("Veri seti yüklenemediği için seçenekler oluşturulamıyor.")
    st.stop()

inputs = {}

# --- ŞELALE FİLTRELEME BAŞLIYOR ---
# Mantık: Her seçimden sonra DataFrame'i filtreleyip bir sonraki aşamaya aktarıyoruz.

# 1. BASAMAK: DEPARTMAN
# Tüm veri setindeki benzersiz departmanlar
dept_list = sorted(df['DEPARTMAN'].astype(str).unique())
secilen_dept = st.selectbox("DEPARTMAN", dept_list)
inputs['DEPARTMAN'] = secilen_dept

# Veriyi departmana göre daraltıyoruz
df_step1 = df[df['DEPARTMAN'] == secilen_dept]


# 2. BASAMAK: MODEL TÜRÜ
# Sadece seçilen departmanda bulunan model türleri gelir
tur_list = sorted(df_step1['MODEL_TURU'].astype(str).unique())
secilen_tur = st.selectbox("MODEL_TURU", tur_list)
inputs['MODEL_TURU'] = secilen_tur

# Veriyi türe göre daraltıyoruz
df_step2 = df_step1[df_step1['MODEL_TURU'] == secilen_tur]


# 3. BASAMAK: MODEL DETAYI
# Sadece seçilen türdeki detaylar gelir (Örn: Bermuda seçildiyse Bermuda detayları)
detay_list = sorted(df_step2['MODEL_DETAYI'].astype(str).unique())
secilen_detay = st.selectbox("MODEL_DETAYI", detay_list)
inputs['MODEL_DETAYI'] = secilen_detay

# Veriyi detaya göre daraltıyoruz
df_step3 = df_step2[df_step2['MODEL_DETAYI'] == secilen_detay]


# 4. BASAMAK: FIT
# Sadece yukarıdaki kombinasyona uygun FIT'ler gelir
fit_list = sorted(df_step3['FIT'].astype(str).unique())
secilen_fit = st.selectbox("FIT", fit_list)
inputs['FIT'] = secilen_fit

# Veriyi FIT'e göre daraltıyoruz
df_step4 = df_step3[df_step3['FIT'] == secilen_fit]


# --- TEKNİK DETAYLAR (İsteğe Bağlı Filtreleme) ---
# Burada iki seçeneğin var:
# A) Bu özellikleri de yukarıdakilere göre kısıtla (Sadece daha önce üretilmiş kombinasyonlar görünür).
# B) Bunları serbest bırak (Tüm pastal türleri görünsün).
# Aşağıda "A" seçeneğini uyguladım ("her kriter" dediğin için).

# 5. PASTAL TURU (Filtrelenmiş veriden)
pastal_turu_list = sorted(df_step4['PASTAL_TURU'].astype(str).unique())
# Eğer liste boş kalırsa (daha önce bu kombinasyon hiç yapılmadıysa) genel listeden getir
if not pastal_turu_list: 
    pastal_turu_list = sorted(df['PASTAL_TURU'].astype(str).unique())
    
secilen_pastal_turu = st.selectbox("PASTAL_TURU", pastal_turu_list)
inputs['PASTAL_TURU'] = secilen_pastal_turu

# 6. PASTAL DETAYI
# Genellikle pastal türüne bağımlı olmadığı için genel listeden de çekebilirsin ama
# veri tutarlılığı için filtrelenmiş veriden çekiyoruz.
pastal_detay_list = sorted(df['PASTAL_DETAYI'].astype(str).unique()) # Bunu genel tuttum çok kısıtlamasın diye
inputs['PASTAL_DETAYI'] = st.selectbox("PASTAL_DETAYI", pastal_detay_list)

# 7. ASORTI
# Modele göre asorti değişebileceği için filtrelenmiş veriden çekmek mantıklı
asorti_list = sorted(df_step2['ASORTI'].astype(str).unique()) # Model Türü bazlı filtreleme yeterli
inputs['ASORTI'] = st.selectbox("ASORTI", asorti_list)


# --- SAYISAL GİRİŞLER (Filtreleme Yok, Kullanıcı Girer) ---
col1, col2 = st.columns(2)
with col1:
    inputs['KUMAS_ENI'] = st.number_input("KUMAS_ENI", 90.0, 195.0, 146.0)
    inputs['KUMAS_CEKME_DEGERI_EN'] = st.number_input("CEKME_EN", -13.0, 0.0, -1.5)
with col2:
    inputs['KUMAS_CEKME_DEGERI_BOY'] = st.number_input("CEKME_BOY", -22.0, 8.0, 1.5)
    inputs['ASORTI_SAYISI'] = st.number_input("ASORTI_SAYISI", 5.0, 20.0, 10.0)

inputs['PARCA_sAYISI'] = st.number_input("PARCA_SAYISI", 1.0, 30.0, 2.0)


# -----------------------------
# 3. TAHMİN İŞLEMİ
# -----------------------------
if st.button("HESAPLA", type="primary"):
    if model:
        # Girdi DataFrame'i oluştur
        X_new = pd.DataFrame([inputs])
        
        # Modelin beklediği sütun sırası ve isimleri (Eğitimdeki ile birebir aynı olmalı)
        # Not: Excel sütun başlıklarınla burası tutmalı.
        cat_features = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT',
                        'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']

        try:
            X_new_pool = Pool(X_new, cat_features=cat_features)
            prediction = model.predict(X_new_pool)[0]
            
            st.divider()
            st.success(f"🧵 Tahmini Birim Sarfiyat: **{prediction:.3f} mt**")
            
            # Seçilen özellikleri özet geçelim
            st.info(f"Seçim: {inputs['MODEL_TURU']} - {inputs['MODEL_DETAYI']} - {inputs['FIT']}")
            
        except Exception as e:
            st.error("Tahmin hatası! Sütun isimlerini veya veri tiplerini kontrol edin.")
            st.code(e)
    else:
        st.error("Model yüklü değil.")