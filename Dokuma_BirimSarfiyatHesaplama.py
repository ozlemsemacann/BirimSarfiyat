import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool
import os

# -----------------------------
# 1. AYARLAR VE MODEL YÜKLEME
# -----------------------------
st.set_page_config(page_title="Sarfiyat Tahmini", layout="wide")

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    try:
        model.load_model("Dokuma_BirimSarfiyatModel.cbm")
        return model
    except Exception as e:
        st.error(f"Model dosyası bulunamadı: {e}")
        return None

model = load_model()

# -----------------------------
# 2. VERİ SETİNİ YÜKLEME (İlişkileri kurmak için şart)
# -----------------------------
st.title("🧵 Akıllı Birim Sarfiyat Tahmini")

# NOT: Buraya eğitimde kullandığın Excel dosyasının tam adını yazmalısın.
# Eğer dosya kodun yanındaysa direkt adını yazman yeterli.
EXCEL_DOSYA_ADI = "_YuklenenDokumaDosya30.1.xlsx" 

@st.cache_data
def load_data():
    # Önce klasörde dosya var mı kontrol edelim
    if os.path.exists(EXCEL_DOSYA_ADI):
        return pd.read_excel(EXCEL_DOSYA_ADI)
    else:
        return None

df = load_data()

# Eğer dosya klasörde yoksa manuel yükleme isteyelim
if df is None:
    st.warning(f"⚠️ '{EXCEL_DOSYA_ADI}' dosyası bulunamadı. İlişkileri kurmak için lütfen dosyayı yükleyin.")
    uploaded_file = st.file_uploader("Veri Setini Yükle (Excel)", type=["xlsx", "xls"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
    else:
        st.stop() # Dosya yoksa durdur

# -----------------------------
# 3. BASAMAKLI FİLTRELEME (CASCADING FILTERS)
# -----------------------------
st.markdown("---")
st.subheader("📋 Model Özellikleri")

inputs = {}

# 1. ADIM: DEPARTMAN
# Tüm departmanları getir
dept_list = sorted(df['DEPARTMAN'].astype(str).unique())
secilen_dept = st.selectbox("DEPARTMAN", dept_list)
inputs['DEPARTMAN'] = secilen_dept

# VERİYİ FİLTRELE: Sadece seçilen departmana ait satırları al
df_step1 = df[df['DEPARTMAN'] == secilen_dept]


# 2. ADIM: MODEL TÜRÜ
# Listeyi filtrelenmiş (df_step1) veriden çekiyoruz.
tur_list = sorted(df_step1['MODEL_TURU'].astype(str).unique())
secilen_tur = st.selectbox("MODEL_TURU", tur_list)
inputs['MODEL_TURU'] = secilen_tur

# VERİYİ TEKRAR FİLTRELE: Sadece seçilen TÜR'e ait satırları al
df_step2 = df_step1[df_step1['MODEL_TURU'] == secilen_tur]


# 3. ADIM: MODEL DETAYI
# Listeyi df_step2'den çekiyoruz. Böylece seçilen türde olmayan detaylar gelmez.
detay_list = sorted(df_step2['MODEL_DETAYI'].astype(str).unique())
secilen_detay = st.selectbox("MODEL_DETAYI", detay_list)
inputs['MODEL_DETAYI'] = secilen_detay

# VERİYİ TEKRAR FİLTRELE
df_step3 = df_step2[df_step2['MODEL_DETAYI'] == secilen_detay]


# 4. ADIM: FIT
# Sadece yukarıdaki kombinasyona uygun FIT'ler gelir.
fit_list = sorted(df_step3['FIT'].astype(str).unique())
secilen_fit = st.selectbox("FIT", fit_list)
inputs['FIT'] = secilen_fit

# -----------------------------
# 4. DİĞER GİRİŞLER (Sabit veya Bağımsız)
# -----------------------------
st.subheader("⚙️ Teknik Detaylar")

col1, col2 = st.columns(2)

with col1:
    # Bu listeleri de istersen veri setinden çekebilirsin: sorted(df['PASTAL_TURU'].unique())
    inputs['PASTAL_TURU'] = st.selectbox("PASTAL_TURU", ['ANA_BEDEN','ASTAR','FILE','TELA','PAT_TELASI'])
    inputs['PASTAL_DETAYI'] = st.selectbox("PASTAL_DETAYI", ['YONLU','YONSUZ'])
    
    # Asorti çok uzun olduğu için veri setinden çekmek daha mantıklı,
    # ama model türüne göre değişiyorsa df_step2'den de çekebilirsin.
    asorti_list = sorted(df['ASORTI'].astype(str).unique())
    inputs['ASORTI'] = st.selectbox("ASORTI", asorti_list)

with col2:
    inputs['KUMAS_ENI'] = st.number_input("KUMAS_ENI", 90.0, 195.0, 152.0)
    inputs['KUMAS_CEKME_DEGERI_EN'] = st.number_input("CEKME_EN", -13.0, 0.0, -3.0)
    inputs['KUMAS_CEKME_DEGERI_BOY'] = st.number_input("CEKME_BOY", -22.0, 8.0, -3.0)

col3, col4 = st.columns(2)
with col3:
    inputs['ASORTI_SAYISI'] = st.number_input("ASORTI_SAYISI", 5.0, 20.0, 10.0)
with col4:
    inputs['PARCA_SAYISI'] = st.number_input("PARCA_SAYISI", 1.0, 30.0, 18.0) # Senin kodunda PARCA_SAYISI (büyük S) dikkat et

# -----------------------------
# 5. TAHMİN BUTONU
# -----------------------------
st.divider()

if st.button("Tüketim Tahmini Yap", type="primary", use_container_width=True):
    if model:
        # DataFrame oluştur
        X_new = pd.DataFrame([inputs])

        # CatBoost feature sırası ve isimleri çok önemlidir.
        # Eğitimde kullandığın isimlerle birebir aynı olmalı.
        cat_features = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT',
                        'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']

        try:
            X_new_pool = Pool(X_new, cat_features=cat_features)
            prediction = model.predict(X_new_pool)[0]
            
            st.success(f"🧵 Tahmini Birim Sarfiyat: **{prediction:.3f} mt**")
            
            # Seçim Özetini Göster
            st.info(f"Seçim: {inputs['MODEL_TURU']} > {inputs['MODEL_DETAYI']} > {inputs['FIT']}")
            
        except Exception as e:
            st.error(f"Tahmin hatası: {e}")
            st.warning("Veri setindeki sütun isimleri ile modelin beklediği isimler uyuşmuyor olabilir.")
    else:
        st.error("Model yüklenemediği için işlem yapılamıyor.")
