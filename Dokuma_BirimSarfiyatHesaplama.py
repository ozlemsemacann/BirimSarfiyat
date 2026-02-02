import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool
import os

# -----------------------------------------------------------------------------
# 1. DOSYA YOLLARINI DİNAMİK OLARAK BULMA (EN ÖNEMLİ KISIM)
# -----------------------------------------------------------------------------
# Bu satır, şu an çalışan kod dosyasının (app.py) nerede olduğunu bulur.
current_dir = os.path.dirname(os.path.abspath(__file__))

# Dosya isimlerini buraya tam olarak yazıyoruz (GitHub'dakiyle birebir aynı olmalı)
EXCEL_NAME = "Yuklenecek.xlsx"
MODEL_NAME = "Dokuma_BirimSarfiyatModel.cbm"

# Tam dosya yollarını oluşturuyoruz (Klasör Yolu + Dosya Adı)
excel_path = os.path.join(current_dir, EXCEL_NAME)
model_path = os.path.join(current_dir, MODEL_NAME)

# -----------------------------------------------------------------------------
# 2. VERİ VE MODEL YÜKLEME FONKSİYONLARI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Sarfiyat Tahmini", layout="wide")
st.title("🧵 Akıllı Birim Sarfiyat Tahmini")

@st.cache_data
def load_data():
    # Dosya var mı kontrol et
    if not os.path.exists(excel_path):
        st.error(f"❌ Excel dosyası bulunamadı!")
        st.write(f"🔍 Aranan Yol: `{excel_path}`")
        st.write("📂 Klasördeki mevcut dosyalar:")
        st.code(os.listdir(current_dir)) # Klasördeki dosyaları listeler (Hata ayıklama için)
        return None
    
    # Varsa oku
    try:
        df = pd.read_excel(excel_path)
        return df
    except Exception as e:
        st.error(f"Dosya bozuk veya okunamıyor: {e}")
        return None

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası bulunamadı! ({MODEL_NAME})")
        return None
        
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model

# Verileri Yükle
df = load_data()
model = load_model()

# Eğer veri yüklenemediyse uygulamayı durdur
if df is None:
    st.stop()

# -----------------------------------------------------------------------------
# 3. BASAMAKLI FİLTRELEME (CASCADING FILTERS)
# -----------------------------------------------------------------------------
inputs = {}
st.markdown("---")
st.success("✅ Veri seti ve Model başarıyla otomatik yüklendi.")

col_sol, col_sag = st.columns([1, 1])

with col_sol:
    st.subheader("📋 Model Seçimi")

    # 1. DEPARTMAN
    dept_list = sorted(df['DEPARTMAN'].astype(str).unique())
    secilen_dept = st.selectbox("DEPARTMAN", dept_list)
    inputs['DEPARTMAN'] = secilen_dept
    
    # Filtre 1
    df_step1 = df[df['DEPARTMAN'] == secilen_dept]

    # 2. MODEL TÜRÜ (Filtrelenmiş listeden gelir)
    tur_list = sorted(df_step1['MODEL_TURU'].astype(str).unique())
    secilen_tur = st.selectbox("MODEL_TURU", tur_list)
    inputs['MODEL_TURU'] = secilen_tur
    
    # Filtre 2
    df_step2 = df_step1[df_step1['MODEL_TURU'] == secilen_tur]

    # 3. MODEL DETAYI (Filtrelenmiş listeden gelir)
    detay_list = sorted(df_step2['MODEL_DETAYI'].astype(str).unique())
    secilen_detay = st.selectbox("MODEL_DETAYI", detay_list)
    inputs['MODEL_DETAYI'] = secilen_detay
    
    # Filtre 3
    df_step3 = df_step2[df_step2['MODEL_DETAYI'] == secilen_detay]

    # 4. FIT (Filtrelenmiş listeden gelir)
    fit_list = sorted(df_step3['FIT'].astype(str).unique())
    secilen_fit = st.selectbox("FIT", fit_list)
    inputs['FIT'] = secilen_fit

with col_sag:
    st.subheader("⚙️ Diğer Özellikler")

    # Sabit Listeler (Veri setinden çekiliyor)
    inputs['PASTAL_TURU'] = st.selectbox("PASTAL_TURU", sorted(df['PASTAL_TURU'].astype(str).unique()))
    inputs['PASTAL_DETAYI'] = st.selectbox("PASTAL_DETAYI", sorted(df['PASTAL_DETAYI'].astype(str).unique()))
    
    # Asorti (Model Türüne göre daraltılmış listeden gelmesi daha iyi)
    asorti_list = sorted(df_step2['ASORTI'].astype(str).unique())
    # Eğer liste boş gelirse (veri eksikliği vs.) genel listeyi kullan
    if not asorti_list:
        asorti_list = sorted(df['ASORTI'].astype(str).unique())
    inputs['ASORTI'] = st.selectbox("ASORTI", asorti_list)

    # Sayısal Girişler
    c1, c2 = st.columns(2)
    inputs['KUMAS_ENI'] = c1.number_input("KUMAS_ENI", 90.0, 195.0, 152.0)
    inputs['KUMAS_CEKME_DEGERI_EN'] = c2.number_input("CEKME_EN", -13.0, 0.0, -3.0)
    
    c3, c4 = st.columns(2)
    inputs['KUMAS_CEKME_DEGERI_BOY'] = c3.number_input("CEKME_BOY", -22.0, 8.0, -3.0)
    inputs['ASORTI_SAYISI'] = c4.number_input("ASORTI_SAYISI", 5.0, 20.0, 10.0)

    inputs['PARCA_SAYISI'] = st.number_input("PARCA_SAYISI", 1.0, 30.0, 18.0) # Büyük-küçük harf dikkat: PARCA_sAYISI yazmıştın eğitimde

# -----------------------------
# 4. HESAPLAMA BUTONU (DÜZELTİLMİŞ HALİ)
# -----------------------------
st.divider()
if st.button("HESAPLA", type="primary", use_container_width=True):
    if model:
        try:
            # 1. Kullanıcı girdilerinden DataFrame oluştur
            X_new = pd.DataFrame([inputs])
            
            # --- KRİTİK DÜZELTME BAŞLANGICI ---
            # Modelin eğitildiği sütun sırasını al ve veriyi ona göre yeniden diz
            # Bu işlem "At position 4..." hatasını kesin olarak çözer.
            beklenen_siralama = model.feature_names_
            X_new = X_new[beklenen_siralama]
            # --- KRİTİK DÜZELTME BİTİŞİ ---

            # Kategorik değişkenlerin listesi (Sadece isim olarak kalmalı)
            cat_features = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT',
                            'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']
            
            # Tahmin Havuzunu Oluştur
            X_new_pool = Pool(X_new, cat_features=cat_features)
            
            # Tahmin Yap
            prediction = model.predict(X_new_pool)[0]
            
            st.success(f"🧵 Tahmini Birim Sarfiyat: **{prediction:.3f} mt**")
            
        except KeyError as e:
            st.error(f"Veri eksik! Model '{e}' isimli bir sütun bekliyor ama girdilerde bu yok.")
        except Exception as e:
            st.error(f"Hesaplama Hatası: {e}")
            st.info("İpucu: Sütun isimleri veya veri tipleri model eğitimiyle uyuşmuyor olabilir.")
    else:
        st.error("Model yüklenemedi.")

