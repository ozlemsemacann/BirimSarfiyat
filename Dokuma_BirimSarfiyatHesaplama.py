import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# -----------------------------------------------------------------------------
# 1. AYARLAR VE OTOMATİK DOSYA BULMA
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Sarfiyat Tahmini", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))

# Dosya adları
EXCEL_NAME = "YuklenenDokumaDosya262.xlsx"
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
        
        # --- VERİ TEMİZLEME (DATA CLEANING) ---
        text_columns = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', 'FIT', 'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                
        return df
    except Exception as e:
        st.error(f"Excel okuma hatası: {e}")
        return None

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası bulunamadı! ({MODEL_NAME})")
        return None
    try:
        model = CatBoostRegressor()
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None

df = load_data()
model = load_model()

if df is None or model is None:
    st.stop()

# -----------------------------------------------------------------------------
# MAİL GÖNDERME FONKSİYONU
# -----------------------------------------------------------------------------
def send_notification_email(prediction_result, user_inputs):
    try:
        # Secrets'tan bilgileri çek
        smtp_server = st.secrets["email"]["smtp_server"]
        port = st.secrets["email"]["port"]
        sender_email = st.secrets["email"]["sender_email"]
        password = st.secrets["email"]["password"]
        receiver_email = "ozlem.semacan@defacto.com"

        # Mail İçeriğini Hazırla
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = "🔔 Yeni Dokuma Birim Sarfiyat Hesaplaması Yapıldı"

        body = f"""
        Merhaba,
        
        Uygulama üzerinden yeni bir dokuma hesaplaması yapıldı. Detaylar aşağıdadır:
        
        ------------------------------------------
        🔮 TAHMİN SONUCU: {prediction_result:.3f} mt
        ------------------------------------------
        
        GİRİLEN VERİLER:
        - MODEL KODU: {user_inputs.get('MODEL_KODU', '-')}
        - DEPARTMAN: {user_inputs.get('DEPARTMAN', '-')}
        - MODEL TURU: {user_inputs.get('MODEL_TURU', '-')}
        - MODEL DETAYI: {user_inputs.get('MODEL_DETAYI', '-')}
        - FIT: {user_inputs.get('FIT', '-')}
        - ASORTI: {user_inputs.get('ASORTI', '-')}
        - PASTAL TURU: {user_inputs.get('PASTAL_TURU', '-')}
        - PASTAL DETAYI: {user_inputs.get('PASTAL_DETAYI', '-')}
        - KUMAS ENI: {user_inputs.get('KUMAS_ENI', '-')}
        - CEKME EN: {user_inputs.get('KUMAS_CEKME_DEGERI_EN', '-')}
        - CEKME BOY: {user_inputs.get('KUMAS_CEKME_DEGERI_BOY', '-')}
        - ASORTI SAYISI: {user_inputs.get('ASORTI_SAYISI', '-')}
        - PARCA SAYISI: {user_inputs.get('PARCA_SAYISI', '-')}
        
        Tarih: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}
        """
        msg.attach(MIMEText(body, 'plain'))

        # Maili Gönder
        server = smtplib.SMTP(smtp_server, port)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Mail gönderme hatası: {e}")
        return False

# -----------------------------------------------------------------------------
# 2. TAM BAĞIMLI (CASCADING) FİLTRELEME ZİNCİRİ
# -----------------------------------------------------------------------------
st.title("🎯 Akıllı Birim Sarfiyat Tahmini")
st.success(f"✅ Modeli önceden eğittik ve yükledik. Şimdi değerleri gir, tahmini al!")

inputs = {}
st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📌 Model Seçimi")

    # MANUEL MODEL KODU GİRİŞİ
    inputs['MODEL_KODU'] = st.text_input("MODEL KODU (Manuel Giriniz)")

    # 1. DEPARTMAN
    dept_list = sorted(df['DEPARTMAN'].unique())
    secilen_dept = st.selectbox("DEPARTMAN", dept_list)
    inputs['DEPARTMAN'] = secilen_dept
    
    # FİLTRE 1
    df_step1 = df[df['DEPARTMAN'] == secilen_dept]

    # 2. MODEL TURU
    tur_list = sorted(df_step1['MODEL_TURU'].unique())
    secilen_tur = st.selectbox("MODEL_TURU", tur_list)
    inputs['MODEL_TURU'] = secilen_tur
    
    # FİLTRE 2
    df_step2 = df_step1[df_step1['MODEL_TURU'] == secilen_tur]

    # 3. MODEL DETAYI
    detay_list = sorted(df_step2['MODEL_DETAYI'].unique())
    secilen_detay = st.selectbox("MODEL_DETAYI", detay_list)
    inputs['MODEL_DETAYI'] = secilen_detay
    
    # FİLTRE 3
    df_step3 = df_step2[df_step2['MODEL_DETAYI'] == secilen_detay]

    # 4. FIT
    fit_list = sorted(df_step3['FIT'].unique())
    secilen_fit = st.selectbox("FIT", fit_list)
    inputs['FIT'] = secilen_fit

    # FİLTRE 4
    df_step4 = df_step3[df_step3['FIT'] == secilen_fit]

with col_right:
    st.subheader("⚙️ Teknik Detaylar")

    # 5. ASORTI
    asorti_list = sorted(df_step4['ASORTI'].unique())
    if not asorti_list:
        asorti_list = sorted(df['ASORTI'].unique())
    inputs['ASORTI'] = st.selectbox("ASORTI", asorti_list)

    # Diğer Sabit Girişler
    inputs['PASTAL_TURU'] = st.selectbox("PASTAL_TURU", sorted(df['PASTAL_TURU'].unique()))
    
    # Büyükten Küçüğe (veya Z-A) sıralama yapıldı
    inputs['PASTAL_DETAYI'] = st.selectbox("PASTAL_DETAYI", sorted(df['PASTAL_DETAYI'].unique(), reverse=True))

    # Sayısal Değerler
    c1, c2 = st.columns(2)
    inputs['KUMAS_ENI'] = c1.number_input("KUMAS_ENI", 90.0, 195.0, 152.0)
    inputs['KUMAS_CEKME_DEGERI_EN'] = c2.number_input("CEKME_EN", -13.0, 0.0, -3.0)
    
    c3, c4 = st.columns(2)
    inputs['KUMAS_CEKME_DEGERI_BOY'] = c3.number_input("CEKME_BOY", -22.0, 8.0, -3.0)
    inputs['ASORTI_SAYISI'] = c4.number_input("ASORTI_SAYISI", 5.0, 20.0, 10.0)

    # PARCA_SAYISI
    inputs['PARCA_SAYISI'] = st.number_input("PARCA_SAYISI", 1.0, 30.0, 18.0)

with col_right:
    st.subheader("⚙️ Teknik Detaylar")

    # 5. ASORTI
    asorti_list = sorted(df_step4['ASORTI'].unique())
    if not asorti_list:
        asorti_list = sorted(df['ASORTI'].unique())
    inputs['ASORTI'] = st.selectbox("ASORTI", asorti_list)

    # Diğer Sabit Girişler
    inputs['PASTAL_TURU'] = st.selectbox("PASTAL_TURU", sorted(df['PASTAL_TURU'].unique()))
    inputs['PASTAL_DETAYI'] = st.selectbox("PASTAL_DETAYI", sorted(df['PASTAL_DETAYI'].unique(), reverse=True))

    # Sayısal Değerler
    c1, c2 = st.columns(2)
    inputs['KUMAS_ENI'] = c1.number_input("KUMAS_ENI", 90.0, 195.0, 152.0)
    inputs['KUMAS_CEKME_DEGERI_EN'] = c2.number_input("CEKME_EN", -13.0, 0.0, -3.0)
    
    c3, c4 = st.columns(2)
    inputs['KUMAS_CEKME_DEGERI_BOY'] = c3.number_input("CEKME_BOY", -22.0, 8.0, -3.0)
    inputs['ASORTI_SAYISI'] = c4.number_input("ASORTI_SAYISI", 5.0, 20.0, 10.0)

    # -------------------------------------------------------------------------
    # ORTALAMA PARCA SAYISI HESAPLAMA (Anlık Çalışır)
    # -------------------------------------------------------------------------
    mask = (
        (df['DEPARTMAN'] == inputs['DEPARTMAN']) &
        (df['MODEL_TURU'] == inputs['MODEL_TURU']) &
        (df['MODEL_DETAYI'] == inputs['MODEL_DETAYI']) &
        (df['PASTAL_TURU'] == inputs['PASTAL_TURU'])
    )
    
    avg_parca = df[mask]['PARCA_SAYISI'].mean()
    
    if pd.isna(avg_parca):
        default_parca = 18.0
        st.warning("⚠️ Bu kombinasyona ait geçmiş veri bulunamadı. Varsayılan değer atanıyor.")
    else:
        default_parca = float(round(avg_parca))
        default_parca = max(1.0, min(30.0, default_parca))
        st.info(f"💡 Seçtiğiniz kriterlere göre geçmiş ortalama parça sayısı **{default_parca}** olarak hesaplandı.")

    # PARCA_SAYISI
    inputs['PARCA_SAYISI'] = st.number_input("PARCA_SAYISI", 1.0, 30.0, value=default_parca)

# -----------------------------------------------------------------------------
# 3. HESAPLAMA VE MAİL GÖNDERME
# -----------------------------------------------------------------------------
st.divider()

if st.button("HESAPLA", type="primary", use_container_width=True):
# ... (Kodun geri kalanı buradan devam eder)
