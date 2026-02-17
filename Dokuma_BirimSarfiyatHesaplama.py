import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool

# 1. Veriyi Yükle (Seçenekleri dinamik getirmek için şart)
@st.cache_data
def load_data():
    # Buraya modelin eğitildiği CSV veya Excel dosyasını koymalısınız
    df = pd.read_csv("egitim_verisi.csv") 
    return df

df = load_data()

# 2. Modeli Yükle
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("Dokuma_BirimSarfiyatModel.cbm")
    return model

model = load_model()

st.title("🧵 Birim Sarfiyat Tahmini")

# --- BİRBİRİNE BAĞLI FİLTRELER ---

# DEPARTMAN (En üst kırılım)
departman_list = df['DEPARTMAN'].unique()
selected_dept = st.selectbox("DEPARTMAN", departman_list)

# MODEL_TURU (Seçilen departmana göre filtreleniyor)
filtered_model_turu = df[df['DEPARTMAN'] == selected_dept]['MODEL_TURU'].unique()
selected_model_turu = st.selectbox("MODEL_TURU", filtered_model_turu)

# MODEL_DETAYI (Seçilen model türüne göre filtreleniyor)
filtered_model_detay = df[
    (df['DEPARTMAN'] == selected_dept) & 
    (df['MODEL_TURU'] == selected_model_turu)
]['MODEL_DETAYI'].unique()
selected_model_detay = st.selectbox("MODEL_DETAYI", filtered_model_detay)

# ... Bu mantığı FIT ve ASORTI için de devam ettirebilirsiniz.

# --- DİĞER GİRİŞLER ---
kumas_eni = st.number_input("KUMAS_ENI", 90.0, 195.0, 145.0)
# ... Diğer number_input girişleriniz

# --- TAHMİN BÖLÜMÜ ---
if st.button("Tahmin Et"):
    # Girişleri modelin beklediği formatta sözlüğe dizin
    input_dict = {
        'DEPARTMAN': selected_dept,
        'MODEL_TURU': selected_model_turu,
        'MODEL_DETAYI': selected_model_detay,
        # Diğer değişkenleri buraya ekleyin...
    }
    
    X_new = pd.DataFrame([input_dict])
    cat_features = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI', ...] # Modelinizdeki kategorik kolonlar
    
    X_new_pool = Pool(X_new, cat_features=cat_features)
    prediction = model.predict(X_new_pool)[0]
    st.success(f"🔮 Tahmini Birim Sarfiyat: **{prediction:.2f}**")
