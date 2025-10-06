import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

# -----------------------------
# Modeli yükle
# -----------------------------
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("Dokuma_BirimSarfiyatModel.cbm")
    return model

model = load_model()

# -----------------------------
# Streamlit Arayüzü
# -----------------------------
st.title("🧵 Birim Sarfiyat Tahmini")

st.markdown("Modeli önceden eğittik ve yükledik. Şimdi değerleri gir, tahmini al!")

# Kullanıcıdan girişler
inputs = {}
inputs['Kumas_Kalitesi'] = st.selectbox("Kumas_Kalitesi", ['Astar','Gabardin','Örme','Crep','PoliViskon','Denim'])
inputs['Departman'] = st.selectbox("Departman", ['Kiz_Cocuk','Erkek_Cocuk','Kadin','Erkek','Kiz_Bebek','Erkek_Bebek'])
inputs['Model_Turu'] = st.selectbox("Model_Turu", ['TROUSERS','SHORT','JACKET','JUMP SUIT','SKIRT','LONG SLEEVE SHIRT','SHORT SLEEVE SHIRT'])
inputs['Model_Detayi'] = st.selectbox("Model_Detayi", ['Yok','5_Cep'])
inputs['Fit'] = st.selectbox("Fit", ['Baggy_Fit','Regular_Fit','New_Wide_Leg_Fit','Straight_Fit','Wide_Leg_Fit','Parachute_Fit','Jogger_Fit','Culotte_Fit','Crop_Flare_Fit','Cargo_jogger_Fit','Slim_Fit','Jogger_Slim_Fit','Carrot_Fit','Oversize_Fit','Wide_Leg_Cargo_Fit','Relax_Fit','Balloon_Fit','Loose_Fit','Flare_Fit','Chino_Fit','Valentina_Fit','Mom_Short_Fit','Gwt_Straigth_Fit','Paperbag_Fit','90s_Slim_Fit','Mia_Flare_Fit','Barrel_Fit','Extra_Baggy_Fit','Relax_Tapared_Fit','Sergio_Fit','Carlo_Fit','Pedro_Fit','Wide_Leg_Tapared_Fit','Bootcut_Fit','Comfort_Straigth_Fit','Short_Etek_Fit','90s_Wide_Leg_Fit'])
# Kumaş Eni Değerini 130 ile 176 arasına sınırlama
inputs['Kumas_Eni'] = st.number_input(
    "Kumas_Eni",
    min_value=125.0,
    max_value=176.0,
    value=146.0) # Başlangıç değeri
# Kumaş Çekme Değerini 1 ile 30 arasına sınırlama
inputs['Kumas_Cekme_Degeri_En'] = st.number_input(
    "Kumas_Cekme_Degeri_En",
    min_value=1.0,
    max_value=30.0,
    value=1.5) # Başlangıç değeri
# Kumaş Çekme Değerini 1 ile 30 arasına sınırlama
inputs['Kumas_Cekme_Degeri_Boy'] = st.number_input(
    "Kumas_Cekme_Degeri_Boy",
    min_value=1.0,
    max_value=30.0,
    value=1.5) # Başlangıç değeri
inputs['Pastal_Turu'] = st.selectbox("Pastal_Turu", ['Ceplik','Ana_Kumas','Garni','Biye'])
inputs['Pastal_Detayi'] = st.selectbox("Pastal_Detayi", ['Yok','En_Boy','Yonsuz'])
inputs['Asorti'] = st.selectbox("Asorti", ['5/6_Y-1/7/8_Y-1/8/9_Y-2/9/10_Y-2/11/12_Y-2/13/14_Y-2','5/6_Y-1/7/8_Y-1/8/9_Y-1/9/10_Y-2/11/12_Y-3/13/14_Y-2','34-1/36-2/38-2/40-2/42-1/','5/6_Y-1/7/8_Y-1/8/9_Y-1/9/10_Y-2/11/12_Y-2/13/14_Y-2','7/8_Y-1/8/9_Y-1/9/10_Y-2/11/12_Y-3/13/14_Y-2','34-1/36-2/38-2/40-2/42-2/44-1/','34-1/36-2/38-3/40-2/42-1/','34-2/36-2/38-2/40-2/42-1/44-1/','34-1/36-2/38-3/40-2/42-2/44-1/','36-1/38-2/40-3/42-3/44-2/46-1/','34-1/36-2/38-3/40-3/42-2/44-1/46-1/','34-1/36-2/38-3/40-3/42-2/44-1/','36-1/38-2/40-2/42-2/','34-2/36-3/38-3/40-2/42-1/','28-1/30-2/32-2/34-2/36-2/38-2/40-1/42-1/','34-1/36-2/38-2/40-2/42-2/','5/6_Y-1/7/8_Y-1/8/9_Y-2/9/10_Y-2/11/12_Y-2/13/14_Y-3','34-1/36-2/38-3/40-3/42-3/44-2/46-1/48-1/','36-1/38-1/40-2/42-2/44-2/46-2/','28-1/30-2/32-3/34-3/36-2/38-1/40-1/','34-1/36-1/38-2/40-2/42-2/','32-1/34-2/36-2/38-2/40-2/42-1/','28-1/30-1/32-2/34-1/','7/8_Y-2/8/9_Y-2/9/10_Y-2/11/12_Y-2/12/13_Y-1/13/14_Y-1','7/8_Y-1/8/9_Y-1/9/10_Y-2/11/12_Y-3/13/14_Y-3','5/6_Y-1/7/8_Y-1/8/9_Y-2/9/10_Y-2/11/12_Y-3/13/14_Y-2','5/6_Y-1/7/8_Y-2/8/9_Y-2/9/10_Y-2/11/12_Y-3/13/14_Y-1','5/6_Y-1/7/8_Y-1/8/9_Y-1/9/10_Y-2/11/12_Y-2/12/13_Y-2/13/14_Y-1','5/6_Y-1/7/8_Y-1/8/9_Y-1/9/10_Y-2/11/12_Y-3/13/14_Y-3','5/6_Y-1/7/8_Y-2/8/9_Y-3/9/10_Y-3/11/12_Y-2/13/14_Y-1','28-1/30-1/32-2/34-2/36-2/38-2/40-1/42-1/','28-1/30-2/32-2/34-2/36-2/38-2/40-1/','28-2/30-3/32-3/34-3/36-2/38-1/40-1/','34-1/36-1/38-2/40-2/42-3/','34-1/36-1/38-2/40-3/42-2/','36-1/38-2/40-3/42-3/','30-1/32-3/34-2/36-1/','32-1/34-2/36-3/38-3/40-2/42-1/','5/6_Y-1/7/8_Y-2/8/9_Y-1/9/10_Y-2/11/12_Y-2/13/14_Y-2','7/8_Y-1/8/9_Y-2/9/10_Y-2/11/12_Y-3/13/14_Y-2','7/8_Y-2/8/9_Y-1/9/10_Y-2/11/12_Y-3/13/14_Y-2','5/6_Y-1/7/8_Y-1/8/9_Y-2/9/10_Y-2/11/12_Y-3/13/14_Y-1','7/8_Y-1/8/9_Y-1/9/10_Y-2/11/12_Y-4/13/14_Y-2','30-1/32-2/34-3/36-2/38-1/','6-9-1/9-12-1/12-18-2/18-24-2/24-36-2/3-4-2/4-5-1/5-6-1/','6-9-1/9-12-1/12-18-2/18-24-2/24-36-2/3-4-2/4-5-2/5-6-2/','30-1/32-2/34-3/36-2/38-2/40-1/42-1/44-1/','28-2/30-3/32-3/34-2/36-2/38-1/','28-1/30-1/32-4/34-4/36-3/38-2/40-1/','30-1/32-3/34-1/36-1/38-2/','28-1/30-2/32-3/34-3/36-2/38-1/','5/6_Y-1/7/8_Y-2/8/9_Y-1/9/10_Y-2/11/12_Y-3/13/14_Y-2','5/6_Y-1/7/8_Y-2/8/9_Y-2/9/10_Y-2/11/12_Y-2/13/14_Y-1','5/6_Y-1/7/8_Y-2/8/9_Y-1/9/10_Y-1/11/12_Y-2/13/14_Y-3','5/6_Y-1/7/8_Y-2/8/9_Y-2/9/10_Y-2/11/12_Y-2/13/14_Y-2','7/8_Y-1/8/9_Y-2/9/10_Y-2/11/12_Y-2/13/14_Y-1','S-1/M-2/L-3/XL-2/XXL-1','S-1/M-3/L-3/XL-2/XXL-1','6-9-1/9-12-1/12-18-1/18-24-2/24-36-2/3-4-2/4-5-2/5-6-2/','XS-1/S-2/M-3/L-3/XL-2/XXL-1/','7/8_Y-1/8/9_Y-1/9/10_Y-2/11/12_Y-2/12/13_Y-1/13/14_Y-2','XS-1/S-2/M-3/L-2/XL-1/','S-1/M-2/L-2/XL-2/XXL-2/3XL-1/','7/8_Y-2/8/9_Y-1/9/10_Y-2/11/12_Y-2/13/14_Y-2'])
# Parça Sayısı Değerini 1 ile 30 arasına sınırlama
inputs['Parca_Sayisi'] = st.number_input(
    "Parca_Sayisi",
    min_value=1.0,
    max_value=30.0,
    value=2.0) # Başlangıç değeri

# DataFrame oluştur
X_new = pd.DataFrame([inputs])

# Tahmin
if st.button("Tahmin Et"):
    from catboost import Pool

    cat_features = ['Kumas_Kalitesi', 'Departman', 'Model_Turu', 'Model_Detayi','Fit',
                    'Pastal_Turu', 'Pastal_Detayi', 'Asorti']

    X_new_pool = Pool(X_new, cat_features=cat_features)
    prediction = model.predict(X_new_pool)[0]
    st.success(f"🔮 Tahmini Birim Sarfiyat: **{prediction:.2f}**")