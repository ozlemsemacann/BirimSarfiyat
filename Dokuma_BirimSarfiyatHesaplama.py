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
inputs['Pastal_Turu'] = st.selectbox("Pastal_Turu", ['Ceplik','Ana_Kumas','Biye'])
inputs['Pastal_Detayi'] = st.selectbox("Pastal_Detayi", ['Yok','En_Boy','Tek_Yon'])
inputs['Asorti'] = st.selectbox("Asorti", ['5/6 Y_7/8 Y_8/9 Y_9/10 Y_11/12 Y_13/14 Y_','34_36_38_40_42_','7/8 Y_8/9 Y_9/10 Y_11/12 Y_13/14 Y_','34_36_38_40_42_44_','36_38_40_42_44_46_','34_36_38_40_42_44_46_','28_30_32_34_36_38_40_42_','34_36_38_40_42_44_46_48_','28_30_32_34_36_38_40_','32_34_36_38_40_42_','28_30_32_34_','7/8 Y_8/9 Y_9/10 Y_11/12 Y_12/13 Y_13/14 Y_','5/6 Y_7/8 Y_8/9 Y_9/10 Y_11/12 Y_12/13 Y_13/14 Y_','S_M_L_XL_XXL_3XL_','5/6 Y_7/8 Y_8/9 Y_9/10 Y_11/12 Y_','36_38_40_42_44_','30_32_34_36_','30_32_34_36_38_','6-9_9-12_12-18_18-24_24-36_3-4_4-5_5-6_','30_32_34_36_38_40_42_44_46_','28_30_32_34_36_38_','S_M_L_XL_XXL_','XS_S_M_L_XL_XXL_','XS_S_M_L_XL_'])
# Asorti Sayısı Değerini 1 ile 30 arasına sınırlama
inputs['Asorti_Sayisi'] = st.number_input(
    "Asorti_Sayisi",
    min_value=5.0,
    max_value=20.0,
    value=10.0) # Başlangıç değeri
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

    cat_features = ['Departman', 'Model_Turu', 'Model_Detayi','Fit',
                    'Pastal_Turu', 'Pastal_Detayi', 'Asorti']

    X_new_pool = Pool(X_new, cat_features=cat_features)
    prediction = model.predict(X_new_pool)[0]
    st.success(f"🔮 Tahmini Birim Sarfiyat: **{prediction:.2f}**")