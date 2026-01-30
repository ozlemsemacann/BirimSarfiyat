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
inputs['DEPARTMAN'] = st.selectbox("DEPARTMAN", ['MAN','BOY','BABYBOY'])
inputs['MODEL_TURU'] = st.selectbox("MODEL_TURU", ['LONG_SLEEVE_SHIRT','TROUSERS','BERMUDA','SHORT_SLEEVE_SHIRT','OVERSHIRT','SWIMMING_SHORT','SHORT','OVERALLS'])
inputs['MODEL_DETAYI'] = st.selectbox("MODEL_DETAYI", ['CIFT_CEP_ROBALI','TEK_CEP_ROBALI','5_CEP','CHINO_CEP_ARKA_1_YAPISTIRMA_CEP_BELI_LASTIKLI','5_CEP_PACA_KATLAMALI','5_CEP_BELI_LASTIKLI','CHINO_CEP_ARKA_2_YAPISTIRMA','CHINO_CEP_ARKA_2_YAPISTIRMA_CARGO','CHINO_CEP_ARKA_2_YAPISTIRMA_BELI_LASTIKLI','5_CEP_BELI_LASTIKLI_CARGO','CHINO_CEP_ARKA_1_YAPISTIRMA_BELI_LASTIKLI','5_CEP_CARGO','5_CEP_CARGO_CEP','CEP_YOK_ROBASIZ','TEK_CEP_ROBASIZ','CEP_YOK_CIFT_KAT_ROBA','CEPSIZ_ROBALI','CEPSIZ_ROBASIZ','TEK_CEP','CEP_YOK_TEK_KAT_ROBA','CIFT_CEP','CIFT_CEP_TEK_KAT_ROBA','CHINO_CEP_ARKA_FILETO','ARKA_FILETO_BELI_LASTIKLI','CHINO_CEP_ARKA_FILETO_BELI_LASTIKLI','CHINO_CEP_CARGO_BEL_PACA_LASTIKLI','CHINO_CEP_ARKA_2_YAPISTIRMA_BELI_LASTIKLI DUYGUYA SOR','CHINO_CEP_CARGO_BELI_LASTIKLI','YAN_CEP_ARKA_FILETO_BELI_LASTIKLI','CHINO_CEP_ARKA_2_YAPISTIRMA_CARGO_BELI_LASTIKLI','CARGO_CEP_BELI_LASTIKLI','YAN_CEP_ARKA_1_YAPISTIRMA_BELI_LASTIKLI','CHINO_CEP_ARKA_1_YAPISTIRMA_CEP','CARGO_CEP_BEL_PACA_LASTIKLI','ARKA_1_YAPISTIRMA_BELI_LASTIKLI','CHINO_CEP_ARKA_1_YAPISTIRMA_CARGO','CHINO_CEP_BELI_LASTIKLI','YAN_CEP_ARKA_FILETO_BEL_PACA_LASTIKLI','CHINO_CEP_BELI_LASTIKLI_PACA_KATLAMALI','YAN_CEP_ARKA_2_YAPISTIRMA_BEL_PACA_LASTIKLI','PUNTEREZ_CARGO','CHINO_CEP_BEL_PACA_LASTIKLI','CHINO_CEP','CIFT_CEP_ROBALI EK 1 CEP VAR','JUMPSUIT_1_CEP','TEK_CEP_ROBALI_APOLETLI','CIFT_CEP_ROBALI_APOLETLI','CEPSIZ_ROBALI_APOLETLI','YAN_CEP_BELI_LASTIKLI','YAN_CEP_BEL_PACA_LASTIKLI','CHINO_CEP_ARKA_2_YAPISTIRMA_CEP_BELI_LASTIKLI','BELI_LASTIKLI','CHINO_CEP_ARKA_1_YAPISTIRMA_BEL_PACA_LASTIKLI','YAN_CEP_CARGO_BEL_PACA_LASTIKLI','CHINO_CEP_ARKA_2_YAPISTIRMA_BEL_PACA_LASTIKLI','ARKA_2_YAPISTIRMA','SALOPET_1_CEP','YAN_CEP_BELI_LASTIKLI_PACA_KATLAMALI','JUMPSUIT_1_CEP_CHINO','YOK'])
inputs['FIT'] = st.selectbox("FIT", ['OVERSIZE_FIT','PEDRO-SLIM_FIT_DENIM','90s_SLIM_FIT','RELAX_FIT','CARLO_SKINNY_FIT_DENIM','SERGIO_REGULAR_FIT','BAGGY_FIT','STRAIGHT_FIT','TAPERED_SLIM','TAPERED_FIT_RELAXED','SLIM_FIT','CARROT_RELAXED_FIT','TAPERED_WIDE_LEG_FIT','REGULAR_FIT','BARREL_FIT','SKATER_FIT','JORT','CARGO_RELAX_FIT','WIDE_LEG_FIT','SLIM_CUT_FIT','RELAXED_SLOUCHY_FIT','BOXY_FIT','MODERN_FIT','LUKE','ANDY','NATHAN','JOGGER_SLIM_FIT','CROPPED_FIT','JOGGER_FIT','RELAXED_JOGGER_FIT','CARGO_REGULAR_SHORT','CARGO_REGULAR_JOGGER','PULL_ON','CARGO_JOGGER_FIT','BALLOON_FIT','CARPENTER_FIT','CARGO_FIT','CARGO_PARACHUTE','LOOSE_FIT','JUMPSUIT','CARROT_FIT','5_POCKET_SHORT','CARGO_SHORT','SALOPET','TAPERED'])
# Kumaş Eni Değerini 130 ile 176 arasına sınırlama
inputs['KUMAS_ENI'] = st.number_input(
    "KUMAS_ENI",
    min_value=90.0,
    max_value=195.0,
    value=146.0) # Başlangıç değeri
# Kumaş Çekme Değerini 1 ile 30 arasına sınırlama
inputs['KUMAS_CEKME_DEGERI_EN'] = st.number_input(
    "KUMAS_CEKME_DEGERI_EN",
    min_value=-13.0,
    max_value=0.0,
    value=1.5) # Başlangıç değeri
# Kumaş Çekme Değerini 1 ile 30 arasına sınırlama
inputs['KUMAS_CEKME_DEGERI_BOY'] = st.number_input(
    "KUMAS_CEKME_DEGERI_BOY",
    min_value=-22.0,
    max_value=8.0,
    value=1.5) # Başlangıç değeri
inputs['PASTAL_TURU'] = st.selectbox("PASTAL_TURU", ['ANA_BEDEN','ASTAR','FILE','TELA','PAT_TELASI'])
inputs['PASTAL_DETAYI'] = st.selectbox("PASTAL_DETAYI", ['YONLU','YONSUZ'])
inputs['ASORTI'] = st.selectbox("ASORTI", ['5/6 Y_7/8 Y_8/9 Y_9/10 Y_11/12 Y_13/14 Y_','34_36_38_40_42_','7/8 Y_8/9 Y_9/10 Y_11/12 Y_13/14 Y_','34_36_38_40_42_44_','36_38_40_42_44_46_','34_36_38_40_42_44_46_','28_30_32_34_36_38_40_42_','34_36_38_40_42_44_46_48_','28_30_32_34_36_38_40_','32_34_36_38_40_42_','28_30_32_34_','7/8 Y_8/9 Y_9/10 Y_11/12 Y_12/13 Y_13/14 Y_','5/6 Y_7/8 Y_8/9 Y_9/10 Y_11/12 Y_12/13 Y_13/14 Y_','S_M_L_XL_XXL_3XL_','5/6 Y_7/8 Y_8/9 Y_9/10 Y_11/12 Y_','36_38_40_42_44_','30_32_34_36_','30_32_34_36_38_','6-9_9-12_12-18_18-24_24-36_3-4_4-5_5-6_','30_32_34_36_38_40_42_44_46_','28_30_32_34_36_38_','S_M_L_XL_XXL_','XS_S_M_L_XL_XXL_','XS_S_M_L_XL_'])
# Asorti Sayısı Değerini 1 ile 30 arasına sınırlama
inputs['ASORTI_SAYISI'] = st.number_input(
    "ASORTI_SAYISI",
    min_value=5.0,
    max_value=20.0,
    value=10.0) # Başlangıç değeri
# Parça Sayısı Değerini 1 ile 30 arasına sınırlama
inputs['PARCA_sAYISI'] = st.number_input(
    "PARCA_SAYISI",
    min_value=1.0,
    max_value=30.0,
    value=2.0) # Başlangıç değeri

# DataFrame oluştur
X_new = pd.DataFrame([inputs])

# Tahmin
if st.button("Tahmin Et"):
    from catboost import Pool

    cat_features = ['DEPARTMAN', 'MODEL_TURU', 'Model_Detayi','FIT',
                    'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']

    X_new_pool = Pool(X_new, cat_features=cat_features)
    prediction = model.predict(X_new_pool)[0]
    st.success(f"🔮 Tahmini Birim Sarfiyat: **{prediction:.2f}**")
