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
    value=-1.5) # Başlangıç değeri
# Kumaş Çekme Değerini 1 ile 30 arasına sınırlama
inputs['KUMAS_CEKME_DEGERI_BOY'] = st.number_input(
    "KUMAS_CEKME_DEGERI_BOY",
    min_value=-22.0,
    max_value=8.0,
    value=-1.5) # Başlangıç değeri
inputs['PASTAL_TURU'] = st.selectbox("PASTAL_TURU", ['ANA_BEDEN','ASTAR','FILE','TELA','PAT_TELASI'])
inputs['PASTAL_DETAYI'] = st.selectbox("PASTAL_DETAYI", ['YONLU','YONSUZ'])
inputs['ASORTI'] = st.selectbox("ASORTI", [("ASORTI", ['5/6(1),7/8(1),8/9(1),9/10(1),11/12(1),13/14(1)','5/6(1),7/8(1),8/9(1),9/10(2),11/12(3),12/13(1),13/14(2)','5/6(1),7/8(1),8/9(1),9/10(2),11/12(3),13/14(1)','5/6(1),7/8(1),8/9(1),9/10(2),11/12(3),13/14(2)','5/6(1),7/8(1),8/9(1),9/10(2),11/12(3),13/14(3)','5/6(1),7/8(1),8/9(2),9/10(2),11/12(2),13/14(2)','5/6(1),7/8(1),8/9(2),9/10(2),11/12(2),13/14(3)','5/6(1),7/8(1),8/9(2),9/10(2),11/12(3),13/14(2)','5/6(1),7/8(1),8/9(2),9/10(2),11/12(3),13/14(3)','5/6(1),7/8(2),8/9(1),9/10(1),11/12(2),13/14(3)','5/6(1),7/8(2),8/9(1),9/10(2),11/12(2),13/14(2)','5/6(1),7/8(2),8/9(1),9/10(2),11/12(3),13/14(1)','5/6(1),7/8(2),8/9(2),9/10(2),11/12(2),13(1)','5/6(1),7/8(2),8/9(2),9/10(2),11/12(2),13/14(1)','5/6(1),7/8(2),8/9(2),9/10(2),11/12(2),13/14(2)','5/6(1),7/8(2),8/9(2),9/10(2),11/12(3),13/14(2)','5/6(2),7/8(2),8/9(1),9/10(2),11/12(2),13/14(2)','5/6(2),7/8(2),8/9(2),9/10(2),11/12(2),13/14(1)','5/6(2),7/8(2),8/9(4),9/10(4),11/12(4),13/14(4)','6/7(1),7/8(1),8/9(2),9/10(3),10/11(3),11/12(2),12/13(1),13/14(1)','6/9(1),9/12(1),12/18(2),18/24(2),24/36(2),3/4(2),4/5(2),5/6(1)','7/8(1),8/9(1),9/10(2),11/12(2),13/14(3)','7/8(1),8/9(1),9/10(2),11/12(3),13(2)','7/8(1),8/9(1),9/10(2),11/12(3),13/14(3)','7/8(1),8/9(2),9/10(2),11/12(3),13/14(2)','7/8(1),8/9(2),9/10(2),11/12(3),13/14(3)','7/8(1),8/9(2),9/10(3),11/12(2),13/14(1)','7/8(1),8/9(2),9/10(3),11/12(3),13/14(1)','7/8(2),8/9(1),9/10(2),11/12(2),13/14(2)','7/8(2),8/9(2),9/10(2),11/12(2),12/13(1),13/14(1)','7/8(2),8/9(2),9/10(4),11/12(6),13/14(2)','8/9(1),9/10(1),11/12(1),13/14(1)','9/12(1),12/18(1),18/24(2),24/36(2),3/4(2),4/5(2),5/6(2)','9/12(1),12/18(2),18/24(2),24/36(2),3/4(2),4/5(2),5/6(1)','9/12(2),12/18(2),18/24(2),24/36(2),3/4(2),4/5(2),5/6(2)','12/18 M(1),18/24 M(2),24/36 M(2),3/4 Y(2),4/5Y(2),5/6Y(2)','12/18(1),18/24(2),24/36(2),3/4(2),4/5(2)','12/18(1),18/24(2),24/36(2),3/4(2),4/5(2),5/6(1)','12/18(1),18/24(2),24/36(2),3/4(2),4/5(2),5/6(2)','12/18(1),18/24(2),24/36(2),4/5(2),5/6(2)','27(1),28(1),29(1),30(1),31(1),32(2)','28(1),29(1),30(1),32(1),34(1)','28(1),29(1),30(1),32(1),34(1),36(1),38(1)','28(1),29(1),30(2),31(2),32(3),33(2),34(2),36(1)','28(1),29(1),30(2),32(2),34(2),36(2),38(1),40(1)','28(1),29(1),30(2),32(3),34(2),36(1),38(1)','28(1),29(1),30(2),32(3),34(3),36(2),38(1)','28(1),29(2),30(2),32(2),34(2),36(2),38(1)','28(1),30(1),31(2),32(2),33(1),34(3),36(2),38(2),40(1)','28(1),30(1),32(2),33(1),34(3),36(2),38(1)','28(1),30(1),32(2),34(1),36(1),38(1),40(1),42(1)','28(1),30(1),32(2),34(2),36(2)','28(1),30(1),32(2),34(3),36(2),38(2)','28(1),30(1),32(2),34(3),36(3),38(2),28(1)','28(1),30(2),32(2),34(2),36(1)','28(1),30(2),32(2),34(2),36(1),38(1)','28(1),30(2),32(2),34(2),36(2),38(1)','28(1),30(2),32(2),34(2),36(2),38(1),40(1)','28(1),30(2),32(2),34(2),36(2),38(1),40(1),42(1)','28(1),30(2),32(2),34(2),36(2),38(2),40(1)','28(1),30(2),32(2),34(2),36(2),38(2),40(1),42(1)','28(1),30(2),32(3),34(2),36(1),38(1)','28(1),30(2),32(3),34(3),36(1)','28(1),30(2),32(3),34(3),36(2)','28(1),30(2),32(3),34(3),36(2),38(1)','28(1),30(2),32(3),34(3),36(2),38(1),40(1)','28(1),30(2),32(3),34(3),36(2),38(2),40(1)','28(1),30(2),32(3),34(3),36(2),38(2),40(1),42(1),44(1)','28(1),30(2),32(3),34(3),36(3),38(1),40(1)','28(1),30(2),32(3),34(3),36(3),38(2),40(1)','28(2),29(2),30(2),32(3),34(1)','28(2),29(2),30(3),32(3),34(2),36(2),38(2)','28(2),30(2),31(1),32(3),33(1),34(2),36(1)','28(2),30(2),32(3),34(3),36(2),38(1),40(1)','28(2),30(2),32(3),34(4),36(2),38(1)','28(2),30(3),32(2),34(2),36(1)','28(2),30(3),32(3),34(2),36(2),38(1)','28(2),30(3),32(3),34(3),36(1)','28(2),30(3),32(3),34(3),36(2),38(1),40(1)','28(2),30(3),32(4),34(2),36(1),38(1)','28(3),30(3),32(3),34(3),36(1),38(1)','28/30(1),29/30(1),30/30(1),31/30(1),32/30(2),34/30(2),36/30(1)  |  32/32(2),33/32(1),34/32(2),36/32(1),38/32(1),40/32(1)  |  36/34(1)','28/30(1),29/30(1),30/30(1),32/30(2)  |  30/32(1),31/32(1),32/32(1),33/32(1),34/32(2),36/32(1),38/32(1)  |  34/34(1),36/34(1)','28/30(1),29/30(1),30/30(1),32/30(2)  |  30/32(1),32/32(1),34/32(2),36/32(1)','28/30(1),29/30(1),30/30(1),32/30(2)  |  30/32(1),32/32(2),34/32(2),36/32(2)  |  32/34(2),34/34(1)','28/30(1),30/30(1),31/30(1),32/30(2),33/30(1),34/30(2),36/30(1)  |  34/32(2),36/32(1),38/32(1),40/32(1),42/32(1)','28/30(1),30/30(1),32/30(1)  |  30/32(1),32/32(1),34/32(2),36/32(1),38/32(1),40/32(1),42/32(1),44/32(1)  |  32/34(1),34/34(1),36/34(1)','28/30(1),30/30(1),32/30(1),34/30(1)  |  30/32(1),32/32(1),34/32(2),36/32(1)','28/30(2),29/30(2),30/30(1),32/30(2)  |  30/32(1),31/32(1),32/32(1),33/32(1),34/32(2),36/32(1),38/32(1),40/32(1)  |  34/34(1),36/34(1)','28/30(2),29/30(2),30/30(2),32/30(3),34/30(1)','29(1),30(1),31(2),32(2),33(2),34(2),36(1),38(1),40(1),42(1)','29(1),30(1),31(2),32(3),33(2),34(1),36(1)','29/30(1),30/30(1),31/30(1),32/30(2),34/30(2)  |  32/32(2),33/32(1),34/32(2),36/32(2),38/32(1)  |  32/34(1),36/34(1),36/34(1)','29/30(1),30/30(1),31/30(1),32/30(2),34/30(2)  |  32/32(2),33/32(1),34/32(2),36/32(2),38/32(1)  |  36/34(1)','30(1),31(1),32(2),33(1),34(3),36(2),38(1),40(1)','30(1),32(2),34(2),36(1)','30(1),32(2),34(2),36(2),38(2),40(1)','30(1),32(2),34(3),36(2),38(1)','30(1),32(2),34(3),36(2),38(2)','30(1),32(3),34(1),36(1),38(2)','30(1),32(3),34(2),36(1)','30(2),32(3),34(2),36(1)','30(2),32(3),34(3),36(2)','30(2),32(3),34(3),36(2),38(1)','30(2),32(3),34(3),36(2),38(1),40(1)','30/30(1),32/30(1),34/30(1),36/30(2)  |  32/32(1),34/32(1),36/32(1),38/32(1)  |  34/34(1),36/34(1),38/34(1)','30/30(1),32/30(1),34/30(2),36/30(2)  |  32/32(1),34/32(2),36/32(1),38/32(2),40/32(1),42/32(1)  |  34/34(1)','30/30(1),32/30(2),34/30(2),36/30(1)  |  31/32(1),32/32(1),33/32(1),34/32(2),36/32(1),38/32(2),40/32(1),42/32(1),44/32(1)  |  36/34(1)','30/30(1),32/30(2),34/30(2),36/30(2)  |  32/32(1),34/32(1),36/32(1),38/32(2),40/32(1),42/32(1)  |  34/34(1),36/34(1)','32(1) | 28(1),29(1),30(1),31(1),32(2),34(2),36(1) | 32(2),33(1),34(2),36(1),38(1),40(1)','32(1) | 28(1),29(1),30(1),31(1),32(2),34(2),36(1) | 32(3),33(1),34(2),36(1),38(1)','32(2),34(3),36(2),38(1)','32/30(1),33/30(2),34/30(2),36/30(2),38/30(1)  |  31/32(1),32/32(2),33/32(1),34/32(2),36/32(1),38/32(1),40/32(1)  |  34/34(1),36/34(1)','32/30(2),34/30(1)  |  30/32(1),32/32(1),34/32(2),36/32(1)','38(1),40(6),42(5),44(2),46(1)','S(1),M(2),L(2),XL(1)','S(1),M(2),L(2),XL(1),XXL(1)','S(1),M(2),L(2),XL(2),XXL(1)','S(1),M(2),L(2),XL(2),XXL(2),3XL(1)','S(1),M(2),L(3),XL(2)','S(1),M(2),L(3),XL(2),XXL(1)','S(1),M(2),L(3),XL(3),XXL(2),3XL(1)','S(1),M(3),L(2),XL(1)','S(1),M(3),L(3),XL(1)','S(2),M(2),L(1),XL(1)','S(2),M(3),L(3),XL(1)','S(2),M(3),L(3),XL(2),XXL(1)','XS(1),S(2),M(2),L(2),XL(1)','XS(1),S(2),M(2),L(2),XL(2),XXL(1)','XS(1),S(2),M(3),L(2),XL(1)','XS(1),S(2),M(3),L(2),XL(1),XXL(1)','XS(1),S(2),M(3),L(3),XL(1)','XS(1),S(2),M(3),L(3),XL(2),XXL(1)','XS(1),S(2),M(3),L(3),XL(3),XXL(2),3XL(1)','XS(1),S(3),M(3),L(3),XL(2),XXL(1),3XL(1)','XS(2),S(3),M(3),L(2),XL(1)','XS(2),S(3),M(3),L(2),XL(1),XXL(1)'])
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

    cat_features = ['DEPARTMAN', 'MODEL_TURU', 'MODEL_DETAYI','FIT',
                    'PASTAL_TURU', 'PASTAL_DETAYI', 'ASORTI']

    X_new_pool = Pool(X_new, cat_features=cat_features)
    prediction = model.predict(X_new_pool)[0]
    st.success(f"🔮 Tahmini Birim Sarfiyat: **{prediction:.2f}**")

