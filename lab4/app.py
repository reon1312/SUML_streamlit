import streamlit as st
import pickle
import h5py
from datetime import datetime

startTime = datetime.now()

filename = "lab4/model.h5"
model = pickle.load(open(filename, "rb"))

sex_d = {0: "Kobieta", 1: "Mężczyzna"}
pclass_d = {0: "Pierwsza", 1: "Druga", 2: "Trzecia"}
embarked_d = {0: "Cherbourg", 1: "Queenstown", 2: "Southampton"}


def main():
    st.set_page_config(page_title="Czy by przeżył?")
    overview = st.container()
    left, right = st.columns(2)  # Fixed: changed from st.container(2) to st.columns(2)
    prediction = st.container()

    with overview:
        st.title("Czy by przeżył?")

    with left:
        sex_radio = st.radio("Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        pclass_radio = st.radio("Klasa", list(pclass_d.keys()),
                                format_func=lambda x: pclass_d[x])  # Fixed: changed sex_d to pclass_d
        embarked_radio = st.radio("Port", list(embarked_d.keys()), format_func=lambda x: embarked_d[
            x])  # Fixed: changed prediction to embarked_d and sex_d to embarked_d

    with right:
        age_slider = st.slider("Wiek", value=50, min_value=1, max_value=120)
        sibsp_slider = st.slider("# Liczba rodzeństwa/partnera", min_value=0, max_value=8)
        parch_slider = st.slider("# Liczba rodziców/dzieci", min_value=0, max_value=6)
        fare_slider = st.slider("Cena biletu", min_value=0, max_value=500, step=10)


    Q = 1 if embarked_radio == 1 else 0
    S = 1 if embarked_radio == 2 else 0
    data = [[pclass_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio, sex_radio]]

    survival = model.predict(data)
    s_confidence = model.predict_proba(data)


    with prediction:
        st.header("Przeżył? {0}".format("Tak" if survival[0] == 1 else "Nie"))
        st.subheader("Pewność predykcji {0:.2f}%".format(s_confidence[0][survival][0] * 100))

    st.image(
        "https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG")

    with st.expander("📊 Informacje o modelu"):
        st.write(f"**Liczba cech:** {model.n_features_in_}")

        if hasattr(model, 'feature_names_in_'):
            st.write("**Nazwy cech:**", list(model.feature_names_in_))

        st.write(f"**Kształt współczynników:** {model.coef_.shape}")
        st.write("**Współczynniki:**")
        st.write(model.coef_)

if __name__ == "__main__":
    main()
