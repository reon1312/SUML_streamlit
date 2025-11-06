import streamlit as st
import pandas as pd
import time
import matplotlib as plt
import os
from transformers import pipeline

# Uwaga: Aby używać tłumaczenia, musisz zainstalować sentencepiece:
# pip install sentencepiece

# zaczynamy od zaimportowania bibliotek

st.success('Gratulacje! Z powodzeniem uruchomiłeś aplikację')
# streamlit jest wykorzystywany do tworzenia aplikacji
# z tego powodu dobrą praktyką jest informowanie użytkownika o postępie, błędach, etc.

# Inne przykłady do wypróbowania:
# st.balloons() # animowane balony ;)
# st.error('Błąd!') # wyświetla informację o błędzie
# st.warning('Ostrzeżenie, działa, ale chyba tak sobie...')
# st.info('Informacja...')
# st.success('Udało się!')

# st.spinner()
# with st.spinner(text='Pracuję...'):
# time.sleep(2)
# st.success('Done')
# możemy dzięki temu "ukryć" późniejsze ładowanie aplikacji

st.title('Lab3 - Streamlit - Analziator tekstu')
# title, jak sama nazwa wskazuje, używamy do wyświetlenia tytułu naszej aplikacji

# Wyświetlanie zdjęć obok siebie
col1, col2 = st.columns(2)

with col1:
    st.image("london.png", caption="Londyn - Big Ben i Parlament", use_container_width=True)

with col2:
    st.image("berlin.png", caption="Berlin - Brama Brandenburska", use_container_width=True)

st.header('Przetwarzanie języka naturalnego')


@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")


@st.cache_resource
def load_translation_model():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")


option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie angielski -> niemiecki",
    ],
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        with st.spinner('Analizuję tekst...'):
            classifier = load_sentiment_model()
            answer = classifier(text)
        st.write(answer)

elif option == "Tłumaczenie angielski -> niemiecki":
    text = st.text_area(label="Wpisz tekst po angielsku")
    if text:
        with st.spinner('Ładuję model tłumaczenia...'):
            try:
                translator = load_translation_model()
                st.info('Model załadowany, tłumaczę...')
                translation = translator(text)
                st.success('Tłumaczenie zakończone!')
                st.write("**Tłumaczenie:**", translation[0]['translation_text'])
                st.balloons()  # animowane balony ;)
            except Exception as e:
                st.error(f'Wystąpił błąd: {str(e)}')

st.text('s25153')
