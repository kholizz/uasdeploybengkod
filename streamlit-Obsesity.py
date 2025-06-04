import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Judul Aplikasi
st.title("Prediksi Obesitas Berdasarkan Data")

# Upload file CSV
uploaded_file = st.file_uploader("Upload file CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Awal")
    st.dataframe(df)

    # Pilih fitur dan target
    st.subheader("Pilih Fitur dan Target")
    all_columns = df.columns.tolist()
    fitur = st.multiselect("Pilih fitur (X):", all_columns)
    target = st.selectbox("Pilih target (y):", all_columns)

    if fitur and target:
        X = df[fitur]
        y = df[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi
        st.subheader("Evaluasi Model")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Visualisasi jika data numerik
        if X.select_dtypes(include=np.number).shape[1] > 1:
            st.subheader("Visualisasi Korelasi")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax)
            st.pyplot(fig)

