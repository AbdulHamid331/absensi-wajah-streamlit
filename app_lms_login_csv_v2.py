
import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt

# Load model dan data
model = pickle.load(open("model_xgb.pkl", "rb"))
data = pd.read_csv("dataset_mahasiswa_812.csv")

# Encode status_akademik_terakhir
data['status_akademik_terakhir'] = data['status_akademik_terakhir'].map({
    'IPK < 2.5': 0, 'IPK 2.5 - 3.0': 1, 'IPK > 3.0': 2
})

# Sidebar - Pilih Mahasiswa
st.sidebar.title("Prediksi Dropout Mahasiswa")
selected = st.sidebar.selectbox("Pilih Mahasiswa", data["Nama"])
mahasiswa = data[data["Nama"] == selected]

# Tampilkan informasi
st.title("Hasil Prediksi Dropout")
st.write("**Nama Mahasiswa:**", selected)

# Persiapkan data untuk prediksi
X = mahasiswa.drop(columns=["ID Mahasiswa", "Nama", "dropout"])

# Prediksi
prediksi = model.predict(X)[0]
proba = model.predict_proba(X)[0][1]

st.write("**Status Prediksi:**", "Dropout" if prediksi == 1 else "Tidak Dropout")
st.write("**Probabilitas Risiko Dropout:**", f"{proba:.2%}")

# Interpretasi dengan SHAP
st.subheader("Penjelasan Prediksi (Visualisasi SHAP)")
explainer = shap.Explainer(model)
shap_values = explainer(X)

fig = shap.plots.waterfall(shap_values[0], show=False)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig)
