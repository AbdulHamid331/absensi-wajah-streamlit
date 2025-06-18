import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from utils import get_embedding, cosine_similarity, knowledge_base

st.set_page_config(page_title="Absensi Wajah", layout="centered")
st.title("📸 Sistem Absensi Berbasis Wajah")
st.markdown("**Kelompok 6 – Knowledge-Based System**")

st.subheader("1️⃣ Ambil Foto Wajah")
img_file_buffer = st.camera_input("Silakan ambil foto untuk absensi")

if img_file_buffer is not None:
    # Konversi gambar ke OpenCV format
    bytes_data = img_file_buffer.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    st.image(image, channels="BGR", caption="📷 Gambar yang diambil")

    if st.button("🔍 Proses & Deteksi Wajah"):
        embedding = get_embedding(image)
        user_id, confidence = None, 0.0

        # Inference dengan rule KBS
        for rule in knowledge_base:
            sim = cosine_similarity(embedding, rule["embedding"])
            if sim > rule["threshold"]:
                user_id, confidence = rule["user"], sim
                break

        if user_id:
            st.success(f"✅ Absensi Berhasil: {user_id} (Confidence: {confidence:.2f})")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("logs/absensi_log.csv", "a") as f:
                f.write(f"{timestamp},{user_id},{confidence:.2f}\n")
        else:
            st.error("❌ Wajah tidak dikenali!")

st.subheader("📊 Riwayat Absensi")
try:
    df = pd.read_csv("logs/absensi_log.csv", names=["Waktu", "User", "Confidence"])
    st.dataframe(df.tail(10))
except FileNotFoundError:
    st.info("Belum ada data absensi yang tercatat.")
