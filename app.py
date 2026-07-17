import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date

DB_PATH = "data_servis.db"

# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS produk (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama TEXT NOT NULL,
            kategori TEXT,
            modal REAL NOT NULL,
            harga_jual REAL NOT NULL,
            stok INTEGER DEFAULT 0,
            dibuat_pada TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS penjualan (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            produk_id INTEGER,
            nama_produk TEXT,
            qty INTEGER,
            modal_satuan REAL,
            harga_satuan REAL,
            total REAL,
            untung REAL,
            tanggal TEXT,
            FOREIGN KEY (produk_id) REFERENCES produk(id)
        )
    """)
    conn.commit()
    conn.close()


def df_produk():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM produk ORDER BY nama ASC", conn)
    conn.close()
    return df


def df_penjualan():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM penjualan ORDER BY tanggal DESC, id DESC", conn)
    conn.close()
    return df


def tambah_produk(nama, kategori, modal, harga_jual, stok):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO produk (nama, kategori, modal, harga_jual, stok, dibuat_pada) VALUES (?,?,?,?,?,?)",
        (nama, kategori, modal, harga_jual, stok, datetime.now().strftime("%Y-%m-%d %H:%M")),
    )
    conn.commit()
    conn.close()


def hapus_produk(produk_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM produk WHERE id=?", (produk_id,))
    conn.commit()
    conn.close()


def catat_penjualan(produk_id, nama_produk, qty, modal_satuan, harga_satuan, tanggal):
    total = qty * harga_satuan
    untung = qty * (harga_satuan - modal_satuan)
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """INSERT INTO penjualan
           (produk_id, nama_produk, qty, modal_satuan, harga_satuan, total, untung, tanggal)
           VALUES (?,?,?,?,?,?,?,?)""",
        (produk_id, nama_produk, qty, modal_satuan, harga_satuan, total, untung, tanggal),
    )
    # kurangi stok kalau tersedia
    c.execute("UPDATE produk SET stok = MAX(stok - ?, 0) WHERE id=?", (qty, produk_id))
    conn.commit()
    conn.close()


def hapus_penjualan(penjualan_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM penjualan WHERE id=?", (penjualan_id,))
    conn.commit()
    conn.close()


def rupiah(n):
    try:
        return "Rp " + f"{n:,.0f}".replace(",", ".")
    except Exception:
        return "Rp 0"


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------

st.set_page_config(page_title="Aplikasi Servis HP", page_icon="🔧", layout="wide")
init_db()

st.sidebar.title("🔧 Servis HP")
st.sidebar.caption("Kelola produk, harga, dan penjualan")
menu = st.sidebar.radio(
    "Menu",
    [
        "➕ Tambah Produk / Bahan",
        "🔍 Cari Produk",
        "🧾 Catat Penjualan",
        "📊 Rekap Penjualan",
    ],
)

KATEGORI_LIST = ["Sparepart", "Alat", "Jasa Servis", "Lainnya"]

# ---------------------------------------------------------
# MENU 1: TAMBAH PRODUK / BAHAN
# ---------------------------------------------------------
if menu == "➕ Tambah Produk / Bahan":
    st.title("➕ Tambah Produk / Bahan")
    st.caption("Masukkan sparepart, alat, atau jasa servis beserta modal dan harga jualnya.")

    with st.form("form_tambah_produk", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            nama = st.text_input("Nama produk / bahan", placeholder="Contoh: LCD Redmi 10")
            kategori = st.selectbox("Kategori", KATEGORI_LIST)
            stok = st.number_input("Stok awal", min_value=0, step=1, value=0)
        with col2:
            modal = st.number_input("Modal (Rp)", min_value=0.0, step=1000.0, format="%.0f")
            harga_jual = st.number_input("Harga jual (Rp)", min_value=0.0, step=1000.0, format="%.0f")

        if modal > 0 and harga_jual > 0:
            untung = harga_jual - modal
            margin = (untung / harga_jual * 100) if harga_jual else 0
            st.info(f"Perkiraan untung per unit: **{rupiah(untung)}** ({margin:.0f}% dari harga jual)")

        submitted = st.form_submit_button("Simpan Produk", use_container_width=True)
        if submitted:
            if not nama.strip():
                st.error("Nama produk wajib diisi.")
            elif modal <= 0 or harga_jual <= 0:
                st.error("Modal dan harga jual harus lebih dari 0.")
            else:
                tambah_produk(nama.strip(), kategori, modal, harga_jual, int(stok))
                st.success(f"Produk '{nama}' berhasil disimpan.")

    st.divider()
    st.subheader("Daftar Produk Tersimpan")
    data = df_produk()
    if data.empty:
        st.info("Belum ada produk. Tambahkan lewat form di atas.")
    else:
        tampil = data.copy()
        tampil["Margin (%)"] = ((tampil["harga_jual"] - tampil["modal"]) / tampil["harga_jual"] * 100).round(0)
        tampil = tampil.rename(columns={
            "nama": "Nama", "kategori": "Kategori", "modal": "Modal",
            "harga_jual": "Harga Jual", "stok": "Stok"
        })
        st.dataframe(
            tampil[["Nama", "Kategori", "Modal", "Harga Jual", "Margin (%)", "Stok"]],
            use_container_width=True, hide_index=True
        )

        with st.expander("Hapus produk"):
            opsi = {f"{row.nama} (id {row.id})": row.id for row in data.itertuples()}
            pilihan = st.selectbox("Pilih produk yang mau dihapus", list(opsi.keys()))
            if st.button("Hapus produk ini", type="secondary"):
                hapus_produk(opsi[pilihan])
                st.success("Produk dihapus.")
                st.rerun()

# ---------------------------------------------------------
# MENU 2: CARI PRODUK
# ---------------------------------------------------------
elif menu == "🔍 Cari Produk":
    st.title("🔍 Cari Produk / Bahan")
    data = df_produk()

    col1, col2 = st.columns([2, 1])
    with col1:
        keyword = st.text_input("Cari nama produk / bahan", placeholder="Ketik nama, misal: LCD, baterai...")
    with col2:
        filter_kategori = st.selectbox("Filter kategori", ["Semua"] + KATEGORI_LIST)

    hasil = data.copy()
    if keyword:
        hasil = hasil[hasil["nama"].str.contains(keyword, case=False, na=False)]
    if filter_kategori != "Semua":
        hasil = hasil[hasil["kategori"] == filter_kategori]

    st.caption(f"Ditemukan {len(hasil)} produk")

    if hasil.empty:
        st.warning("Produk tidak ditemukan.")
    else:
        tampil = hasil.copy()
        tampil["Margin (%)"] = ((tampil["harga_jual"] - tampil["modal"]) / tampil["harga_jual"] * 100).round(0)
        tampil = tampil.rename(columns={
            "nama": "Nama", "kategori": "Kategori", "modal": "Modal",
            "harga_jual": "Harga Jual", "stok": "Stok"
        })
        st.dataframe(
            tampil[["Nama", "Kategori", "Modal", "Harga Jual", "Margin (%)", "Stok"]],
            use_container_width=True, hide_index=True
        )

# ---------------------------------------------------------
# MENU 3: CATAT PENJUALAN
# ---------------------------------------------------------
elif menu == "🧾 Catat Penjualan":
    st.title("🧾 Catat Penjualan")
    data = df_produk()

    if data.empty:
        st.warning("Belum ada produk. Tambahkan produk dulu di menu 'Tambah Produk / Bahan'.")
    else:
        with st.form("form_penjualan", clear_on_submit=True):
            opsi = {f"{row.nama} — stok: {row.stok}": row.id for row in data.itertuples()}
            pilihan_nama = st.selectbox("Pilih produk / jasa", list(opsi.keys()))
            produk_id = opsi[pilihan_nama]
            baris = data[data["id"] == produk_id].iloc[0]

            col1, col2 = st.columns(2)
            with col1:
                qty = st.number_input("Jumlah terjual", min_value=1, step=1, value=1)
                tanggal = st.date_input("Tanggal", value=date.today())
            with col2:
                st.metric("Modal satuan", rupiah(baris["modal"]))
                st.metric("Harga jual satuan", rupiah(baris["harga_jual"]))

            total = qty * baris["harga_jual"]
            untung = qty * (baris["harga_jual"] - baris["modal"])
            st.info(f"Total: **{rupiah(total)}**  ·  Untung: **{rupiah(untung)}**")

            submitted = st.form_submit_button("Simpan Penjualan", use_container_width=True)
            if submitted:
                catat_penjualan(
                    int(produk_id), baris["nama"], int(qty),
                    baris["modal"], baris["harga_jual"], tanggal.strftime("%Y-%m-%d")
                )
                st.success("Penjualan berhasil dicatat.")
                st.rerun()

    st.divider()
    st.subheader("Transaksi Terbaru")
    penjualan = df_penjualan()
    if penjualan.empty:
        st.info("Belum ada transaksi.")
    else:
        tampil = penjualan.head(15).rename(columns={
            "nama_produk": "Produk", "qty": "Qty", "modal_satuan": "Modal Satuan",
            "harga_satuan": "Harga Satuan", "total": "Total", "untung": "Untung", "tanggal": "Tanggal"
        })
        st.dataframe(
            tampil[["Tanggal", "Produk", "Qty", "Modal Satuan", "Harga Satuan", "Total", "Untung"]],
            use_container_width=True, hide_index=True
        )

# ---------------------------------------------------------
# MENU 4: REKAP PENJUALAN
# ---------------------------------------------------------
elif menu == "📊 Rekap Penjualan":
    st.title("📊 Rekap Penjualan")
    penjualan = df_penjualan()

    if penjualan.empty:
        st.info("Belum ada data penjualan untuk direkap.")
    else:
        penjualan["tanggal"] = pd.to_datetime(penjualan["tanggal"])
        min_tgl, max_tgl = penjualan["tanggal"].min().date(), penjualan["tanggal"].max().date()

        col1, col2 = st.columns(2)
        with col1:
            tgl_mulai = st.date_input("Dari tanggal", value=min_tgl)
        with col2:
            tgl_akhir = st.date_input("Sampai tanggal", value=max_tgl)

        filtered = penjualan[
            (penjualan["tanggal"] >= pd.to_datetime(tgl_mulai)) &
            (penjualan["tanggal"] <= pd.to_datetime(tgl_akhir))
        ]

        total_omzet = filtered["total"].sum()
        total_untung = filtered["untung"].sum()
        jumlah_transaksi = len(filtered)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Omzet", rupiah(total_omzet))
        m2.metric("Total Keuntungan", rupiah(total_untung))
        m3.metric("Jumlah Transaksi", jumlah_transaksi)

        st.divider()

        if not filtered.empty:
            st.subheader("Omzet per Hari")
            per_hari = filtered.groupby(filtered["tanggal"].dt.date)[["total", "untung"]].sum()
            st.bar_chart(per_hari)

            st.subheader("Produk Terlaris (berdasarkan qty terjual)")
            terlaris = (
                filtered.groupby("nama_produk")["qty"].sum()
                .sort_values(ascending=False).head(10)
            )
            st.bar_chart(terlaris)

            st.subheader("Detail Transaksi")
            tampil = filtered.rename(columns={
                "nama_produk": "Produk", "qty": "Qty", "modal_satuan": "Modal Satuan",
                "harga_satuan": "Harga Satuan", "total": "Total", "untung": "Untung", "tanggal": "Tanggal"
            })
            tampil["Tanggal"] = tampil["Tanggal"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                tampil[["Tanggal", "Produk", "Qty", "Modal Satuan", "Harga Satuan", "Total", "Untung"]],
                use_container_width=True, hide_index=True
            )

            csv = tampil.to_csv(index=False).encode("utf-8")
            st.download_button("Unduh rekap (CSV)", csv, "rekap_penjualan.csv", "text/csv")
        else:
            st.warning("Tidak ada transaksi pada rentang tanggal ini.")
