import string
import streamlit as st
import pandas as pd
import numpy as np
import re
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="Dashboard Analisis Kasus Narkoba",
    layout="wide"
)

st.title("📊 Dashboard Analisis Kasus Penyalahgunaan Narkoba")

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    df = pd.read_excel("KasusClean.xlsx")
    return df

df = load_data()

# =============================
# DATA CLEANING
# =============================
df = df.drop(columns=['No', 'Jumlah Tersangka'])
df['Pekerjaan'] = df['Pekerjaan'].str.strip()
df['Kecamatan'] = df['Kecamatan'].str.strip()
df['Besaran/Jumlah BB'] = df['Besaran/Jumlah BB'].astype(str)

# Extract BB gram
df['BB_gram'] = (
    df['Besaran/Jumlah BB']
    .str.findall(r'([\d,]+)\s*gram')
    .apply(lambda x: sum(float(i.replace(',', '.')) for i in x) if x else 0)
)

# Extract BB butir
df['BB_butir'] = (
    df['Besaran/Jumlah BB']
    .str.findall(r'([\d\.]+)\s*butir')
    .apply(lambda x: sum(float(i.replace('.', '')) for i in x) if x else 0)
)

# =============================
# SIDEBAR FILTER
# =============================
st.sidebar.header("🔍 Filter Data")

tahun_filter = st.sidebar.multiselect(
    "Pilih Tahun",
    sorted(df['Tahun'].unique()),
    default=sorted(df['Tahun'].unique())
)

df = df[df['Tahun'].isin(tahun_filter)]

# =============================
# METRIK UTAMA
# =============================
col1, col2, col3 = st.columns(3)

col1.metric("Total Kasus", len(df))
col2.metric("Total BB (Gram)", f"{df['BB_gram'].sum():.2f}")
col3.metric("Total BB (Butir)", int(df['BB_butir'].sum()))

# =============================
# 1. JUMLAH KASUS PER KECAMATAN
# =============================
st.subheader("📍 Jumlah Kasus per Kecamatan")

kasus_kecamatan = df.groupby('Kecamatan').size().sort_values(ascending=False)

st.bar_chart(kasus_kecamatan)

# ============================0
# 2. JUMLAH KASUS PER TAHUN
# =============================
st.subheader("📆 Jumlah Kasus per Tahun")

df['Tahun'] = (
    df['Tahun']
    .astype(str)
    .str.extract(r'(\d{4})')
    .astype(int)
)

kasus_per_tahun = (
    df.groupby('Tahun')
      .size()
      .sort_index()
)

kasus_per_tahun.index = kasus_per_tahun.index.astype(str)

st.bar_chart(
    kasus_per_tahun,
    height= 700)

# =============================
# 3. JUMLAH KASUS PER TAHUN DI TIAP KECAMATAN
# =============================
st.subheader("🏘️ Jumlah Kasus per Tahun di Tiap Kecamatan")

kasus_tahun_kecamatan = (
    df.groupby(['Tahun', 'Kecamatan'])
      .size()
      .unstack(fill_value=0)
)

kasus_tahun_kecamatan.index = kasus_tahun_kecamatan.index.astype(str)

st.dataframe(kasus_tahun_kecamatan)

st.bar_chart(
    kasus_tahun_kecamatan,
    height=1200  # default biasanya sekitar 400
)

# =============================
# 4. JUMLAH KASUS PER PEKERJAAN
# =============================
df["Pekerjaan"] = df["Pekerjaan"].str.strip()

st.subheader("👷 Jumlah Kasus per Pekerjaan")

kasus_pekerjaan = (
    df.groupby('Pekerjaan')
       .size()
       .sort_values(ascending=False)
)
st.bar_chart(kasus_pekerjaan)

kasus_tahun_pekerjaan = (
    df.groupby(['Tahun', 'Pekerjaan'])
      .size()
      .unstack(fill_value=0)
)

st.bar_chart(
    kasus_tahun_pekerjaan,
    height=1200)




# =============================
# PARSING BB (VERSI VALID COLAB)
# =============================

df['Besaran/Jumlah BB'] = df['Besaran/Jumlah BB'].astype(str)

df['Besaran/Jumlah BB'] = df['Besaran/Jumlah BB'].str.replace(
    r',\s+',
    ',',
    regex=True
)

# Ngatasin spasi setelah koma
df['bb_satuan'] = df['Besaran/Jumlah BB'].str.extract(
    r'(gram|butir)', flags=re.IGNORECASE
)

# Untuk satuan gram
df['BB_gram'] = (
    df['Besaran/Jumlah BB']
    .str.findall(r'([\d,]+)\s*gram')
    .apply(
        lambda x: sum(float(i.replace(',', '.')) for i in x) if len(x) > 0 else 0
    )
)

# Untuk satuan butir
df['BB_butir'] = (
    df['Besaran/Jumlah BB']
    .str.findall(r'([\d\.]+)\s*butir')
    .apply(
        lambda x: sum(float(i.replace('.', '')) for i in x) if len(x) > 0 else 0
    )
)

# =============================
# 5. GRAFIK TOTAL BB PER KECAMATAN
# =============================
st.subheader("⚖️ Total Barang Bukti per Kecamatan")

bb_per_kecamatan = (
    df.groupby('Kecamatan')
      .agg(
          Total_Gram=('BB_gram', 'sum'),
          Total_Butir=('BB_butir', 'sum'),
          Jumlah_Kasus=('Kecamatan', 'size')
      )
      .sort_values(by='Jumlah_Kasus', ascending=False)
)

st.dataframe(bb_per_kecamatan)

bb_gram_chart = (
    bb_per_kecamatan
    .reset_index()                     
    .sort_values('Total_Gram', ascending=False)
)

st.bar_chart(
    bb_gram_chart,
    x='Kecamatan',
    y='Total_Gram'
    )

bb_butir_chart = (
    bb_per_kecamatan
    .reset_index()
    .sort_values('Total_Butir', ascending=False)
)

st.bar_chart(
    bb_butir_chart,
    x='Kecamatan',
    y='Total_Butir'
    )

# =============================
# 6. CLUSTERING & MAP
# =============================
st.subheader("🗺️ Peta Persebaran Daerah Rawan")

fitur_kecamatan = (
    df.groupby('Kecamatan')
      .agg(
          jumlah_kasus=('Kecamatan','count'),
          total_gram=('BB_gram','sum'),
          total_butir=('BB_butir','sum')
      )
      .reset_index()
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    fitur_kecamatan[['jumlah_kasus','total_gram','total_butir']]
)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
fitur_kecamatan['cluster'] = kmeans.fit_predict(X_scaled)
fitur_kecamatan['cluster'] = fitur_kecamatan['cluster'].map({0: 1, 1: 0})

# Load peta
gdf_map = gpd.read_file("34.04_kecamatan.geojson")
gdf_map['kecamatan'] = gdf_map['nm_kecamatan'].str.lower().str.strip()
fitur_kecamatan['Kecamatan'] = fitur_kecamatan['Kecamatan'].str.lower().str.strip()

gdf_merge = gdf_map.merge(
    fitur_kecamatan,
    left_on='kecamatan',
    right_on='Kecamatan',
    how='left'
)

fig, ax = plt.subplots(figsize=(8,8))
def generate_labels(n):
    labels = []
    alphabet = string.ascii_uppercase
    i = 0
    while len(labels) < n:
        label = ""
        x = i
        while True:
            label = alphabet[x % 26] + label
            x = x // 26 - 1
            if x < 0:
                break
        labels.append(label)
        i += 1
    return labels

labels = generate_labels(len(gdf_merge))
gdf_merge['label'] = labels

gdf_merge['centroid'] = gdf_merge.geometry.centroid

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

gdf_merge.plot(
    column='cluster',
    cmap='RdYlGn_r',
    legend=True,
    edgecolor='black',
    missing_kwds={
        "color": "white",
        "edgecolor": "black",
        "label": "Tidak ada data"
    },
    ax=ax
)

for idx, row in gdf_merge.iterrows():
    if row['centroid'] is not None:
        ax.text(
            row['centroid'].x,
            row['centroid'].y,
            row['label'],
            fontsize=9,
            ha='center',
            va='center',
            fontweight='bold'
        )

ax.set_title('Peta Tingkat Kerawanan Penyalahgunaan Narkoba per Kecamatan')
ax.axis('off')
plt.show()

legend_table = gdf_merge[['label', 'nm_kecamatan']].sort_values('label')
legend_table.rename(
    columns={
        'label': 'Kode',
        'nm_kecamatan': 'Nama Kecamatan'
    },
    inplace=True
)
legend_table

st.pyplot(fig)