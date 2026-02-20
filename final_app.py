import streamlit as st
import pandas as pd
import re
import string
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Dashboard Narkoba Sleman",
    layout="wide"
)

st.title("📊 Dashboard Analisis Narkoba Sleman")

# =====================================================
# SIDEBAR MENU (2 HALAMAN SAJA)
# =====================================================
#menu = st.sidebar.radio(
#   "Pilih Halaman",
#    ["🚔 Ungkap Kasus", "🏥 Rehabilitasi", ]
#)

tab1, tab2, tab3 = st.tabs(["🚔 Ungkap Kasus", "🏥 Rehabilitasi", "Insight dan Kesimpulan"])

with tab1:
    st.header("Halaman Ungkap Kasus")
    # =====================================================
    # =====================================================
    # 🚔 HALAMAN 1 : UNGKAP KASUS
    # =====================================================
    # =====================================================

    #if menu == "🚔 Ungkap Kasus":

    
    @st.cache_data
    def load_data():
        dfkasus = pd.read_excel("KasusClean.xlsx")
        return dfkasus

    dfkasus = load_data()

     # =============================
    # DATA CLEANING
    # =============================
    dfkasus = dfkasus.drop(columns=['No', 'Jumlah Tersangka'])
    dfkasus['Pekerjaan'] = dfkasus['Pekerjaan'].str.strip()
    dfkasus['Kecamatan'] = dfkasus['Kecamatan'].str.strip()
    dfkasus['Besaran/Jumlah BB'] = dfkasus['Besaran/Jumlah BB'].astype(str)

    # Extract BB gram
    dfkasus['BB_gram'] = (
        dfkasus['Besaran/Jumlah BB']
        .str.findall(r'([\d,]+)\s*gram')
        .apply(lambda x: sum(float(i.replace(',', '.')) for i in x) if x else 0)
    )

    # Extract BB butir
    dfkasus['BB_butir'] = (
        dfkasus['Besaran/Jumlah BB']
        .str.findall(r'([\d\.]+)\s*butir')
        .apply(lambda x: sum(float(i.replace('.', '')) for i in x) if x else 0)
    )

    # =============================
    # SIDEBAR FILTER
    # =============================
    #st.sidebar.header("🔍 Filter Data")

    #tahun_filter = st.sidebar.multiselect(
    #    "Pilih Tahun",
    #    sorted(dfkasus['Tahun'].unique()),
    #    default=sorted(dfkasus['Tahun'].unique()),
    #    key="tahun_kasus"
    #)

    #dfkasus = dfkasus[dfkasus['Tahun'].isin(tahun_filter)]
    tahun_filter = st.multiselect(
        "Pilih Tahun Kasus",
        sorted(dfkasus['Tahun'].unique()),
        default=sorted(dfkasus['Tahun'].unique())
    )

    

    # =============================
    # METRIK UTAMA
    # =============================
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Kasus", len(dfkasus))
    col2.metric("Total BB (Gram)", f"{dfkasus['BB_gram'].sum():.2f}")
    col3.metric("Total BB (Butir)", int(dfkasus['BB_butir'].sum()))    

    # =============================
    # 1. JUMLAH KASUS PER KECAMATAN
    # =============================
    st.subheader("📍 Jumlah Kasus per Kecamatan")

    kasus_kecamatan = dfkasus.groupby('Kecamatan').size().sort_values(ascending=False)

    st.bar_chart(kasus_kecamatan)

    # ============================0
    # 2. JUMLAH KASUS PER TAHUN
    # =============================
    st.subheader("📆 Jumlah Kasus per Tahun")

    dfkasus['Tahun'] = (
        dfkasus['Tahun']
        .astype(str)
        .str.extract(r'(\d{4})')
        .astype(int)
    )

    kasus_per_tahun = (
            dfkasus.groupby('Tahun')
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
        dfkasus.groupby(['Tahun', 'Kecamatan'])
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
    dfkasus["Pekerjaan"] = dfkasus["Pekerjaan"].str.strip()

    st.subheader("👷 Jumlah Kasus per Pekerjaan")

    kasus_pekerjaan = (
        dfkasus.groupby('Pekerjaan')
        .size()
        .sort_values(ascending=False)
    )
    st.bar_chart(kasus_pekerjaan)

    kasus_tahun_pekerjaan = (
        dfkasus.groupby(['Tahun', 'Pekerjaan'])
        .size()
        .unstack(fill_value=0)
    )

    st.bar_chart(
        kasus_tahun_pekerjaan,
        height=1200)




    # =============================
    # PARSING BB (VERSI VALID COLAB)
    # =============================

    dfkasus['Besaran/Jumlah BB'] = dfkasus['Besaran/Jumlah BB'].astype(str)

    dfkasus['Besaran/Jumlah BB'] = dfkasus['Besaran/Jumlah BB'].str.replace(
        r',\s+',
        ',',
            regex=True
    )

    # Ngatasin spasi setelah koma
    dfkasus['bb_satuan'] = dfkasus['Besaran/Jumlah BB'].str.extract(
        r'(gram|butir)', flags=re.IGNORECASE
    )

    # Untuk satuan gram
    dfkasus['BB_gram'] = (
        dfkasus['Besaran/Jumlah BB']
        .str.findall(r'([\d,]+)\s*gram')
        .apply(
            lambda x: sum(float(i.replace(',', '.')) for i in x) if len(x) > 0 else 0
        )
    )

    # Untuk satuan butir
    dfkasus['BB_butir'] = (
        dfkasus['Besaran/Jumlah BB']
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
        dfkasus.groupby('Kecamatan')
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
        dfkasus.groupby('Kecamatan')
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
    fitur_kecamatan['cluster'] = fitur_kecamatan['cluster'].map({0: 0, 1: 1})
    st.dataframe(fitur_kecamatan, hide_index=True)

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
        inplace=True,
    )
    st.dataframe(legend_table, hide_index=True)

    st.pyplot(fig)   

with tab2:  
    st.header("Halaman Rehabilitasi")

    # =====================================================
    # =====================================================
    # 🏥 HALAMAN 2 : REHABILITASI
    # =====================================================
    # =====================================================

    #elif menu == "🏥 Rehabilitasi":
    def load_rehab():
        return pd.read_csv("datarehabdone.csv")

    dfrehab = load_rehab()

    def load_map():
        return gpd.read_file("34.04_kecamatan.geojson")
    gdf_map = load_map()


    dfrehab = dfrehab.drop(columns=['NO', 'STATUS PERNIKAHAN', 'AGAMA', 'GOL. DARAH'])

    dfrehab['PEKERJAAN'] = (
        dfrehab['PEKERJAAN']
            .fillna('TIDAK DIKETAHUI')
    )

    def clean_tanggal(text):
        if pd.isna(text):
            return pd.NaT

        text = str(text).upper().strip()

        bulan_map = {
            'JANUARI': '01', 'JAN': '01',
            'FEBRUARI': '02', 'FEB': '02',
            'MARET': '03', 'MAR': '03',
            'APRIL': '04', 'APR': '04',
            'MEI': '05', 'MAY': '05',
            'JUNI': '06', 'JUN': '06',
            'JULI': '07', 'JUL': '07',
            'AGUSTUS': '08', 'AGU': '08',
            'SEPTEMBER': '09', 'SEP': '09',
            'OKTOBER': '10', 'OKT': '10', 'OCT': '10',
            'NOVEMBER': '11', 'NOV': '11',
            'DESEMBER': '12', 'DES': '12'
        }

        for bulan, angka in bulan_map.items():
            text = re.sub(rf'\b{bulan}\b', angka, text)

        text = text.replace('-', ' ').replace('/', ' ')
        text = re.sub(r'\s+', ' ', text)

        try:
            return pd.to_datetime(text, dayfirst=True)
        except:
            return pd.NaT

    dfrehab['TANGGAL MASUK'] = dfrehab['TANGGAL MASUK'].apply(clean_tanggal)
    dfrehab['tahun'] = dfrehab['TANGGAL MASUK'].dt.year.astype('Int64')

    dfrehab['PEKERJAAN'] = (
        dfrehab['PEKERJAAN']
            .astype(str)
            .str.strip()
            .str.upper()
    )



    dfrehab['DIAGNOSA UTAMA'] = (
        dfrehab['DIAGNOSA UTAMA']
            .astype(str)
            .str.strip()
            .str.upper()
    )

    dfrehab['DIAGNOSA UTAMA'] = dfrehab['DIAGNOSA UTAMA'].str.replace(r'F\.', 'F', regex=True)

    dfrehab['ALAMAT DOMISILI'] = dfrehab['ALAMAT DOMISILI'].str.upper().str.strip()
    gdf_map['nm_kecamatan'] = gdf_map['nm_kecamatan'].str.upper().str.strip()

    kecamatan_Sleman = gdf_map['nm_kecamatan'].unique().tolist()
    
    #Feature engineering
    dfrehab['tahun'] = dfrehab['TANGGAL MASUK'].dt.year.astype('Int64')

    dfrehab = dfrehab.dropna(subset=['tahun'])

    def ekstrak_kecamatan_sleman(alamat):
        for kec in kecamatan_Sleman:
            if kec in alamat:
                return kec
        return None    

    dfrehab['kecamatan_Sleman'] = dfrehab['ALAMAT DOMISILI'].apply(ekstrak_kecamatan_sleman)

    dfrehab['status_wilayah'] = dfrehab['kecamatan_Sleman'].apply(
        lambda x: 'SLEMAN' if pd.notna(x) else 'LUAR SLEMAN'
    )

    dfrehab_sleman = dfrehab[dfrehab['kecamatan_Sleman'].notna()].copy()

    dfrehab_sleman.reset_index(drop=True, inplace=True)

    #=========================
    # SIDEBAR TAHUH
    #=========================

    #st.sidebar.header("🔍 Filter Data")

    #tahun_filter_rehab = st.sidebar.multiselect(
    #"Pilih Tahun",
    #sorted(dfrehab_sleman['tahun'].unique()),
    #default=sorted(dfrehab_sleman['tahun'].unique()),
    #key="tahun_rehab"
    #)

    #dfrehab_sleman = dfrehab_sleman[dfrehab_sleman['tahun'].isin(tahun_filter_rehab)]
    tahun_filter_rehab = st.multiselect(
        "Pilih Tahun Rehabilitasi",
        sorted(dfrehab_sleman['tahun'].unique()),
        default=sorted(dfrehab_sleman['tahun'].unique())
    )
    # ========================
    # METRIK
    # ========================
    col1, col2 = st.columns(2)
    col1.metric("Total Rehabilitasi", len(dfrehab))
    col2.metric("Jumlah Tahun Data", dfrehab['tahun'].nunique())

    #=========================
    #JUMLAH PER KECAMATAN
    #=========================
    st.subheader("📍 Jumlah Kasus per Kecamatan")

    kasus_kecamatan = dfrehab_sleman.groupby('kecamatan_Sleman').size().sort_values(ascending=False)

    st.bar_chart(kasus_kecamatan)

  # ============================0
    # 2. JUMLAH KASUS PER TAHUN
    # =============================
    st.subheader("📆 Jumlah Kasus per Tahun")

    dfrehab_sleman['tahun'] = (
        dfrehab_sleman['tahun']
        .astype(str)
        .str.extract(r'(\d{4})')
        .astype(int)
    )

    rehab_per_tahun = (
            dfrehab_sleman.groupby('tahun')
            .size()
            .sort_index()
        )

    rehab_per_tahun.index = rehab_per_tahun.index.astype(str)

    st.bar_chart(
        rehab_per_tahun,
        height= 700)

    # =============================
    # 3. JUMLAH KASUS PER TAHUN DI TIAP KECAMATAN
    # =============================
    st.subheader("🏘️ Jumlah Kasus per Tahun di Tiap Kecamatan")

    rehab_tahun_kecamatan = (
        dfrehab_sleman.groupby(['tahun', 'kecamatan_Sleman'])
        .size()
        .unstack(fill_value=0)
    )

    rehab_tahun_kecamatan.index = rehab_tahun_kecamatan.index.astype(str)

    st.dataframe(rehab_tahun_kecamatan)

    st.bar_chart(
        rehab_tahun_kecamatan,
        height=1200  # default biasanya sekitar 400
    )

    # =============================
    # 4. JUMLAH KASUS PER PEKERJAAN
    # =============================
    dfrehab_sleman["PEKERJAAN"] = dfrehab_sleman["PEKERJAAN"].str.strip()

    st.subheader("👷 Jumlah Kasus per Pekerjaan")

    rehab_pekerjaan = (
        dfrehab_sleman.groupby('PEKERJAAN')
        .size()
        .sort_values(ascending=False)
    )
    st.bar_chart(rehab_pekerjaan)

    rehab_tahun_pekerjaan = (
        dfrehab_sleman.groupby(['tahun', 'PEKERJAAN'])
        .size()
        .unstack(fill_value=0)
    )

    st.bar_chart(
        rehab_tahun_pekerjaan,
        height=1200)
    

    #=========================
    #REHAB PER UMUR
    #=========================
    bins = [0, 19, 24, 29, 34, 100]
    labels = ['<20', '20-24', '25-29', '30-34', '35+']

    dfrehab_sleman['kelompok_usia'] = pd.cut(
        dfrehab_sleman['USIA'],
        bins=bins,
        labels=labels
    )    

    rehab_usia = (
        dfrehab_sleman.groupby('kelompok_usia')
        .size()
        .sort_values(ascending=False)
    )
    st.bar_chart(rehab_usia)

    rehab_tahun_usia = (
        dfrehab_sleman.groupby(['tahun', 'kelompok_usia'])
        .size()
        .unstack(fill_value=0)
    )

    st.bar_chart(
        rehab_tahun_usia,
        height=1200)
    

    #=========================
    # DIAGNOSA UTAMA REHAB
    #=========================

    dfrehab_sleman['KODE_DIAGNOSA'] = (
        dfrehab_sleman['DIAGNOSA UTAMA']
            .astype(str)
            .str.upper()
            .str.extract(r'([A-Z]\d{2})')
    )

    #rehab_tahun_diagnosa = (
        #dfrehab_sleman.groupby(['tahun', 'KODE_DIAGNOSA'])
           #.size()
            #.unstack(fill_value=0)
            #.sort_index(axis=1)
    #)

    rehab_diagnosa = (
        dfrehab_sleman.groupby('KODE_DIAGNOSA')
        .size()
        .sort_values(ascending=False)
    )
    st.bar_chart(rehab_diagnosa)

    rehab_tahun_diagnosa = (
        dfrehab_sleman.groupby(['tahun', 'KODE_DIAGNOSA'])
        .size()
        .unstack(fill_value=0)
    )

    st.bar_chart(
        rehab_tahun_diagnosa,
        height=1200)

    st.text("Keterangan Kode Diagnosa")
    st.text("F10 = Gangguan akibat alkohol")
    st.text("F12 = Gangguan akibat kanabis (ganja)")
    st.text("F13 = Gangguan akibat sedatif / benzodiazepin")
    st.text("F15 = Gangguan akibat stimulan (shabu / amfetamin)")
    st.text("F16 = Gangguan akibat halusinogen")
    st.text("F19 = Gangguan akibat multiple drugs")
    st.text("Dxx = Penyakit lain (bukan gangguan zat, perlu verifikasi data)")
    # ========================
    # CLUSTERING PETA
    # ========================
    #======================
    # BETA
    #======================
    st.subheader("🗺️ Peta Persebaran Daerah Rawan")

    fitur_kecamatan_rehab = (
        dfrehab_sleman.groupby('kecamatan_Sleman')
        .agg(
            jumlah_kasus=('kecamatan_Sleman','count'),
        )
        .reset_index()
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(
        fitur_kecamatan_rehab[['jumlah_kasus']]
    )

    inertia = []
    silhouette_scores = []

    K = range(1, 8)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        inertia.append(kmeans.inertia_)

        if k > 1:
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(None)
 
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    fitur_kecamatan_rehab['cluster'] = kmeans.fit_predict(X_scaled)

    fitur_kecamatan_rehab.sort_values(
    'jumlah_kasus', ascending=False
    )

    cluster_rehab = fitur_kecamatan_rehab.copy()
    # Load peta
    gdf_map = gpd.read_file("34.04_kecamatan.geojson")
    gdf_map['kecamatan'] = gdf_map['nm_kecamatan'].str.lower().str.strip()
    fitur_kecamatan_rehab['Kecamatan'] = fitur_kecamatan_rehab['kecamatan_Sleman'].str.lower().str.strip()

    gdf_merge = gdf_map.merge(
        fitur_kecamatan_rehab,
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

with tab3:
    st.header("Insight dan Kesimpulan")
    st.markdown("---")
    st.subheader("📊 Ringkasan Statistik Utama")

    # ==============================
    # RINGKASAN UNGKAP KASUS
    # ==============================
    total_kasus = len(dfkasus)
    kecamatan_tertinggi_kasus = (
        dfkasus['Kecamatan']
        .value_counts()
        .idxmax()
    )

    tahun_tertinggi_kasus = (
        dfkasus['Tahun']
        .value_counts()
        .idxmax()
    )

    # ==============================
    # RINGKASAN REHABILITASI
    # ==============================
    total_rehab = len(dfrehab_sleman)

    kecamatan_tertinggi_rehab = (
        dfrehab_sleman['kecamatan_Sleman']
        .value_counts()
        .idxmax()
    )

    usia_dominan = (
        dfrehab_sleman['kelompok_usia']
        .value_counts()
        .idxmax()
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Kasus", total_kasus)
        st.write("Kecamatan Kasus Tertinggi:", kecamatan_tertinggi_kasus)
        st.write("Tahun Kasus Tertinggi:", tahun_tertinggi_kasus)

    with col2:
        st.metric("Total Rehabilitasi", total_rehab)
        st.write("Kecamatan Rehab Tertinggi:", kecamatan_tertinggi_rehab)
        st.write("Kelompok Usia Dominan:", usia_dominan)

    st.markdown("---")

    # ===================================
    # VISUALISASI PERBANDINGAN PER KECAMATAN
    # ===================================
    st.subheader("📍 Perbandingan Kasus vs Rehabilitasi per Kecamatan")

    kasus_per_kec = (
        dfkasus
        .groupby('Kecamatan')
        .size()
        .reset_index(name='Jumlah_Kasus')
    )

    rehab_per_kec = (
        dfrehab_sleman
        .groupby('kecamatan_Sleman')
        .size()
        .reset_index(name='Jumlah_Rehabilitasi')
    )

    kasus_per_kec['Kecamatan'] = kasus_per_kec['Kecamatan'].str.upper().str.strip()
    rehab_per_kec['kecamatan_Sleman'] = rehab_per_kec['kecamatan_Sleman'].str.upper().str.strip()

    df_compare = kasus_per_kec.merge(
        rehab_per_kec,
        left_on='Kecamatan',
        right_on='kecamatan_Sleman',
        how='outer'
    )

    df_compare['Kecamatan'] = df_compare['Kecamatan'].fillna(
        df_compare['kecamatan_Sleman']
    )

    df_compare[['Jumlah_Kasus', 'Jumlah_Rehabilitasi']] = (
        df_compare[['Jumlah_Kasus', 'Jumlah_Rehabilitasi']].fillna(0).astype(int)
    )

    df_compare = (
        df_compare[['Kecamatan', 'Jumlah_Kasus', 'Jumlah_Rehabilitasi']]
        .sort_values('Kecamatan')
        .reset_index(drop=True)
    )

    st.dataframe(
        df_compare, 
        use_container_width=True,
        hide_index=True)

    st.bar_chart(
        df_compare.set_index('Kecamatan')
    )

    st.markdown("---")

    # ===================================
    # ANALISIS CLUSTER UNGKAP
    # ===================================
    st.subheader("🧠 Insight Clustering Ungkap Kasus")

    cluster_summary = fitur_kecamatan.groupby('cluster')['Kecamatan'].count()

    st.write("Distribusi Cluster:")
    st.dataframe(cluster_summary)

    st.markdown("""
    Cluster dengan nilai lebih tinggi menunjukkan kecamatan dengan kombinasi:
    - Jumlah kasus tinggi
    - Total barang bukti besar
    """)

    st.markdown("---")

    # ===================================
    # ANALISIS CLUSTER REHAB
    # ===================================
    st.subheader("🧠 Insight Clustering Rehabilitasi")

    cluster_rehab_summary = fitur_kecamatan_rehab.groupby('cluster')['kecamatan_Sleman'].count()

    st.write("Distribusi Cluster Rehabilitasi:")
    st.dataframe(cluster_rehab_summary)

    st.markdown("---")

    # ===================================
    # KESIMPULAN OTOMATIS
    # ===================================
    st.subheader("📌 Kesimpulan Otomatis Berbasis Data")

    st.markdown(f"""
    1. Kecamatan dengan kasus tertinggi adalah **{kecamatan_tertinggi_kasus}**.
    2. Kecamatan dengan rehabilitasi tertinggi adalah **{kecamatan_tertinggi_rehab}**.
    3. Kelompok usia paling rentan adalah **{usia_dominan}**.
    4. Clustering menunjukkan adanya segmentasi wilayah risiko yang dapat dijadikan dasar kebijakan.
    """)

    st.markdown("Dengan insight di atas dapat disimpulkan bahwa banyak kasus pelaku penyalahgunaan atau pengedar narkoba yang banyak tersebar " \
    "di kecamatan Depok, Sleman, dan Ngaglik sehingga pemberantasan narkoba dapat difokuskan ke 3 kecamatan tersebut. Lalu terdapat 4 kecamatan " \
    "yang tidak memiliki catatan kasus sehingga dapat dikatakan aman, walaupun begitu tidak menutup kemungkinan bahwa di daerah tersebut tidak terjadi" \
    "tindakan penyalahgunaan dan peredaran narkoba")

    st.markdown("Untuk rehabilitasi tercatat ada banyak pengguna narkoba yang berasal daerah Sleman dan pengguna paling banyak berkisar pada umur di bawah " \
    "20 tahun, sehingga untuk langkah penyuluhan mengenai narkoba dapat difokuskan pada daerah Sleman dan kepada kelompok pemuda atau pelajar. " \
    "Lalu ada 3 wilayah yang dapat dikatakan aman karena tidak ada data pengguna yang melakukakn rehabilitasi dari daerah tersebut, namun juga tidak " \
    "menutup kemungkinan bahwa di daerah-daerah tersebut tidak ada korban penyalahgunaan narkoba")

    st.success("Dashboard ini dapat digunakan sebagai sistem pendukung keputusan berbasis data.")
