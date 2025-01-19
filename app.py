import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit app configuration
st.set_page_config(page_title="Medical Charges Analysis", layout="wide")
st.title("Medical Charges Analysis")
st.markdown("""
<style>
    .reportview-container { background-color: #f4f1ec; color: #333333; }
    .sidebar .sidebar-content { background-color: #e6e2d3; padding: 2rem; border-radius: 10px; }
    html, body, [class*="css"] { font-family: 'Georgia', serif; font-size: 18px; }
    h1, h2, h3, h4, h5, h6 { color: #4a4a4a; }
    .stDataFrame { margin-left: auto; margin-right: auto; border: 1px solid #ccc; border-radius: 8px; background-color: #ffffff; }
    button { background-color: #5f9ea0 !important; color: white !important; border-radius: 5px !important; padding: 0.5rem 1rem !important; }
    button:hover { background-color: #4682b4 !important; }
    .block-container { padding: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("**Selamat datang!** Aplikasi ini dirancang untuk menganalisis data biaya medis. Silakan eksplorasi data yang tersedia.")

# Load dataset
file_path = "C:/Users/agung/UAS/Regression.csv"
if os.path.exists(file_path):
    data = pd.read_csv(file_path)
else:
    st.error(f"File tidak ditemukan di: {file_path}")
    data = pd.DataFrame()  # Buat dataset kosong sebagai pengganti

# Sidebar Navigation
st.sidebar.title("Navigasi")
pages = ["Data Overview", "Data Analysis", "Model Training"]
selected_page = st.sidebar.radio("Pilih Halaman", pages)

# Page 1: Data Overview
if selected_page == "Data Overview":
    st.header("Overview Dataset")
    st.markdown("Pada halaman ini, Anda dapat melihat gambaran umum dari dataset yang digunakan.")

    if not data.empty:
        if st.button("🔍 Preview Dataset"):
            st.dataframe(data.head())

        if st.button("📊 Distribusi Variabel Kategori"):
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                for col in categorical_cols:
                    st.subheader(f"Distribusi {col.capitalize()}:")
                    st.bar_chart(data[col].value_counts())
            else:
                st.error("Tidak ada kolom kategori yang ditemukan dalam dataset.")
    else:
        st.error("Dataset kosong atau belum dimuat.")

# Page 2: Data Analysis
elif selected_page == "Data Analysis":
    st.header("Analisis Data")
    st.write("Halaman ini memberikan analisis mendalam mengenai hubungan antar variabel dalam dataset.")

    if not data.empty:
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        if not numeric_data.empty:
            st.subheader("Heatmap Korelasi")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)

            st.subheader("Pairplot (Hubungan Antar Variabel)")
            pairplot_cols = st.multiselect("Pilih Kolom untuk Pairplot", numeric_data.columns.tolist())
            if pairplot_cols:
                fig = sns.pairplot(numeric_data[pairplot_cols])
                st.pyplot(fig)

            st.subheader("Distribusi Biaya Medis (Charges)")
            if 'charges' in data.columns:
                fig, ax = plt.subplots()
                sns.histplot(data['charges'], kde=True, ax=ax, color="blue")
                ax.set_title("Distribusi Charges")
                st.pyplot(fig)
            else:
                st.error("Kolom 'charges' tidak ditemukan dalam dataset.")
        else:
            st.error("Tidak ada kolom numerik untuk dianalisis.")
    else:
        st.error("Dataset kosong atau belum dimuat.")

# Page 3: Model Training
elif selected_page == "Model Training":
    st.header("Pelatihan Model")
    st.write("Halaman ini membangun model prediksi biaya medis menggunakan regresi linear.")

    if not data.empty:
        if st.button("🔧 Preprocessing Data"):
            data_encoded = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
            st.write("Data yang telah di-encode:", data_encoded.head())

        if st.button("📊 Membagi Data"):
            data_encoded = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
            X = data_encoded.drop(columns=['charges'], errors='ignore')
            y = data_encoded['charges'] if 'charges' in data_encoded.columns else None

            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
            else:
                st.error("Kolom target 'charges' tidak ditemukan dalam dataset.")

        if st.button("🧑‍💻 Latih Model"):
            if 'X_train' in st.session_state:
                model = LinearRegression()
                model.fit(st.session_state.X_train, st.session_state.y_train)
                st.session_state.model = model

                y_pred = model.predict(st.session_state.X_test)
                mse = mean_squared_error(st.session_state.y_test, y_pred)
                r2 = r2_score(st.session_state.y_test, y_pred)
                st.write(f"Mean Squared Error: Rp {mse * 15000:,.2f}")
                st.write(f"R2 Score: {r2:.2f}")
            else:
                st.warning("Data belum dibagi! Silakan lakukan preprocessing dan pembagian data terlebih dahulu.")

        st.subheader("Prediksi Biaya Medis")
        if 'model' in st.session_state:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            sex = st.selectbox("Sex", ['Male', 'Female'])
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
            children = st.number_input("Children", min_value=0, max_value=10, value=0)
            smoker = st.selectbox("Smoker", ['Yes', 'No'])
            region = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])

            input_data = pd.DataFrame({
                'age': [age],
                'sex_male': [1 if sex == "Male" else 0],
                'bmi': [bmi],
                'children': [children],
                'smoker_yes': [1 if smoker == "Yes" else 0],
                'region_northeast': [1 if region == "northeast" else 0],
                'region_southeast': [1 if region == "southeast" else 0],
                'region_southwest': [1 if region == "southwest" else 0]
            })

            input_data = input_data.reindex(columns=st.session_state.X_train.columns, fill_value=0)
            prediction_usd = st.session_state.model.predict(input_data)
            prediction_idr = prediction_usd[0] * 15000
            st.write(f"Prediksi Biaya Medis: Rp {prediction_idr:,.2f}")
        else:
            st.warning("Model belum dilatih. Harap latih model terlebih dahulu.")
    else:
        st.error("Dataset kosong atau belum dimuat.")
