import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit app configuration
st.set_page_config(page_title="Medical Charges Analysis", layout="wide")
st.title("Medical Charges Analysis")
st.markdown("""<style>
    /* Background color */
    .reportview-container {
        background-color: #f4f1ec;
        color: #333333;
    }
    /* Sidebar style */
    .sidebar .sidebar-content {
        background-color: #e6e2d3;
        padding: 2rem;
        border-radius: 10px;
    }
    /* General font styling */
    html, body, [class*="css"] {
        font-family: 'Georgia', serif;
        font-size: 18px;
    }
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #4a4a4a;
        font-family: 'Georgia', serif;
    }
    /* Centering DataFrame previews */
    .stDataFrame {
        margin-left: auto;
        margin-right: auto;
        border: 1px solid #ccc;
        border-radius: 8px;
        background-color: #ffffff;
    }
    /* Button styling */
    button {
        background-color: #5f9ea0 !important;
        color: white !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
    }
    button:hover {
        background-color: #4682b4 !important;
    }
    /* Boxplot and charts padding */
    .block-container {
        padding: 2rem;
    }
</style>""", unsafe_allow_html=True)

st.markdown("""**Selamat datang!** Aplikasi ini dirancang untuk menganalisis data biaya medis. Silakan eksplorasi data yang telah tersedia.""", unsafe_allow_html=True)

# Load dataset
file_path = "d:\\KULIAH MUHAMADIYAH\\SEMESTER 3\\DATA MINING\\UAS\\Regression.csv"
data = pd.read_csv(file_path)

# Sidebar Navigation
st.sidebar.title("Navigasi")
pages = ["Data Overview", "Data Analysis", "Model Training"]
selected_page = st.sidebar.radio("Pilih Halaman", pages)

# Page 1: Data Overview
if selected_page == "Data Overview":
    st.header("Overview Dataset")
    st.markdown("""Pada halaman ini, Anda dapat melihat gambaran umum dari dataset yang digunakan. Dataset ini berisi informasi mengenai biaya medis yang dapat dianalisis untuk memahami berbagai faktor yang memengaruhi biaya tersebut.""", unsafe_allow_html=True)

    if st.button("🔍 Preview Dataset"):
        st.write("<h3 style='text-align: left;'>Preview Dataset</h3>", unsafe_allow_html=True)
        st.write("""Menampilkan 5 baris pertama dari dataset, yang memberikan gambaran awal mengenai data yang ada.""", unsafe_allow_html=True)
        st.dataframe(data.head())

    if st.button("📊 Distribusi Variabel Kategori"):
        st.write("<h3 style='text-align: left;'>Distribusi Variabel Kategori</h3>", unsafe_allow_html=True)
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            st.subheader(f"Distribusi {col.capitalize()}:")
            st.bar_chart(data[col].value_counts())

# Page 2: Data Analysis
elif selected_page == "Data Analysis":
    st.header("Analisis Data")
    st.write("""Halaman ini memberikan analisis mendalam mengenai hubungan antar variabel dalam dataset serta distribusi biaya medis (charges).""")

    # Filter only numeric columns for correlation
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    st.subheader("Heatmap Korelasi")
    st.write("Heatmap di bawah ini menunjukkan hubungan antara variabel numerik dalam dataset.")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("Pairplot (Hubungan Antar Variabel)")
    st.markdown("""Pairplot membantu untuk memvisualisasikan hubungan antara beberapa variabel numerik. Pilih kolom yang ingin Anda analisis lebih lanjut.""")
    pairplot_cols = st.multiselect("Pilih Kolom untuk Pairplot", numeric_data.columns.tolist())

    if pairplot_cols:
        fig = sns.pairplot(numeric_data[pairplot_cols])
        st.pyplot(fig)

    st.subheader("Distribusi Biaya Medis (Charges)")
    st.markdown("""Distribusi biaya medis menunjukkan bagaimana biaya tersebut tersebar di seluruh data.""")
    fig, ax = plt.subplots()
    sns.histplot(data['charges'], kde=True, ax=ax, color="blue")
    ax.set_title("Distribusi Charges")
    ax.set_xlabel("Charges")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Analisis Biaya Berdasarkan Kategori")
    for col in ['sex', 'smoker', 'region']:
        fig, ax = plt.subplots()
        sns.boxplot(x=col, y='charges', data=data, ax=ax)
        ax.set_title(f"Biaya Medis Berdasarkan {col.capitalize()}")
        st.pyplot(fig)

# Page 3: Model Training
elif selected_page == "Model Training":
    st.header("Pelatihan Model")
    st.write("""Halaman ini memberikan pengalaman dalam membangun model prediksi biaya medis menggunakan regresi linear. Anda dapat melihat bagaimana model dibangun dan dievaluasi.""")
    
    # Preprocess Data Button
    if st.button("🔧 Preprocessing Data"):
        st.write("### Preprocessing Data")
        st.write("""Pada tahap ini, data kategorikal diubah menjadi variabel dummy menggunakan get_dummies untuk memungkinkan model regresi linear bekerja dengan baik.""")
        data_encoded = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
        st.write("Data yang telah di-encode:", data_encoded.head())

    # Split Data Button
    if st.button("📊 Membagi Data"):
        st.write("### Membagi Data")
        st.write("""Data dibagi menjadi dua bagian: data pelatihan (80%) dan data pengujian (20%).""")
        data_encoded = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
        X = data_encoded.drop(columns=['charges'])
        y = data_encoded['charges']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        # Save training data to session state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

    # Train Model Button
    if st.button("🧑‍💻 Latih Model"):
        st.write("### Membangun dan Melatih Model")
        if 'X_train' in st.session_state:
            model = LinearRegression()
            model.fit(st.session_state.X_train, st.session_state.y_train)

            st.session_state.model = model

            # Evaluate Model
            st.write("### Evaluasi Model")
            y_pred = model.predict(st.session_state.X_test)
            mse = mean_squared_error(st.session_state.y_test, y_pred)
            r2 = r2_score(st.session_state.y_test, y_pred)

            # Convert MSE to IDR
            conversion_rate = 15000  # 1 USD = 15,000 IDR
            mse_idr = mse * conversion_rate  # MSE dalam IDR

            st.write(f"Mean Squared Error: Rp {mse_idr:,.2f}")
            st.write(f"R2 Score: {r2:.2f}")

            st.write("Model telah dilatih dengan sukses!")

        else:
            st.write("Data belum dibagi! Silakan lakukan preprocessing dan pembagian data terlebih dahulu.")

    # Make Prediction
    st.subheader("Prediksi Biaya Medis")
    if 'model' in st.session_state:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ['Male', 'Female'])
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        children = st.number_input("Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoker", ['Yes', 'No'])
        region = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])

        # Preprocess the input data
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

        # Add any missing columns with 0 values to match the model's feature set
        for col in st.session_state.X_train.columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match the training data
        input_data = input_data[st.session_state.X_train.columns]

        # Make prediction
        prediction_usd = st.session_state.model.predict(input_data)
        conversion_rate = 15000  # 1 USD = 15,000 IDR
        prediction_idr = prediction_usd[0] * conversion_rate  # Convert to IDR

        st.write(f"Prediksi Biaya Medis: Rp {prediction_idr:,.2f}")
