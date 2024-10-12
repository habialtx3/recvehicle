import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load dataset
file_path = 'sigma.xlsx'  # Path ke dataset Anda
df = pd.read_excel(file_path)

# Label encoding
le_domisili = LabelEncoder()
le_tujuan = LabelEncoder()
le_kendaraan = LabelEncoder()

# Pastikan kolom yang ada di DataFrame
if 'domisili' in df.columns and 'tujuan' in df.columns and 'nama_kendaraan' in df.columns and 'alasan' in df.columns:
    df['domisili'] = le_domisili.fit_transform(df['domisili'])
    df['tujuan'] = le_tujuan.fit_transform(df['tujuan'])
    df['kendaraan'] = le_kendaraan.fit_transform(df['nama_kendaraan'])

    X = df[['domisili', 'tujuan']]
    y_kendaraan = df['kendaraan']

    X_train, X_test, y_train_kendaraan, y_test_kendaraan = train_test_split(X, y_kendaraan, test_size=0.2, random_state=42)

    knn_kendaraan = KNeighborsClassifier(n_neighbors=3)
    knn_kendaraan.fit(X_train, y_train_kendaraan)

    def main():
        st.title("Rekomendasi Kendaraan Menggunakan KNN")

        # Unique options for selectbox
        domisili_options = le_domisili.inverse_transform(df['domisili'].unique())
        tujuan_options = le_tujuan.inverse_transform(df['tujuan'].unique())

        selected_domisili = st.selectbox("Pilih Domisili:", domisili_options)
        selected_tujuan = st.selectbox("Pilih Tujuan Wisata:", tujuan_options)

        if st.button("Tampilkan Rekomendasi"):
            try:
                domisili_num = le_domisili.transform([selected_domisili])[0]
                tujuan_num = le_tujuan.transform([selected_tujuan])[0]
                user_input = [[domisili_num, tujuan_num]]

                # Main prediction
                kendaraan_pred = knn_kendaraan.predict(user_input)
                kendaraan_main = le_kendaraan.inverse_transform(kendaraan_pred)[0]
                alasan_main = df[df['kendaraan'] == kendaraan_pred[0]]['alasan'].values[0]

                # Nearest neighbors
                distances, indices = knn_kendaraan.kneighbors(user_input)

                if len(indices[0]) >= 3:
                    alt_indices = indices[0][1:3]
                    alt_kendaraan_pred = knn_kendaraan.predict(X_train.iloc[alt_indices])
                    kendaraan_alternatif = le_kendaraan.inverse_transform(alt_kendaraan_pred)
                    alasan_alternatif = [df[df['kendaraan'] == alt]['alasan'].values[0] for alt in alt_kendaraan_pred]

                    main_proba = knn_kendaraan.predict_proba(user_input)[0][kendaraan_pred[0]] * 100
                    kendaraan_sudah_rekomendasi = {kendaraan_main}

                    st.markdown(f"<div style='padding: 20px; background-color: #e0f7fa; border-radius: 10px;'>"
                                 f"<h3 style='color: black;'>Rekomendasi Utama Kendaraan:</h3>"
                                 f"<h2 style='color: #000000;'>{kendaraan_main} ({main_proba:.2f}%)</h2>"
                                 f"<p style='color: black;'>Alasan: {alasan_main}</p>"
                                 f"</div>", unsafe_allow_html=True)

                    for i, (kendaraan, alasan) in enumerate(zip(kendaraan_alternatif, alasan_alternatif)):
                        if kendaraan not in kendaraan_sudah_rekomendasi:
                            st.markdown(f"<br> <div style='padding: 10px; background-color: #f1f8e9; border-radius: 10px;'>"
                                         f"<h4 style='color: black;'>Alternatif Kendaraan {i + 1}:</h4>"
                                         f"<h5 style='color: #000000;'>{kendaraan}</h5>"
                                         f"<p style='color: black;'>Alasan: {alasan}</p>"
                                         f"</div>", unsafe_allow_html=True)
                            kendaraan_sudah_rekomendasi.add(kendaraan)
                        if len(kendaraan_sudah_rekomendasi) == 3:
                            break
                else:
                    st.error("Tidak cukup tetangga untuk menghasilkan alternatif kendaraan.")

                # Menampilkan Google Maps Embedded berdasarkan tujuan
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                
                if selected_tujuan == "Istana Maimun":
                    st.markdown("""
                    <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3982.248643398496!2d98.66008771047859!3d3.5299489964294883!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x30313ab287c92185%3A0x944ea83f9578f7a9!2sTaman%20Hutan%20Kota%20Cadika!5e0!3m2!1sid!2sid!4v1728759139075!5m2!1sid!2sid" 
                    style="border:0; width: 100%; height: 450px;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
                    """, unsafe_allow_html=True)

                elif selected_tujuan == "Masjid Raya Medan":
                    st.markdown("""
                    <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d7964.040377639248!2d98.68121696205642!3d3.5828374855050464!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3031304954318df1%3A0xd9cfba5dbd38bae5!2sMasjid%20Raya%20Al-Mashun!5e0!3m2!1sid!2sid!4v1728758897119!5m2!1sid!2sid" 
                    style="border:0; width: 100%; height: 450px;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
                    """, unsafe_allow_html=True)

                elif selected_tujuan == "Stadion Teladan":
                    st.markdown("""
                    <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3982.0979124225914!2d98.6930619104787!3d3.564931996394289!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3031305c871e2f09%3A0xb4c92b13ab4bcb8e!2sStadion%20Teladan%20Medan!5e0!3m2!1sid!2sid!4v1728759100076!5m2!1sid!2sid" 
                    style="border:0; width: 100%; height: 450px;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
                    """, unsafe_allow_html=True)

                elif selected_tujuan == "Maha Vihara Adhi Maitreya Cemara":
                    st.markdown("""
                    <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3982.301881013287!2d98.7920200104785!3d3.517510096442012!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3031398892a0aab9%3A0x9673d22fe6c33bc4!2sVihara%20Adhi%20Maitreya!5e0!3m2!1sid!2sid!4v1728759161706!5m2!1sid!2sid" 
                    style="border:0; width: 100%; height: 450px;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
                    """, unsafe_allow_html=True)

                elif selected_tujuan == "Cadika":
                    st.markdown("""
                    <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d31655.50449181871!2d98.6535794604774!3d3.5615138314841804!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x303136776c2da3db%3A0x27917cf0e44b91d1!2sCadika%20Medan!5e0!3m2!1sid!2sid!4v1728759200085!5m2!1sid!2sid" 
                    style="border:0; width: 100%; height: 450px;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")

    if __name__ == "__main__":
        main()
