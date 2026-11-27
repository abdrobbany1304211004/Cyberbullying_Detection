# Aplikasi Deteksi Ujaran Kebencian & Perundungan Siber (Media Sosial X)

Aplikasi web interaktif berbasis **Streamlit** untuk menguji dan membandingkan performa 3 arsitektur Transformer (**DistilBERT Monolingual Indonesia**, **DistilBERT Multilingual**, dan **XLM-RoBERTa Base**) hasil penelitian Tugas Akhir.

## Cara Menjalankan Aplikasi

1. **Selesaikan pelatihan** `Cyberbullying_Audit_Dataset.ipynb`.
2. **Unduh file `paket_model_streamlit.zip`** , lalu ekstrak seluruh isinya ke dalam folder `Aplikasi Detecting Cyberbullying/`.
   Pastikan susunan foldernya menjadi seperti ini:
   ```text
   Aplikasi Detecting Cyberbullying/
   ├── app.py
   ├── preprocessing.py
   ├── requirements.txt
   ├── saved_models/
   │   ├── models_registry.json
   │   ├── distilbert_indo/
   │   ├── distilbert_multi/
   │   └── xlm_roberta/
   ├── hasil_komparasi_multiseed_3model.csv
   ├── tabel_cm_3_model_final.csv
   ├── tabel_konfigurasi_parameter.csv
   └── *.png
   ```
3. **Instal pustaka (jika belum ada):**
   ```bash
   pip install -r requirements.txt
   ```
4. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```

