import json
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing import preprocess_pipeline

# ==============================================================================
# KONFIGURASI HALAMAN & GAYA TAMPILAN
# ==============================================================================
st.set_page_config(
    page_title="Deteksi Cyberbullying & Hate Speech (X)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .verdict-hs {
        background-color: #fef2f2;
        border-left: 6px solid #dc2626;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .verdict-nonhs {
        background-color: #f0fdf4;
        border-left: 6px solid #16a34a;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .verdict-title-hs { color: #991b1b; font-size: 1.35rem; font-weight: 700; margin: 0; }
    .verdict-title-nonhs { color: #166534; font-size: 1.35rem; font-weight: 700; margin: 0; }
    .small-muted { color: #64748b; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
BEST_MODEL_FALLBACK_DIR = os.path.join(BASE_DIR, "best_model_cyberbullying")
MAX_LEN = 128


# ==============================================================================
# PENEMUAN MODEL TERSIMPAN & CACHING
# ==============================================================================
def discover_available_models():
    """Mendeteksi model hasil fine-tuning lokal dari Colab (saved_models/ atau best_model_cyberbullying/)."""
    models = []
    registry_path = os.path.join(SAVED_MODELS_DIR, "models_registry.json")

    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        for item in reg:
            model_path = os.path.join(SAVED_MODELS_DIR, item["folder"])
            if os.path.isdir(model_path):
                item["path"] = model_path
                models.append(item)

    if not models and os.path.isdir(SAVED_MODELS_DIR):
        for folder in sorted(os.listdir(SAVED_MODELS_DIR)):
            cand_dir = os.path.join(SAVED_MODELS_DIR, folder)
            info_path = os.path.join(cand_dir, "model_info.json")
            if os.path.isdir(cand_dir) and os.path.exists(os.path.join(cand_dir, "config.json")):
                if os.path.exists(info_path):
                    with open(info_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                else:
                    meta = {
                        "key": folder,
                        "folder": folder,
                        "display_name": folder,
                        "model_architecture": folder,
                        "is_best": False
                    }
                meta["path"] = cand_dir
                models.append(meta)

    if not models and os.path.isdir(BEST_MODEL_FALLBACK_DIR):
        info_path = os.path.join(BEST_MODEL_FALLBACK_DIR, "model_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {"display_name": "Model Terbaik Tersimpan", "model_architecture": "local"}
        meta["key"] = "best_model"
        meta["is_best"] = True
        meta["path"] = BEST_MODEL_FALLBACK_DIR
        models.append(meta)

    # Urutkan agar model terbaik berada di urutan pertama (Default)
    models.sort(key=lambda x: (not x.get("is_best", False), x.get("display_name", "")))
    return models


@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_path: str):
    """Memuat model Transformer dan tokenizer ke memori secara efisien."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()
    return tokenizer, model, device


def predict_single(text: str, tokenizer, model, device, threshold: float = 0.50):
    """Menjalankan preprocessing, tokenisasi, dan inferensi Softmax pada 1 kalimat."""
    t_start = time.perf_counter()
    prep = preprocess_pipeline(text)
    norm_text = prep["normalized"] if prep["normalized"] else prep["cleaned"]

    inputs = tokenizer(
        norm_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_token_type_ids=False
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    inference_ms = (time.perf_counter() - t_start) * 1000.0
    prob_non_hs = float(probs[0])
    prob_hs = float(probs[1])
    is_hate_speech = prob_hs >= threshold
    subword_tokens = tokenizer.tokenize(norm_text)

    return {
        "prep": prep,
        "tokens": subword_tokens,
        "prob_non_hs": prob_non_hs,
        "prob_hs": prob_hs,
        "inference_ms": inference_ms,
        "label": 1 if is_hate_speech else 0,
        "verdict": "HATE SPEECH (Perundungan Siber)" if is_hate_speech else "BUKAN HATE SPEECH (Aman)"
    }


def predict_batch(texts, tokenizer, model, device, threshold: float = 0.50, batch_size: int = 32):
    """Memproses daftar teks CSV secara batch agar cepat."""
    t_start = time.perf_counter()
    cleaned_list = [preprocess_pipeline(str(t))["normalized"] for t in texts]
    all_probs_hs = []

    progress = st.progress(0.0, text="Memproses klasifikasi batch...")
    total = len(cleaned_list)

    for start in range(0, total, batch_size):
        batch_texts = cleaned_list[start:start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_token_type_ids=False
        ).to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"]
            )
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
        all_probs_hs.extend(probs.tolist())
        progress.progress(min(1.0, (start + len(batch_texts)) / total))

    progress.empty()
    elapsed_sec = time.perf_counter() - t_start
    preds = [1 if p >= threshold else 0 for p in all_probs_hs]
    return cleaned_list, all_probs_hs, preds, elapsed_sec


# ==============================================================================
# INISIALISASI SESSION STATE
# ==============================================================================
if "history" not in st.session_state:
    st.session_state["history"] = []
if "sample_input" not in st.session_state:
    st.session_state["sample_input"] = ""

available_models = discover_available_models()


# ==============================================================================
# SIDEBAR: IDENTITAS, PEMILIH MODEL, & THRESHOLD
# ==============================================================================
with st.sidebar:
    st.title("Cyberbullying Detector")
    st.caption("Implementasi Pengujian Skripsi — Media Sosial X")
    st.markdown(
        "**Peneliti:** Abdurrohman Robbany  \n"
        "**NIM:** 1304211004  \n"
        "**Prodi:** S1 Informatika PJJ - Telkom University"
    )
    st.divider()

    st.subheader("⚙️ Pengaturan Inferensi")
    if available_models:
        model_labels = []
        for m in available_models:
            tag = " ⭐ (Terbaik / Default)" if m.get("is_best") else ""
            model_labels.append(f"{m['display_name']}{tag}")

        selected_idx = st.selectbox(
            "Pilih Arsitektur Model:",
            options=list(range(len(available_models))),
            format_func=lambda idx: model_labels[idx]
        )
        selected_model_meta = available_models[selected_idx]

        if "macro_f1_mean" in selected_model_meta:
            st.info(
                f"**Statistik 5-Seed Model Dipilih:**\n"
                f"- Macro F1: **{selected_model_meta['macro_f1_mean']*100:.2f}%**\n"
                f"- PR-AUC: **{selected_model_meta.get('pr_auc_mean', 0)*100:.2f}%**\n"
                f"- ROC-AUC: **{selected_model_meta.get('roc_auc_mean', 0)*100:.2f}%**"
            )
    else:
        selected_model_meta = None
        st.warning("Belum ada folder model tersimpan di `saved_models/`.")

    threshold = st.slider(
        "Ambang Batas Probabilitas Hate Speech (Threshold):",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Default = 0.50. Nilai lebih rendah membuat deteksi lebih sensitif (Recall naik), nilai lebih tinggi membuat prediksi lebih selektif (Precision naik)."
    )

    st.divider()
    st.markdown(
        "<div class='small-muted'>Konfigurasi Pelatihan:<br>"
        "LR = 2e-5 | Batch = 32 | Class Weighting 2.34x | 5 Seeds</div>",
        unsafe_allow_html=True
    )


# ==============================================================================
# HEADER UTAMA
# ==============================================================================
st.title("Aplikasi Deteksi Ujaran Kebencian & Perundungan Siber")
st.markdown(
    "Prototype pengujian model Transformer (**DistilBERT Indo, DistilBERT Multilingual, dan XLM-RoBERTa**) "
    "untuk klasifikasi teks Bahasa Indonesia pada kasus perundungan siber di Media Sosial X."
)

if not available_models:
    st.error(
        "Folder model terlatih belum ditemukan di direktori aplikasi.\n\n"
        "**Cara Mengaktifkan Model Hasil Pelatihan Colab:**\n"
        "1. Jalankan notebook `Cyberbullying_Audit_Dataset.ipynb`.\n"
        "2. Unduh file **`paket_model_streamlit.zip`** .\n"
        f"3. Ekstrak seluruh isinya langsung ke folder: `{BASE_DIR}` (sehingga muncul folder `saved_models/`)."
    )

tab_single, tab_batch, tab_research = st.tabs([
    "🔍 Deteksi Kalimat & Komparasi Model",
    "📁 Uji Massal (Upload CSV)",
    "📊 Dashboard Performa Riset"
])


# ==============================================================================
# TAB 1: DETEKSI KALIMAT TUNGGAL & KOMPARASI 3 MODEL
# ==============================================================================
with tab_single:
    st.subheader("Uji Kalimat Baru")
    st.caption("Pilih contoh preset di bawah untuk demo cepat, atau ketik kalimat baru:")

    c1, c2, c3, c4 = st.columns(4)
    presets = {
        "Contoh 1 (Hate Speech)": "Dasar lu cewek goblok ga tau malu, pantesan gaada yg mau sama lu!",
        "Contoh 2 (Perundungan Fisik)": "Muka jelek item kayak lu gausah banyak gaya di sosmed deh najis bgt",
        "Contoh 3 (Umpatan Non-HS)": "Anjir macet parah bgt hari ini ditambah motor gue mogok di jalan raya",
        "Contoh 4 (Netral / Positif)": "Selamat ya atas pencapaian skripsinya hari ini, keren banget presentasinya!"
    }
    if c1.button("🔴 Hate Speech Eksplisit", use_container_width=True):
        st.session_state["sample_input"] = presets["Contoh 1 (Hate Speech)"]
    if c2.button("🔴 Perundungan / Slang", use_container_width=True):
        st.session_state["sample_input"] = presets["Contoh 2 (Perundungan Fisik)"]
    if c3.button("🟢 Kata Kasar Non-HS", use_container_width=True):
        st.session_state["sample_input"] = presets["Contoh 3 (Umpatan Non-HS)"]
    if c4.button("🟢 Kalimat Positif", use_container_width=True):
        st.session_state["sample_input"] = presets["Contoh 4 (Netral / Positif)"]

    user_text = st.text_area(
        "Masukkan teks cuitan atau komentar:",
        value=st.session_state["sample_input"],
        height=110,
        placeholder="Ketik kalimat Bahasa Indonesia di sini..."
    )

    col_btn1, col_btn2 = st.columns([1, 1])
    run_single = col_btn1.button("🔍 Deteksi dengan Model Terpilih", type="primary", use_container_width=True)
    run_compare = col_btn2.button("⚖️ Bandingkan Semua Model Sekaligus", use_container_width=True)

    if run_single:
        if not user_text.strip():
            st.warning("Silakan masukkan teks terlebih dahulu.")
        elif selected_model_meta is None:
            st.error("Model belum tersedia. Ekstrak `paket_model_streamlit.zip` terlebih dahulu.")
        else:
            with st.spinner(f"Memuat & memprediksi dengan {selected_model_meta['display_name']}..."):
                tok, mdl, dev = load_model_and_tokenizer(selected_model_meta["path"])
                res = predict_single(user_text, tok, mdl, dev, threshold=threshold)

            if res["label"] == 1:
                st.markdown(
                    f"<div class='verdict-hs'><p class='verdict-title-hs'>🚨 {res['verdict']}</p>"
                    f"<p class='small-muted'>Model: <b>{selected_model_meta['display_name']}</b> | Ambang Batas: {threshold:.2f}</p></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='verdict-nonhs'><p class='verdict-title-nonhs'>✅ {res['verdict']}</p>"
                    f"<p class='small-muted'>Model: <b>{selected_model_meta['display_name']}</b> | Ambang Batas: {threshold:.2f}</p></div>",
                    unsafe_allow_html=True
                )

            m1, m2, m3 = st.columns(3)
            m1.metric("Probabilitas Hate Speech (Kelas 1)", f"{res['prob_hs']*100:.2f}%")
            m2.metric("Probabilitas Bukan HS (Kelas 0)", f"{res['prob_non_hs']*100:.2f}%")
            m3.metric("⏱️ Waktu Deteksi (Inferensi)", f"{res['inference_ms']:.1f} ms")
            st.progress(res["prob_hs"], text=f"Tingkat Keyakinan Hate Speech: {res['prob_hs']*100:.2f}%")

            with st.expander("🔎 Lihat Transparansi Preprocessing & Tokenisasi Subword"):
                st.markdown(f"**1. Teks Asli:** `{res['prep']['raw']}`")
                st.markdown(f"**2. Cleaning (Regex & Case Folding):** `{res['prep']['cleaned']}`")
                st.markdown(f"**3. Normalisasi Slang:** `{res['prep']['normalized']}`")
                st.markdown(f"**4. Subword Tokens ({len(res['tokens'])} token):**")
                st.code(" | ".join(res["tokens"]), language="text")

            st.session_state["history"].insert(0, {
                "Waktu": datetime.datetime.now().strftime("%H:%M:%S"),
                "Model": selected_model_meta["display_name"],
                "Teks Input": user_text[:80] + ("..." if len(user_text) > 80 else ""),
                "Hasil Prediksi": "Hate Speech (1)" if res["label"] == 1 else "Bukan HS (0)",
                "Skor HS (%)": f"{res['prob_hs']*100:.2f}%",
                "Kecepatan": f"{res['inference_ms']:.1f} ms"
            })

    if run_compare:
        if not user_text.strip():
            st.warning("Silakan masukkan teks terlebih dahulu.")
        elif len(available_models) == 0:
            st.error("Model belum tersedia. Ekstrak `paket_model_streamlit.zip` terlebih dahulu.")
        else:
            st.markdown("#### ⚖️ Hasil Perbandingan Langsung Antar-Model")
            cols = st.columns(len(available_models))
            compare_timings = []
            for idx, m_meta in enumerate(available_models):
                with cols[idx]:
                    with st.spinner(f"Menguji {m_meta['display_name']}..."):
                        tok, mdl, dev = load_model_and_tokenizer(m_meta["path"])
                        res = predict_single(user_text, tok, mdl, dev, threshold=threshold)
                    compare_timings.append((m_meta["display_name"], res["inference_ms"]))
                    badge = "🚨 **HATE SPEECH**" if res["label"] == 1 else "✅ **BUKAN HS**"
                    st.markdown(f"**{m_meta['display_name']}** {'⭐' if m_meta.get('is_best') else ''}")
                    if res["label"] == 1:
                        st.error(f"{badge}\n\nSkor HS: **{res['prob_hs']*100:.2f}%**  \n⏱️ Waktu: **{res['inference_ms']:.1f} ms**")
                    else:
                        st.success(f"{badge}\n\nSkor HS: **{res['prob_hs']*100:.2f}%**  \n⏱️ Waktu: **{res['inference_ms']:.1f} ms**")
                    st.progress(res["prob_hs"])
            if len(compare_timings) >= 2:
                fastest_name, fastest_ms = min(compare_timings, key=lambda x: x[1])
                slowest_name, slowest_ms = max(compare_timings, key=lambda x: x[1])
                ratio = slowest_ms / fastest_ms if fastest_ms > 0 else 1.0
                st.caption(
                    f"⚡ **Efisiensi Inferensi:** `{fastest_name}` paling cepat (**{fastest_ms:.1f} ms**), "
                    f"sekitar **{ratio:.1f}x lebih cepat** dibanding `{slowest_name}` (**{slowest_ms:.1f} ms**) pada perangkat ini."
                )

    if st.session_state["history"]:
        st.divider()
        h_col1, h_col2 = st.columns([4, 1])
        h_col1.subheader("🕒 Riwayat Pengujian Sesi Ini")
        if h_col2.button("Hapus Riwayat"):
            st.session_state["history"] = []
            st.rerun()
        st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 2: UJI MASSAL VIA UPLOAD CSV
# ==============================================================================
with tab_batch:
    st.subheader("Klasifikasi Massal dari File CSV")
    st.markdown("Unggah file `.csv` berisi kumpulan komentar/tweet media sosial X untuk dimoderasi secara otomatis.")

    uploaded_csv = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded_csv is not None:
        df_up = pd.read_csv(uploaded_csv)
        st.write(f"Jumlah baris terdeteksi: **{len(df_up):,} baris** | Kolom: `{list(df_up.columns)}`")
        text_column = st.selectbox("Pilih kolom yang berisi teks cuitan:", options=list(df_up.columns))

        if st.button("🚀 Jalankan Klasifikasi Massal", type="primary"):
            if selected_model_meta is None:
                st.error("Model belum tersedia. Ekstrak `paket_model_streamlit.zip` terlebih dahulu.")
            else:
                tok, mdl, dev = load_model_and_tokenizer(selected_model_meta["path"])
                norm_texts, probs_hs, preds, elapsed_sec = predict_batch(df_up[text_column].fillna("").tolist(), tok, mdl, dev, threshold=threshold)

                df_out = df_up.copy()
                df_out["Teks_Normalized"] = norm_texts
                df_out["Probabilitas_HS (%)"] = [round(p * 100, 2) for p in probs_hs]
                df_out["Prediksi_Label"] = preds
                df_out["Kategori"] = ["Hate Speech" if p == 1 else "Bukan Hate Speech" for p in preds]

                n_hs = int(sum(preds))
                n_non = len(preds) - n_hs
                ms_per_tweet = (elapsed_sec / max(1, len(preds))) * 1000.0
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Total Tweet Diproses", f"{len(preds):,}")
                b2.metric("Terdeteksi Hate Speech", f"{n_hs:,} ({n_hs/len(preds)*100:.1f}%)")
                b3.metric("Bukan Hate Speech", f"{n_non:,} ({n_non/len(preds)*100:.1f}%)")
                b4.metric("⏱️ Total Waktu", f"{elapsed_sec:.2f} dtk", f"{ms_per_tweet:.1f} ms/tweet")

                fig, ax = plt.subplots(figsize=(5.5, 3.2))
                ax.bar(["Bukan Hate Speech", "Hate Speech"], [n_non, n_hs], color=["#16a34a", "#dc2626"], edgecolor="black")
                ax.set_ylabel("Jumlah Tweet")
                ax.set_title(f"Ringkasan Deteksi ({selected_model_meta['display_name']})")
                st.pyplot(fig)

                st.dataframe(df_out.head(50), use_container_width=True)
                csv_bytes = df_out.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 Download Hasil Klasifikasi (.csv)",
                    data=csv_bytes,
                    file_name="hasil_deteksi_cyberbullying.csv",
                    mime="text/csv"
                )


# ==============================================================================
# TAB 3: DASHBOARD PERFORMA & METODOLOGI RISET
# ==============================================================================
with tab_research:
    st.subheader("Ringkasan Hasil Eksperimen Multi-Seed (5 Random Seeds)")
    st.markdown(
        "Seluruh angka dan visualisasi di bawah dimuat otomatis dari file hasil eksperimen "
        "`Cyberbullying_Detection.ipynb`."
    )

    def resolve_artifact_path(filename: str):
        for candidate_dir in [
            BASE_DIR,
            os.path.join(BASE_DIR, "paket_model_streamlit"),
            os.path.dirname(BASE_DIR),
        ]:
            fpath = os.path.join(candidate_dir, filename)
            if os.path.exists(fpath):
                return fpath
        return None

    csv_candidates = [
        ("Tabel Komparasi Final 3 Model (Mean ± Std)", "hasil_komparasi_multiseed_3model.csv"),
        ("Tabel Sebaran Confusion Matrix", "tabel_cm_3_model_final.csv"),
        ("Tabel Konfigurasi Hyperparameter", "tabel_konfigurasi_parameter.csv"),
    ]
    for title, fname in csv_candidates:
        fpath = resolve_artifact_path(fname)
        if fpath:
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_csv(fpath), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Visualisasi Grafik Evaluasi Model")
    img_candidates = [
        ("Grafik Bar Chart Multi-Seed (± Std Dev)", "grafik_bar_komparasi_3model_multiseed.png"),
        ("Kurva ROC-AUC & PR-AUC Komparasi 3 Model", "kurva_roc_dan_pr_auc_3_model.png"),
        ("Confusion Matrix 3 Model (Orientasi Hate Speech)", "confusion_matrix_3_model_final.png"),
        ("Visualisasi Balancing Class Weighting", "visualisasi_balancing_class_weighting.png"),
    ]
    for title, fname in img_candidates:
        fpath = resolve_artifact_path(fname)
        if fpath:
            st.markdown(f"**{title}**")
            st.image(fpath, use_container_width=True)
