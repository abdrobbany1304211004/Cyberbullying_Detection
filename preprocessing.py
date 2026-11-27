import re

# Kamus normalisasi kata slang / singkatan bahasa Indonesia (identik dengan notebook pelatihan)
SLANG_DICT = {
    'yg': 'yang',
    'udh': 'sudah',
    'udah': 'sudah',
    'gpp': 'tidak apa-apa',
    'klo': 'kalau',
    'kalo': 'kalau',
    'bgt': 'banget',
    'dlm': 'dalam',
    'krn': 'karena',
    'kyk': 'seperti',
    'trs': 'terus',
    'ntar': 'nanti',
    'jd': 'jadi',
    'jdi': 'jadi',
    'ga': 'tidak',
    'gak': 'tidak',
    'ngga': 'tidak',
    'gmn': 'bagaimana',
    'gimana': 'bagaimana',
    'lg': 'lagi',
    'dgn': 'dengan',
    'tdk': 'tidak',
    'tp': 'tapi',
    'tpi': 'tapi',
    'bener': 'benar',
    'beneran': 'benar-benar',
    'aja': 'saja',
    'dr': 'dari',
    'sm': 'sama',
    'dpt': 'dapat',
    'lu': 'kamu',
    'lo': 'kamu',
    'loe': 'kamu',
    'gw': 'saya',
    'gue': 'saya',
    'dkk': 'dan kawan-kawan',
}


def clean_text(text: str) -> str:
    """Membersihkan URL, tag RT/USER, noise hex, tanda baca, angka, dan case folding."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\\+x[0-9a-fA-F]{2}', ' ', text)
    text = re.sub(r'\b(RT|USER)\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).lower().strip()
    return text


def normalize_text(text: str) -> str:
    """Mengubah kata slang/singkatan menjadi kata baku sesuai SLANG_DICT."""
    if not text:
        return ""
    words = text.split()
    normalized_words = [SLANG_DICT.get(w, w) for w in words]
    return " ".join(normalized_words)


def preprocess_pipeline(raw_text: str) -> dict:
    """Menjalankan preprocessing lengkap dan mengembalikan setiap tahapnya untuk transparansi UI."""
    cleaned = clean_text(raw_text)
    normalized = normalize_text(cleaned)
    return {
        'raw': raw_text,
        'cleaned': cleaned,
        'normalized': normalized,
    }

