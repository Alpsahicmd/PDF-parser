import os
import re
import json
import requests
import pdfplumber
# import cv2 /for OCR
# import numpy as np /for OCR
# from pdf2image import convert_from_path /for OCR
# from PIL import Image /for OCR
from collections import Counter
import time

import nltk

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

#########################################
# 0. PATH: PDF'lerin bulunduğu klasör
#########################################
PDF_DIR = r"C:\Users\mustafa\Python\pythonProject\PdfParse\pdfs" #CHANGE HERE WITH YOUR PDFS FOLDER !!!


###################################################
# Gelişmiş Wikipedia Arama Fonksiyonu (İnsan doğrulaması dahil)
###################################################
def search_in_wikipedia(search_query, require_human=True, verbose=True):
    """
    Verilen 'search_query' için Wikipedia'da arama yapar.
    Eğer require_human=True ise, gelen sonucun bir 'insan' olduğunu
    (kategorilerinde 'people' veya 'birth' kelimeleri geçiyorsa) doğrulamaya çalışır.
    Bulamazsa veya hata alırsa 'Meçhul Yazar' döndürür.
    """
    base_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": search_query,
        "format": "json"
    }

    try:
        response = requests.get(base_url, params=search_params, timeout=10)
        response.raise_for_status()
        data = response.json()
        search_results = data.get("query", {}).get("search", [])

        if verbose:
            print(f"[INFO] Wikipedia araması ('{search_query}') sonuçları:")
            for r in search_results:
                print(f" - {r.get('title')} (snippet: {r.get('snippet')})")

        if not search_results:
            if verbose:
                print("[INFO] Wikipedia'da arama sonucu bulunamadı.")
            return "Meçhul Yazar"

        # İlk kaydı al
        page_id = search_results[0].get("pageid")
        page_title = search_results[0].get("title")

        # İlgili sayfanın extract bilgisini çek
        details_params = {
            "action": "query",
            "prop": "extracts",
            "pageids": page_id,
            "exintro": True,
            "explaintext": True,
            "format": "json"
        }
        details_response = requests.get(base_url, params=details_params, timeout=10)
        details_response.raise_for_status()
        details_data = details_response.json()
        pages = details_data.get("query", {}).get("pages", {})
        page_info = pages.get(str(page_id), {})
        extract = page_info.get("extract", "")

        if verbose:
            print(f"[INFO] Wikipedia sayfa başlığı: {page_title}")
            print("[INFO] Sayfa özeti:\n", extract)

        # Eğer 'require_human' isteniyorsa, kategoriler içinde 'people' ya da 'birth' gibi terimler var mı bakalım
        if require_human:
            cat_params = {
                "action": "query",
                "prop": "categories",
                "pageids": page_id,
                "format": "json"
            }
            cat_response = requests.get(base_url, params=cat_params, timeout=10)
            cat_response.raise_for_status()
            cat_data = cat_response.json()
            cat_pages = cat_data.get("query", {}).get("pages", {})
            page_cats = cat_pages.get(str(page_id), {}).get("categories", [])

            # 'people' veya 'birth' içeren kategori aranıyor (örnek: 'Category:Living people', 'Category:1964 births')
            is_human = False
            for c in page_cats:
                cat_title = c.get("title", "").lower()
                if "people" in cat_title or "birth" in cat_title:
                    is_human = True
                    break

            if not is_human:
                if verbose:
                    print("[INFO] Kategorilerde bir 'insan' emaresi (people/birth) bulunamadı.")
                return "Meçhul Yazar"

        return page_title

    except Exception as e:
        print(f"[ERROR] Wikipedia sorgusu başarısız: {e}")
        return "Meçhul Yazar"


###################################################
# Yardımcı Fonksiyonlar
###################################################
def clean_text(text):
    """Basit metin temizliği."""
    if not text:
        return ""
    text = re.sub(r'cid:\d+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_keywords(text, top_n=5):
    """
    1) Stopword ve 3'ten kısa kelimeleri ele
    2) Frekansı 3'ün altında olanları çıkar
    3) En çok tekrar eden ilk top_n kelimeyi döndür
    """
    words = re.findall(r'\w+', text.lower())
    stop_words = set(stopwords.words('english')) | set(stopwords.words('turkish'))
    filtered = [w for w in words if w not in stop_words and len(w) > 3]

    counter = Counter(filtered)
    all_common = counter.most_common()
    # Frekansı 3'ün altında olanları çıkar
    filtered_common = [(w, freq) for (w, freq) in all_common if freq >= 3]
    top_keywords = filtered_common[:top_n]
    return [w for (w, freq) in top_keywords]


###################################################
# AdvancedPDFParser Sınıfı
###################################################
class AdvancedPDFParser:
    def __init__(self, pdf_path, config=None):
        self.pdf_path = pdf_path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF bulunamadı: {pdf_path}")

        default_config = {
            'dpi': 200,
            'text_threshold': 50,
            'analyze_layout': True,
            'max_chunk_words': 300
        }
        self.config = default_config
        if config:
            self.config.update(config)

    def parse(self):
        """
        1) PDF'i pdfplumber ile açmaya çalış (iç içe try-except)
        2) Tüm sayfaların metnini birleştir ve sayfa sayısını hesapla
        3) Yazar doğrulaması yap (PDF 30 sayfa veya daha az ise yazar "Meçhul Yazar" olarak işaretlenecek)
        4) Top Keywords
        5) Chunk'ları oluştur
        6) { "metadata": {...}, "chunks": [...] } döndür
        """
        meta_info = {
            "pdf_path": self.pdf_path,
            "deduced_author": None,
            "verification": None,
            "top_keywords_in_pdf": []
        }

        pdf_text = ""
        pages_count = 0

        # ================================================
        # 1) Güvenli PDF Açma (İÇ İÇE TRY-EXCEPT) ve sayfa sayısını hesaplama
        # ================================================
        try:
            with pdfplumber.open(self.pdf_path, repair=False) as pdf:
                pages_count = len(pdf.pages)
                print(f"[INFO] PDF sayfa sayısı: {pages_count}")
                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        extracted = page.extract_text() or ""
                    except Exception as e_page:
                        print(f"[WARNING] Sayfa {i} okunamadı: {e_page}")
                        extracted = ""
                    pdf_text += extracted + "\n"
        except Exception as e_open:
            print(f"[ERROR] PDF açılamadı: {self.pdf_path}\nSebep: {e_open}")
            return None  # PDF tamamen açılamıyorsa None dön

        pdf_text = pdf_text.strip()
        if not pdf_text:
            print("[INFO] PDF metin çıkarılamadı veya çok bozuk.")
            return None

        # ================================================
        # 2) Yazar Doğrulaması (Multi-Source: Metadata, Dosya Adı, + Kombinasyonlar)
        # ================================================
        candidate_author, verification = self._deduce_author(pdf_text)
        # Eğer PDF 30 sayfadan uzun değilse, yazar Meçhul Yazar olarak işaretlenecek
        if pages_count <= 30:
            candidate_author = "Meçhul Yazar"
            verification = "PDF too short (<= 30 pages)"
            print("[INFO] PDF sayfa sayısı 30 veya daha az, yazar Meçhul Yazar olarak belirlendi.")
        meta_info["deduced_author"] = candidate_author
        meta_info["verification"] = verification

        # ================================================
        # 3) PDF düzeyinde en çok geçen 5 kelime
        # ================================================
        pdf_level_top_keywords = extract_keywords(pdf_text, top_n=5)
        meta_info["top_keywords_in_pdf"] = pdf_level_top_keywords

        # ================================================
        # 4) Chunk işlemi (cümle bazlı, chunk başı ve sonu cümle bütünlüğü)
        # ================================================
        chunks = self._chunk_into_sentences(pdf_text)

        # ================================================
        # 5) Sonuç JSON
        # ================================================
        result = {
            "metadata": meta_info,
            "chunks": chunks  # pages yok, sadece chunks
        }
        return result

    def _deduce_author(self, pdf_text):
        """
        Gelişmiş yazar tahmini:
        1) PDF'in gömülü metadata'sındaki 'Author' bilgisini al.
           - Eğer 4 harften kısaysa ignore et.
           - Kitap metninde (pdf_text) geçmiyorsa prioritize etme (kullanma).
        2) Dosya adından aday yazar ismi çıkar (ilk iki büyük harfle başlayan kelime).
        3) Bu adaylar Wikipedia'da aranır, eğer sonuç insansa ve arananla eşleşiyorsa "doğrulandı" say.
        4) Eğer bunlar başarısız olursa, dosya adındaki kelimelerle 'yan yana kombinasyonlar'
           bulunup PDF içinde geçenleri tekrar Wikipedia'da dene.
        5) Son olarak hangisi bulunmuşsa ona göre bir karar ver, yoksa Meçhul Yazar.
        """
        candidate_metadata = None
        candidate_filename = None
        verification = ""
        verified_metadata = None
        verified_filename = None
        verified_combination = None

        # 1) PDF Metadata’dan yazar bilgisi alma
        try:
            with pdfplumber.open(self.pdf_path, repair=False) as pdf:
                meta = pdf.metadata
                if meta and meta.get("Author"):
                    m_author = meta.get("Author").strip()
                    if len(m_author) < 4:
                        print(f"[INFO] Metadata yazar bilgisi çok kısa (<4): '{m_author}' -> ignore")
                    else:
                        if m_author.lower() not in pdf_text.lower():
                            print(f"[INFO] Metadata yazar ismi PDF içinde bulunamadı: '{m_author}' -> prioritize edilmesin.")
                        else:
                            candidate_metadata = m_author
                            print(f"[INFO] PDF metadata yazar: {candidate_metadata}")
        except Exception as e:
            print(f"[WARNING] Metadata okunamadı: {e}")

        # 2) Dosya adından yazar çıkarma (örn. ilk iki büyük harfle başlayan kelime)
        filename_no_ext = os.path.splitext(os.path.basename(self.pdf_path))[0]
        name_parts = re.findall(r'[A-Z][a-z]+', filename_no_ext)
        if len(name_parts) >= 2:
            candidate_filename = " ".join(name_parts[:2])
            print(f"[INFO] Dosya adından aday yazar: {candidate_filename}")

        # 3) Metadata ve Dosya adını Wikipedia'da doğrula (insan olması + kelime eşleşmesi)
        if candidate_metadata:
            wiki_meta = search_in_wikipedia(candidate_metadata, require_human=True, verbose=False)
            if wiki_meta != "Meçhul Yazar" and candidate_metadata.lower() in wiki_meta.lower():
                verified_metadata = candidate_metadata
                print(f"[INFO] Metadata yazar Wikipedia tarafından doğrulandı: {verified_metadata}")

        if candidate_filename:
            wiki_file = search_in_wikipedia(candidate_filename, require_human=True, verbose=False)
            if wiki_file != "Meçhul Yazar" and candidate_filename.lower() in wiki_file.lower():
                verified_filename = candidate_filename
                print(f"[INFO] Dosya adı yazar Wikipedia tarafından doğrulandı: {verified_filename}")

        # 4) Eğer her ikisi de doğrulanmazsa, dosya adındaki kelimelerin PDF'te geçen yan yana kombinasyonlarını dene
        if not verified_metadata and not verified_filename:
            if len(name_parts) > 1:
                for i in range(len(name_parts) - 1):
                    combo = f"{name_parts[i]} {name_parts[i+1]}"
                    if combo.lower() in pdf_text.lower():
                        wiki_combo = search_in_wikipedia(combo, require_human=True, verbose=False)
                        if wiki_combo != "Meçhul Yazar":
                            if name_parts[i].lower() in wiki_combo.lower() or name_parts[i+1].lower() in wiki_combo.lower():
                                verified_combination = combo
                                print(f"[INFO] '{combo}' kombinasyonu Wikipedia tarafından doğrulandı: {wiki_combo}")
                                break

        # 5) Karar Aşaması
        if verified_metadata and verified_filename:
            final_author = verified_metadata
            verification = "metadata and filename confirmed (metadata prioritized)"
        elif verified_metadata:
            final_author = verified_metadata
            verification = "metadata confirmed"
        elif verified_filename:
            final_author = verified_filename
            verification = "filename confirmed"
        elif verified_combination:
            final_author = verified_combination
            verification = "combination confirmed"
        else:
            if candidate_metadata:
                final_author = candidate_metadata
                verification = "metadata (not confirmed via Wikipedia)"
            elif candidate_filename:
                final_author = candidate_filename
                verification = "filename (not confirmed via Wikipedia)"
            else:
                final_author = "Meçhul Yazar"
                verification = "no author found"

        return final_author, verification

    def _chunk_into_sentences(self, pdf_text):
        """
        Cümle bazlı chunk işlemi.
        1) Tüm PDF metnini cümlelere ayır.
        2) Sabit kelime sınırını (max_chunk_words) aşmamak üzere chunk oluştur.
        3) Chunk başlangıç ve sonu cümle bütünlüğüne denk gelir.
        """
        max_words = self.config['max_chunk_words']
        chunks = []

        cleaned = clean_text(pdf_text)
        sentences = sent_tokenize(cleaned)

        chunk_text = ""
        chunk_word_count = 0
        chunk_counter = 1

        for sentence in sentences:
            sentence_word_count = len(sentence.split())

            if chunk_word_count + sentence_word_count < max_words:
                if chunk_text:
                    chunk_text += " " + sentence
                else:
                    chunk_text = sentence
                chunk_word_count += sentence_word_count
            else:
                if chunk_text:
                    chunks.append({
                        "chunk_index": str(chunk_counter),
                        "text": chunk_text,
                        "word_count": chunk_word_count,
                        "keywords": extract_keywords(chunk_text)
                    })
                    chunk_counter += 1

                chunk_text = sentence
                chunk_word_count = sentence_word_count

        if chunk_text:
            chunks.append({
                "chunk_index": str(chunk_counter),
                "text": chunk_text,
                "word_count": chunk_word_count,
                "keywords": extract_keywords(chunk_text)
            })

        return chunks


#############################################
# Main Program
#############################################
if __name__ == "__main__":
    overall_start_time = time.time()

    if not os.path.isdir(PDF_DIR):
        print(f"[ERROR] PDF_DIR mevcut değil: {PDF_DIR}")
    else:
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"[INFO] {PDF_DIR} içinde PDF bulunamadı.")
        else:
            for pdf_name in pdf_files:
                pdf_path = os.path.join(PDF_DIR, pdf_name)
                print(f"\nProcessing: {pdf_name}")

                parser = AdvancedPDFParser(pdf_path)
                result = parser.parse()
                if result is None:
                    print(f"Skipped: {pdf_name} (PDF bozuk olabilir veya tamamen okunamadı.)")
                    continue

                out_file = os.path.splitext(pdf_name)[0] + "_metadata.json"
                out_path = os.path.join(PDF_DIR, out_file)
                with open(out_path, "w", encoding="utf-8") as fw:
                    json.dump(result, fw, ensure_ascii=False, indent=2)

                print(f"Done: {pdf_name} -> {out_file}")

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    print(f"\n[TIME] Tüm işlemin süresi: {total_duration:.2f} saniye.")
