import torch, os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from google.cloud import translate_v2 as translate 

dic_list_langs = {
    "mgsm":['en','de','ru','fr','zh','es','ja','sw','th','bn','te'],
    "xcopa":['zh', 'it', 'vi', 'tr', 'id', 'sw', 'th', 'et', 'ta', 'ht', 'qu'],
    "xnli":['en', 'de', 'ru', 'fr', 'zh', 'es', 'vi', 'tr', 'sw', 'ar', 'el', 'th', 'bg', 'hi', 'ur'],
    "paws-x":['en', 'de', 'fr', 'zh', 'es', 'ja', 'ko'],
    "xlsum":['en', 'fr', 'zh', 'es', 'vi', 'tr'],
    "mkqa": ['en', 'de', 'ru', 'fr', 'zh_cn', 'es', 'ja', 'vi', 'tr', 'th'],
    "shareGPT": ['ja', 'zh', 'es', 'fr', 'vi', 'id', 'ko', 'ro', 'uk', 'no'],
    "shareGPT_filter": ['ja', 'zh', 'es', 'fr', 'ko'],
}

langs = ['zh', 'zh_cn', 'it', 'es', 'vi', 'ar', 'et', 'tr', 'el', 'qu', 'bg', 'te', 'bn', 'sw', 'ta', 'ht', 'id', 'de', 'ur', 'ja', 'hi', 'en', 'th', 'fr', 'ko', 'ru', 'pt']

lang_codes = {'zh': "Chinese", 'zh_cn': "Chinese", 'it': "Italian", 'es': "Spanish", 'vi': "Vietnamese", 'ar': "Arabic", 'et': "Estonian", 'tr': "Turkish", 'el': "Greek", 
              'qu': "Quechua", 'bg': "Bulgarian", 'te': "Telugu", 'bn': "Bengali", 'sw': "Swahili", 'ta': "Tamil", 'ht': "Haitian Creole", 'id': "Indonesian", 'de': "German", 
              'ur': "Urdu", 'ja': "Japanese", 'hi': "Hindi", 'en': "English", 'th': "Thai", 'fr': "French", 'ko': "Korean", 'ru': "Russian", 'pt': "Portuguese",
              'ms': "Malay", 'fi': "Finnish", 'nl': "Dutch", 'cs': "Czech", 'uk': "Ukrainian", 'sv': "Swedish", 'ro': "Romanian", 'no': "Norwegian"}

lang_codes_nllb = {
  "Acehnese (Arabic script)": "ace_Arab",
  "Acehnese (Latin script)": "ace_Latn",
  "Mesopotamian Arabic": "acm_Arab",
  "Ta\u2019izzi-Adeni Arabic": "acq_Arab",
  "Tunisian Arabic": "aeb_Arab",
  "Afrikaans": "afr_Latn",
  "South Levantine Arabic": "ajp_Arab",
  "Akan": "aka_Latn",
  "Amharic": "amh_Ethi",
  "North Levantine Arabic": "apc_Arab",
  "Modern Standard Arabic": "arb_Arab",
  "Arabic": "arb_Arab",
  "Modern Standard Arabic (Romanized)": "arb_Latn",
  "Najdi Arabic": "ars_Arab",
  "Moroccan Arabic": "ary_Arab",
  "Egyptian Arabic": "arz_Arab",
  "Assamese": "asm_Beng",
  "Asturian": "ast_Latn",
  "Awadhi": "awa_Deva",
  "Central Aymara": "ayr_Latn",
  "South Azerbaijani": "azb_Arab",
  "North Azerbaijani": "azj_Latn",
  "Bashkir": "bak_Cyrl",
  "Bambara": "bam_Latn",
  "Balinese": "ban_Latn",
  "Belarusian": "bel_Cyrl",
  "Bemba": "bem_Latn",
  "Bengali": "ben_Beng",
  "Bhojpuri": "bho_Deva",
  "Banjar (Arabic script)": "bjn_Arab",
  "Banjar (Latin script)": "bjn_Latn",
  "Standard Tibetan": "bod_Tibt",
  "Bosnian": "bos_Latn",
  "Buginese": "bug_Latn",
  "Bulgarian": "bul_Cyrl",
  "Catalan": "cat_Latn",
  "Cebuano": "ceb_Latn",
  "Czech": "ces_Latn",
  "Chokwe": "cjk_Latn",
  "Central Kurdish": "ckb_Arab",
  "Crimean Tatar": "crh_Latn",
  "Welsh": "cym_Latn",
  "Danish": "dan_Latn",
  "German": "deu_Latn",
  "Southwestern Dinka": "dik_Latn",
  "Dyula": "dyu_Latn",
  "Dzongkha": "dzo_Tibt",
  "Greek": "ell_Grek",
  "English": "eng_Latn",
  "Esperanto": "epo_Latn",
  "Estonian": "est_Latn",
  "Basque": "eus_Latn",
  "Ewe": "ewe_Latn",
  "Faroese": "fao_Latn",
  "Fijian": "fij_Latn",
  "Finnish": "fin_Latn",
  "Fon": "fon_Latn",
  "French": "fra_Latn",
  "Friulian": "fur_Latn",
  "Nigerian Fulfulde": "fuv_Latn",
  "Scottish Gaelic": "gla_Latn",
  "Irish": "gle_Latn",
  "Galician": "glg_Latn",
  "Guarani": "grn_Latn",
  "Gujarati": "guj_Gujr",
  "Haitian Creole": "hat_Latn",
  "Hausa": "hau_Latn",
  "Hebrew": "heb_Hebr",
  "Hindi": "hin_Deva",
  "Chhattisgarhi": "hne_Deva",
  "Croatian": "hrv_Latn",
  "Hungarian": "hun_Latn",
  "Armenian": "hye_Armn",
  "Igbo": "ibo_Latn",
  "Ilocano": "ilo_Latn",
  "Indonesian": "ind_Latn",
  "Icelandic": "isl_Latn",
  "Italian": "ita_Latn",
  "Javanese": "jav_Latn",
  "Japanese": "jpn_Jpan",
  "Kabyle": "kab_Latn",
  "Jingpho": "kac_Latn",
  "Kamba": "kam_Latn",
  "Kannada": "kan_Knda",
  "Kashmiri (Arabic script)": "kas_Arab",
  "Kashmiri (Devanagari script)": "kas_Deva",
  "Georgian": "kat_Geor",
  "Central Kanuri (Arabic script)": "knc_Arab",
  "Central Kanuri (Latin script)": "knc_Latn",
  "Kazakh": "kaz_Cyrl",
  "Kabiy\u00e8": "kbp_Latn",
  "Kabuverdianu": "kea_Latn",
  "Khmer": "khm_Khmr",
  "Kikuyu": "kik_Latn",
  "Kinyarwanda": "kin_Latn",
  "Kyrgyz": "kir_Cyrl",
  "Kimbundu": "kmb_Latn",
  "Northern Kurdish": "kmr_Latn",
  "Kikongo": "kon_Latn",
  "Korean": "kor_Hang",
  "Lao": "lao_Laoo",
  "Ligurian": "lij_Latn",
  "Limburgish": "lim_Latn",
  "Lingala": "lin_Latn",
  "Lithuanian": "lit_Latn",
  "Lombard": "lmo_Latn",
  "Latgalian": "ltg_Latn",
  "Luxembourgish": "ltz_Latn",
  "Luba-Kasai": "lua_Latn",
  "Ganda": "lug_Latn",
  "Luo": "luo_Latn",
  "Mizo": "lus_Latn",
  "Standard Latvian": "lvs_Latn",
  "Magahi": "mag_Deva",
  "Maithili": "mai_Deva",
  "Malayalam": "mal_Mlym",
  "Marathi": "mar_Deva",
  "Minangkabau (Arabic script)": "min_Arab",
  "Minangkabau (Latin script)": "min_Latn",
  "Macedonian": "mkd_Cyrl",
  "Plateau Malagasy": "plt_Latn",
  "Maltese": "mlt_Latn",
  "Meitei (Bengali script)": "mni_Beng",
  "Halh Mongolian": "khk_Cyrl",
  "Mossi": "mos_Latn",
  "Maori": "mri_Latn",
  "Burmese": "mya_Mymr",
  "Dutch": "nld_Latn",
  "Norwegian Nynorsk": "nno_Latn",
  "Norwegian Bokm\u00e5l": "nob_Latn",
  "Nepali": "npi_Deva",
  "Northern Sotho": "nso_Latn",
  "Nuer": "nus_Latn",
  "Nyanja": "nya_Latn",
  "Occitan": "oci_Latn",
  "West Central Oromo": "gaz_Latn",
  "Odia": "ory_Orya",
  "Pangasinan": "pag_Latn",
  "Eastern Panjabi": "pan_Guru",
  "Papiamento": "pap_Latn",
  "Western Persian": "pes_Arab",
  "Polish": "pol_Latn",
  "Portuguese": "por_Latn",
  "Dari": "prs_Arab",
  "Southern Pashto": "pbt_Arab",
  "Quechua": "quy_Latn",
  "Romanian": "ron_Latn",
  "Rundi": "run_Latn",
  "Russian": "rus_Cyrl",
  "Sango": "sag_Latn",
  "Sanskrit": "san_Deva",
  "Santali": "sat_Olck",
  "Sicilian": "scn_Latn",
  "Shan": "shn_Mymr",
  "Sinhala": "sin_Sinh",
  "Slovak": "slk_Latn",
  "Slovenian": "slv_Latn",
  "Samoan": "smo_Latn",
  "Shona": "sna_Latn",
  "Sindhi": "snd_Arab",
  "Somali": "som_Latn",
  "Southern Sotho": "sot_Latn",
  "Spanish": "spa_Latn",
  "Tosk Albanian": "als_Latn",
  "Sardinian": "srd_Latn",
  "Serbian": "srp_Cyrl",
  "Swati": "ssw_Latn",
  "Sundanese": "sun_Latn",
  "Swedish": "swe_Latn",
  "Swahili": "swh_Latn",
  "Silesian": "szl_Latn",
  "Tamil": "tam_Taml",
  "Tatar": "tat_Cyrl",
  "Telugu": "tel_Telu",
  "Tajik": "tgk_Cyrl",
  "Tagalog": "tgl_Latn",
  "Thai": "tha_Thai",
  "Tigrinya": "tir_Ethi",
  "Tamasheq (Latin script)": "taq_Latn",
  "Tamasheq (Tifinagh script)": "taq_Tfng",
  "Tok Pisin": "tpi_Latn",
  "Tswana": "tsn_Latn",
  "Tsonga": "tso_Latn",
  "Turkmen": "tuk_Latn",
  "Tumbuka": "tum_Latn",
  "Turkish": "tur_Latn",
  "Twi": "twi_Latn",
  "Central Atlas Tamazight": "tzm_Tfng",
  "Uyghur": "uig_Arab",
  "Ukrainian": "ukr_Cyrl",
  "Umbundu": "umb_Latn",
  "Urdu": "urd_Arab",
  "Northern Uzbek": "uzn_Latn",
  "Venetian": "vec_Latn",
  "Vietnamese": "vie_Latn",
  "Waray": "war_Latn",
  "Wolof": "wol_Latn",
  "Xhosa": "xho_Latn",
  "Eastern Yiddish": "ydd_Hebr",
  "Yoruba": "yor_Latn",
  "Yue Chinese": "yue_Hant",
  "Chinese": "zho_Hans",
  "Standard Malay": "zsm_Latn",
  "Zulu": "zul_Latn"
}

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="../keys/key_tts_robert.json"
def get_translation_google(text: str, dest: str) -> str:
    """Translates text into the target language.

    Args:
        text: The text to translate.
        dest: The ISO 639-1 code for the target language (e.g., "es", "fr").
        project_id: Your Google Cloud project ID.

    Returns:
        The translated text.
        Returns original text if translation fails.
    """
    try:
        translate_client = translate.Client() # Assumes GOOGLE_APPLICATION_CREDENTIALS is set

        # The result is a dictionary or list of dictionaries
        result = translate_client.translate(
            text,
            target_language=dest
            # source_language="en" # Optional: Specify source language
        )

        # print(f"Raw API response: {result}") # See the full structure

        # Extract the translated text
        # For single input, result is a dict:
        if isinstance(result, dict):
             translated_text = result['translatedText']
             detected_language = result.get('detectedSourceLanguage', 'N/A') # v2 might provide detected language
            #  print(f"Detected source language: {detected_language}")
             return translated_text
         # For multiple inputs (if text were a list), result is a list:
        elif isinstance(result, list) and len(result) > 0:
             translated_text = result[0]['translatedText'] # Get first translation
             detected_language = result[0].get('detectedSourceLanguage', 'N/A')
            #  print(f"Detected source language: {detected_language}")
             return translated_text
        else:
            print("Warning: Unexpected translation result format.")
            return text # Return original on unexpected result

    except Exception as e:
        print(f"Error during translation: {e}")
        return text # Return original text on error

def prepare_pipeline_nllb(src_lang='zh', tgt_lang='en', max_length=400,
                          CKPT="facebook/nllb-200-3.3B", device=None):
    """
    Prepare the pipeline for nllb translation
    """
    # CKPT = "facebook/nllb-200-distilled-600M"
    # CKPT = "facebook/nllb-200-3.3B"
    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT)
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    if device is None:
        device = torch.cuda.device_count() -1

    src_lang = lang_codes[src_lang]
    tgt_lang = lang_codes[tgt_lang]
    LANGS_map = lang_codes_nllb

    assert src_lang in LANGS_map.keys(), "Source language not supported"
    assert tgt_lang in LANGS_map.keys(), "Target language not supported"

    translation_pipeline = pipeline("translation",
                                    model=model,
                                    tokenizer=tokenizer,
                                    src_lang=LANGS_map[src_lang],
                                    tgt_lang=LANGS_map[tgt_lang],
                                    max_length=max_length,
                                    device=device)
    
    return translation_pipeline

def get_translation_nllb(translation_pipeline, text, repetition_penalty=1.0):
    result = translation_pipeline(text, repetition_penalty=repetition_penalty)
    # add repeatation penalty
    # result = translation_pipeline(text, repetition_penalty=2.5)
    return result[0]['translation_text']

def _test_translator(lang_code):
    # text = "今天天气真好，我想出去玩。"
    text = "I want to go out and play today."
    # res = get_translation_google(text, dest=lang_code)
    # print(f'{lang_code}: {res}')
    
    print("nllb translate:")
    translation_pipeline = prepare_pipeline_nllb(src_lang='en', tgt_lang=lang_code, 
                                                    max_length=400,device='cpu')
    for i in range(1):
        res = get_translation_nllb(translation_pipeline, text)
        print(res)

if __name__ == '__main__':
    # print(dic_list_langs.keys())
    for lang in ['de', 'ru', 'fr', 'zh', 'es', 'ja', 'vi', 'tr', 'th']:
    # for lang in ['zh']:
        _test_translator(lang)