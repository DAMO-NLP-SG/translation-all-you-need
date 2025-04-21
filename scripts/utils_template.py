import re
from datasets import load_from_disk, load_dataset
import pandas as pd
from util_mkqa import compute_max_score_over_answers, calculate_f1
from utils_langs import *
from utils import *

dic_answer_word = {'zh': '答案', 'zh_cn': '答案','it': 'Risposta', 'es': 'Respuesta', 'vi': 'Trả lời', 'ar': 'إجابة', 'et': 'Vastus', 'tr': 'Cevap', 'el': 'Απάντηση', 'qu': 'kutichiy', 'bg': 'Отговор', 'te': 'సమాధానం', 'bn': 'উত্তর', 'sw': 'Jibu', 'ta': 'பதில்', 'ht': 'Repons', 'id': 'Jawaban', 'de': 'Antwort', 'ur': 'جواب', 'ja': '答え', 'hi': 'उत्तर', 'en': 'Answer', 'th': 'คำตอบ', 'fr': 'Réponse', 'ko': '대답', 'ru': 'Ответ', 'pt': 'Resposta'}
dic_lang_native = {'zh': "中文", 'zh_cn': "中文", 'it': "Italiano", 'es': "Español", 'vi': "Tiếng Việt", 'ar': "العربية", 'et': "Eesti", 'tr': "Türkçe", 'el': "Ελληνικά",
                        'qu': "Quechua", 'bg': "български", 'te': "తెలుగు", 'bn': "বাংলা", 'sw': "Kiswahili", 'ta': "தமிழ்", 'ht': "Kreyòl Ayisyen", 'id': "Bahasa Indonesia",'de': "Deutsch", 'ur': "اردو", 'ja': "日本語", 'hi': "हिन्दी", 'en': "English", 'th': "ไทย", 'fr': "Français", 'ko': "한국어", 'ru': "русский", 'pt': "Português"}
dic_template_cot = {
  "cot": {
    "th": "ลองคิดทีละขั้นตอน",
    "de": "Denken wir Schritt für Schritt.",
    "en": "Let's think step by step.",
    "ko": "단계별로 생각해 봅시다.",
    "zh": "让我们一步步思考。",
    "bg": "Нека помислим стъпка по стъпка.",
    "el": "Ας σκεφτούμε βήμα βήμα.",
    "vi": "Hãy suy nghĩ từng bước một.",
    "et": "Mõelgem samm-sammult.",
    "ru": "Давайте думать поэтапно.",
    "ur": "آئیے قدم بہ قدم سوچتے ہیں۔",
    "zh_cn": "让我们一步步思考。",
    "fr": "Réfléchissons étape par étape.",
    "te": "అంచెలంచెలుగా ఆలోచిద్దాం.",
    "it": "Pensiamo passo dopo passo.",
    "bn": "আসুন ধাপে ধাপে চিন্তা করি।",
    "hi": "आइए चरण दर चरण सोचें.",
    "sw": "Hebu fikiria hatua kwa hatua.",
    "ht": "Ann reflechi etap pa etap.",
    "es": "Pensemos paso a paso.",
    "id": "Mari kita berpikir selangkah demi selangkah.",
    "ja": "段階的に考えてみましょう。",
    "tr": "Adım adım düşünelim.",
    "ar": "دعونا نفكر خطوة بخطوة.",
    "ta": "படிப்படியாக சிந்திப்போம்.",
    "qu": "Hukmanta hukmanta yuyaymanasun."
  },
  "cot_en": {
    "th": "ลองคิดทีละขั้นตอนเป็นภาษาอังกฤษ",
    "de": "Denken wir Schritt für Schritt auf Englisch.",
    "en": "Let's think step by step in English.",
    "ko": "영어로 단계별로 생각해 봅시다.",
    "zh": "让我们用英语一步步思考。",
    "bg": "Нека помислим стъпка по стъпка на английски.",
    "el": "Ας σκεφτούμε βήμα-βήμα στα αγγλικά.",
    "vi": "Hãy suy nghĩ từng bước bằng tiếng Anh.",
    "et": "Mõelgem samm-sammult inglise keeles.",
    "ru": "Давайте думать шаг за шагом на английском языке.",
    "ur": "آئیے انگریزی میں قدم بہ قدم سوچتے ہیں۔",
    "zh_cn": "让我们用英语一步步思考。",
    "fr": "Pensons étape par étape en anglais.",
    "te": "ఇంగ్లీషులో స్టెప్ బై స్టెప్ ఆలోచిద్దాం.",
    "it": "Pensiamo passo dopo passo in inglese.",
    "bn": "আসুন ইংরেজিতে ধাপে ধাপে চিন্তা করি।",
    "hi": "आइए चरण दर चरण अंग्रेजी में सोचें।",
    "sw": "Wacha tufikirie hatua kwa hatua kwa Kiingereza.",
    "ht": "Ann panse etap pa etap nan lang angle.",
    "es": "Pensemos paso a paso en inglés.",
    "id": "Mari kita berpikir langkah demi langkah dalam bahasa Inggris.",
    "ja": "ステップバイステップで英語で考えてみましょう。",
    "tr": "Adım adım İngilizce düşünelim.",
    "ar": "دعونا نفكر خطوة بخطوة باللغة الإنجليزية.",
    "ta": "ஆங்கிலத்தில் படிப்படியாக சிந்திப்போம்.",
    "qu": "Qhichwa simipi hukmanta hukmanta yuyaykusun."
  }
}

dic_template_basic = {
  "mgsm": {
    "en": "{}",
    "de": "{}",
    "ru": "{}",
    "fr": "{}",
    "zh": "{}",
    "es": "{}",
    "ja": "{}",
    "sw": "{}",
    "th": "{}",
    "bn": "{}",
    "te": "{}"
  },
  "xcopa": {
    "zh": "这里有一个前提：{}。是什么 {}？帮我选择更合理的选项： - 选择 1：{}， - 选择 2：{}",
    "it": "Ecco una premessa: {}. Quale è {}? Aiutami a scegliere l'opzione più plausibile: - scelta 1: {}, - scelta 2: {}",
    "vi": "Đây là tiền đề: {}. {} là gì? Giúp tôi chọn phương án hợp lý hơn: - lựa chọn 1: {}, - lựa chọn 2: {}",
    "tr": "İşte bir öncül: {}. Nedir {}? Daha makul seçeneği seçmeme yardım et: - seçenek 1: {}, - seçenek 2: {}",
    "id": "Berikut ini premisnya: {}. Apakah yang {}? Bantu saya memilih opsi yang lebih masuk akal: - pilihan 1: {}, - pilihan 2: {}",
    "sw": "Hapa kuna dhana: {}. {} ni nini? Nisaidie kuchagua chaguo linalokubalika zaidi: - chaguo la 1: {}, - chaguo la 2: {}",
    "th": "นี่คือหลักฐาน: {} อะไรคือ {}? ช่วยฉันเลือกตัวเลือกที่เป็นไปได้มากกว่า: - ตัวเลือกที่ 1: {} - ตัวเลือกที่ 2: {}",
    "et": "Siin on eeldus: {}. Mis on {}? Aidake mul valida usutavam variant: - valik 1: {}, - valik 2: {}",
    "ta": "இங்கே ஒரு முன்மாதிரி உள்ளது: {}. {} என்றால் என்ன? மிகவும் நம்பத்தகுந்த விருப்பத்தைத் தேர்வுசெய்ய எனக்கு உதவுங்கள்: - தேர்வு 1: {}, - தேர்வு 2: {}",
    "ht": "Isit la se yon site: {}. Ki sa ki {} a? Ede m chwazi opsyon ki pi posib: - chwa 1: {}, - chwa 2: {}",
    "qu": "Kaypi huk premisa kachkan: {}. ¿Imataq {}? Yanapaway aswan iñiypaq akllanata akllanaypaq: - 1 kaq akllana: {}, - 2 kaq akllana: {}"
  },
  "xnli": {
    "en": "{} Based on previous passage, is it true that {}? 1:Yes, 2:No, 3:Maybe?",
    "de": "{} Stimmt es basierend auf der vorherigen Passage, dass {}? 1:Ja, 2:Nein, 3:Vielleicht?",
    "ru": "{} Судя по предыдущему отрывку, верно ли, что {}? 1:Да, 2:Нет, 3:Может быть?",
    "fr": "{} D'après le passage précédent, est-il vrai que {} ? 1 : Oui, 2 : Non, 3 : Peut-être ?",
    "zh": "{} 根据之前的段落，{} 是真的吗？ 1：是，2：否，3：也许？",
    "es": "{} Según el pasaje anterior, ¿es cierto que {}? 1:Sí, 2:No, 3:¿Quizás?",
    "vi": "{} Dựa vào đoạn văn trước, điều đó có đúng không {}? 1:Có, 2:Không, 3:Có thể?",
    "tr": "{} Önceki pasaja göre, {} doğru mu? 1:Evet, 2:Hayır, 3:Belki?",
    "sw": "{} Kulingana na kifungu kilichotangulia, je ni kweli kwamba {}? 1:Ndiyo, 2:Hapana, 3:Labda?",
    "ar": "{} بناءً على المقطع السابق، هل صحيح أن {}؟ 1: نعم، 2: لا، 3: ربما؟",
    "el": "{} Με βάση το προηγούμενο απόσπασμα, είναι αλήθεια ότι {}; 1: Ναι, 2: Όχι, 3: Ίσως;",
    "th": "{} จากข้อความที่แล้ว {} จริงหรือ? 1:ใช่, 2:ไม่ใช่, 3:อาจจะ?",
    "bg": "{} Въз основа на предишния пасаж, вярно ли е, че {}? 1: Да, 2: Не, 3: Може би?",
    "hi": "{} पिछले अनुच्छेद के आधार पर, क्या यह सच है कि {}? 1:हाँ, 2:नहीं, 3:शायद?",
    "ur": "{} پچھلے حوالے کی بنیاد پر، کیا یہ درست ہے کہ {}؟ 1:ہاں، 2:نہیں، 3:شاید؟"
  },
  "paws-x": {
    "en": "Sentence 1: {} Sentence 2: {} Question: Does Sentence 1 paraphrase Sentence 2? 1:Yes, 2:No",
    "de": "Satz 1: {} Satz 2: {} Frage: Paraphrasiert Satz 1 Satz 2? 1:Ja, 2:Nein",
    "fr": "Phrase 1 : {} Phrase 2 : {} Question : La phrase 1 paraphrase-t-elle la phrase 2 ? 1 : Oui, 2 : Non",
    "zh": "句子 1：{} 句子 2：{} 问题：句子 1 是否解释了句子 2？ 1：是，2：否",
    "es": "Oración 1: {} Oración 2: {} Pregunta: ¿La oración 1 parafrasea la oración 2? 1: Sí, 2: No",
    "ja": "文 1: {} 文 2: {} 質問: 文 1 は文 2 を言い換えていますか? 1:はい、2:いいえ",
    "ko": "문장 1: {} 문장 2: {} 질문: 문장 1이 문장 2를 바꾸어 표현합니까? 1:예, 2:아니요"
  },
  "xlsum": {
    "en": "{}\nSummarize the article.",
    "fr": "{}\nRésumez l’article.",
    "zh": "{}\n总结一下这篇文章。",
    "es": "{}\nResume el artículo.",
    "vi": "{}\nTóm tắt bài viết.",
    "tr": "{}\nMakaleyi özetleyin."
  },
  "mkqa": {
    "en": "Answer the question in one or a few words in {}: {}?",
    "de": "Beantworten Sie die Frage in einem oder mehreren Worten in {}: {}?",
    "ru": "Ответьте на вопрос одним или несколькими словами в {}: {}?",
    "fr": "Répondez à la question en un ou quelques mots en {} : {} ?",
    "zh_cn": "用{}中的一个或几个词回答问题：{}？",
    "es": "Responda la pregunta en una o varias palabras en {}: ¿{}?",
    "ja": "{} 内の 1 つまたはいくつかの単語で質問に答えてください: {}?",
    "vi": "Trả lời câu hỏi bằng một hoặc vài từ trong {}: {}?",
    "tr": "Soruyu bir veya birkaç kelimeyle yanıtlayın {}: {}?",
    "th": "ตอบคำถามด้วยคำเดียวหรือสองสามคำใน {}: {}?"
  }
}

dic_tempalte_format = {
  "general": {
    "ja": "最終的な回答は次のようにフォーマットする必要があります。",
    "ru": "Вы должны отформатировать свой окончательный ответ следующим образом:",
    "fr": "Vous devez formater votre réponse finale comme suit :",
    "tr": "Son cevabınızı şu şekilde biçimlendirmelisiniz:",
    "vi": "Bạn nên định dạng câu trả lời cuối cùng của mình là:",
    "bg": "Трябва да форматирате окончателния си отговор като:",
    "ht": "Ou ta dwe fòmate repons final ou a tankou:",
    "ur": "آپ کو اپنا حتمی جواب اس طرح فارمیٹ کرنا چاہیے:",
    "de": "Sie sollten Ihre endgültige Antwort wie folgt formatieren:",
    "et": "Peaksite oma lõpliku vastuse vormistama järgmiselt:",
    "it": "Dovresti formattare la tua risposta finale come:",
    "es": "Debes formatear tu respuesta final como:",
    "zh_cn": "您的最终答案的格式应为：",
    "zh": "您的最终答案的格式应为：",
    "th": "คุณควรจัดรูปแบบคำตอบสุดท้ายเป็น:",
    "id": "Anda harus memformat jawaban akhir Anda sebagai:",
    "bn": "আপনি আপনার চূড়ান্ত উত্তর ফর্ম্যাট করা উচিত:",
    "te": "మీరు మీ చివరి సమాధానాన్ని ఇలా ఫార్మాట్ చేయాలి:",
    "hi": "आपको अपना अंतिम उत्तर इस प्रकार प्रारूपित करना चाहिए:",
    "en": "You should format your final answer as:",
    "sw": "Unapaswa kupanga jibu lako la mwisho kama:",
    "ko": "최종 답변의 형식을 다음과 같이 지정해야 합니다.",
    "el": "Θα πρέπει να μορφοποιήσετε την τελική σας απάντηση ως εξής:",
    "ta": "உங்கள் இறுதி பதிலை பின்வருமாறு வடிவமைக்க வேண்டும்:",
    "ar": "يجب عليك تنسيق إجابتك النهائية على النحو التالي:",
    "qu": "Qhipa kutichiyniykitaqa kayhinata formato ruwanayki tiyan:"
  },
  "mgsm": {
    "en": "Arabic numeral",
    "de": "Arabische Ziffer",
    "ru": "арабская цифра",
    "fr": "Chiffre arabe",
    "zh": "阿拉伯数字",
    "es": "número arábigo",
    "ja": "アラビア数字",
    "sw": "Nambari ya Kiarabu",
    "th": "เลขอารบิค",
    "bn": "আরবি সংখ্যা",
    "te": "అరబిక్ సంఖ్య"
  },
  "xcopa": {
    "zh": "1 或 2",
    "it": "1 o 2",
    "vi": "1 hoặc 2",
    "tr": "1 yada 2",
    "id": "1 atau 2",
    "sw": "1 au 2",
    "th": "1 หรือ 2",
    "et": "1 või 2",
    "ta": "1 அல்லது 2",
    "ht": "1 oswa 2",
    "qu": "1 utaq 2"
  },
  "xnli": {
    "en": "1,2 or 3",
    "de": "1,2 oder 3",
    "ru": "1,2 или 3",
    "fr": "1,2 ou 3",
    "zh": "1,2 或 3",
    "es": "1,2 o 3",
    "vi": "1,2 hoặc 3",
    "tr": "1,2 veya 3",
    "sw": "1, 2 au 3",
    "ar": "1،2 أو 3",
    "el": "1,2 ή 3",
    "th": "1,2 หรือ 3",
    "bg": "1,2 или 3",
    "hi": "1,2 या 3",
    "ur": "1،2 یا 3"
  },
  "paws-x": {
    "en": "1 or 2",
    "de": "1 oder 2",
    "fr": "1 ou 2",
    "zh": "1 或 2",
    "es": "1 o 2",
    "ja": "1 または 2",
    "ko": "1 또는 2"
  },
  "xlsum": {
    "en": "You should format your final summary in one sentence in {} as:",
    "fr": "Vous devez formater votre résumé final en une phrase entre {} comme :",
    "zh": "您应将最终摘要的格式设置为 {} 中的一句话：",
    "es": "Debes formatear tu resumen final en una oración en {} como:",
    "vi": "Bạn nên định dạng bản tóm tắt cuối cùng của mình bằng một câu trong {} như sau:",
    "tr": "{} dilindeki tek cümlelik son özetinizi şu şekilde biçimlendirmelisiniz:"
  },
  "mkqa": {
    "en": "You should format your final answer in one or a few words as:",
    "de": "Sie sollten Ihre endgültige Antwort in einem oder mehreren Worten wie folgt formulieren:",
    "ru": "Вам следует отформатировать свой окончательный ответ в одном или нескольких словах следующим образом:",
    "fr": "Vous devez formater votre réponse finale en un ou quelques mots comme :",
    "zh_cn": "您应该用一个或几个词来格式化您的最终答案，如下所示：",
    "es": "Debes formatear tu respuesta final en una o pocas palabras como:",
    "ja": "最終的な回答は次のように 1 語または数語でフォーマットする必要があります。",
    "vi": "Bạn nên định dạng câu trả lời cuối cùng của mình bằng một hoặc một vài từ như sau:",
    "tr": "Son cevabınızı bir veya birkaç kelimeyle şu şekilde biçimlendirmelisiniz:",
    "th": "คุณควรจัดรูปแบบคำตอบสุดท้ายเป็นคำเดียวหรือสองสามคำดังนี้:"
  }
}

dic_tempalte_round2 = {
  "mgsm": {
    "en": "Therefore, the answer (Arabic numerals) is",
    "de": "Daher lautet die Antwort (arabische Ziffern).",
    "ru": "Следовательно, ответ (арабские цифры) такой:",
    "fr": "Par conséquent, la réponse (chiffres arabes) est",
    "zh": "因此，答案（阿拉伯数字）是",
    "es": "Por lo tanto, la respuesta (números arábigos) es",
    "ja": "したがって、答え（アラビア数字）は、",
    "sw": "Kwa hiyo, jibu (nambari za Kiarabu) ni",
    "th": "ดังนั้นคำตอบ (เลขอารบิค) ก็คือ",
    "bn": "অতএব, উত্তর (আরবি সংখ্যা) হল",
    "te": "కాబట్టి, సమాధానం (అరబిక్ సంఖ్యలు)."
  },
  "xcopa": {
    "zh": "因此，答案（1或2）是",
    "it": "Pertanto, la risposta (1 o 2) è",
    "vi": "Do đó, câu trả lời (1 hoặc 2) là",
    "tr": "Bu nedenle cevap (1 veya 2)",
    "id": "Oleh karena itu, jawabannya (1 atau 2) adalah",
    "sw": "Kwa hivyo, jibu (1 au 2) ni",
    "th": "ดังนั้นคำตอบ (1 หรือ 2) ก็คือ",
    "et": "Seetõttu on vastus (1 või 2).",
    "ta": "எனவே, பதில் (1 அல்லது 2).",
    "ht": "Se poutèt sa, repons lan (1 oswa 2) se",
    "qu": "Chayrayku, kutichiyqa (1 utaq 2) kaymi"
  },
  "xnli": {
    "en": "Therefore, the answer (1,2 or 3) is",
    "de": "Daher lautet die Antwort (1,2 oder 3).",
    "ru": "Следовательно, ответ (1,2 или 3)",
    "fr": "Par conséquent, la réponse (1,2 ou 3) est",
    "zh": "因此，答案（1,2或3）是",
    "es": "Por tanto, la respuesta (1,2 o 3) es",
    "vi": "Do đó, câu trả lời (1,2 hoặc 3) là",
    "tr": "Bu nedenle cevap (1,2 veya 3)",
    "sw": "Kwa hivyo, jibu (1,2 au 3) ni",
    "ar": "وبالتالي فإن الجواب (1،2 أو 3) هو",
    "el": "Επομένως, η απάντηση (1,2 ή 3) είναι",
    "th": "ดังนั้นคำตอบ (1,2 หรือ 3) ก็คือ",
    "bg": "Следователно отговорът (1, 2 или 3) е",
    "hi": "इसलिए, उत्तर (1,2 या 3) है",
    "ur": "لہذا، جواب (1،2 یا 3) ہے۔"
  },
  "paws-x": {
    "en": "Therefore, the answer (1 or 2) is",
    "de": "Daher lautet die Antwort (1 oder 2).",
    "fr": "La réponse (1 ou 2) est donc",
    "zh": "因此，答案（1或2）是",
    "es": "Por tanto, la respuesta (1 o 2) es",
    "ja": "したがって、答え(1または2)は、",
    "ko": "따라서 답(1 또는 2)은 다음과 같습니다."
  },
  "xlsum": {
    "en": "Therefore, the answer is",
    "fr": "La réponse est donc",
    "zh": "因此，答案是",
    "es": "Por lo tanto, la respuesta es",
    "vi": "Vì vậy, câu trả lời là",
    "tr": "Bu nedenle cevap"
  },
  "mkqa": {
    "en": "Therefore, the answer is",
    "de": "Daher lautet die Antwort",
    "ru": "Поэтому ответ",
    "fr": "La réponse est donc",
    "zh_cn": "因此，答案是",
    "es": "Por lo tanto, la respuesta es",
    "ja": "したがって、答えは次のとおりです。",
    "vi": "Vì vậy, câu trả lời là",
    "tr": "Bu nedenle cevap",
    "th": "ดังนั้นคำตอบก็คือ"
  }
}

dic_cause_effect = {
  "cause": {
    "zh": "原因",
    "it": "causa",
    "vi": "gây ra",
    "tr": "neden",
    "id": "menyebabkan",
    "sw": "sababu",
    "th": "สาเหตุ",
    "et": "põhjus",
    "ta": "காரணம்",
    "ht": "koz",
    "qu": "causa"
  },
  "effect": {
    "zh": "影响",
    "it": "effetto",
    "vi": "tác dụng",
    "tr": "etki",
    "id": "memengaruhi",
    "sw": "athari",
    "th": "ผล",
    "et": "mõju",
    "ta": "விளைவு",
    "ht": "efè",
    "qu": "qatiynin"
  }
}

dic_template_xlt = {
    "mgsm": """I want you to act as an arithmetic reasoning expert for {lang} . 
Request: {Q} 
You should retell the request in English. 
You should do step-by-step answer to obtain a number answer . 
You should step-by-step answer the request. 
You should tell me the answer in this format 'Answer:'.""",
    "xcopa": """I want you to act as a commonsense reasoning expert for {lang} . 
Here is a premise: {premise}. What is the {question}? Help me pick the more plausible option: -choice1: {choice1}, -choice2: {choice2}
You should retell the premise and the options in English. 
You should do step-by-step answer to pick a choice . 
You should step-by-step answer the request. 
You should tell me the choice number in this format 'Choice number:'.""",
    "xnli": """I want you to act as a natural language inference expert for {lang} . 
Premise: {premise}. 
Hypothesis: {hypothesis}. 
You should retell the premise and hypothesis in English. 
You should judge whether the hypothesis is true (entailment), false (contradiction), or undetermined (neutral) given the premise. 
The relationship can be chosen from entailment, contradiction, and neutral . 
You should step-by-step answer the request. 
You should tell me the relationship in this format 'Relationship:'.""",
    "paws-x": """I want you to act as a paraphrase identification expert for {lang}. 
Sentence 1: {sentence1}
Sentence 2: {sentence2}
Question: Does Sentence 1 paraphrase Sentence 2? Yes or No? 
You should retell the sentence 1 and sentence 2 in English. 
You should provide a yes or no answer to the question: Does Sentence 1 paraphrase Sentence 2? 
You should step-by-step answer the request. 
You should tell me the answer choosing either yes or no in this format 'Answer:'.""",
    "mkqa": """I want you to act as a question answering expert for {lang}.
Question: {Q}
You should retell the question in English. 
You should answer the question in English in one or a few words. 
You should step-by-step answer the request. 
You should tell me the answer in one or a few words in {lang} in this format 'Answer:'.""",
    "xlsum": """I want you to act as a multilingual summarization expert for {lang} . 
Text: {text}
You should repeat the entire text in English. 
You should think step-by-step to summarize the entire text in a maximum of two sentences. 
You should step-by-step answer the request. 
You should tell me the summary into one sentence in {lang} in this format 'Summary:'. """
}

def get_output_folder(args):
    output_folder = f'{args.results_folder}/{args.task}/{args.model.split("/")[-1]}_{args.prompt_type}/{args.lang}'
    create_folder_if_not_exist(output_folder)
    return output_folder

def get_data(args):
    if args.task == 'mgsm':
        if args.prompt_type == 'google' or args.prompt_type == 'nllb': 
            df = pd.read_csv(f'datasets/{args.task}/translations_{args.prompt_type}/mgsm_{args.lang}.tsv',delimiter='\t',header=None)
        elif args.prompt_type == 'google_direct':
            df = pd.read_csv(f'datasets/{args.task}/translations_google/mgsm_{args.lang}.tsv',delimiter='\t',header=None)
        elif args.prompt_type == "self_trans":
            if "gpt-3.5-turbo" in args.model:
                df = pd.read_csv(f'datasets/{args.task}/translations_chatgpt_1106/mgsm_{args.lang}.tsv',delimiter='\t',header=None)
            elif "Mistral-7B" in args.model:
                df = pd.read_csv(f'datasets/{args.task}/translations_mistral/mgsm_{args.lang}.tsv',delimiter='\t',header=None)
            elif "Llama-2-70B-Chat-AWQ" in args.model:
                output_folder = get_output_folder(args)
                file_path_cleaned = os.path.join(output_folder,f'data_translated_cleaned.json')
                with open(file_path_cleaned) as f:
                    ds = json.load(f)
                return ds
        else:
            ds = load_dataset("juletxara/mgsm", args.lang, split='test')
            ds = ds.remove_columns(['answer','equation_solution'])
            ds = ds.rename_column("question", "Q").rename_column("answer_number", "A")
            df = ds.to_pandas()
            # df = pd.read_csv(f'datasets/{args.task}/mgsm_{args.lang}.tsv',delimiter='\t',header=None)
        df.columns = ['Q','A']
        ds = df.to_dict('records')
    elif args.task == 'wildchat':
        file_path = f'results/wildchat/data_translated/{args.lang}_{args.culture_type}.json'
        with open(file_path) as f:
                ds = json.load(f)
        if args.prompt_type == 'google' or args.prompt_type == 'nllb': 
            for item in ds:
                item['question_native'] = item['question']
                item['question'] = item['question_translation']
    else:
        if args.prompt_type == 'google' or args.prompt_type == 'nllb':
            ds = load_from_disk(f"datasets/{args.task}/translations_{args.prompt_type}/"+args.lang)
        elif args.prompt_type == 'google_direct':
            ds = load_from_disk(f"datasets/{args.task}/translations_google/"+args.lang)
        else:
            ds = load_from_disk(f"datasets/{args.task}/"+args.lang)
    return ds

def gen_prompt(args, item):
    if args.task == "mgsm":
        if args.prompt_type == 'direct':
            template = "{Q}"
            template = template + '\nYou should format your final answer as "Answer: <Arabic numeral>".'
            prompt = template.format(Q=item['Q'])
        elif args.prompt_type == 'en_cot':
            template = "{Q}\nLet's think step by step in English."
            template = template + '\nYou should format your final answer as "Answer: <Arabic numeral>".'
            prompt = template.format(Q=item['Q'])
        # elif args.prompt_type == 'google' or args.prompt_type == 'nllb':
        elif args.prompt_type in ['google','nllb','self_trans']:
            template = "{Q}\nLet's think step by step."
            template = template + '\nYou should format your final answer as "Answer: <Arabic numeral>".'
            prompt = template.format(Q=item['Q'])
        elif args.prompt_type == 'google_direct':
            template = "{Q}"
            template = template + '\nYou should format your final answer as "Answer: <Arabic numeral>".'
            prompt = template.format(Q=item['Q'])
        elif args.prompt_type == 'direct_native':
            template = dic_template_basic[args.task][args.lang]
            output_format = f'{dic_tempalte_format["general"][args.lang]}"{dic_answer_word[args.lang]}: <{dic_tempalte_format["mgsm"][args.lang]}>".'
            template = template + '\n' + output_format
            # print(template)
            prompt = template.format(item['Q'])
        elif args.prompt_type == 'native_cot':
            template = dic_template_basic[args.task][args.lang]
            template += f'\n{dic_template_cot["cot"][args.lang]}'  
            template += f'\n{dic_tempalte_format["general"][args.lang]}"{dic_answer_word[args.lang]}: <{dic_tempalte_format["mgsm"][args.lang]}>".'
            # print(template)
            prompt = template.format(item['Q'])
        elif args.prompt_type == 'xlt':
            template = dic_template_xlt[args.task]
            prompt = template.format(lang=lang_codes[args.lang], Q=item['Q'])
        else:
            raise NotImplementedError

    elif args.task == "xcopa":
        if args.prompt_type == 'direct':
            template = "Here is a premise: {premise}. What is the {question}? Help me pick the more plausible option: -choice1: {choice1}, -choice2: {choice2}"
            template = template + '\nYou should format your final answer as "Answer: 1" or "Answer: 2".'
            prompt = template.format(premise=item['premise'], question=item['question'], choice1=item['choice1'], choice2=item['choice2'])
        elif args.prompt_type == 'en_cot':
            template = "Here is a premise: {premise}. What is the {question}? Help me pick the more plausible option: -choice1: {choice1}, -choice2: {choice2}\nLet's think step by step in English."
            template = template + '\nYou should format your final answer as "Answer: 1" or "Answer: 2".'
            prompt = template.format(premise=item['premise'], question=item['question'], choice1=item['choice1'], choice2=item['choice2'])
        elif args.prompt_type == 'google' or args.prompt_type == 'nllb':
            template = "Here is a premise: {premise}. What is the {question}? Help me pick the more plausible option: -choice1: {choice1}, -choice2: {choice2}\nLet's think step by step."
            template = template + '\nYou should format your final answer as "Answer: 1" or "Answer: 2".'
            prompt = template.format(premise=item['premise'], question=item['question'], choice1=item['choice1'], choice2=item['choice2'])
        elif args.prompt_type == 'google_direct':
            template = "Here is a premise: {premise}. What is the {question}? Help me pick the more plausible option: -choice1: {choice1}, -choice2: {choice2}"
            template = template + '\nYou should format your final answer as "Answer: 1" or "Answer: 2".'
            prompt = template.format(premise=item['premise'], question=item['question'], choice1=item['choice1'], choice2=item['choice2'])
        elif args.prompt_type == 'direct_native':
            template = dic_template_basic[args.task][args.lang]
            output_format = f'{dic_tempalte_format["general"][args.lang]}"{dic_answer_word[args.lang]}: <{dic_tempalte_format["xcopa"][args.lang]}>".'
            template = template + '\n' + output_format
            # print(template)
            prompt = template.format(item['premise'], dic_cause_effect[item['question']][args.lang], item['choice1'], item['choice2'])
        elif args.prompt_type == 'native_cot':
            template = dic_template_basic[args.task][args.lang]
            template += f'\n{dic_template_cot["cot"][args.lang]}'  
            template += f'\n{dic_tempalte_format["general"][args.lang]}"{dic_answer_word[args.lang]}: <{dic_tempalte_format["xcopa"][args.lang]}>".'
            # print(template)
            prompt = template.format(item['premise'], dic_cause_effect[item['question']][args.lang], item['choice1'], item['choice2'])
        elif args.prompt_type == 'xlt':
            template = dic_template_xlt[args.task]
            prompt = template.format(lang=lang_codes[args.lang], premise=item['premise'], question=item['question'], choice1=item['choice1'], choice2=item['choice2'])
        else:
            raise NotImplementedError
        # template = template + '\nYou should format your final answer as "Answer: 1" or "Answer: 2".'
        # prompt = template.format(premise=item['premise'], question=item['question'], choice1=item['choice1'], choice2=item['choice2'])
    
    elif args.task == "xnli":
        if args.prompt_type == 'direct':
            template = "{premise} Based on previous passage, is it true that {hypothesis}? Yes, No, or Maybe?"
            template += '\nYou should format your final answer as "Answer: Yes", "Answer: No", or "Answer: Maybe".'
            prompt = template.format(premise=item['premise'], hypothesis=item['hypothesis'])
        elif args.prompt_type == 'en_cot':
            template = "{premise} Based on previous passage, is it true that {hypothesis}? Yes, No, or Maybe?\nLet's think step by step in English."
            template += " You should format your final answer as \"Answer: Yes\", \"Answer: No\", or \"Answer: Maybe\"."
            prompt = template.format(premise=item['premise'], hypothesis=item['hypothesis'])
        elif args.prompt_type == 'google' or args.prompt_type == 'nllb':
            template = "{premise} Based on previous passage, is it true that {hypothesis}? Yes, No, or Maybe?\nLet's think step by step."
            template += " You should format your final answer as \"Answer: Yes\", \"Answer: No\", or \"Answer: Maybe\"."
            prompt = template.format(premise=item['premise'], hypothesis=item['hypothesis'])
        elif args.prompt_type == 'google_direct':
            template = "{premise} Based on previous passage, is it true that {hypothesis}? Yes, No, or Maybe?"
            template += '\nYou should format your final answer as "Answer: Yes", "Answer: No", or "Answer: Maybe".'
            prompt = template.format(premise=item['premise'], hypothesis=item['hypothesis'])
        elif args.prompt_type == 'direct_native':
            template = dic_template_basic[args.task][args.lang]
            output_format = f'{dic_tempalte_format["general"][args.lang]}"{dic_answer_word[args.lang]}: <{dic_tempalte_format["xnli"][args.lang]}>".'
            template = template + '\n' + output_format
            # print(template)
            prompt = template.format(item['premise'], item['hypothesis'])
        elif args.prompt_type == 'native_cot':
            template = dic_template_basic[args.task][args.lang]
            template += f'\n{dic_template_cot["cot"][args.lang]}'  
            template += f'\n{dic_tempalte_format["general"][args.lang]}"{dic_answer_word[args.lang]}: <{dic_tempalte_format["xnli"][args.lang]}>".'
            # print(template)
            prompt = template.format(item['premise'], item['hypothesis'])
        elif args.prompt_type == 'xlt':
            template = dic_template_xlt[args.task]
            prompt = template.format(lang=lang_codes[args.lang], premise=item['premise'], hypothesis=item['hypothesis'])
        else:
            raise NotImplementedError
        # template = template + '\nYou should format your final answer as "Answer: Yes", "Answer: No", or "Answer: Maybe".'
        # prompt = template.format(premise=item['premise'], hypothesis=item['hypothesis'])

    elif args.task == "paws-x":
        if args.prompt_type == 'direct':
            template = "Sentence 1: {sentence1} Sentence 2: {sentence2} Question: Does Sentence 1 paraphrase Sentence 2? Yes or No?"
            template = template + '\nYou should format your final answer as "Answer: Yes" or "Answer: No".'
            prompt = template.format(sentence1=item['sentence1'], sentence2=item['sentence2'])
        elif args.prompt_type == 'en_cot':
            template = "Sentence 1: {sentence1} Sentence 2: {sentence2} Question: Does Sentence 1 paraphrase Sentence 2? Yes or No?\nLet's think step by step in English."
            template = template + '\nYou should format your final answer as "Answer: Yes" or "Answer: No".'
            prompt = template.format(sentence1=item['sentence1'], sentence2=item['sentence2'])
        elif args.prompt_type == 'google' or args.prompt_type == 'nllb':
            template = "Sentence 1: {sentence1} Sentence 2: {sentence2} Question: Does Sentence 1 paraphrase Sentence 2? Yes or No?\nLet's think step by step."
            template = template + '\nYou should format your final answer as "Answer: Yes" or "Answer: No".'
            prompt = template.format(sentence1=item['sentence1'], sentence2=item['sentence2'])
        elif args.prompt_type == 'google_direct':
            template = "Sentence 1: {sentence1} Sentence 2: {sentence2} Question: Does Sentence 1 paraphrase Sentence 2? Yes or No?"
            template = template + '\nYou should format your final answer as "Answer: Yes" or "Answer: No".'
            prompt = template.format(sentence1=item['sentence1'], sentence2=item['sentence2'])
        elif args.prompt_type == 'direct_native':
            template = dic_template_basic[args.task][args.lang]
            output_format = f'{dic_tempalte_format["general"][args.lang]}"{dic_answer_word[args.lang]}: <{dic_tempalte_format["paws-x"][args.lang]}>".'
            template = template + '\n' + output_format
            # print(template)
            prompt = template.format(item['sentence1'], item['sentence2'])
        elif args.prompt_type == 'native_cot':
            template = dic_template_basic[args.task][args.lang]
            template += f'\n{dic_template_cot["cot"][args.lang]}'  
            template += f'\n{dic_tempalte_format["general"][args.lang]}"{dic_answer_word[args.lang]}: <{dic_tempalte_format["paws-x"][args.lang]}>".'
            # print(template)
            prompt = template.format(item['sentence1'], item['sentence2'])
        elif args.prompt_type == 'xlt':
            template = dic_template_xlt[args.task]
            prompt = template.format(lang=lang_codes[args.lang], sentence1=item['sentence1'], sentence2=item['sentence2'])
        else:
            raise NotImplementedError
        # template = template + '\nYou should format your final answer as "Answer: Yes" or "Answer: No".'
        # prompt = template.format(sentence1=item['sentence1'], sentence2=item['sentence2'])
    elif args.task == "mkqa":
        if args.prompt_type == 'direct':
            output_format = '\nYou should format your final answer as \"Answer:\".'
            template = "Answer the question in one or a few words in {language}: {question}?"+output_format
            prompt = template.format(language=lang_codes[args.lang], question=item['question'])
        elif args.prompt_type == 'en_cot':
            template = "{question}?\nLet's think step by step in English."
            template += " You should format your final answer in one or a few words in {language} as \"Answer:\"."
            prompt = template.format(language=lang_codes[args.lang], question=item['question'])
        elif args.prompt_type == 'google' or args.prompt_type == 'nllb':
            template = "{question}?\nLet's think step by step."
            template += " You should format your final answer in one or a few words as \"Answer:\"."
            prompt = template.format(question=item['question'])
        elif args.prompt_type == 'google_direct':
            template = "{question}?"
            template += " You should format your final answer in one or a few words as \"Answer:\"."
            prompt = template.format(question=item['question'])
        elif args.prompt_type == 'direct_native':
            template = dic_template_basic[args.task][args.lang]
            output_format = f'{dic_tempalte_format["general"][args.lang]}"{dic_answer_word[args.lang]}:".'
            template = template + '\n' + output_format
            # print(template)
            prompt = template.format(dic_lang_native[args.lang],item['question'])
        elif args.prompt_type == 'native_cot':
            template = dic_template_basic[args.task][args.lang]
            template += f'\n{dic_template_cot["cot"][args.lang]}'  
            template += f'\n{dic_tempalte_format["general"][args.lang]}"{dic_answer_word[args.lang]}:".'
            # print(template)
            prompt = template.format(dic_lang_native[args.lang],item['question'])
        elif args.prompt_type == 'xlt':
            template = dic_template_xlt[args.task]
            prompt = template.format(lang=lang_codes[args.lang], Q=item['question'])
        else:
            raise NotImplementedError
    elif args.task == "xlsum":
        if args.prompt_type == 'direct':
            template = "{article}\nSummarize the article."
            template += " You should format your final summary in one sentence in {language} as \"Answer:\"."
            prompt = template.format(article=item['text'], language=lang_codes[args.lang])
        elif args.prompt_type == 'en_cot':
            template = "{article}\nSummarize the article.\nLet's think step by step in English."
            template += " You should format your final summary in one sentence in {language} as \"Answer:\"."
            prompt = template.format(article=item['text'], language=lang_codes[args.lang])
        elif args.prompt_type == 'google' or args.prompt_type == 'nllb':
            template = "{article}\nSummarize the article.\nLet's think step by step."
            template += " You should format your final summary in one sentence as \"Answer:\"."
            prompt = template.format(article=item['text'])
        elif args.prompt_type == 'google_direct':
            template = "{article}\nSummarize the article."
            template += " You should format your final summary in one sentence as \"Answer:\"."
            prompt = template.format(article=item['text'])
        elif args.prompt_type == 'direct_native':
            template = dic_template_basic[args.task][args.lang]
            output_format = f'{dic_tempalte_format["xlsum"][args.lang]}"{dic_answer_word[args.lang]}:".'
            template = template + '\n' + output_format
            # print(template)
            prompt = template.format(item['text'],dic_lang_native[args.lang])
        elif args.prompt_type == 'native_cot':
            template = dic_template_basic[args.task][args.lang]
            template += f'\n{dic_template_cot["cot"][args.lang]}'  
            template += f'\n{dic_tempalte_format["xlsum"][args.lang]}"{dic_answer_word[args.lang]}:".'
            # print(template)
            prompt = template.format(item['text'],dic_lang_native[args.lang])
        elif args.prompt_type == 'xlt':
            template = dic_template_xlt[args.task]
            prompt = template.format(lang=lang_codes[args.lang], text=item['text'])
        else:
            raise NotImplementedError
    elif args.task == "shareGPT" or args.task == "wildchat" or args.task == "shareGPT_filter":
        # if args.prompt_type == 'direct':
        prompt = item['question']
        # else: 
        #     raise NotImplementedError
    else:
        raise NotImplementedError
        
    return prompt

def get_prompt_ans(args):
    if args.prompt_type == 'direct_native' or args.prompt_type == 'native_cot':
        prompt_for_ans = dic_tempalte_round2[args.task][args.lang]
    elif args.prompt_type == 'xlt':
        prompt_for_ans = 'Therefore, the answer is'
    else:
        if args.task == "mgsm" or args.task == "xcopa":
            prompt_for_ans = 'Therefore, the answer (Arabic numerals) is'
        elif args.task == "xnli":
            prompt_for_ans = 'Therefore, the answer (Yes/No/Maybe) is'
        elif args.task == "paws-x":
            prompt_for_ans = 'Therefore, the answer (Yes/No) is'
        elif args.task == "mkqa":
            prompt_for_ans = 'Therefore, the answer is'
        elif args.task == "xlsum":
            if args.prompt_type == 'direct' or args.prompt_type == 'en_cot':
                prompt_for_ans = 'Therefore, the answer in {lang} is'
                prompt_for_ans = prompt_for_ans.format(lang=lang_codes[args.lang])
            else: 
                prompt_for_ans = 'Therefore, the answer is'
        else:
            raise NotImplementedError
    return prompt_for_ans

def text_2_float(x):
    try: 
        return float(str(x).replace(',',''))
    except Exception:
        return -1

def text_2_int(x):
    try: 
        return int(float(str(x).replace(',','')))
    except Exception:
        return -1

def clean_ans(args,pred_str):
    if args.prompt_type == "xlt":
        return clean_ans_xlt(args, pred_str)
    answer_word = dic_answer_word[args.lang] if args.prompt_type == 'direct_native' or args.prompt_type == 'native_cot' else dic_answer_word['en']
    if args.task == "mgsm" or args.task == "xcopa":
        pred_str = str(pred_str).replace(',','')
        pattern = f'{answer_word}:\s*(-?\d*\.?\d+)'
        pattern2 = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        pred2 = re.findall(pattern2, pred_str)
        if(len(pred) >= 1):
            pred = pred[-1]
        elif(len(pred2) >= 1):
            pred = pred2[-1]
        else: pred = -1
        if args.task == "mgsm":
            return text_2_int(pred)
        elif args.task == "xcopa":
            return text_2_int(pred) - 1
    elif args.task == "xnli":
        # 1:Yes, 2:No, 3:Maybe?",
        # pattern1 = 'Answer: (Yes|No|Maybe)'
        if args.prompt_type in ['direct_native','native_cot']:
            pattern1 = f'{answer_word}:\s*(\d)'
            pred1 = re.findall(pattern1, pred_str)
            pattern = '(\d)'
            pred = re.findall(pattern, pred_str.lower())
            if(len(pred1) >= 1):
                pred = pred1[-1]
                return 0 if pred == '1' else 2 if pred == '2' else 1
                # # if "Yes" in pred:
                # if pred == '1':
                #     return 0
                # # elif "No" in pred:
                # elif pred == '2':
                #     return 2
                # else:
                #     return 1
            elif(len(pred) >= 1):
                pred = pred[-1]
                return 0 if pred == '1' else 2 if pred == '2' else 1
                # # if 'yes' in pred.lower():
                # if pred == '1':
                #     return 0
                # # elif 'no' in pred.lower():
                # elif pred == '2':
                #     return 2
                # else:
                #     return 1
            else: return pred
        else:
            pattern1 = 'Answer: (Yes|No|Maybe)'
            pred1 = re.findall(pattern1, pred_str)
            if(len(pred1) >= 1):
                return 0 if "Yes" in pred1[-1] else 2 if "No" in pred1[-1] else 1
            else:
                return 0 if "yes" in pred_str.lower() else 2 if "no" in pred_str.lower() else 1
            
    elif args.task == "paws-x":
        
        if args.prompt_type in ['direct_native','native_cot']:
            pattern1 = f'{answer_word}:\s*(\d)'
            pred1 = re.findall(pattern1, pred_str)
            pattern = '(\d)'
            pred = re.findall(pattern, pred_str.lower())
            if(len(pred1) >= 1):
                pred = pred1[-1]
                return 1 if pred == '1' else 0 if pred == '2' else -1
            elif(len(pred) >= 1):
                pred = pred[-1]
                return 1 if pred == '1' else 0 if pred == '2' else -1
            else: return -1
        else: 
            pattern1 = 'Answer: (Yes|No)'
            pred1 = re.findall(pattern1, pred_str)
            if(len(pred1) >= 1):
                return 1 if "Yes" in pred1[-1] else 0 if "No" in pred1[-1] else -1
            else:
                return 1 if "yes" in pred_str.lower() else 0 if "no" in pred_str.lower() else -1

    elif args.task == "mkqa" or args.task == "xlsum":
        patterns = [answer_word, "Answer: ","answer: ","answer is ","Answer is ","Answer is: ","answer is: ","Answer is:","answer is:","Answer is","answer is","Answer:","answer:"]
        for pattern in patterns:
            if pattern in pred_str:
                pred_str = pred_str.split(pattern)[-1].strip()
                break
        if args.prompt_type == 'google' or args.prompt_type == 'google_direct':
            pred_str = get_translation_google(pred_str, dest=args.lang)
        elif args.prompt_type == 'nllb':
            # translation_pipeline = prepare_pipeline_nllb(src_lang='en', tgt_lang=args.lang, max_length=400)
            pred_str = get_translation_nllb(args.translation_pipeline, pred_str)
        return pred_str
    elif args.task == "shareGPT" or args.task == "wildchat" or args.task == "shareGPT_filter":
        if args.prompt_type == 'direct':
            pred_str = pred_str
        elif args.prompt_type == 'google' or args.prompt_type == 'google_direct':
            pred_str = get_translation_google(pred_str, dest=args.lang)
        elif args.prompt_type == 'nllb':
            pred_str = get_translation_nllb(args.translation_pipeline, pred_str)
        return pred_str
    else:
        return NotImplementedError
    
def clean_ans_xlt(args, pred_str):
    answer_word = "Answer"
    if args.task == "mgsm" or args.task == "xcopa":
        answer_word = 'Choice number' if args.task == "xcopa" else 'Answer'
        pred_str = str(pred_str).replace(',','')
        pattern = f'{answer_word}:\s*(-?\d*\.?\d+)'
        pattern2 = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        pred2 = re.findall(pattern2, pred_str)
        if(len(pred) >= 1):
            pred = pred[-1]
        elif(len(pred2) >= 1):
            pred = pred2[-1]
        else: pred = -1
        if args.task == "mgsm":
            return text_2_int(pred)
        elif args.task == "xcopa":
            return text_2_int(pred) - 1
    elif args.task == "xnli":
        # print("the task is xnli")
        pred_str = pred_str.lower()
        # answer_word = "Relationship"
        # pattern1 = f'{answer_word}:\s*(\d)'
        answer_word = "relationship"
        pattern1 = f'{answer_word}:\s*(\w+)'
        pred1 = re.findall(pattern1, pred_str)
        if(len(pred1) >= 1):
            pred = pred1[-1]
            if "entailment" in pred.lower():
            # if pred == '1':
                return 0
            elif "contradiction" in pred.lower():
            # elif pred == '2':
                return 2
            else:
                return 1
        elif 'entailment' in pred_str.lower():
            return 0
        elif 'contradiction' in pred_str.lower():
            return 2
        else:
            return 1
    elif args.task == "paws-x":
        pattern = 'answer:\s*(yes|no|maybe)'
        pred = re.findall(pattern, pred_str.lower())
        if(len(pred) >= 1):
            pred = pred[-1]
            if "yes" in pred:
                return 1
            elif "no" in pred:
                return 0
            else:
                return -1
        elif 'yes' in pred_str.lower():
            return 1
        elif 'no' in pred_str.lower():
            return 0
        else:
            return -1
    elif args.task == "mkqa" or args.task == "xlsum":
        patterns = [answer_word, "Answer: ","answer: ","answer is ","Answer is ","Answer is: ","answer is: ","Answer is:","answer is:","Answer is","answer is","Answer:","answer:",
                    "Summary:","summary:","summary is","Summary is","Summary is:","summary is:","Summary is:","summary is:"]
        for pattern in patterns:
            if pattern in pred_str:
                pred_str = pred_str.split(pattern)[-1].strip()
                break
        return pred_str
    
    
def check_ans(args, pred):
    if args.prompt_type == "xlt":
        for ans in ["Answer:","Relationship:","Choice number:","Summary:"]:
            if ans in pred:
                return True
        return False
    answer_word = dic_answer_word[args.lang] if args.prompt_type == 'direct_native' or args.prompt_type == 'native_cot' else dic_answer_word['en']
    if args.task == "mgsm" or args.task == "mkqa" or args.task == "xlsum":
        # return "Answer: " in pred
        return f'{answer_word}:' in pred
    if args.task == "xcopa":
        # return "Answer: 1" in pred or "Answer: 2" in pred
        return f'{answer_word}: 1' in pred or f'{answer_word}: 2' in pred
    elif args.task == "xnli":
        # return "Answer: Yes" in pred or "Answer: No" in pred or "Answer: Maybe" in pred
        return f'{answer_word}: 1' in pred or f'{answer_word}: 2' in pred or f'{answer_word}: 3' in pred
    elif args.task == "paws-x":
        # return "Answer: Yes" in pred or "Answer: No" in pred
        return f'{answer_word}: 1' in pred or f'{answer_word}: 2' in pred
    # elif args.task == "mkqa" or args.task == "xlsum":
    #     return "Answer: " in pred
        # return True
    else:
        return NotImplementedError
    
def evaluate_item(args, item):
    if args.task == "mgsm":
        item['label'] = text_2_int(item['A'])
        return text_2_int(item['pred']) == text_2_int(item['A'])
    elif args.task == "xcopa" or args.task == "xnli" or args.task == "paws-x":
        return item['pred'] == item['label']
    elif args.task == "mkqa":
        item['label'] = item['answer']
        item['pred'] = item['pred'].split('\n\n')[0]
        return compute_max_score_over_answers(calculate_f1, item['pred'], item['answer'], args.lang)
    elif args.task == "xlsum":
        item['label'] = item['summary']
        scores = args.rouge_scorer.score(item['summary'], item['pred'])
        item['scores'] = scores
        return scores[0]
    else:
        return NotImplementedError

template_judge = """[System] 
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, expected language and level of detail of the response. Begin your evaluation by providing a short explanation (up to 100 words). Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format: "Rating: <rating>", for example: "Rating: 5".

[Question]
{question}

[The Start of Assistant’s Answer]
{answer}
[The End of Assistant’s Answer]
"""

def gen_prompt_judge(question,answer):
    return template_judge.format(question=question, answer=answer)

# Extract the rating from the judge's response "Rating: 5".
def clean_judge(judgement):
    pattern = 'Rating: (\d+)'
    pred = re.findall(pattern, judgement)
    if(len(pred) >= 1):
        pred = pred[-1]
    else: pred = -1
    return int(pred)

