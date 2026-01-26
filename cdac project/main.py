import json
import re
import unicodedata
import ollama
from langdetect import detect_langs
from collections import OrderedDict


class UnicodeScriptDetector:
    
    SCRIPT_RANGES = {
        "Hindi": (0x0900, 0x097F),
        "Bengali": (0x0980, 0x09FF),
        "Punjabi": (0x0A00, 0x0A7F),
        "Gujarati": (0x0A80, 0x0AFF),
        "Odia": (0x0B00, 0x0B7F),
        "Tamil": (0x0B80, 0x0BFF),
        "Telugu": (0x0C00, 0x0C7F),
        "Kannada": (0x0C80, 0x0CFF),
        "Malayalam": (0x0D00, 0x0D7F),
    }
    
    @staticmethod
    def detect(text):
        if not text or len(text) < 10:
            return None, 0.0
        
        counts = {lang: 0 for lang in UnicodeScriptDetector.SCRIPT_RANGES}
        total = 0
        
        for char in text:
            if not char.isalpha():
                continue
            total += 1
            cp = ord(char)
            for lang, (start, end) in UnicodeScriptDetector.SCRIPT_RANGES.items():
                if start <= cp <= end:
                    counts[lang] += 1
                    break
        
        if total == 0:
            return None, 0.0
        
        best = max(counts.items(), key=lambda x: x[1])
        if best[1] > 0:
            conf = best[1] / total
            if conf >= 0.3:
                return best[0], min(conf, 0.99)
        
        return None, 0.0


class RomanizedHindiDetector:
    
    MARKERS = frozenset({
        "hai", "haan", "han", "nahi", "nahin", "kyu", "kyun", "kya", "kaise",
        "mera", "meri", "mere", "tera", "teri", "tum", "aap", "ap",
        "kar", "karo", "karna", "kiya", "hum", "ham", "main",
        "bahut", "bohot", "thoda", "jaldi", "abhi", "kal", "aaj",
        "wala", "wali", "wale", "se", "ko", "me", "mein", "par",
        "achha", "acha", "bura", "bhi", "baat", "sahi", "galat",
        "mat", "kariye", "krdo", "krna", "kr", "diya"
    })
    
    ALPHA_RE = re.compile(r"[^a-zA-Z ]")
    
    @classmethod
    def detect(cls, text):
        if not text or len(text) < 20:
            return None, 0.0
        
        if sum(1 for c in text if ord(c) < 128) / len(text) < 0.85:
            return None, 0.0
        
        words = set(cls.ALPHA_RE.sub(" ", text.lower()).split())
        hits = len(words & cls.MARKERS)
        
        if hits < 2:
            return None, 0.0
        
        conf = min(hits / max(len(text.split()) * 0.12, 1), 0.95)
        return ("Hindi (Romanized)", conf) if conf >= 0.25 else (None, 0.0)


class MarathiDetector:
    
    MARKERS = frozenset({
        "आहे", "आहेत", "नाही", "नाहीत", "झाला", "झाली", "झाले", "झालेला", "झालेली", "झालेल्या",
        "करतो", "करते", "करतात", "केले", "केली", "केलेला", "केलेली",
        "होता", "होती", "होते", "होत", "असतात", "असे", "असून",
        "यांनी", "यांचा", "यांची", "यांचे", "यांना", "यांच्यावर", "यांच्याकडून",
        "चा", "ची", "चे", "च्या", "ला", "ना", "मध्ये", "वर", "साठी", "मुळे", "पर्यंत",
        "कडून", "प्रमाणे", "संबंधित", "प्रस्ताव", "अधिवेशन", "विधेयक", "सभागृह", "खासदार",
        "म्हणजे", "म्हणून", "काय", "का", "की", "बघा", "दिला", "दिली", "दिले",
        "गेला", "गेली", "गेलो", "पाठवले", "आम्ही", "तुम्ही", "त्यांनी", "त्याचा"
    })
    
    ALPHA_RE = re.compile(r"[^ऀ-ॿ ]")
    
    @classmethod
    def detect(cls, text):
        if not text or len(text) < 12:
            return None, 0.0
        
        dev_count = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        if dev_count / max(len(text), 1) < 0.65:
            return None, 0.0
        
        normalized = text.replace("।", " ").replace(",", " ").replace("!", " ").replace("?", " ")
        words = set(cls.ALPHA_RE.sub(" ", normalized).split())
        
        hits = len(words & cls.MARKERS)
        
        if hits < 1:
            return None, 0.0
        
        word_count = max(len(words), 1)
        conf = min(hits / word_count * 4.0, 0.96)
        return ("Marathi", conf) if conf >= 0.22 else (None, 0.0)


class NLPOrchestrator:
    def __init__(self, model_id="llama3.1:8b-instruct-q4_K_M"):
        print(f"Using Ollama model: {model_id}")
        self.model_id = model_id
        
        self.cleanup_trans = str.maketrans({"\u200c": "", "\u200d": "", "\ufeff": ""})
        self.space_re = re.compile(r"\s+")
        self.trailing_comma_re = re.compile(r",\s*([}\]])")
        self.json_block_re = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
        
        self._llm_cache = OrderedDict()
        self._cache_max_size = 300
    
    def callllm(self, prompt, max_tokens=800):
        cache_key = f"{max_tokens}:{prompt[:400]}"
        
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        
        resp = ollama.chat(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,
                "num_predict": max_tokens,
                "top_p": 0.9
            },
        )
        result = resp["message"]["content"].strip()
        
        self._llm_cache[cache_key] = result
        if len(self._llm_cache) > self._cache_max_size:
            self._llm_cache.popitem(last=False)
        
        return result
    
    def clean(self, text):
        text = unicodedata.normalize("NFKC", text).translate(self.cleanup_trans)
        return self.space_re.sub(" ", text).strip()
    
    def parsejson(self, response):
        if not response:
            return None
        
        response = response.replace(""", '"').replace(""", '"').replace("'", "'")
        response = self.trailing_comma_re.sub(r"\1", response)
        
        match = self.json_block_re.search(response)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(response[start:end])
            except:
                pass
        
        return None
    
    def llmdetectlanguage(self, text):
        prompt = f"""What language is this text written in?
Only return the language name in English. Examples:
- Hindi
- Marathi
- Punjabi
- English
- Bengali

Text:
{text[:400]}"""
        
        resp = self.callllm(prompt, max_tokens=30)
        lang = resp.strip()
        
        lang = lang.replace("Hindi.", "Hindi").replace("Marathi.", "Marathi")
        if "marathi" in lang.lower():
            lang = "Marathi"
        elif "hindi" in lang.lower():
            lang = "Hindi"
        
        return {"primary_lang": lang, "confidence": 0.92, "method": "llm"}
    
    def detectlanguage(self, text):
        dev_count = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        total_len = max(len(text), 1)
        dev_ratio = dev_count / total_len
        
        if dev_ratio >= 0.55:
            m_lang, m_conf = MarathiDetector.detect(text)
            if m_lang:
                return {"primary_lang": m_lang, "confidence": round(m_conf, 3), "method": "marathi_first"}
            
            u_lang, u_conf = UnicodeScriptDetector.detect(text)
            if u_lang == "Hindi":
                return {"primary_lang": "Hindi", "confidence": round(u_conf, 3), "method": "unicode_hindi"}
            
            return self.llmdetectlanguage(text)
        
        lang, conf = UnicodeScriptDetector.detect(text)
        if lang and conf >= 0.3:
            if lang == "Hindi":
                m_lang, m_conf = MarathiDetector.detect(text)
                if m_lang and m_conf >= 0.30:
                    return {"primary_lang": m_lang, "confidence": round(m_conf, 3), "method": "marathi_markers"}
            return {"primary_lang": lang, "confidence": round(conf, 3), "method": "unicode"}
        
        lang, conf = RomanizedHindiDetector.detect(text)
        if lang and conf >= 0.25:
            return {"primary_lang": lang, "confidence": round(conf, 3), "method": "romanized"}
        
        try:
            detected = detect_langs(text)[0]
            lang_name = detected.lang.upper()
            if lang_name == "HI":
                lang_name = "Hindi"
            elif lang_name == "MR":
                lang_name = "Marathi"
            elif lang_name == "EN":
                lang_name = "English"
            elif lang_name == "PA":
                lang_name = "Punjabi"
            return {"primary_lang": lang_name, "confidence": round(detected.prob, 3), "method": "langdetect"}
        except:
            pass
        
        return self.llmdetectlanguage(text)
    
    def translate(self, text, lang):
        if "english" in lang.lower():
            return text, 1.0
        
        prompt = f"""Translate to fluent English. Preserve proper nouns.
Return ONLY: {{"translated_text": "...", "confidence": 0.0}}

Text: {text}"""
        
        resp = self.callllm(prompt, max_tokens=200)
        resp = resp.strip()
        
        parsed = self.parsejson(resp)
        
        if parsed and "translated_text" in parsed:
            translated = str(parsed["translated_text"]).strip()
            conf = float(parsed.get("confidence", 0.88))
            if conf == 0.0:
                conf = 0.75
        else:
            # Aggressive cleanup for artifacts
            for prefix in ['{"translated_text": "', '{ "translated_text": "', '"translated_text": ']:
                if resp.startswith(prefix):
                    translated = resp[len(prefix):].rstrip('"}').strip()
                    break
            else:
                translated = resp.strip('"').strip()
            
            conf = 0.7
        
        return translated, conf
    
    def analyze(self, text):
        prompt = f"""Analyze this text and return ONLY valid JSON — no markdown, no explanations, no extra text.

{{
  "domain_ident": [],
  "domain_confidence": 0.0,
  "sentiment": "",
  "sentiment_confidence": 0.0,
  "NER": {{"Person": [], "Location": [], "Organisation": [], "Event": [], "Product": []}},
  "ner_confidence": 0.0,
  "Event_calendar": [],
  "event_calendar_confidence": 0.0,
  "Country_iden": "",
  "country_confidence": 0.0,
  "Fact_checker": {{"relevant_topics": [], "confidence_level": 0.0, "relevance_rating": ""}},
  "Summary": "",
  "summary_confidence": 0.0
}}

STRICT RULES:
- domain_ident: Choose ONLY ONE from this list — [Politics, Crime, Military, Terrorism, Radicalisation, Extremism in J&K, Law and Order, Narcotics, Left Wing Extremism, General]
  - Use "Politics" ONLY if the main topic is elections, policy debates, political parties, government formation, or political rivalry.
  - Use "Law and Order" for riots, protests, accidents, fires, disasters, public safety incidents, rescue operations.
  - Use "General" for everyday news that doesn't fit above (most accidents/fires go here unless political angle dominates).
- sentiment: "Positive", "Negative", "Neutral", "Anti-National"
  - Negative: deaths, destruction, suffering, tragedy, criticism of authorities, chaos
  - Neutral: factual reporting without strong emotion
  - Anti-National: ONLY direct threats/calls against India or its unity
- NER: ONLY proper named entities
  - Person: real individual names (e.g. "Arup Biswas")
  - Location: specific places (e.g. "Nazirabad", "Anandpur")
  - Organisation: named groups (e.g. "Fire Department", "West Bengal Government")
  - Event: named events only (leave empty if none)
  - Product: named products only (leave empty if none)
- Summary: One single continuous sentence. High-level overview in your own words. Do NOT copy sentences from the text. Focus on what happened, who was affected, and outcome.
- All confidence scores: 0.0-1.0 — be realistic, 0.9+ only for very clear cases

Text:
{text}"""
        
        max_tokens = min(300, 500 + len(text.split()) // 5)
        resp = self.callllm(prompt, max_tokens=max_tokens)
        result = self.parsejson(resp)
        
        if not result:
            result = {
                "domain_ident": ["General"], "domain_confidence": 0.3,
                "sentiment": "Neutral", "sentiment_confidence": 0.3,
                "NER": {"Person": [], "Location": [], "Organisation": [], "Event": [], "Product": []},
                "ner_confidence": 0.0,
                "Event_calendar": [], "event_calendar_confidence": 0.0,
                "Country_iden": "Unknown", "country_confidence": 0.0,
                "Fact_checker": {"relevant_topics": [], "confidence_level": 0.0, "relevance_rating": "Low"},
                "Summary": "Analysis failed.", "summary_confidence": 0.0
            }
        
        return result
    
    @staticmethod
    def tostr(val):
        if isinstance(val, list):
            return ", ".join(str(v) for v in val if v)
        return str(val) if val else ""
    
    def process(self, text):
        cleaned = self.clean(text)
        
        print("Detecting language...")
        lang_info = self.detectlanguage(cleaned)
        
        if "english" in lang_info["primary_lang"].lower() and lang_info["confidence"] >= 0.88:
            print("Skipping translation (confident English detected)...")
            translated = cleaned
            trans_conf = 1.0
        else:
            print("Translating...")
            translated, trans_conf = self.translate(cleaned, lang_info["primary_lang"])
        
        print("Analyzing...")
        analysis = self.analyze(translated)
        
        ner = analysis.get("NER", {})
        ner_formatted = {
            "Person": self.tostr(ner.get("Person", [])),
            "Location": self.tostr(ner.get("Location", [])),
            "Organisation": self.tostr(ner.get("Organisation", [])),
            "Event": self.tostr(ner.get("Event", [])),
            "Product": self.tostr(ner.get("Product", []))
        }
        
        fc = analysis.get("Fact_checker", {})
        
        return {
            "Cleaned_content": cleaned,
            "domain_ident": self.tostr(analysis.get("domain_ident", [])),
            "domain_ident_confidence_score": round(float(analysis.get("domain_confidence", 0.0)), 3),
            "lang_det": lang_info["primary_lang"],
            "lang_det_confidence_score": lang_info["confidence"],
            "sentiment": analysis.get("sentiment", "Neutral"),
            "sentiment_confidence_score": round(float(analysis.get("sentiment_confidence", 0.0)), 3),
            "NER": ner_formatted,
            "NER_confidence_score": round(float(analysis.get("ner_confidence", 0.0)), 3),
            "Event_calender": self.tostr(analysis.get("Event_calendar", [])),
            "Event_calender_confidence_score": round(float(analysis.get("event_calendar_confidence", 0.0)), 3),
            "Country_iden": analysis.get("Country_iden", "Unknown"),
            "Country_iden_confidence_score": round(float(analysis.get("country_confidence", 0.0)), 3),
            "Summary": str(analysis.get("Summary", "")),
            "Summary_confidence_score": round(float(analysis.get("summary_confidence", 0.0)), 3),
            "Fact_checker": self.tostr(fc.get("relevant_topics", [])),
            "Fact_checker_relevance_rating": fc.get("relevance_rating", "Low"),
            "Fact_checker_confidence_score": round(float(fc.get("confidence_level", 0.0)), 3),
            "Translation": translated,
            "Translation_confidence_score": round(trans_conf, 3)
        }


if __name__ == "__main__":
    content = '''अग्निशमन दलानं आगीवर थोड्या प्रमाणात नियंत्रण मिळवल्यानंतर गॅस कटर घेत इमारतीत गेले. आनंदपूरच्या नाझिराबादमधील गोदामात प्रामुख्यानं कोरड्या स्वरुपाची खाद्य पदार्थाची पॅकेटस आणि सॉफ्ट ड्रिंक्सच्या बाटल्या होत्या. अग्निशमन दलाच्या माहितीनुसार आग दोन गोदामांमध्ये पसरली. यामुळं सर्व काही जळून खाक झालं. 

अग्निशमन दलाला या आगीसंदर्भात माहिती मिळाली होती. मात्र, गोदाम अरुंद रस्ता असणाऱ्या ठिकाणी असल्यानं अग्निशमन दलाच्या वाहनांना घटनास्थळापर्यंत पोहोचण्यास अडचणी निर्माण झाल्या यामुळं आगीवर नियंत्रण आणण्यात वेळ लागला. 

पश्चिम बंगालचे ऊर्जा मंत्री अरुप विश्वास घटनास्थळी दाखल झाले आणि त्यांनी मृतांच्या नातेवाईकांचं सांत्वन केलं.  याशिवाय त्यांच्याकडून बचाव कार्याचा आढावा घेण्यात आला. आगीच्या घटनेत ज्यांचे नातेवाईक बेपत्ता आहेत त्यांच्या नातेवाईकांसोबत अरुप विश्वास यांनी चर्चा केली. मंत्री अरुप विश्वास यांनी दिलेल्या माहितीनुसार पोलीस आणि अग्निशमन दलाकडून आगीवर नियंत्रण आणण्याचं काम सुरु आहे. आगीवर मोठ्या प्रमाणात नियंत्रण मिळवण्यात आल्याची माहिती देत फायर फायटर्स इमारतीत दाखल झाल्याची माहिती त्यांनी दिली. अरुप विश्वास यांनी ही राजकारण करण्याची वेळ नाही, पोलीस आणि अग्निशमन दलाला त्यांचं काम करु देण्याची वेळ असल्याचं म्हटलं. 

गोदामात आग कशामुळं लागली हे स्पष्ट झालेलं नाही. रात्रीच्या ड्युटीवर असणारे कर्मचारी आत अडकले होते. याशिवाय सुरक्षा रक्षक म्हणून काम करणारे सहा लोक देखील आत अडकल्याची माहिती आहे. स्थानिकांच्या माहितीनुसार गोदाम बाहेरुन बंद होतं, त्यामुळं लोक आत जाऊ शकले नाहीत.'''
    
    nlp = NLPOrchestrator()
    print("=" * 60)
    print("NLP ORCHESTRATION PIPELINE")
    print("=" * 60)
    
    result = nlp.process(content)
    
    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))