import json
import re
import unicodedata
import ollama
from langdetect import detect_langs
from collections import OrderedDict
from pydantic import BaseModel, Field, ValidationError, field_validator, AliasChoices
from typing import List, Optional

class NERData(BaseModel):
    Person: List[str] = Field(default_factory=list)
    Location: List[str] = Field(default_factory=list)
    Organisation: List[str] = Field(
        default_factory = list, 
        validation_alias = AliasChoices('Organisation', 'Organization')
    )
    Event: List[str] = Field(default_factory=list)
    Product: List[str] = Field(default_factory=list)

class FactCheckerData(BaseModel):
    relevant_topics: List[str] = Field(default_factory=list)
    confidence_level: float = Field(default=0.0)
    relevance_rating: str = Field(default="Low")

class AnalysisResult(BaseModel):
    domain_ident: List[str] = Field(default_factory=list)
    domain_confidence: float = Field(default=0.0)
    sentiment: str = Field(default="Neutral")
    sentiment_confidence: float = Field(default=0.0)
    NER: NERData = Field(default_factory=NERData)
    ner_confidence: float = Field(default=0.0)
    Event_calendar: List[str] = Field(default_factory=list)
    event_calendar_confidence: float = Field(default=0.0)
    Country_iden: str = Field(default="Unknown")
    country_confidence: float = Field(default=0.0)
    Fact_checker: FactCheckerData = Field(default_factory=FactCheckerData)
    Summary: str = Field(default="")
    summary_confidence: float = Field(default=0.0)

    @field_validator('domain_confidence', 'sentiment_confidence', 'ner_confidence', 
                     'event_calendar_confidence', 'country_confidence', 'summary_confidence', mode='before')
    @classmethod
    def normalize_confidence(cls, v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

class TranslationResult(BaseModel):
    translated_text: str
    confidence: float = 0.0

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
        "Urdu": (0x0600, 0x06FF),
        "Santali": (0x1C50, 0x1C7F),
        "Manipuri": (0xABC0, 0xABFF),
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
    
    def parse_with_pydantic(self, response, pydantic_model):
        if not response:
            return None
        
        response = response.replace(""", '"').replace(""", '"').replace("'", "'")
        response = self.trailing_comma_re.sub(r"\1", response)
        
        json_str = None
        match = self.json_block_re.search(response)
        
        if match:
            json_str = match.group(1)
        else:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
        
        if not json_str:
            return None

        try:
            data = json.loads(json_str)
            return pydantic_model(**data)
        except (json.JSONDecodeError, ValidationError):
            return None
    
    def llmdetectlanguage(self, text):
        prompt = f"""Identify the language of this text from the 22 official languages of India:
Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu.

Return ONLY the language name in English.

Text:
{text[:400]}"""
        
        resp = self.callllm(prompt, max_tokens=30)
        lang = resp.strip()
        
        if "marathi" in lang.lower(): lang = "Marathi"
        elif "hindi" in lang.lower(): lang = "Hindi"
        elif "bengali" in lang.lower(): lang = "Bengali"
        
        return {"primary_lang": lang, "confidence": 0.92, "method": "llm"}
    
    def detectlanguage(self, text):
        dev_count = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        total_len = max(len(text), 1)
        dev_ratio = dev_count / total_len
        
        if dev_ratio >= 0.55:
            m_lang, m_conf = MarathiDetector.detect(text)
            if m_lang:
                return {"primary_lang": m_lang, "confidence": round(m_conf, 3), "method": "marathi_first"}
            
            try:
                matches = detect_langs(text)
                for match in matches:
                    if match.lang == 'ne' and match.prob > 0.8:
                        return {"primary_lang": "Nepali", "confidence": round(match.prob, 3), "method": "langdetect"}
                    if match.lang == 'hi' and match.prob > 0.9:
                        return {"primary_lang": "Hindi", "confidence": round(match.prob, 3), "method": "langdetect"}
            except:
                pass
            
            return self.llmdetectlanguage(text)
        
        lang, conf = UnicodeScriptDetector.detect(text)
        if lang and conf >= 0.3:
            
            if lang == "Bengali":
                if '\u09F0' in text or '\u09F1' in text:
                    return {"primary_lang": "Assamese", "confidence": 0.95, "method": "unicode_char"}
                return {"primary_lang": "Bengali", "confidence": round(conf, 3), "method": "unicode"}
            
            if lang == "Urdu":
                try:
                    matches = detect_langs(text)
                    if matches[0].lang == 'ur':
                         return {"primary_lang": "Urdu", "confidence": round(matches[0].prob, 3), "method": "langdetect"}
                    if matches[0].lang == 'sd':
                         return {"primary_lang": "Sindhi", "confidence": round(matches[0].prob, 3), "method": "langdetect"}
                except:
                    pass
                return self.llmdetectlanguage(text)
            
            if lang == "Hindi":
                m_lang, m_conf = MarathiDetector.detect(text)
                if m_lang and m_conf >= 0.30:
                    return {"primary_lang": m_lang, "confidence": round(m_conf, 3), "method": "marathi_markers"}
                return self.llmdetectlanguage(text)

            return {"primary_lang": lang, "confidence": round(conf, 3), "method": "unicode"}
        
        lang, conf = RomanizedHindiDetector.detect(text)
        if lang and conf >= 0.25:
            return {"primary_lang": lang, "confidence": round(conf, 3), "method": "romanized"}
        
        try:
            detected = detect_langs(text)[0]
            lang_map = {'hi': 'Hindi', 'mr': 'Marathi', 'ne': 'Nepali', 'pa': 'Punjabi', 'gu': 'Gujarati', 
                        'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu', 'kn': 'Kannada', 'ml': 'Malayalam', 
                        'or': 'Odia', 'ur': 'Urdu', 'sd': 'Sindhi', 'en': 'English'}
            lang_name = lang_map.get(detected.lang, detected.lang.upper())
            
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
        
        result_model = self.parse_with_pydantic(resp, TranslationResult)
        
        if result_model:
            translated = result_model.translated_text
            conf = result_model.confidence
            if conf == 0.0:
                conf = 0.88
        else:
            resp = resp.strip()
            for prefix in ['{"translated_text": "', '{ "translated_text": "', '"translated_text": ']:
                if resp.startswith(prefix):
                    translated = resp[len(prefix):].rstrip('"}').strip()
                    break
            else:
                translated = resp.strip('"').strip()
            conf = 0.7
        
        return translated, conf
    
    def analyze(self, text):
        prompt = f"""Analyze this text and return ONLY valid JSON — no markdown, no explanations.

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
- domain_ident: Choose ONLY ONE from: [Politics, Crime, Military, Terrorism, Radicalisation, Extremism in J&K, Law and Order, Narcotics, Left Wing Extremism, General]
  - "Politics": ONLY for elections, political parties, government formation, parliament, or policy debates. (Do NOT use for business, product launches, or technology).
  - "Law and Order": Riots, protests, accidents, fires, public safety.
  - "General": Use this for EVERYTHING else, including: Technology (Smartphones, AI), Business, Sports, Entertainment, Lifestyle, and ordinary accidents.
- sentiment: "Positive", "Negative", "Neutral", "Anti-National"
- NER: ONLY proper named entities
  - Organisation: Government bodies, Companies (e.g. Samsung), Agencies (e.g. Police).
  - Product: Specific product names (e.g. Galaxy A07, iPhone 15).
- Summary: One single continuous sentence.

Text:
{text}"""
        
        max_tokens = min(300, 500 + len(text.split()) // 5)
        resp = self.callllm(prompt, max_tokens=max_tokens)
        
        result = self.parse_with_pydantic(resp, AnalysisResult)
        
        if not result:
            result = AnalysisResult(
                domain_ident=["General"],
                domain_confidence=0.3,
                Summary="Analysis failed."
            )
        
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
        
        ner = analysis.NER
        ner_formatted = {
            "Person": self.tostr(ner.Person),
            "Location": self.tostr(ner.Location),
            "Organisation": self.tostr(ner.Organisation),
            "Event": self.tostr(ner.Event),
            "Product": self.tostr(ner.Product)
        }
        
        fc = analysis.Fact_checker
        
        return {
            "Cleaned_content": cleaned,
            "domain_ident": self.tostr(analysis.domain_ident),
            "domain_ident_confidence_score": round(analysis.domain_confidence, 3),
            "lang_det": lang_info["primary_lang"],
            "lang_det_confidence_score": lang_info["confidence"],
            "sentiment": analysis.sentiment,
            "sentiment_confidence_score": round(analysis.sentiment_confidence, 3),
            "NER": ner_formatted,
            "NER_confidence_score": round(analysis.ner_confidence, 3),
            "Event_calender": self.tostr(analysis.Event_calendar),
            "Event_calender_confidence_score": round(analysis.event_calendar_confidence, 3),
            "Country_iden": analysis.Country_iden,
            "Country_iden_confidence_score": round(analysis.country_confidence, 3),
            "Summary": analysis.Summary,
            "Summary_confidence_score": round(analysis.summary_confidence, 3),
            "Fact_checker": self.tostr(fc.relevant_topics),
            "Fact_checker_relevance_rating": fc.relevance_rating,
            "Fact_checker_confidence_score": round(fc.confidence_level, 3),
            "Translation": translated,
            "Translation_confidence_score": round(trans_conf, 3)
        }

if __name__ == "__main__":
    content = '''Samsung Galaxy A07 5G Smartphone : శాంసంగ్ గెలాక్సీ A07 4G స్మార్ట్‌ఫోన్‌ భారత్‌ మార్కెట్‌ లో ఇప్పటికే విడుదల అయింది. త్వరలో ఈ స్మార్ట్‌ఫోన్ 5G వేరియంట్‌ త్వరలో లాంచ్‌ కానుందని తెలుస్తోంది. అయితే గెలాక్సీ A07 5G స్మార్ట్‌ఫోన్‌ భారత్‌ లో విడుదలపై శాంసంగ్‌ ఎటువంటి ప్రకటన చేయలేదు. అయితే తాజాగా ఈ హ్యాండ్‌సెట్ గురించి అనేక వివరాలు లీక్‌ అయ్యాయి. కీలక వివరాలు లీక్‌ : గ్లోబల్‌ మార్కెట్‌ లో గెలాక్సీ A07 5G స్మార్ట్‌ఫోన్ ఇప్పటికే అందుబాటులో ఉంది. దీని ఆధారంగా భారత్‌ వేరియంట్‌ స్పెసిఫికేషన్‌లు, ఫీచర్లను కొంత వరకు అంచనా వేయవచ్చు. ఈ ఫోన్ ఇండియా వేరియంట్‌ 120Hz రీఫ్రెష్‌ రేట్‌ తో 6.7 అంగుళాల IPS LCD డిస్‌ప్లేను కలిగి ఉండే అవకాశం ఉంది.'''
    
    nlp = NLPOrchestrator()
    print("=" * 60)
    print("NLP ORCHESTRATION PIPELINE")
    print("=" * 60)
    
    result = nlp.process(content)
    
    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))