import json
import re
import unicodedata
import ollama
from langdetect import detect_langs
from collections import OrderedDict  # for simple LRU-like cache


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
        if not text or len(text) < 15:
            return None, 0.0
        
        devanagari_count = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        if devanagari_count / max(len(text), 1) < 0.6:
            return None, 0.0
        
        normalized = text.replace("।", " ").replace(",", " ").replace("!", " ").replace("?", " ")
        words = set(cls.ALPHA_RE.sub(" ", normalized).split())
        
        hits = len(words & cls.MARKERS)
        
        if hits < 1:
            return None, 0.0
        
        word_count = max(len(words), 1)
        conf = min(hits / word_count * 3.0, 0.95)
        return ("Marathi", conf) if conf >= 0.25 else (None, 0.0)


class NLPOrchestrator:
    def __init__(self, model_id="llama3.1:8b-instruct-q4_K_M"):
        print(f"Using Ollama model: {model_id}")
        self.model_id = model_id
        
        self.cleanup_trans = str.maketrans({"\u200c": "", "\u200d": "", "\ufeff": ""})
        self.space_re = re.compile(r"\s+")
        self.trailing_comma_re = re.compile(r",\s*([}\]])")
        self.json_block_re = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
        
        # Simple cache for LLM calls (prompt prefix + max_tokens)
        self._llm_cache = OrderedDict()
        self._cache_max_size = 300  # prevent unbounded memory growth
    
    def callllm(self, prompt, max_tokens=800):
        # Cache key: first 400 chars of prompt + max_tokens
        cache_key = f"{max_tokens}:{prompt[:400]}"
        
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        
        resp = ollama.chat(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,
                "num_predict": max_tokens,
                "top_p": 0.9   # added for slightly faster/more focused output
            },
        )
        result = resp["message"]["content"].strip()
        
        # Store in cache
        self._llm_cache[cache_key] = result
        if len(self._llm_cache) > self._cache_max_size:
            self._llm_cache.popitem(last=False)  # remove oldest
        
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
        lang, conf = UnicodeScriptDetector.detect(text)
        if lang and conf >= 0.3:
            if lang == "Hindi":
                m_lang, m_conf = MarathiDetector.detect(text)
                if m_lang and m_conf >= 0.35:
                    return {"primary_lang": m_lang, "confidence": round(m_conf, 3), "method": "marathi_markers"}
                
                if conf < 0.70 or len(text.split()) < 25:
                    return self.llmdetectlanguage(text)
            
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
Provide a confidence score 0.0-1.0 based on translation accuracy: 
0.9+ for very clear and natural, 0.6-0.8 for mostly accurate, <0.6 if ambiguous or difficult phrasing.
Return ONLY: {{"translated_text": "...", "confidence": 0.XX}}

Text: {text}"""
        
        # Reduced max_tokens
        resp = self.callllm(prompt, max_tokens=200)
        parsed = self.parsejson(resp)
        
        if parsed and "translated_text" in parsed:
            translated = str(parsed["translated_text"])
            conf = float(parsed.get("confidence", 0.88))
            if conf == 0.0:
                conf = 0.75
        else:
            translated = resp.strip().strip('"').strip("'")
            conf = 0.7
        
        return translated, conf
    
    def analyze(self, text):
        prompt = f"""Analyze this text and return ONLY valid JSON (no markdown, no extra text):

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

RULES:
- domain_ident: ONLY from [Politics, Crime, Military, Terrorism, Radicalisation, Extremism in J&K, Law and Order, Narcotics, Left Wing Extremism, General]
- Country_iden: Single country only
- Event_calendar: dd/mm/yyyy format, only explicit dates
- sentiment: Positive/Negative/Neutral/Anti-National (Anti-National only for direct threats to India)
- Summary: One continuous sentence, no newlines. Make it a concise, high-level overview in your own words — do NOT copy or closely paraphrase the input text or translation. Focus on the core event, actors, and implication.
- Summary must differ meaningfully from any direct English rendering of the original text — rephrase using different structure and vocabulary.
- All confidence scores: realistic values 0.0-1.0
- NER: Extract only proper named entities. Person: specific individuals (e.g., "Eknath Shinde"), not groups like "farmers". Location: specific places (e.g., "Maharashtra"). Organisation: specific entities (e.g., "Maharashtra Legislative Assembly"). Event: specific named events (e.g., "2026 winter session"). Product: specific named products (leave empty if none).

Text: {text}"""
        
        # Tighter max_tokens cap
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
        
        # Skip translation if confident English
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
    content = '''महाराष्ट्र विधानसभेच्या २०२६ च्या हिवाळी अधिवेशनात शेतकरी कर्जमाफी आणि बेरोजगारीवरून खडाजंगी झाली. मुख्यमंत्री एकनाथ शिंदे यांनी आर्थिक स्थितीवर श्वेतपत्रिका काढण्याचे आश्वासन दिले आहे.'''
    
    nlp = NLPOrchestrator()
    
    print("=" * 60)
    print("NLP ORCHESTRATION PIPELINE")
    print("=" * 60)
    
    result = nlp.process(content)
    
    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))