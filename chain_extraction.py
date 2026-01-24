import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import gc

class ChainExtractor:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        print(f"Loading model {model_id} in 4-bit to save VRAM...")
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=self.quantization_config,
            device_map="auto",
        )
        
    def _generate(self, system_prompt, user_input, max_tokens=512):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Clean up cache
        del inputs
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return response.strip()

    def step_1_classify(self, text):
        prompt = """Detect language and basic metadata. Output ONLY JSON:
        {"lang_detected": "...", "domain": ["..."], "sentiment": "...", "country": "..."}
        Domains: [Politics, Crime, Military, Terrorism, Radicalisation, Extremism in J&K, Law and Order, Narcotics, Left Wing Extremism].
        Sentiments: [Positive, Negative, Neutral, Anti-National]."""
        response = self._generate(prompt, text, max_tokens=100)
        try:
            return json.loads(response)
        except:
            # Fallback if model fails JSON
            return {"lang_detected": "Unknown", "domain": [], "sentiment": "Neutral", "country": "Abroad"}

    def step_2_translate(self, text, lang):
        if lang.lower() == "english":
            return text
        prompt = f"Translate the following {lang} text to plain English. Output ONLY the translation."
        return self._generate(prompt, text, max_tokens=1024)

    def step_3_ner(self, english_text):
        prompt = """Extract NER entities. Output ONLY JSON:
        {"Person": "...", "Location": "...", "Organisation": "...", "Event": "...", "Product": "..."}
        Rules: Person (full names), Location (geography only), Organisation (deep think), Product (vessels/weapons/drugs)."""
        response = self._generate(prompt, english_text, max_tokens=300)
        try:
            return json.loads(response)
        except:
            return {"Person": "", "Location": "", "Organisation": "", "Event": "", "Product": ""}

    def step_4_summarize(self, english_text):
        prompt = "Summarize the text in 3-4 sentences (key who/what/where/impact). Output ONLY the summary."
        return self._generate(prompt, english_text, max_tokens=400)

    def process(self, text):
        # Master method to orchestrate
        meta = self.step_1_classify(text)
        translated = self.step_2_translate(text, meta['lang_detected'])
        ner = self.step_3_ner(translated)
        summary = self.step_4_summarize(translated)
        
        final_json = {
            "Cleaned_content": text,
            "domain_ident": meta.get("domain", []),
            "lang_detected": meta.get("lang_detected", ""),
            "sentiment": meta.get("sentiment", ""),
            "NER": ner,
            "Event_calender": "", # Placeholder or can be extracted in NER step
            "Country_iden": meta.get("country", ""),
            "Summary": summary,
            "Fact_checker": "", # Can add another step if needed
            "Translation": translated if meta['lang_detected'].lower() != "english" else ""
        }
        return final_json

if __name__ == "__main__":
    # Example usage
    extractor = ChainExtractor()
    sample_text = "Delhi Police arrested 3 Lashkar-e-Taiba terrorists near Red Fort on 15/03/2025."
    result = extractor.process(sample_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
