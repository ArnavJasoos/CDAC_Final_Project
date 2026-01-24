# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from huggingface_hub import login
import os

import os
from dotenv import load_dotenv
load_dotenv() # Load variables from .env file

# Now you can access the token
hf_token = os.environ.get("HF_TOKEN") 

# Check if the token was loaded correctly
if hf_token:
    print("HF_TOKEN successfully loaded.")
    # Use the token, e.g., for logging into the Hub
    from huggingface_hub import login
    login(token=hf_token) #
else:
    print("HF_TOKEN not found.")



quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=quantization_config,
    device_map="auto", # This helps distribute the model across available devices
)
role='''<system_role>
NLP analyzer: detect lang→translate to English→analyze. Output ONLY valid JSON. Follow schema exactly.
</system_role>

<rules>
1. JSON ONLY, no extras
2. All analysis on final English text
3. Dates: dd/mm/yyyy | Domains: "[D1,D2,D3]" | Sentiment: 1 value
4. Empty: "" strings, [] arrays
5. NO translation justification
</rules>

<schema>
{
  "lang_list": ["English","Hindi","Roman Hindi",...],
  "primary_lang": "dominant language",
  "translated_english_text": "final English for analysis",
  "domain_ident": "[D1,D2,D3]|''",
  "sentiment": "Positive|Negative|Neutral|Anti-National",
  "NER": {
    "Person": "full names only",
    "Location": "cities/places ONLY (no Mt Everest/objects/generic/org-only)",
    "Organisation": "govt/companies/NGOs/boards (New Zealand Cricket Board=Org)",
    "Event": "Event|Loc|Date format",
    "Product": "exact names (AK-47, Heroin, C-4)"
  },
  "Event_calendar": "[Event|dd/mm/yyyy,...]|[]",
  "Country_iden": "Indian|Pakistan|Sri Lanka|Afghanistan|Nepal|Bangladesh|China|Abroad",
  "Fact_checker": {"relevant_topics": [...], "confidence_level": 0.0-1.0, "relevance_rating": "High|Medium|Low"},
  "Summary": "3-4 sentences"
}
</schema>

<tasks>
1. **Lang+Translate**: Detect lang_list. primary_lang=dominant. English→use as-is. Native Indian→translate English. Roman [Lang]→transliterate→translate (show English only).
2. **Domain**: ≤3 ranked: Politics,Crime,Military,Terrorism,Radicalisation,Extremism in J&K,Law and Order,Narcotics,Left Wing Extremism.
3. **NER**: Person=names. Location=geographies only. Organisation=deep think (sports boards=Org). Event=with context. Product=exact names.
4. **Sentiment**: ONE value. Anti-National=sovereignty attack/terrorism support.
5. **Events**: Real events with dd/mm/yyyy. [] if vague.
6. **Country**: Content perspective (India operations=Indian).
7. **Fact_checker**: Relevant topics ≥0.5 confidence. High=≥0.8.
8. **Summary**: 3-4 sentences, 25% length.
</tasks>

<example>
INPUT: "Delhi Police arrested 3 Lashkar-e-Taiba terrorists near Red Fort on 15/03/2025 with AK-47s."

JSON:
{
  "lang_list": ["English"],
  "primary_lang": "English",
  "translated_english_text": "Delhi Police arrested 3 Lashkar-e-Taiba terrorists near Red Fort on 15/03/2025 with AK-47s.",
  "domain_ident": "[Terrorism,Law and Order]",
  "sentiment": "Neutral",
  "NER": {
    "Person": "",
    "Location": "Delhi, Red Fort",
    "Organisation": "Delhi Police, Lashkar-e-Taiba",
    "Event": "Terrorist arrest|Red Fort|15/03/2025",
    "Product": "AK-47"
  },
  "Event_calendar": "[Terrorist arrest|15/03/2025]",
  "Country_iden": "Indian",
  "Fact_checker": {"relevant_topics": ["Terrorism"], "confidence_level": 0.95, "relevance_rating": "High"},
  "Summary": "Delhi Police arrested 3 Lashkar-e-Taiba terrorists near Red Fort on 15 March 2025."
}
</example>

<final>
Analyze input following rules/schema/example. Output ONLY JSON.
</final>
'''

content='''கடந்த பதினைந்து நாட்களாக உலகில் என்ன நடந்தது?

இந்த மாத தொடக்கத்தில் வெனிசுவேலாவில் மேற்கொள்ளப்பட்ட வெற்றிகரமான ராணுவ நடவடிக்கையால் உற்சாகமடைந்த டொனால்ட் டிரம்ப் , கிரீன்லாந்து விவகாரத்தில் தனது ஆக்ரோஷமான பேச்சைத் தொடங்கினார்.

கிரீன்லாந்து மீதான உரிமை கோரல்கள், ராணுவ நடவடிக்கை குறித்த எச்சரிக்கைகள் மற்றும் ஐரோப்பாவின் பாரம்பரிய நட்பு நாடுகளுக்கு எதிரான வர்த்தக வரிகள் என உலகம் தினமும் ஒரு செய்தியை எதிர்கொண்டது.

ஆனால் இப்போது, இவை அனைத்தும் ஒரு புகையைப் போல மறைந்துவிட்டதாகத் தெரிகிறது.

டிரம்பை கையாள்வதில் வல்லவர் என்று கருதப்படும் நேட்டோ பொதுச் செயலாளர் மார்க் ருட்டே, அதிபரின் இந்த ஆபத்தான போக்கைக் கட்டுப்படுத்தியதாகத் தெரிகிறது.

கடந்த வாரம் டென்மார்க் மற்றும் கிரீன்லாந்து வெளியுறவு அமைச்சர்கள் அமெரிக்காவுக்கு மேற்கொண்ட பயணத்தின் போது இதற்கான அடித்தளம் அமைக்கப்பட்டிருக்கலாம்.

அந்தப் பயணத்தின் முடிவில், கிரீன்லாந்தின் எதிர்காலம் குறித்து விவாதிக்க ஒரு "செயற்குழு" அமைக்க ஒப்புக்கொள்ளப்பட்டது.

வடக்கு அட்லாண்டிக் கூட்டணியையே சிதைக்கக்கூடிய ஒரு சிக்கலை ரூட்டே மிக நுட்பமாக கையாண்டுள்ளதாகத் தெரிகிறது.'''
messages = [
    {"role": role, "content": content},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
 tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))