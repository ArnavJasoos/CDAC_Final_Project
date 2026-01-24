from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import os
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
hf_token = os.environ.get("HF_TOKEN") 

if hf_token:
    login(token=hf_token)
else:
    print("HF_TOKEN not found.")

# 2. Setup for CPU (Remove BitsAndBytes for CPU execution)
model_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Load Model with Accelerate on CPU
# We use torch_dtype=torch.float32 because CPUs usually handle float32 best, 
# though some modern CPUs support bfloat16.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",         # Explicitly set to CPU
    torch_dtype=torch.float32, # Standard for CPU
    low_cpu_mem_usage=True    # Accelerate optimization
)
role='''You are an NLP analysis assistant. Given the input text, perform these tasks sequentially:

1. **Language Detection & Cleaned Content:** Detect the text's language. If it is Romanized (e.g. Hindi in Latin script), label it as "Roman [Language]" and transliterate the core content to the native script. Put this transliterated or cleaned text in `Cleaned_content`. If not Romanized, just output the original text in `Cleaned_content` and set `lang_den` to the language name (e.g. "English").

2. **Domain Identification:** Identify up to 3 relevant domains from [Politics, Crime, Military, Terrorism, Radicalisation, Extremism in J&K, Law and Order, Narcotics, Left Wing Extremism]. Rank them by relevance and output as a JSON list in `domain_ident`, e.g. `["Politics", "Law and Order", "Crime"]`. If none match, output `"General"`.

3. **Named Entities (NER):** Extract unique entities of type Person, Location, Organisation, Event, and Product. For each category, list the entities (no duplicates) as a comma-separated string under the corresponding `NER` subfield (e.g. `"Person": "Alice,Bob"`). Use an empty string if no entity is found.

4. **Sentiment Analysis:** Classify the sentiment of the text as exactly one of `"Positive"`, `"Negative"`, `"Neutral"`, or `"Anti-National"` (from an Indian perspective). Output this single tag in `sentiment`.

5. **Event & Date Extraction:** Identify any events mentioned and their dates (format `DD/MM/YYYY`). If events are gatherings or protests, include any mentioned persons/organisations involved. Place events in `Event_calender` as a JSON array of objects, each with keys `"Event"` (description), `"Date"`, and `"Participants"` (comma-separated list). Example: `[{"Event":"Protest","Date":"15/08/2023","Participants":"Alice,Police Dept"}]`. If no events, output an empty array (`[]`).

6. **Country Identification:** Determine the country context: output `"Indian"` if related to India, or the appropriate neighboring country (Pakistan, Sri Lanka, Afghanistan, Nepal, Bangladesh, China) if applicable; otherwise use `"Abroad"`. Place this in `Country_iden`.

7. **Relevancy Check:** (Internal use) Note relevance to topics Narcotics, Extremism in J&K, Terrorism, Radicalisation, Law and Order, Left Wing Extremism. *Do not add extra output fields for this; it may influence domain identification but is not directly output.*

8. **Translation to English:** If the content language is not English **and** the domain is not `"General"`, translate the content into English. Put the translated text in `Translation`. If the content is already English or the domain is `"General"`, set `Translation` to `null`. (You may optionally prepend the translation with a brief justification or confidence, since no separate field is provided.)

9. **Summary:** Summarize the content in 3–4 sentences (about 25% of original length), focusing on key information. Place this in `Summary`.

**Output:** Only produce a JSON object with exactly these keys: 
`Cleaned_content`, `domain_ident`, `lang_den`, `sentiment`, `NER` (with subkeys `Person`, `Location`, `Organisation`, `Event`, `Product`), `Event_calender`, `Country_iden`, `Summary`, `Fact_checker`, and `Translation`.  
Use `null`, empty string, or empty array if no relevant data exists for a field. **Do not include any additional keys or commentary.**'''

content='''கடந்த பதினைந்து நாட்களாக உலகில் என்ன நடந்தது?

இந்த மாத தொடக்கத்தில் வெனிசுவேலாவில் மேற்கொள்ளப்பட்ட வெற்றிகரமான ராணுவ நடவடிக்கையால் உற்சாகமடைந்த டொனால்ட் டிரம்ப் , கிரீன்லாந்து விவகாரத்தில் தனது ஆக்ரோஷமான பேச்சைத் தொடங்கினார்.

கிரீன்லாந்து மீதான உரிமை கோரல்கள், ராணுவ நடவடிக்கை குறித்த எச்சரிக்கைகள் மற்றும் ஐரோப்பாவின் பாரம்பரிய நட்பு நாடுகளுக்கு எதிரான வர்த்தக வரிகள் என உலகம் தினமும் ஒரு செய்தியை எதிர்கொண்டது.

ஆனால் இப்போது, இவை அனைத்தும் ஒரு புகையைப் போல மறைந்துவிட்டதாகத் தெரிகிறது.

டிரம்பை கையாள்வதில் வல்லவர் என்று கருதப்படும் நேட்டோ பொதுச் செயலாளர் மார்க் ருட்டே, அதிபரின் இந்த ஆபத்தான போக்கைக் கட்டுப்படுத்தியதாகத் தெரிகிறது.

கடந்த வாரம் டென்மார்க் மற்றும் கிரீன்லாந்து வெளியுறவு அமைச்சர்கள் அமெரிக்காவுக்கு மேற்கொண்ட பயணத்தின் போது இதற்கான அடித்தளம் அமைக்கப்பட்டிருக்கலாம்.

அந்தப் பயணத்தின் முடிவில், கிரீன்லாந்தின் எதிர்காலம் குறித்து விவாதிக்க ஒரு "செயற்குழு" அமைக்க ஒப்புக்கொள்ளப்பட்டது.

வடக்கு அட்லாண்டிக் கூட்டணியையே சிதைக்கக்கூடிய ஒரு சிக்கலை ரூட்டே மிக நுட்பமாக கையாண்டுள்ளதாகத் தெரிகிறது.'''
# 4. Prepare Input
messages = [
    {"role": role, "content": content},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to("cpu") # Ensure inputs are on CPU

# 5. Generate
outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))