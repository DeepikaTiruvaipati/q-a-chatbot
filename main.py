# main.py — Robust notebook-matching version for deployment
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json, numpy as np
import spacy, re, unicodedata
from dateutil import parser as date_parser
from sentence_transformers import SentenceTransformer

# ---------- Load artifacts ----------
with open("messages_saved.json", "r", encoding="utf-8") as f:
    messages = json.load(f)

with open("idx_by_user.json", "r", encoding="utf-8") as f:
    raw_idx_by_user = json.load(f)   # keys may be normalized or raw depending on how saved

embeddings = np.load("message_embeddings.npy")

# load models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Helpers: normalize ----------
def normalize_text(s):
    if not s:
        return ""
    s = s.strip().lower()
    s = ''.join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Build a normalized idx_by_user mapping (normalized_name -> list[int indices])
idx_by_user = {}
for k, v in raw_idx_by_user.items():
    nk = normalize_text(k)
    # ensure indices are ints
    idxs = [int(x) for x in v] if isinstance(v, (list, tuple)) else []
    idx_by_user[nk] = idxs

canonical_users = sorted(list(idx_by_user.keys()))

# ---------- Simple, stable similarity (no external libs) ----------
def simple_similarity(a, b):
    """Normalized token-level overlap / char prefix heuristic."""
    if not a or not b:
        return 0.0
    # token overlap
    ta = a.split()
    tb = b.split()
    common = len(set(ta) & set(tb))
    score_tokens = common / max(len(set(ta) | set(tb)), 1)
    # char prefix/sequence match (lightweight)
    seq = sum(1 for x, y in zip(a, b) if x == y) / max(len(a), len(b), 1)
    # combined
    return 0.6 * score_tokens + 0.4 * seq

def find_best_user_match_norm(norm_name, threshold=0.35):
    best = None
    best_score = 0.0
    for cu in canonical_users:
        s = simple_similarity(norm_name, cu)
        if s > best_score:
            best_score = s
            best = cu
    if best_score >= threshold:
        return best, best_score
    return None, 0

# ---------- Person extraction + resolution ----------
def extract_persons_from_question(q):
    doc = nlp(q)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

def resolve_person(question):
    # 1) NER persons first
    persons = extract_persons_from_question(question)
    for p in persons:
        npn = normalize_text(p)
        match, score = find_best_user_match_norm(npn)
        if match:
            return match, score
    # 2) fallback: try to match any name-looking substring: extract capitalized sequences
    caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", question)
    for c in caps:
        npn = normalize_text(c)
        match, score = find_best_user_match_norm(npn)
        if match:
            return match, score
    # 3) try whole question normalized
    return find_best_user_match_norm(normalize_text(question))

# ---------- Retrieval (use saved embeddings, compute cosine) ----------
def retrieve_semantic(query, top_k=5, candidate_idxs=None):
    # encode q and normalize
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    if candidate_idxs:
        cand_emb = emb[candidate_idxs]
        sims = (q_emb @ cand_emb.T)[0]
        top_local = sims.argsort()[::-1][:top_k]
        return [candidate_idxs[int(i)] for i in top_local]
    else:
        sims = (q_emb @ emb.T)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [int(i) for i in top_idx]

def retrieve_user_aware(query, top_k=5):
    user_norm, score = resolve_person(query)
    if user_norm:
        candidate_idxs = idx_by_user.get(user_norm, [])
        if candidate_idxs:
            top_idx = retrieve_semantic(query, top_k, candidate_idxs)
            return {"matched_user": user_norm, "results": [(i, messages[i]) for i in top_idx]}
    # fallback global
    top_idx = retrieve_semantic(query, top_k)
    return {"matched_user": None, "results": [(i, messages[i]) for i in top_idx]}

# ---------- Extraction helpers (wider heuristics) ----------
def extract_dates(text):
    doc = nlp(text)
    found = []
    for ent in doc.ents:
        if ent.label_ == "DATE":
            try:
                found.append(date_parser.parse(ent.text, fuzzy=True).date().isoformat())
            except:
                found.append(ent.text)
    # regex fallback: dd/mm/yyyy or month names
    m = re.findall(r"\b(?:\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b[^\.,]{0,15}\d{1,2}?)", text, flags=re.I)
    for mm in m:
        try:
            found.append(date_parser.parse(mm, fuzzy=True).date().isoformat())
        except:
            found.append(mm)
    return list(dict.fromkeys(found))

def extract_numbers(text):
    nums = re.findall(r"\b\d+\b", text)
    return list(dict.fromkeys([int(x) for x in nums]))

def extract_locations(text):
    doc = nlp(text)
    locs = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
    return list(dict.fromkeys(locs))

def extract_restaurants(text):
    doc = nlp(text)
    res = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "FAC")]
    # pattern "at X" or "at the X" capturing capitalized names
    pat = re.findall(r"\bat (?:the )?([A-Z][A-Za-z0-9&' ]{2,50})", text)
    for p in pat:
        res.append(p.strip())
    # pattern "restaurant" nearby nouns: "restaurant X" or "X restaurant"
    nearby = re.findall(r"([A-Z][A-Za-z0-9&' ]{2,40} (?:restaurant|Restaurant))", text)
    for n in nearby:
        res.append(n.strip())
    # drop short or obviously non-names
    res = [r for r in res if len(r) > 2 and not r.isdigit()]
    # unique preserve order
    seen = set()
    out = []
    for r in res:
        if r not in seen:
            seen.add(r); out.append(r)
    return out

# ---------- Intent detection ----------
def detect_intent(question):
    q = question.lower()
    if "when" in q or "date" in q or "time" in q:
        return "when"
    if "how many" in q or "number of" in q:
        return "how_many"
    if "where" in q or "location" in q or "going" in q:
        return "where"
    if "restaurant" in q or "dinner" in q or "eat" in q or "favorite restaurants" in q:
        return "restaurant"
    if "trip" in q or "travel" in q or "flight" in q:
        return "travel"
    return "general"

# ---------- Synthesis (expects retrieved["results"] = [(idx, msg_obj), ...]) ----------
def synthesize_answer_full(question, retrieved):
    intent = detect_intent(question)
    pairs = retrieved.get("results", [])
    texts = [m["message"] for (_, m) in pairs]

    all_dates = []
    all_nums = []
    all_locs = []
    all_rest = []

    for t in texts:
        all_dates.extend(extract_dates(t))
        all_nums.extend(extract_numbers(t))
        all_locs.extend(extract_locations(t))
        all_rest.extend(extract_restaurants(t))

    # dedupe preserving order
    def dedupe(lst):
        seen = set(); out = []
        for x in lst:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    all_dates = dedupe(all_dates)
    all_nums = dedupe(all_nums)
    all_locs = dedupe(all_locs)
    all_rest = dedupe(all_rest)

    if intent == "when" or intent == "travel":
        if all_dates:
            return f"Found date references: {all_dates}"
        if all_locs:
            return f"Travel locations: {all_locs}"
        return "Travel-related messages present but no clear location/date."

    if intent == "how_many":
        if all_nums:
            return f"Found numeric mentions: {all_nums}"
        return "I could not find any relevant quantities."

    if intent == "where":
        if all_locs:
            return f"Locations mentioned: {all_locs}"
        return "No locations found."

    if intent == "restaurant":
        if all_rest:
            return f"Restaurants referenced: {', '.join(all_rest)}"
        # fallback: if no restaurants from top hits, try searching full user messages if a user matched
        matched_user = retrieved.get("matched_user")
        if matched_user:
            # search all messages of that user
            idxs = idx_by_user.get(matched_user, [])
            found = []
            for i in idxs:
                found.extend(extract_restaurants(messages[i]["message"]))
            found = dedupe(found)
            if found:
                return f"Restaurants referenced: {', '.join(found)}"
        return "Restaurant details not clearly mentioned."

    # fallback: return top message snippet
    if texts:
        return texts[0][:400]
    return "No relevant information found."

# ---------- FastAPI ----------
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    if not q.question or not q.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    try:
        retrieved = retrieve_user_aware(q.question, top_k=5)
        answer = synthesize_answer_full(q.question, retrieved)
        return {"answer": answer}
    except Exception as e:
        # return error for debugging in HF logs
        raise HTTPException(status_code=500, detail=str(e))
