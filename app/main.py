# Cell 4: FastAPI full app code
import hashlib
import tempfile
import asyncio
import threading
import json
import wave
from typing import List, Optional, Dict, Any

import httpx
from fastapi import (
    FastAPI, HTTPException, UploadFile, File, Form, Query, BackgroundTasks, Request
)
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import RequestValidationError
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from aiocache import cached, Cache
from transformers import MarianMTModel, MarianTokenizer, pipeline
from gtts import gTTS
import vosk
import pymongo
from pywebpush import webpush, WebPushException
from fpdf import FPDF
from datetime import datetime

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
client = pymongo.MongoClient(MONGO_URL)
db = client["moroccotourai"]

VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY", "")
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY", "")
HF_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_CHAT_MODEL = os.environ.get("HF_CHAT_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "/content/vosk-model-small-multilingual-0.4")

app = FastAPI(
    title="MoroccoTour AI Backend - Free, Unlimited, AI/UX Upgraded",
    version="4.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"error": "Invalid input", "details": exc.errors()})

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": "Internal server error", "details": str(exc)})

class PlanTripRequest(BaseModel):
    city: str = Field(..., example="Agadir")
    time: str = Field("afternoon", example="afternoon")
    mood: str = Field("relaxing", example="adventurous")
    budget: str = Field("cheap", example="cheap")

class Place(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    coordinates: Dict[str, float]

class PlanTripResponse(BaseModel):
    city: str
    time: str
    mood: str
    budget: str
    suggestions: List[Place]

class GetPlacesResponse(BaseModel):
    city: str
    type: str
    page: int
    page_size: int
    total: int
    places: List[Place]

class WeatherResponse(BaseModel):
    city: str
    temperature: Optional[float]
    conditions: Optional[int]
    wind_speed: Optional[float]
    sunrise: Optional[str]
    sunset: Optional[str]

class TranslateRequest(BaseModel):
    text: str
    from_lang: str
    to_lang: str

class TranslateResponse(BaseModel):
    translation: str

class EmergencyPlace(BaseModel):
    type: Optional[str]
    name: Optional[str]
    address: Optional[str]
    coordinates: Dict[str, float]

class EmergencyInfoResponse(BaseModel):
    query: str
    emergency_places: List[EmergencyPlace]

class CommunityPostRequest(BaseModel):
    author: str
    tip: str

class CommunityPostResponse(BaseModel):
    success: bool
    tip: Dict[str, Any]

class CommunityPostsResponse(BaseModel):
    posts: List[Dict[str, Any]]

class VoiceToTextResponse(BaseModel):
    text: str

class PushSubscription(BaseModel):
    endpoint: str
    keys: Dict[str, str]

class PushNotifyRequest(BaseModel):
    subscription: PushSubscription
    message: str

class ChatRequest(BaseModel):
    history: List[Dict[str, str]]  # [{"role": "user"/"assistant", "content": "..."}]
    user_message: str
    user_id: Optional[str] = None
    use_ollama: Optional[bool] = False

class ChatResponse(BaseModel):
    reply: str

class UserPreferences(BaseModel):
    user_id: str
    language: Optional[str]
    favorite_cities: Optional[List[str]]
    interests: Optional[List[str]]

class UserHistory(BaseModel):
    user_id: str
    trips: Optional[List[Dict[str, Any]]]  # [{city, date, activities, notes}]

class BadgeRequest(BaseModel):
    user_id: str
    badge: str

class BadgeResponse(BaseModel):
    user_id: str
    badges: List[str]

class DailyChallengeResponse(BaseModel):
    challenge: str
    completed: bool

_TRANSLATOR_MODELS = {
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
    ("fr", "ar"): "Helsinki-NLP/opus-mt-fr-ar",
    ("ar", "fr"): "Helsinki-NLP/opus-mt-ar-fr",
}
_TRANSLATOR_PIPELINES = {}
_MODEL_LOCK = threading.Lock()

def get_translator_pipeline(from_lang: str, to_lang: str):
    pair = (from_lang, to_lang)
    if pair not in _TRANSLATOR_MODELS:
        return None
    if pair not in _TRANSLATOR_PIPELINES:
        with _MODEL_LOCK:
            if pair in _TRANSLATOR_PIPELINES:
                return _TRANSLATOR_PIPELINES[pair]
            model_name = _TRANSLATOR_MODELS[pair]
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            translator = pipeline("translation", model=model, tokenizer=tokenizer)
            _TRANSLATOR_PIPELINES[pair] = translator
    return _TRANSLATOR_PIPELINES[pair]

_CLASSIFIER = None
_CLASSIFIER_LOCK = threading.Lock()
def get_classifier():
    global _CLASSIFIER
    if _CLASSIFIER is None:
        with _CLASSIFIER_LOCK:
            if _CLASSIFIER is None:
                _CLASSIFIER = pipeline("text-classification", model="unitary/toxic-bert")
    return _CLASSIFIER

@app.post("/plan-trip", response_model=PlanTripResponse)
@limiter.limit("10/minute")
@cached(ttl=600, cache=Cache.MEMORY)
async def plan_trip(request: Request, data: PlanTripRequest):
    mood_tags = {
        "relaxing": ["park", "spa", "garden", "viewpoint"],
        "adventurous": ["hiking", "climbing", "water_park", "beach", "adventure"],
        "cultural": ["museum", "theatre", "art_gallery", "historic", "mosque"],
        "romantic": ["cafe", "restaurant", "viewpoint", "garden"]
    }
    time_tags = {
        "morning": ["cafe", "park", "market", "museum"],
        "afternoon": ["beach", "market", "museum", "park", "restaurant"],
        "evening": ["restaurant", "theatre", "cafe", "viewpoint"]
    }
    tags = set(mood_tags.get(data.mood, [])) | set(time_tags.get(data.time, []))
    overpass_url = "https://overpass-api.de/api/interpreter"
    search_tags = "|".join(tags)
    query = f"""
    [out:json][timeout:25];
    area["name"="{data.city}"][admin_level=8]->.searchArea;
    (
      node["amenity"~"{search_tags}"](area.searchArea);
      node["tourism"~"{search_tags}"](area.searchArea);
      node["leisure"~"{search_tags}"](area.searchArea);
    );
    out body 10;
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(overpass_url, data={"data": query})
        pois = resp.json().get("elements", [])
    suggestions = []
    for poi in pois:
        name = poi.get("tags", {}).get("name")
        lat, lon = poi.get("lat"), poi.get("lon")
        typ = poi.get("tags", {}).get("amenity") or poi.get("tags", {}).get("tourism") or poi.get("tags", {}).get("leisure") or "unknown"
        suggestions.append({"name": name, "type": typ, "coordinates": {"lat": lat, "lon": lon}})
        if len(suggestions) >= 5: break
    return PlanTripResponse(city=data.city, time=data.time, mood=data.mood, budget=data.budget, suggestions=suggestions)

@app.get("/weather", response_model=WeatherResponse)
@cached(ttl=600, cache=Cache.MEMORY)
async def weather(city: str = Query(
    ...,
    examples={
        "default": {
            "summary": "A city in Morocco",
            "description": "Example city name",
            "value": "Agadir"
        }
    }
)):
    async with httpx.AsyncClient(timeout=20.0) as client:
        url = f"https://nominatim.openstreetmap.org/search"
        georesp = await client.get(url, params={"q": city, "country": "Morocco", "format": "json"})
        results = georesp.json()
        if not results:
            raise HTTPException(404, "City not found")
        lat, lon = float(results[0]["lat"]), float(results[0]["lon"])
        meteo_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true&daily=sunrise,sunset"
            f"&timezone=auto"
        )
        meteo_resp = await client.get(meteo_url)
        w = meteo_resp.json()
        cw = w.get("current_weather", {})
        sunrise = w.get("daily", {}).get("sunrise", [""])[0]
        sunset = w.get("daily", {}).get("sunset", [""])[0]
        return WeatherResponse(
            city=city,
            temperature=cw.get("temperature"),
            conditions=cw.get("weathercode"),
            wind_speed=cw.get("windspeed"),
            sunrise=sunrise,
            sunset=sunset
        )

@app.post("/speak")
@limiter.limit("20/minute")
async def speak(
    request: Request,
    text: str = Form(...),
    lang: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    supported_langs = {"en", "fr", "ar"}
    if lang not in supported_langs:
        raise HTTPException(400, f"Unsupported language '{lang}'")
    cache_dir = os.path.join(tempfile.gettempdir(), "moroccotour_tts_cache")
    os.makedirs(cache_dir, exist_ok=True)
    fname = hashlib.sha256(f"{lang}-{text}".encode("utf-8")).hexdigest() + ".mp3"
    fpath = os.path.join(cache_dir, fname)
    def generate_tts():
        tts = gTTS(text, lang=lang)
        tts.save(fpath)
    if not os.path.exists(fpath):
        if background_tasks:
            background_tasks.add_task(generate_tts)
        else:
            generate_tts()
        for _ in range(30):
            if os.path.exists(fpath):
                break
            asyncio.sleep(0.1)
    audio_stream = open(fpath, "rb")
    return StreamingResponse(audio_stream, media_type="audio/mpeg")

@app.post("/translate", response_model=TranslateResponse)
@limiter.limit("30/minute")
async def translate(request: Request, data: TranslateRequest):
    supported_langs = {"en", "fr", "ar"}
    if data.from_lang not in supported_langs or data.to_lang not in supported_langs:
        raise HTTPException(400, "Languages supported: English (en), French (fr), Arabic (ar).")
    if data.from_lang == data.to_lang:
        return TranslateResponse(translation=data.text)
    pipeline_translator = get_translator_pipeline(data.from_lang, data.to_lang)
    if not pipeline_translator:
        raise HTTPException(400, f"Translation from {data.from_lang} to {data.to_lang} not supported.")
    try:
        result = pipeline_translator(data.text)
        translation = result[0]["translation_text"]
        return TranslateResponse(translation=translation)
    except Exception as e:
        raise HTTPException(500, f"Translation failed: {str(e)}")

@app.get("/get-places", response_model=GetPlacesResponse)
@cached(ttl=600, cache=Cache.MEMORY)
async def get_places(
    city: str = Query(...),
    type: str = Query(..., description="POI type"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
):
    osm_type_map = {
        "restaurant": ('amenity', 'restaurant'),
        "cafe": ('amenity', 'cafe'),
        "beach": ('natural', 'beach'),
        "mosque": ('amenity', 'place_of_worship'),
        "museum": ('tourism', 'museum'),
        "hotel": ('tourism', 'hotel'),
        "park": ('leisure', 'park'),
        "market": ('amenity', 'marketplace'),
        "atm": ('amenity', 'atm'),
        "hospital": ('amenity', 'hospital'),
        "embassy": ('amenity', 'embassy'),
        "police": ('amenity', 'police'),
    }
    tag_key, tag_val = osm_type_map.get(type.lower(), (None, None))
    if not tag_key:
        raise HTTPException(400, f"POI type '{type}' not supported.")
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    area["name"="{city}"][admin_level=8]->.searchArea;
    (
      node["{tag_key}"="{tag_val}"](area.searchArea);
      way["{tag_key}"="{tag_val}"](area.searchArea);
      relation["{tag_key}"="{tag_val}"](area.searchArea);
    );
    out center 100;
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(overpass_url, data={"data": query})
        results = resp.json().get("elements", [])
    places = []
    for el in results:
        tags = el.get("tags", {})
        name = tags.get("name")
        if "lat" in el and "lon" in el:
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        places.append({"name": name, "type": tag_val, "coordinates": {"lat": lat, "lon": lon}})
    total = len(places)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = places[start:end]
    return GetPlacesResponse(city=city, type=type, page=page, page_size=page_size, total=total, places=paginated)

@app.get("/emergency-info", response_model=EmergencyInfoResponse)
@cached(ttl=600, cache=Cache.MEMORY)
async def emergency_info(
    city: Optional[str] = Query(None, description="City name (preferred)"),
    lat: Optional[float] = Query(None, description="Latitude (optional, for GPS search)"),
    lon: Optional[float] = Query(None, description="Longitude (optional, for GPS search)")
):
    overpass_url = "https://overpass-api.de/api/interpreter"
    if city:
        area_query = f'area["name"="{city}"][admin_level=8]->.searchArea;'
        bbox_query = "(area.searchArea);"
    elif lat is not None and lon is not None:
        d = 0.045
        bbox_query = f"({lat-d},{lon-d},{lat+d},{lon+d});"
        area_query = ""
    else:
        raise HTTPException(400, "Provide city or lat/lon")
    query = f"""
    [out:json][timeout:25];
    {area_query}
    (
      node["amenity"="hospital"]{bbox_query};
      node["amenity"="embassy"]{bbox_query};
      node["amenity"="police"]{bbox_query};
      way["amenity"="hospital"]{bbox_query};
      way["amenity"="embassy"]{bbox_query};
      way["amenity"="police"]{bbox_query};
    );
    out center 20;
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(overpass_url, data={"data": query})
        results = resp.json().get("elements", [])
    pois = []
    for el in results:
        tags = el.get("tags", {})
        kind = tags.get("amenity", "unknown")
        name = tags.get("name", None)
        address = tags.get("address", None) or tags.get("addr:full", None) or tags.get("addr:street", None)
        if "lat" in el and "lon" in el:
            lat_, lon_ = el["lat"], el["lon"]
        elif "center" in el:
            lat_, lon_ = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        pois.append({"type": kind, "name": name, "address": address, "coordinates": {"lat": lat_, "lon": lon_}})
        if len(pois) >= 20:
            break
    query_repr = city if city else f"{lat},{lon}"
    return EmergencyInfoResponse(query=query_repr, emergency_places=pois)

@app.post("/voice-to-text", response_model=VoiceToTextResponse)
@limiter.limit("10/minute")
async def voice_to_text(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, mono preferred, max ~1min)")
):
    if not os.path.exists(VOSK_MODEL_PATH):
        raise HTTPException(500, f"Vosk model not found. Please download 'vosk-model-small-multilingual-0.4' and set VOSK_MODEL_PATH.")
    try:
        model = vosk.Model(VOSK_MODEL_PATH)
    except Exception as e:
        raise HTTPException(500, f"Could not load Vosk model: {str(e)}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        wav_path = tmp.name
    try:
        wf = wave.open(wav_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [16000, 8000]:
            raise HTTPException(400, "Audio must be mono WAV, 16kHz or 8kHz, 16-bit PCM.")
        rec = vosk.KaldiRecognizer(model, wf.getframerate())
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part = rec.Result()
                results.append(part)
        results.append(rec.FinalResult())
        texts = [json.loads(r).get("text", "") for r in results if r]
        transcript = " ".join(texts).strip()
        wf.close()
    except Exception as e:
        raise HTTPException(500, f"Audio transcription failed: {str(e)}")
    finally:
        os.unlink(wav_path)
    return VoiceToTextResponse(text=transcript)

@app.post("/community-posts", response_model=CommunityPostResponse)
@limiter.limit("10/minute")
async def post_community_tip(
    request: Request,
    author: str = Form(..., description="Username or anonymous"),
    tip: str = Form(..., description="Travel tip or story")
):
    classifier = get_classifier()
    result = classifier(tip)[0]
    label = result["label"]
    score = result["score"]
    is_toxic = (label == "toxic" and score > 0.5)
    if is_toxic:
        raise HTTPException(400, "Tip detected as toxic or profane and was rejected.")
    tip_obj = {"author": author, "tip": tip}
    db["community_posts"].insert_one(tip_obj)
    return CommunityPostResponse(success=True, tip=tip_obj)

@app.get("/community-posts", response_model=CommunityPostsResponse)
async def get_community_posts():
    posts = list(db["community_posts"].find({}, {"_id": 0}).sort("$natural", -1).limit(100))
    return CommunityPostsResponse(posts=posts)

@app.post("/subscribe")
async def subscribe_push(subscription: PushSubscription):
    db["push_subscriptions"].update_one(
        {"endpoint": subscription.endpoint},
        {"$set": subscription.dict()},
        upsert=True
    )
    return {"success": True}

@app.post("/notify")
async def notify_push(req: PushNotifyRequest):
    try:
        webpush(
            subscription_info=req.subscription.dict(),
            data=req.message,
            vapid_private_key=VAPID_PRIVATE_KEY,
            vapid_claims={"sub": "mailto:admin@moroccotour.com"}
        )
        return {"success": True}
    except WebPushException as ex:
        return JSONResponse({"error": str(ex)}, status_code=400)

@app.post("/analytics/event")
async def track_event(event_name: str = Form(...), props: Optional[str] = Form(None)):
    return {"success": True}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_llm(data: ChatRequest):
    if not data.use_ollama:
        if not HF_TOKEN:
            raise HTTPException(500, "HF_API_TOKEN not set")
        url = f"https://api-inference.huggingface.co/models/{HF_CHAT_MODEL}"
        messages = data.history + [{"role": "user", "content": data.user_message}]
        prompt = ""
        for m in messages:
            prompt += f"{m['role'].capitalize()}: {m['content']}\n"
        prompt += "Assistant:"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256}}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code != 200:
                raise HTTPException(500, f"LLM error: {r.text}")
            result = r.json()
            output = (
                result[0]["generated_text"][len(prompt):]
                if isinstance(result, list) and "generated_text" in result[0] else str(result)
            )
        return ChatResponse(reply=output.strip())
    else:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": data.history + [{"role": "user", "content": data.user_message}]
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(OLLAMA_URL, json=payload)
            result = r.json()
            output = result.get("message", {}).get("content") or result.get("response") or ""
        return ChatResponse(reply=output.strip())

@app.post("/user/preferences")
async def save_user_prefs(prefs: UserPreferences):
    db["user_prefs"].update_one({"user_id": prefs.user_id}, {"$set": prefs.dict()}, upsert=True)
    return {"success": True}

@app.get("/user/preferences")
async def get_user_prefs(user_id: str):
    prefs = db["user_prefs"].find_one({"user_id": user_id}, {"_id": 0})
    return prefs or {}

@app.post("/user/history")
async def save_user_history(hist: UserHistory):
    db["user_history"].update_one({"user_id": hist.user_id}, {"$set": hist.dict()}, upsert=True)
    return {"success": True}

@app.get("/user/history")
async def get_user_history(user_id: str):
    hist = db["user_history"].find_one({"user_id": user_id}, {"_id": 0})
    return hist or {}

@app.post("/user/badges", response_model=BadgeResponse)
async def award_badge(req: BadgeRequest):
    db["user_badges"].update_one(
        {"user_id": req.user_id},
        {"$addToSet": {"badges": req.badge}},
        upsert=True
    )
    badges_doc = db["user_badges"].find_one({"user_id": req.user_id}, {"_id": 0})
    return BadgeResponse(user_id=req.user_id, badges=badges_doc.get("badges", []))

@app.get("/user/badges", response_model=BadgeResponse)
async def get_badges(user_id: str):
    badges_doc = db["user_badges"].find_one({"user_id": user_id}, {"_id": 0})
    return BadgeResponse(user_id=user_id, badges=badges_doc.get("badges", []))

DAILY_CHALLENGES = [
    "Try a dish you've never eaten before.",
    "Visit a local market and buy something unique.",
    "Take a photo in front of a famous landmark.",
    "Talk to a local and learn a new phrase.",
    "Share a tip in the community board."
]

@app.get("/challenge/daily", response_model=DailyChallengeResponse)
async def daily_challenge(user_id: str):
    day = datetime.utcnow().date().toordinal() % len(DAILY_CHALLENGES)
    challenge = DAILY_CHALLENGES[day]
    doc = db["user_challenge"].find_one({"user_id": user_id}, {"_id": 0, "date": 1})
    completed = bool(doc and doc.get("date") == datetime.utcnow().strftime("%Y-%m-%d"))
    return DailyChallengeResponse(challenge=challenge, completed=completed)

@app.post("/challenge/daily/complete")
async def complete_daily_challenge(user_id: str):
    db["user_challenge"].update_one(
        {"user_id": user_id},
        {"$set": {"date": datetime.utcnow().strftime("%Y-%m-%d")}},
        upsert=True
    )
    return {"success": True}

@app.post("/trip/export/pdf")
async def export_trip_pdf(user_id: str):
    trip = db["user_history"].find_one({"user_id": user_id}, {"_id": 0})
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Your Morocco Trip!", ln=True, align='C')
    if trip and trip.get("trips"):
        for t in trip["trips"]:
            pdf.ln(8)
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, f"City: {t.get('city', '')} - Date: {t.get('date', '')}", ln=True)
            pdf.multi_cell(0, 8, f"Activities: {', '.join(t.get('activities', [])) if t.get('activities') else ''}")
            if t.get("notes"):
                pdf.multi_cell(0, 8, f"Notes: {t['notes']}")
    out_path = f"/tmp/trip_{user_id}.pdf"
    pdf.output(out_path)
    return FileResponse(out_path, filename="my_morocco_trip.pdf")
