import uvicorn
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import Config
from schema import (
    ProductCatalog,
    VoiceTranscriptionResponse,
    ErrorResponse,
    PricingInput,
    PricingEstimateResponse,
    MarketplaceSearchQuery,
    MarketplaceSearchResponse,
    MarketIntelligenceRequest,
    MarketIntelligenceResponse,
    SchemeMatchingRequest,
    SchemeMatchingResponse,
)
from service import CatalogAIService
from voice import VoiceService
from pricing import PricingEngine
from marketplace import MarketplaceEngine
from market_intelligence import MarketIntelligenceEngine
from scheme_matching import SchemeMatchingEngine
from exceptions import (
    CatalogAIException,
    MissingImageError,
    InvalidImageError,
    MissingAudioError,
    InvalidAudioError,
    UnsupportedAudioFormatError,
    EmptyTranscriptionError,
    MissingAPIKeyError,
    AIServiceError,
    InvalidAIResponseError,
    PricingError,
    MissingPricingInputError,
    InvalidPricingInputError,
    PricingDatasetError,
    MarketplaceError,
    InvalidMarketplaceQueryError,
    MarketIntelligenceError,
    InvalidMarketIntelligenceRequestError,
    SchemeMatchingError,
    InvalidSchemeRequestError,
)

app = FastAPI(
    title="Smart Artisan Platform AI & Marketplace Microservice",
    description="AI-driven product cataloging, multilingual speech-to-text, fair pricing, buyer marketplace, market intelligence, and government scheme matching engine.",
    version="1.5.0",
)





# Enable CORS for frontend/backend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(CatalogAIException)
async def catalog_exception_handler(request, exc: CatalogAIException):
    """Global exception handler mapping domain errors to structured HTTP JSON responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.message,
            status_code=exc.status_code
        ).model_dump()
    )


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "Smart Catalog & Multilingual Voice AI", "version": "1.1.0"}


@app.post(
    "/api/v1/voice/transcribe",
    response_model=VoiceTranscriptionResponse,
    responses={
        200: {"description": "Voice audio successfully transcribed."},
        400: {"model": ErrorResponse, "description": "Missing, invalid, or unsupported audio format."},
        401: {"model": ErrorResponse, "description": "Missing API key."},
        422: {"model": ErrorResponse, "description": "Empty or inaudible speech transcript."},
        502: {"model": ErrorResponse, "description": "Speech recognition service failure."}
    },
    tags=["Voice AI"]
)
async def transcribe_artisan_voice(
    audio: UploadFile = File(..., description="Artisan voice audio file (WAV, MP3, M4A, OGG, FLAC, AAC)."),
    language_hint: Optional[str] = Form(None, description="Optional expected language code hint (e.g., kn-IN, hi-IN, ta-IN).")
):
    """
    Standalone Speech-to-Text endpoint for transcribing artisan voice input in Indian languages.
    """
    if not audio or not audio.filename:
        raise MissingAudioError("Voice audio file upload is required.")

    try:
        audio_bytes = await audio.read()
    except Exception as e:
        raise InvalidAudioError(f"Could not read uploaded audio file: {str(e)}")

    if not audio_bytes or len(audio_bytes) == 0:
        raise MissingAudioError("Uploaded voice audio file is empty.")

    mime_type = audio.content_type or "audio/wav"

    voice_service = VoiceService()
    response = voice_service.transcribe_audio(
        audio_bytes=audio_bytes,
        mime_type=mime_type,
        language_hint=language_hint
    )

    return response


@app.post(
    "/api/v1/catalog/generate",
    response_model=ProductCatalog,
    responses={
        200: {"description": "Structured product catalog successfully generated."},
        400: {"model": ErrorResponse, "description": "Missing or invalid image upload."},
        401: {"model": ErrorResponse, "description": "Missing or invalid API key configuration."},
        502: {"model": ErrorResponse, "description": "AI service or response parsing failure."}
    },
    tags=["Catalog AI"]
)
async def generate_product_catalog(
    image: UploadFile = File(..., description="Product image file (JPEG, PNG, WEBP)."),
    artisan_description: Optional[str] = Form(None, description="Optional text description from artisan."),
    artisan_voice: Optional[UploadFile] = File(None, description="Optional voice audio file from artisan."),
    language_hint: Optional[str] = Form(None, description="Optional language hint for voice audio (e.g. kn-IN).")
):
    """
    Generates a structured product catalog from an image, optional text description, and optional voice audio description.
    
    Priority/Combination Rule:
    - If both text description and voice audio are provided:
      Transcribes voice audio, preserves both inputs, and combines them clearly.
    - If only text is provided:
      Uses text description directly.
    - If only voice audio is provided:
      Transcribes voice audio and uses transcript as description.
    """
    if not image or not image.filename:
        raise MissingImageError("Product image file must be uploaded.")

    try:
        image_bytes = await image.read()
    except Exception as e:
        raise InvalidImageError(f"Could not read uploaded image file: {str(e)}")

    if not image_bytes:
        raise MissingImageError("Uploaded product image file is empty.")

    audio_bytes = None
    audio_mime_type = None

    if artisan_voice and artisan_voice.filename:
        try:
            audio_bytes = await artisan_voice.read()
            audio_mime_type = artisan_voice.content_type or "audio/wav"
        except Exception as e:
            raise InvalidAudioError(f"Could not read uploaded voice audio file: {str(e)}")

    service = CatalogAIService()
    catalog = service.generate_catalog_combined(
        image_bytes=image_bytes,
        artisan_description=artisan_description,
        audio_bytes=audio_bytes,
        audio_mime_type=audio_mime_type,
        language_hint=language_hint
    )

    return catalog


@app.post(
    "/api/v1/pricing/estimate",
    response_model=PricingEstimateResponse,
    responses={
        200: {"description": "Smart pricing estimate successfully generated."},
        400: {"model": ErrorResponse, "description": "Missing or invalid pricing input parameters."},
        401: {"model": ErrorResponse, "description": "Missing or invalid API key configuration."},
        500: {"model": ErrorResponse, "description": "Marketplace benchmark dataset error."},
        502: {"model": ErrorResponse, "description": "AI service or response parsing failure."}
    },
    tags=["Pricing Intelligence"]
)
async def estimate_product_pricing(pricing_input: PricingInput):
    """
    Generates a data-backed selling price recommendation, price range (floor, recommended, max),
    itemized cost breakdown, and market pricing rationale.
    """
    if not pricing_input or not pricing_input.catalog:
        raise MissingPricingInputError("Pricing input must contain a valid ProductCatalog object.")

    pricing_engine = PricingEngine()
    response = pricing_engine.estimate_price(pricing_input)
    return response


@app.get("/api/v1/marketplace/health", tags=["Marketplace"])
async def marketplace_health_check():
    """Marketplace health check endpoint."""
    return {"status": "ok", "service": "Buyer Marketplace & Artisan Matching Engine", "version": "1.3.0"}


@app.post(
    "/api/v1/marketplace/search",
    response_model=MarketplaceSearchResponse,
    responses={
        200: {"description": "Marketplace search successfully executed."},
        400: {"model": ErrorResponse, "description": "Invalid search parameters."},
        500: {"model": ErrorResponse, "description": "Marketplace dataset error."}
    },
    tags=["Marketplace"]
)
async def search_marketplace_products(search_query: MarketplaceSearchQuery):
    """
    Search artisan product catalog with multi-criteria weighted scoring
    (category, keyword, craft type, location, product similarity, price compatibility).
    """
    engine = MarketplaceEngine()
    response = engine.search_products(search_query)
    return response


@app.get("/api/v1/market-intelligence/health", tags=["Market Intelligence"])
async def market_intelligence_health_check():
    """Market intelligence health check endpoint."""
    return {"status": "ok", "service": "Market Intelligence & Trend Detection Engine", "version": "1.4.0"}


@app.post(
    "/api/v1/market-intelligence/analyze",
    response_model=MarketIntelligenceResponse,
    responses={
        200: {"description": "Market intelligence report successfully generated."},
        400: {"model": ErrorResponse, "description": "Invalid analysis parameters."},
        500: {"model": ErrorResponse, "description": "Activity dataset error."}
    },
    tags=["Market Intelligence"]
)
async def analyze_market_intelligence(request: MarketIntelligenceRequest):
    """
    Analyzes marketplace buyer demand, detects trends, and generates actionable, explainable insights for artisans.
    """
    engine = MarketIntelligenceEngine()
    response = engine.analyze(request)
    return response


@app.get("/api/v1/schemes/health", tags=["Government Schemes"])
async def schemes_health_check():
    """Government Scheme Matching health check endpoint."""
    return {"status": "ok", "service": "Government Scheme Matching Engine", "version": "1.5.0"}


@app.post(
    "/api/v1/schemes/match",
    response_model=SchemeMatchingResponse,
    responses={
        200: {"description": "Government schemes successfully matched for artisan profile."},
        400: {"model": ErrorResponse, "description": "Invalid matching request parameters."},
        500: {"model": ErrorResponse, "description": "Schemes dataset error."}
    },
    tags=["Government Schemes"]
)
async def match_government_schemes(request: SchemeMatchingRequest):
    """
    Connects artisan craft profile, location, business status, and registration status
    with verified official Government of India / State Government welfare & credit schemes.
    """
    engine = SchemeMatchingEngine()
    response = engine.match_schemes(request)
    return response


if __name__ == "__main__":




    host = Config.get_host()
    port = Config.get_port()
    uvicorn.run("api:app", host=host, port=port, reload=True)
