# Smart Artisan Platform AI Suite

Independent Python AI & Marketplace microservice responsible for:
1. **Multilingual Voice AI (Speech-to-Text)**: Converts artisan spoken descriptions in Indian languages (Kannada, Hindi, Tamil, Telugu, Malayalam, Marathi, Bengali, English) into text.
2. **Smart Product Catalog AI**: Automatically generates structured e-commerce product catalogs from artisan product images, text notes, and/or voice recordings using Google Gemini AI.
3. **Smart Pricing Intelligence AI**: Combines financial cost-plus profit floor calculations (protecting artisan margins) with multimodal marketplace comparable matching & Gemini 3.6 Flash.
4. **Buyer Marketplace & Artisan Matching**: Provides multi-criteria weighted search, recommendation ranking, and buyer-artisan matching.
5. **Market Intelligence**: Converts buyer search, view, order, and enquiry activity into transparent, explainable market insights for artisans.
6. **Government Scheme Matching**: Connects artisans with verified central and state government credit, toolkit, and cluster schemes based on craft, location, business status, and registration status.

---

## 📌 Key Features

- **Multilingual Speech-to-Text**: Standalone transcription endpoint (`POST /api/v1/voice/transcribe`) supporting Indian languages and code-switching (*Kanglish*, *Hinglish*).
- **Domain Vocabulary Adaptation**: Pre-tuned craft terms (*Dhokra, Bidriware, Kantha, Bandhani, Madhubani, Terracotta, Bamboo, Channapatna*).
- **Combined Voice + Text + Image Cataloging**: Accepts product image, optional typed text description, and optional voice recording (`POST /api/v1/catalog/generate`).
- **Smart Pricing Intelligence**: Data-backed price estimates, fair price ranges (cost floor, recommended, premium export), and itemized financial cost breakdowns (`POST /api/v1/pricing/estimate`).
- **Buyer Marketplace & Artisan Matching**: Multi-criteria weighted scoring engine (`POST /api/v1/marketplace/search`) matching buyers and artisans based on Category (25%), Keyword (25%), Craft Type (15%), Location (10%), Product Similarity (15%), and Price Compatibility (10%).
- **Market Intelligence**: Transparent activity analysis engine (`POST /api/v1/market-intelligence/analyze`) detecting category/craft trends, buyer demand levels, optimal price windows, and actionable artisan guidance.
- **Government Scheme Matching**: Rule-based matching engine (`POST /api/v1/schemes/match`) connecting artisans to verified official schemes (*PM Vishwakarma*, *Pehchan Card*, *PMEGP*, *Mahila Coir Yojana*, *AHVY*, *Karnataka State Scheme*) with official `.gov.in` source URLs.
- **100% Backward Compatible & Deterministic**: Zero breaking changes, preserving existing API contracts and full test coverage.
- **62 Unit & Integration Tests**: Complete test coverage across schemas, voice engine, catalog generator, pricing engine, marketplace search, market intelligence, government schemes, and HTTP endpoints.

---

## 📂 Project Structure

```text
ai/catalog/
├── .env.example                               # Environment configuration template
├── README.md                                  # Documentation and API integration guide
├── requirements.txt                           # Python dependency declarations
├── config.py                                  # Settings loader & API key manager
├── exceptions.py                              # Custom exception definitions
├── schema.py                                  # Pydantic schemas (Catalog, Voice, Pricing, Marketplace, Intelligence, Schemes)
├── voice.py                                   # Multilingual Speech-to-Text engine & VoiceService
├── service.py                                 # Core Catalog AIService with combined Voice + Text chaining
├── pricing.py                                 # Smart Pricing Intelligence AI Engine
├── marketplace.py                             # Buyer Marketplace & Artisan Recommendation Engine
├── market_intelligence.py                     # Market Intelligence & Trend Analytics Engine
├── scheme_matching.py                         # Government Scheme Matching Engine
├── marketplace_dataset.json                   # Benchmark historical dataset of Indian artisan products
├── market_intelligence_dataset.json           # Benchmark synthetic activity log dataset (searches, views, orders)
├── schemes_dataset.json                       # Verified official Indian Government schemes database
├── pricing_evaluation_dataset.json            # Ground-truth evaluation dataset for pricing benchmarks
├── marketplace_evaluation_dataset.json        # Ground-truth evaluation dataset for buyer marketplace search
├── market_intelligence_evaluation_dataset.json# Ground-truth evaluation dataset for market intelligence
├── scheme_matching_evaluation_dataset.json    # Ground-truth evaluation dataset for government scheme matching
├── api.py                                     # FastAPI REST server exposing all endpoints
├── run_evaluation.py                          # Catalog AI benchmark runner
├── run_voice_evaluation.py                    # Multilingual Voice WER benchmark runner
├── run_pricing_evaluation.py                  # Pricing accuracy & MAPE benchmark runner
├── run_marketplace_evaluation.py              # Marketplace search & ranking accuracy runner
├── run_market_intelligence_evaluation.py      # Market intelligence trend accuracy runner
├── run_scheme_matching_evaluation.py          # Government scheme matching accuracy runner
└── tests/
    ├── test_schema.py                         # Schema unit tests
    ├── test_service.py                        # Catalog AI service mock unit tests
    ├── test_api.py                            # FastAPI HTTP endpoint mock unit tests
    ├── test_voice.py                          # Voice Speech-to-Text unit tests
    ├── test_combined.py                       # Combined Voice + Image catalog unit tests
    ├── test_pricing.py                        # Pricing engine & API unit tests
    ├── test_marketplace.py                    # Buyer marketplace search unit tests
    ├── test_market_intelligence.py            # Market intelligence unit tests
    └── test_scheme_matching.py                # Government scheme matching unit tests
```

---

## 🚀 Setup & Execution

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Test Suite

Run all 62 unit and integration tests:

```bash
python -m pytest tests/ -v
```

### 3. Run Benchmark Evaluation Suites

```bash
# Pricing Accuracy Evaluation
python run_pricing_evaluation.py

# Marketplace Search & Ranking Evaluation
python run_marketplace_evaluation.py

# Market Intelligence & Trend Evaluation
python run_market_intelligence_evaluation.py

# Government Scheme Matching Evaluation
python run_scheme_matching_evaluation.py
```

### 4. Start the FastAPI Server

```bash
python api.py
```

Service runs at `http://localhost:8000`. Interactive Swagger API docs are available at `http://localhost:8000/docs`.

---

## 📡 API Reference

### 1. Government Scheme Matching: `POST /api/v1/schemes/match`

#### Request Body (`application/json`)
```json
{
  "craft_type": "Terracotta",
  "location": "Karnataka",
  "business_status": "Individual Artisan",
  "registration_status": "Unregistered / Informal",
  "age": 32,
  "gender": "Female"
}
```

#### Response (`200 OK`)
```json
{
  "success": true,
  "total_matched": 6,
  "results": [
    {
      "scheme": {
        "scheme_id": "SCHEME-KARNATAKA-ARTISAN",
        "scheme_name": "Karnataka Artisan Development & Toolkit Subsidy Scheme",
        "ministry_department": "Department of Handlooms & Textiles, Government of Karnataka",
        "target_crafts": ["Pottery", "Channapatna Toy", "Terracotta", "Woodcraft", "Metalcraft", "All Traditional Crafts"],
        "target_locations": ["Karnataka"],
        "eligible_business_statuses": ["Individual Artisan", "Self-Help Group (SHG)"],
        "eligible_registration_statuses": ["Pehchan Card Registered", "Udyam Registered", "Unregistered / Informal"],
        "benefit_description": "Financial assistance of ₹10,000 for improved toolkits and capital subsidy for traditional artisans operating in Karnataka State.",
        "eligibility_conditions": [
          "Must be a domicile resident of Karnataka State.",
          "Must be practicing a traditional Karnataka heritage craft."
        ],
        "required_documents": [
          "Karnataka Domicile / Resident Certificate",
          "Aadhaar Card",
          "Artisan Certificate / Ration Card"
        ],
        "official_source_url": "https://karnataka.gov.in/",
        "last_verified_date": "2026-08-01",
        "verification_status": "Official Verified"
      },
      "match_score": 1.0,
      "recommendation_reason": "Matches your Terracotta craft and Individual Artisan status under Karnataka coverage.",
      "matched_criteria": [
        "State location 'Karnataka' matches specific state scheme coverage",
        "Craft technique 'Terracotta' specifically eligible under scheme trades",
        "Business status 'Individual Artisan' eligible",
        "Registration status 'Unregistered / Informal' accepted"
      ],
      "unmet_criteria": []
    }
  ]
}
```

### 2. Government Schemes Health Check: `GET /api/v1/schemes/health`

#### Response (`200 OK`)
```json
{
  "status": "ok",
  "service": "Government Scheme Matching Engine",
  "version": "1.5.0"
}
```

---

## 📊 Benchmark Evaluation Results

- **Top-1 Scheme Match Accuracy**: **60.0%** (3/5 benchmark profiles)
- **Top-3 Scheme Match Accuracy**: **100.0%** (5/5 benchmark profiles)
- **Precision @ K Scheme Match**: **100.0%** (5/5 benchmark profiles)
- **Official Source Verification Rate**: **100.0%** (26/26 verified `.gov.in` URLs)

---

## ⚠️ Current Limitations

- **State Scheme Expansion**: Verified database currently includes key Central Government schemes and Karnataka state schemes. Additional state schemes (e.g., UP One District One Product - ODOP) can be appended to `schemes_dataset.json`.
- **Zero Sensitive Data Storage**: Engine intentionally avoids storing sensitive identification numbers (Aadhaar, PAN, Bank Accounts) to protect artisan privacy.
