"""Custom exceptions for Smart Catalog & Multilingual Voice AI module."""

class CatalogAIException(Exception):
    """Base exception for Catalog AI & Voice module."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class MissingImageError(CatalogAIException):
    """Raised when no image data is provided."""
    def __init__(self, message: str = "Product image is required."):
        super().__init__(message, status_code=400)


class InvalidImageError(CatalogAIException):
    """Raised when the provided image data cannot be opened or parsed."""
    def __init__(self, message: str = "Invalid or corrupted image format."):
        super().__init__(message, status_code=400)


class MissingAudioError(CatalogAIException):
    """Raised when no audio data is provided for voice transcription."""
    def __init__(self, message: str = "Voice audio file is required."):
        super().__init__(message, status_code=400)


class InvalidAudioError(CatalogAIException):
    """Raised when the provided audio data is corrupted or empty."""
    def __init__(self, message: str = "Invalid or corrupted audio file."):
        super().__init__(message, status_code=400)


class UnsupportedAudioFormatError(CatalogAIException):
    """Raised when uploaded audio format/MIME type is not supported."""
    def __init__(self, message: str = "Unsupported audio format. Allowed: WAV, MP3, M4A, OGG, FLAC, AAC."):
        super().__init__(message, status_code=400)


class EmptyTranscriptionError(CatalogAIException):
    """Raised when speech recognition returns no audible text."""
    def __init__(self, message: str = "Audio contained no clear or audible speech to transcribe."):
        super().__init__(message, status_code=422)


class MissingAPIKeyError(CatalogAIException):
    """Raised when GEMINI_API_KEY environment variable is missing or empty."""
    def __init__(self, message: str = "GEMINI_API_KEY environment variable is missing or empty."):
        super().__init__(message, status_code=401)


class AIServiceError(CatalogAIException):
    """Raised when call to Gemini API fails due to network, rate limit, or upstream server errors."""
    def __init__(self, message: str = "AI service call failed."):
        super().__init__(message, status_code=502)


class InvalidAIResponseError(CatalogAIException):
    """Raised when AI output fails JSON schema parsing or field validation."""
    def __init__(self, message: str = "Failed to parse structured catalog response from AI output."):
        super().__init__(message, status_code=502)


class PricingError(CatalogAIException):
    """Base exception for pricing module errors."""
    def __init__(self, message: str = "Pricing calculation error occurred.", status_code: int = 500):
        super().__init__(message, status_code=status_code)


class MissingPricingInputError(PricingError):
    """Raised when required pricing parameters are missing."""
    def __init__(self, message: str = "Required pricing input parameters are missing."):
        super().__init__(message, status_code=400)


class InvalidPricingInputError(PricingError):
    """Raised when pricing input parameters are invalid (e.g. negative costs or hours)."""
    def __init__(self, message: str = "Invalid pricing input parameters."):
        super().__init__(message, status_code=400)


class PricingDatasetError(PricingError):
    """Raised when historical marketplace dataset cannot be loaded."""
    def __init__(self, message: str = "Marketplace benchmark dataset unavailable or invalid."):
        super().__init__(message, status_code=500)


class MarketplaceError(CatalogAIException):
    """Base exception for marketplace module errors."""
    def __init__(self, message: str = "Marketplace operation error occurred.", status_code: int = 500):
        super().__init__(message, status_code=status_code)


class InvalidMarketplaceQueryError(MarketplaceError):
    """Raised when marketplace search parameters are invalid."""
    def __init__(self, message: str = "Invalid marketplace search parameters."):
        super().__init__(message, status_code=400)


class MarketIntelligenceError(CatalogAIException):
    """Base exception for market intelligence module errors."""
    def __init__(self, message: str = "Market intelligence error occurred.", status_code: int = 500):
        super().__init__(message, status_code=status_code)


class InvalidMarketIntelligenceRequestError(MarketIntelligenceError):
    """Raised when market intelligence request parameters are invalid."""
    def __init__(self, message: str = "Invalid market intelligence request parameters."):
        super().__init__(message, status_code=400)


class SchemeMatchingError(CatalogAIException):
    """Base exception for government scheme matching module errors."""
    def __init__(self, message: str = "Government scheme matching error occurred.", status_code: int = 500):
        super().__init__(message, status_code=status_code)


class InvalidSchemeRequestError(SchemeMatchingError):
    """Raised when scheme matching request parameters are invalid."""
    def __init__(self, message: str = "Invalid scheme matching request parameters."):
        super().__init__(message, status_code=400)




