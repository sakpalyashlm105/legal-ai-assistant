"""
extraction/ocr.py
-----------------
OCR fallback module for the Legal AI Assistant.

What this file does:
    Receives a single page rendered as an image (PNG bytes) and asks
    GPT-4o-mini Vision to read the text inside that image. This is
    called by pdf_parser.py ONLY when a page's PyMuPDF extraction
    returned fewer than 100 meaningful characters (i.e. the page is
    likely a scanned image with no embedded text).

Why per-page, not per-document?
    A single PDF can mix digital-born pages (PyMuPDF handles these fine)
    with scanned pages (need OCR). Sending the WHOLE document to Vision
    every time would be slow and far more expensive than necessary.
    We only pay the OCR cost for the specific pages that actually need it.

Dependencies:
    openai (official OpenAI Python SDK)
    schemas/document.py
    python-dotenv (loads OPENAI_API_KEY from .env)
"""

import base64
import logging
import os

from openai import OpenAI, APIError, APITimeoutError, RateLimitError, AuthenticationError

from schemas.document import ExtractionMethod, PageExtraction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI client setup
# ---------------------------------------------------------------------------

# Which model handles the Vision OCR calls.
# Centralized here so if OpenAI renames the model, we change it in one place.
VISION_MODEL = "gpt-4o-mini"

# The client is initialized lazily (on first use) rather than at module load
# time. This is because config.py loads the .env file, but only when config.py
# itself is imported — which happens AFTER this module is already imported.
# Creating OpenAI() at module level would run before the key is in os.environ,
# causing an AuthenticationError even when the key exists in .env.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Return the shared OpenAI client, creating it on first call."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file and ensure "
                "config.py (or load_dotenv) is imported before calling OCR."
            )
        _client = OpenAI(api_key=api_key)
    return _client

# Maximum number of times we retry a failed Vision call before giving up
# on this page. A "retry" means we try the exact same request again,
# usually because of a temporary network hiccup or rate limit.
MAX_RETRIES = 2

# How many seconds to wait for a response before giving up and timing out.
# Vision calls on a single page image should not realistically take this long;
# this protects us from a hung connection blocking the whole pipeline.
REQUEST_TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# Helper: convert raw image bytes to base64 text
# ---------------------------------------------------------------------------

def _encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Convert raw PNG image bytes into a base64-encoded text string.

    Why this is needed:
        image_bytes is raw binary data (the actual pixels of the page,
        produced by PyMuPDF's pixmap.tobytes("png") in pdf_parser.py).
        The OpenAI API expects images as text-safe base64 strings inside
        a JSON message, not as raw binary. This function does that
        binary-to-text conversion.

    Parameters
    ----------
    image_bytes : bytes
        Raw PNG image data.

    Returns
    -------
    str
        A long string of base64 text representing the same image.

    Example
    -------
        Input:  b'\\x89PNG\\r\\n...' (raw binary, unreadable as text)
        Output: 'iVBORw0KGgoAAAANSUhEUgAA...' (safe text, same information)
    """
    # base64.b64encode() does the binary → base64 conversion.
    # The result is itself a bytes object containing only safe ASCII
    # characters, so we call .decode("utf-8") to turn it into a normal
    # Python string we can embed in a JSON request.
    return base64.b64encode(image_bytes).decode("utf-8")


# ---------------------------------------------------------------------------
# Helper: build the prompt sent to Vision
# ---------------------------------------------------------------------------

def _build_ocr_prompt() -> str:
    """
    Build the text instruction sent alongside the page image.

    Why do we need careful wording here?
        GPT-4o-mini Vision is a general-purpose model — by default it might
        try to "describe" the image ("This appears to be a legal document
        with a heading...") instead of transcribing it word-for-word.
        This prompt explicitly tells it: transcribe exactly, don't summarize,
        don't comment, don't add anything that isn't on the page.

    Returns
    -------
    str
        The instruction text for the Vision model.
    """
    return (
        "You are looking at a single page from a scanned legal document "
        "(such as a contract, NDA, or amendment). "
        "Transcribe ALL visible text on this page exactly as it appears, "
        "preserving paragraph breaks, numbered clauses, and line structure "
        "as closely as possible. "
        "Do NOT summarize, explain, or comment on the content. "
        "Do NOT add any text that is not visibly printed on the page. "
        "If a word or section is illegible, write [illegible] in its place "
        "rather than guessing. "
        "Output ONLY the transcribed text, with no preamble like 'Here is the text:'."
    )


# ---------------------------------------------------------------------------
# Main public function: OCR one page
# ---------------------------------------------------------------------------

def extract_page_with_vision(
    image_bytes: bytes,
    page_number: int,
) -> PageExtraction:
    """
    Send a single page image to GPT-4o-mini Vision and return the
    transcribed text as a PageExtraction object.

    This function is called by pdf_parser.py's extract_page_text()
    ONLY when PyMuPDF's direct text extraction returned too few
    characters for a given page.

    How it works, step by step:
        1. Convert the raw image bytes to base64 text (so it can travel
           inside a JSON API request).
        2. Build the message we send to the model: the instruction text
           PLUS the image, attached together.
        3. Call the OpenAI API, with retries if a temporary error occurs.
        4. Read the model's text response.
        5. Count meaningful characters in that response.
        6. Package everything into a PageExtraction and return it.

    Parameters
    ----------
    image_bytes : bytes
        The page rendered as a PNG image (produced by PyMuPDF's
        page.get_pixmap().tobytes("png") in pdf_parser.py).

    page_number : int
        The 1-based page number this image came from. Used only for
        labeling the result and for log messages — we trust the caller
        to have already converted from PyMuPDF's 0-based index.

    Returns
    -------
    PageExtraction
        method will be OCR_VISION if the call succeeded with usable text,
        or FAILED if every retry failed or the model returned no usable text.
    """
    # Import here, locally, to avoid a circular import: pdf_parser.py imports
    # this function, and this module imports from schemas — keeping the
    # heavier dependency (count_meaningful_chars) duplicated as a tiny
    # local helper avoids pdf_parser.py and ocr.py needing to import each
    # other back and forth.
    def _count_meaningful_chars(text: str) -> int:
        return len("".join(text.split()))

    logger.info(f"Sending page {page_number} to GPT-4o-mini Vision for OCR")

    # --- Step 1: Encode the image ---
    base64_image = _encode_image_to_base64(image_bytes)

    # --- Step 2: Build the message payload ---
    # The OpenAI Chat Completions API expects a list of "messages".
    # Each message has a "role" (who is speaking) and "content".
    # For Vision calls, "content" is a LIST containing both:
    #   - a text block (our instructions)
    #   - an image block (the page picture, as a base64 "data URL")
    #
    # A "data URL" is a special text format that says "this string IS an
    # image, encoded as base64, in PNG format" — it looks like:
    #   data:image/png;base64,iVBORw0KGgoAAAANSU...
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": _build_ocr_prompt(),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        # "detail": "high" tells the model to examine the
                        # image at full resolution rather than a quick
                        # low-res pass. Legal text is small and dense,
                        # so we want maximum reading accuracy.
                        "detail": "high",
                    },
                },
            ],
        }
    ]

    # --- Step 3: Call the API, with retry logic ---
    last_error: Exception | None = None

    # range(MAX_RETRIES + 1) means: try once, then retry up to MAX_RETRIES
    # more times if needed. With MAX_RETRIES=2, this tries a total of 3 times.
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = _get_client().chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                timeout=REQUEST_TIMEOUT_SECONDS,
                # temperature=0 means "be as deterministic and literal as
                # possible, don't get creative." For transcription tasks,
                # we want the same image to produce the same text every time,
                # not creative variation.
                temperature=0,
                max_tokens=4096,  # generous ceiling for a single page of text
            )

            # The actual text reply lives at response.choices[0].message.content
            transcribed_text = response.choices[0].message.content or ""
            transcribed_text = transcribed_text.strip()

            char_count = _count_meaningful_chars(transcribed_text)

            if char_count == 0:
                # The model responded but gave us nothing usable —
                # treat this the same as a failure, but don't retry
                # (retrying won't fix a blank page).
                logger.warning(f"Page {page_number}: Vision OCR returned empty text")
                return PageExtraction(
                    page_number=page_number,
                    text="",
                    char_count=0,
                    method=ExtractionMethod.FAILED,
                    extraction_notes="Vision OCR returned no usable text.",
                )

            logger.info(
                f"Page {page_number}: Vision OCR succeeded "
                f"({char_count} chars, attempt {attempt + 1})"
            )

            return PageExtraction(
                page_number=page_number,
                text=transcribed_text,
                char_count=char_count,
                method=ExtractionMethod.OCR_VISION,
                extraction_notes=(
                    None if attempt == 0
                    else f"Succeeded after {attempt + 1} attempts"
                ),
            )

        # ---------------------------------------------------------------
        # Error handling — each error type gets its own branch because
        # the right RESPONSE differs depending on what went wrong.
        # ---------------------------------------------------------------

        except AuthenticationError as e:
            # The API key is missing, invalid, or revoked.
            # Retrying will NEVER fix this — it's not a temporary problem.
            # We stop immediately rather than wasting attempts.
            logger.error(f"Page {page_number}: OpenAI authentication failed: {e}")
            return PageExtraction(
                page_number=page_number,
                text="",
                char_count=0,
                method=ExtractionMethod.FAILED,
                extraction_notes=f"OpenAI authentication error: {e}",
            )

        except RateLimitError as e:
            # We've sent too many requests too quickly, or hit a usage cap.
            # This IS often temporary — worth retrying after the loop's
            # natural delay, unless we're already on our last attempt.
            last_error = e
            logger.warning(
                f"Page {page_number}: rate limit hit (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}"
            )

        except APITimeoutError as e:
            # The request took longer than REQUEST_TIMEOUT_SECONDS.
            # Worth retrying — could be a temporary network slowdown.
            last_error = e
            logger.warning(
                f"Page {page_number}: request timed out (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}"
            )

        except APIError as e:
            # A general OpenAI API error (e.g. their servers had a problem).
            # Worth retrying once or twice.
            last_error = e
            logger.warning(
                f"Page {page_number}: API error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}"
            )

        except Exception as e:
            # Anything else unexpected (network dropped, malformed response, etc.)
            last_error = e
            logger.warning(
                f"Page {page_number}: unexpected error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}"
            )

    # ------------------------------------------------------------------------
    # If we reach this point, every attempt failed (and none of them was an
    # AuthenticationError, which would have returned early above).
    # ------------------------------------------------------------------------
    logger.error(
        f"Page {page_number}: Vision OCR failed after {MAX_RETRIES + 1} attempts. "
        f"Last error: {last_error}"
    )
    return PageExtraction(
        page_number=page_number,
        text="",
        char_count=0,
        method=ExtractionMethod.FAILED,
        extraction_notes=f"Vision OCR failed after {MAX_RETRIES + 1} attempts: {last_error}",
    )