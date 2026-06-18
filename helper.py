import functools
import time
from io import BytesIO
from PIL import Image

# Extract image from response
def extract_image(response):
    for part in response.candidates[0].content.parts:
        if (
            part.inline_data
            and part.inline_data.mime_type
            .startswith("image/")
        ):
            return Image.open(
                BytesIO(part.inline_data.data)
            )
    return None

# Clean non-printable characters
def clean(s):
    """Remove non-printable characters."""
    return "".join(c for c in str(s) if c.isprintable())

# Tool wrapper with timing
def make_display_tool(fn):
    """Wrap a tool to log timing."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"\n>> {fn.__name__}")
        start = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        print(f"   Done in {elapsed:.1f}s")
        return result
    return wrapper
