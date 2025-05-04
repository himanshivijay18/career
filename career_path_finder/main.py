from app import app  # noqa: F401
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
