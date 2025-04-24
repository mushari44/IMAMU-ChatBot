# api/index.py
from mangum import Mangum
from app.main import app   # your FastAPI instance

handler = Mangum(app)      # Vercel will invoke this Lambda-style handler
