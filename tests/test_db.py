import os
from dotenv import load_dotenv
from app.db import test_connection

load_dotenv()

print(os.getenv("DATABASE_URL"))
print(test_connection())