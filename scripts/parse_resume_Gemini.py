from google import genai
import os
from dotenv import load_dotenv

load_dotenv()


client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

file = client.files.upload(file='scripts/RESUME Sumeet Sahu_PAT.pdf')
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=['Could you summarize this file?', file]
)
print(response.text)