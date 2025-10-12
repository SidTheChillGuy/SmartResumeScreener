from google import genai
import os
from dotenv import load_dotenv
import json
import pandas as pd

load_dotenv()

# load current working dir
cwd = os.getcwd()
cwd = cwd.replace("\\","/")

# API key
if os.environ["GEMINI_API_KEY"] is not None:
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
else:
    print("API Key cannot be None, update the values in the .env file and retry")
    exit(1)

# make client
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

# Vaiables
MODEL_NAME = "gemini-2.5-flash"

# The JSON schema we expect the model to return so we can maintain a Database
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_name": {
            "type": "string",
            "description": "The name  of the candidate extracted from the Resume file"
        },
        "candidate_score": {
            "type": "integer",
            "description": "A score from 0 to 100 representing how well the candidate's skills and experience from the attached resume PDF match the job description."
        },
        "summary": {
            "type": "string",
            "description": "A brief two-sentence summary explaining whether the candidate is suitable for the role or not, and why."
        },
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of the candidate's key strengths that may align with the job description, or NONE."
        },
        "weaknesses": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of potential weaknesses or areas where the candidate's experience is lacking compared to the job description."
        },
        "missing_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of important keywords from the job description that were not found in the resume."
        }
    },
    "required": ["candidate_name", "candidate_score", "summary", "strengths", "weaknesses", "missing_keywords"]
}

# template to create a prompt
def get_jd_prompt(jd_text):

    return f"""
    You are an expert HR recruitment manager. Your task is to analyze the attached resume PDF file against the following job description and provide a structured JSON output.

    Carefully evaluate the resume based on the skills, experience, and qualifications outlined in the job description below.
    Provide a score from 0 to 100, where 100 is a perfect match.
    Also, provide a concise summary, and lists of strengths, weaknesses, and missing keywords.

    **Job Description:**
    ---
    {jd_text}
    ---

    Return ONLY the JSON object based on the schema provided. Do not include any other text or markdown formatting.
    **JSON Schema:**
    ---
    {JSON_SCHEMA}
    ---
    """

# move the parsed files
def movefile(xmp):
    os.rename(f"user_resumes_unparsed/{xmp}", f"user_resumes_parsed/{xmp}")

# cleaning the Json formatting from LLM
def clean_output(response):
    raw_text = response
    
    # Find the first '{' and the last '}' to isolate the JSON object
    start_index = raw_text.find('{')
    end_index = raw_text.rfind('}')
    
    if start_index != -1 and end_index != -1 and end_index > start_index:
        json_string = raw_text[start_index : end_index + 1]
        result = json.loads(json_string)

    return result

# processing the output json
def process_output(f):
    for i in ["candidate_name","summary","strengths", "weaknesses","missing_keywords"]:
        if type(f[i])==list:
            f[i] = [(" ".join(f[i]))]
        else:
            f[i] = [f[i]]
    
    return f


# actual function
def analyze_resume_pdf(jd_text, userdata):
    # load all the unparsed files
    resfiles = [i for i in os.walk("user_resumes_unparsed")][0][2]
    
    for i in resfiles:
        if i[0] == ".":
            continue
        
        prompt = get_jd_prompt(jd_text)
        file = client.files.upload(file=f"user_resumes_unparsed/{i}")

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, file]
        )
        
        object = clean_output(response.text)
        object = process_output(object)
        
        object["resume_link"] = ["file:///" + cwd + "/user_resumes_parsed/" + i]
        
        userdata = pd.concat([userdata, pd.DataFrame(object)])
        
        movefile(i)
        
    return userdata


def runjob(jd_text):
    # load parsed resume data
    userdata = pd.read_csv("src/mdData.csv")
    
    userdata = analyze_resume_pdf(jd_text, userdata)
    
    userdata.reset_index(drop=True)
    userdata.to_csv("src/mdData.csv", index=False)