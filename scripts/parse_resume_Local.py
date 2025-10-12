# in progress

"""
Based on my previous projects involving VLMs, the script code should work. However, I have not tested this out. You may report errors or fixes.
"""

from pdf2image import convert_from_path
import torch
from transformers import AutoTokenizer, AutoModel
import math
import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import os
import json
import pandas as pd

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = 'OpenGVLab/InternVL3-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map="cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

cwd = os.getcwd().replace("\\","/")

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
        
        # set the max number of tiles in `max_num`
        pixel_values = load_image(convert_from_path("user_resumes_unparsed/" + i)[0], max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=10000, do_sample=True)
        
        prompt = get_jd_prompt(jd_text)
        question = f'<image>\nGiven is the resume file.\n{prompt}'
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        
        object = clean_output(response)
        object = process_output(object)
        
        object["resume_link"] = ["file:///" + cwd + "/user_resumes_parsed/" + i]
        
        userdata = pd.concat([userdata, pd.DataFrame(object)])
        
        movefile(i)
        
    return userdata


def runjob(jd_text):
    # load parsed resume data
    userdata = pd.read_csv("src/mdData.csv")
    
    yield "Starting Execution"
    
    userdata = analyze_resume_pdf(jd_text, userdata)
    
    userdata.reset_index(drop=True)
    userdata.to_csv("src/mdData.csv", index=False)
    
    yield "Completed Parsing. You can refresh the table."