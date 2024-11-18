import os
import json
import nest_asyncio
from openai import OpenAI
from pydantic import BaseModel
from llama_parse import LlamaParse
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from llama_index.core import SimpleDirectoryReader
from typing import List

nest_asyncio.apply()

# Setup fastAPI and client
app = FastAPI()

# Model for response
class QAResponse(BaseModel):
    Question: str
    Answer: str
    Filename: str
    Verify: bool

# Function to extract text from PDF
def extract_text_from_pdf(file_path: str) -> str:
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="text",
        language="vi",
        fast_mode=True,
        continuous_mode=True,
        do_not_unroll_columns=True,
        parsing_instruction="You are parsing files from word(.pdf) format to text(.txt) format. Remove unnecessary space between words!."
    )

    # Use SimpleDirectoryReader to parse a single file
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
    contents = [doc.text for doc in documents]

    return contents

# Function to generate QA pairs using OpenAI API
def generate_qa_pairs(content: str, filename: str, client) -> List[QAResponse]:
    client = client
    prompt = f"""
    Trong ngữ cảnh người dùng sẽ hỏi các thông tin liên quan tới quy trình trong nội dung, hãy tạo ra ít nhất 10 cặp (tầm 12-13) Question-Answer là những câu hỏi có khả năng cao bị hỏi và câu trả lời tương ứng.
    Hãy bỏ qua các mục tài liệu viện dẫn và định nghĩa/tóm tắt. Tập trung tạo Question-Answer cho mục "5. nội dung quy trình" trở đi.
    Nội dung:
    {content}

    Output dưới dạng JSON gồm: 
    - "Question" (câu hỏi liên quan),
    - "Answer" (câu trả lời chính xác),
    - "Filename" (tên file PDF),
    - "Verify" (giá trị mặc định là True).
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4o",
            temperature=0.7,
        )
        if response.choices[0].message.content.startswith("```json") and response.choices[0].message.content.endswith("```"):
            response.choices[0].message.content = response.choices[0].message.content[7:-3].strip()
        qa_pairs = json.loads(response.choices[0].message.content)
        return [
            QAResponse(
                Question=qa["Question"],
                Answer=qa["Answer"],
                Filename=filename,
                Verify=True,
            )
            for qa in qa_pairs
        ]
    except Exception as e:
        print(f"Error generating QA pairs: {e}")
        return []

@app.api_route("/generate-qa/", methods=["POST", "GET"])
async def create_qa(file: UploadFile = File(...), request: Request = None):
    # Lấy user_api_key từ headers
    openai_api_key = request.headers.get('user_api_key')
    if not openai_api_key:
        raise HTTPException(status_code=400, detail="API key is required.")

    # User's API key
    client = OpenAI(api_key=openai_api_key)
    try:
        # Save uploaded file locally
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Extract text from PDF
        content = extract_text_from_pdf(file_path)
        
        # Generate QA pairs
        qa_pairs = generate_qa_pairs(content, file.filename, client)
        
        # Prepare JSON response
        result = [qa.dict() for qa in qa_pairs]
        json_file_path = f"{file.filename.split('.')[0]}_qa.json"
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)
        
        # Cleanup temporary file
        os.remove(file_path)
        
        return {
            "message": "QA pairs generated successfully",
            "file": json_file_path,
            "data": result,
        }
    except Exception as e:
        return {"error": str(e)}