from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import re
import torch
import logging
import uvicorn
import numpy as np
import faiss
import pickle
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bidi.algorithm import get_display
from arabic_reshaper import reshape
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
from pypdf import PdfReader
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class Config:
    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 120 
    SIMILARITY_THRESHOLD = 0.68
    MAX_FILE_SIZE_MB = 10
    INDEX_PATH = "faiss_index.bin"
    CHUNK_MAP_PATH = "chunk_map.pkl"
    GEMINI_API_KEY = "AIzaSyCjQakHsM7ypOrtYOBmHd6dZ-4MB8iXzUk" 

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def initialize_services():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(Config.EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(Config.EMBEDDING_MODEL_NAME).to(device)
    model.eval()

    executor = ThreadPoolExecutor(max_workers=4)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.35,
        google_api_key=Config.GEMINI_API_KEY,
        max_output_tokens=2048,
        convert_system_message_to_human=True
    )

    return {
        "tokenizer": tokenizer,
        "embedding_model": model,
        "llm": llm,
        "executor": executor,
        "device": device
    }

services = initialize_services()

class VectorStore:
    def __init__(self):
        hidden_size = services["embedding_model"].config.hidden_size
        self.index = faiss.IndexFlatIP(hidden_size)
        self.chunk_map = {}
        self.current_id = 0
        self.file_metadata = {}
        self.load_index()

    def add_document(self, chunks: List[str], filename: str):
        try:
            display_chunks = [ArabicTextProcessor.process_file(chunk.encode('utf-8')) for chunk in chunks]
            embedding_chunks = [ArabicTextProcessor.normalize_text(chunk) for chunk in chunks]

            embeddings = list(services["executor"].map(self._generate_embedding, embedding_chunks))
            self.index.add(np.array(embeddings, dtype=np.float32))

            for idx, chunk in enumerate(display_chunks):
                self.chunk_map[self.current_id] = {
                    "text": chunk,
                    "file": filename,
                    "chunk_num": idx + 1
                }
                self.current_id += 1

            self.file_metadata[filename] = {
                "num_chunks": len(chunks),
                "start_id": self.current_id - len(chunks),
                "end_id": self.current_id - 1
            }
            self._save_index()
        except Exception as e:
            logger.error(f"Indexing error: {str(e)}")
            raise

    def _generate_embedding(self, text: str) -> np.ndarray:
        try:
            tokenizer = services["tokenizer"]
            model = services["embedding_model"]
            device = services["device"]

            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                pooled = mean_pooling(outputs, inputs['attention_mask'])
                normalized_embedding = F.normalize(pooled, p=2, dim=1)
            
            return normalized_embedding[0].cpu().numpy()
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        try:
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            distances, indices = self.index.search(query_embedding, top_k * 3)

            results = []
            for idx, score in zip(indices[0], distances[0]):
                if score > Config.SIMILARITY_THRESHOLD and idx in self.chunk_map:
                    results.append({
                        "score": float(score),
                        "text": self.chunk_map[idx]["text"],
                        "metadata": {
                            "file": self.chunk_map[idx]["file"],
                            "chunk": self.chunk_map[idx]["chunk_num"]
                        }
                    })

            return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def _save_index(self):
        try:
            faiss.write_index(self.index, Config.INDEX_PATH)
            with open(Config.CHUNK_MAP_PATH, "wb") as f:
                pickle.dump((self.chunk_map, self.file_metadata), f)
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise

    def load_index(self):
        try:
            if os.path.exists(Config.INDEX_PATH) and os.path.exists(Config.CHUNK_MAP_PATH):
                self.index = faiss.read_index(Config.INDEX_PATH)
                with open(Config.CHUNK_MAP_PATH, "rb") as f:
                    self.chunk_map, self.file_metadata = pickle.load(f)
                self.current_id = len(self.chunk_map)
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            self.index = faiss.IndexFlatIP(services["embedding_model"].config.hidden_size)

vector_store = VectorStore()

class ArabicTextProcessor:
    @staticmethod
    def normalize_text(text: str) -> str:
        allowed_punctuation = r'\.\?\!\،\؛\:\-\–\—\(\)\[\]'
        text = re.sub(fr'[^\w\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s{allowed_punctuation}]', '', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    @staticmethod
    def process_file(content: bytes) -> str:
        encodings = ['utf-8-sig', 'utf-16', 'windows-1256', 'latin-1']
        text = ""
        
        for encoding in encodings:
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Failed to decode file with supported encodings")

        normalized = ArabicTextProcessor.normalize_text(text)
        reshaped = reshape(normalized)
        return get_display(reshaped)

    @staticmethod
    def chunk_text(text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "۔ ", ".\n", "!\n", "؟\n", "\n", " "]
        )
        return splitter.split_text(text)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

def extract_text_from_pdf(content: bytes) -> str:
    try:
        pdf_reader = PdfReader(BytesIO(content))
        extracted_text = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text.append(page_text)
        combined_text = "\n".join(extracted_text)
        return ArabicTextProcessor.process_file(combined_text.encode('utf-8'))
    except Exception as e:
        logger.error(f"Failed to extract PDF text: {str(e)}")
        raise HTTPException(400, "Could not parse PDF content") from e

@app.post("/upload", status_code=201)
async def upload_file(file: UploadFile = File(...)):
    allowed_types = [
        "text/plain",
        "application/octet-stream",
        "text/plain; charset=utf-8",
        "application/pdf"
    ]
    
    if file.content_type not in allowed_types:
        logger.error(f"Invalid file type: {file.content_type}")
        raise HTTPException(400, "Only TXT and PDF files are supported")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(400, "Empty file uploaded")

        if len(content) > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                413, 
                f"File size exceeds {Config.MAX_FILE_SIZE_MB}MB limit"
            )

        if file.content_type == "application/pdf":
            text = extract_text_from_pdf(content)
        else:
            text = ArabicTextProcessor.process_file(content)

        if len(text.strip()) < 50:
            raise HTTPException(
                422, 
                "File content is too short (min 50 characters required)"
            )

        chunks = ArabicTextProcessor.chunk_text(text)
        vector_store.add_document(chunks, file.filename)
        return {"message": f"Successfully processed {len(chunks)} chunks"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(500, "File upload processing failed") from e
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        question = ArabicTextProcessor.normalize_text(request.question)
        print ("QUESTION : ",question)
        query_embedding = vector_store._generate_embedding(question)
        results = vector_store.search(query_embedding)
        print("RESULT : ",results)
        if not results:
            return JSONResponse(
                content={"answer": "لا تتوفر معلومات كافية للإجابة"},
                media_type="application/json; charset=utf-8"
            )

        context = "\n\n".join([
            f"المقطع {res['metadata']['chunk']} من {res['metadata']['file']}:\n{res['text']}"
            for res in results
        ])

        prompt = PromptTemplate(
            template="""
            # الدور: 
            مساعد أكاديمي متخصص في الوثائق الجامعية. يجب الإجابة بناءً على السياق المرفق فقط.

            ## السياق:
            {context}

            ## السؤال: 
            {question}

            ## التعليمات:
            1. الإجابة باللغة العربية الفصحى مع استخدام علامات الترقيم المناسبة
            2. تنظيم الإجابة في نقاط مرقمة (١، ٢، ٣) عند وجود معلومات متعددة
            3. وضع المراجع مباشرة بعد كل معلومة بين أقواس مربعة: [رقم]
            4. إضافة قسم "المصادر" في النهاية يحتوي على:
            - أسماء الملفات
            - أرقام المقاطع المستخدمة
            5. إذا كانت المعلومات غير كافية:
            - اذكر ذلك بوضوح في بداية الإجابة
            - لا تخترع أي معلومات

            ## تنسيق الإجابة المطلوب:
            الإجابة:
            ......
            
            المصادر:
            [١] اسم الملف (المقطع X)
            [٢] اسم الملف (المقطع Y)

            ## أمثلة مقبولة:
            الإجابة:
            ١. الشرط الأساسي للتسجيل هو ...[١]
            ٢. يجب تقديم الأوراق خلال ...[٢]
            
            المصادر:
            [١] اللائحة_الأكاديمية.pdf (المقطع ٣)
            [٢] دليل_الطالب.docx (المقطع ٥)

            ## تحذيرات:
            - ممنوع استخدام تنسيق Markdown
            - تجنب العبارات العامة غير المحددة
            - الحفاظ على التسلسل المنطقي في الإجابة
            - التأكد من تطابق أرقام المراجع مع المصادر

            الإجابة:
            """,
            input_variables=["context", "question"]
        )


        chain = LLMChain(
            llm=services["llm"],
            prompt=prompt,
            output_parser=StrOutputParser()
        )

        answer = await chain.apredict(context=context, question=question)
        print("FINAL ANSWER : ",answer)
        return JSONResponse(
            content={
                "answer": answer,
                "sources": [{"file": res["metadata"]["file"], "chunk": res["metadata"]["chunk"]} for res in results]
            },
        )
    except Exception as e:
        logger.error(f"Q&A failed: {str(e)}")
        raise HTTPException(500, "Question processing failed") from e

@app.get("/documents")
def list_documents():
    return {"documents": list(vector_store.file_metadata.keys())}

@app.delete("/reset")
def reset_index():
    try:
        vector_store.index.reset()
        vector_store.chunk_map.clear()
        vector_store.file_metadata.clear()
        vector_store.current_id = 0
        vector_store._save_index()
        return {"message": "Index reset successfully"}
    except Exception as e:
        logger.error(f"Reset failed: {str(e)}")
        raise HTTPException(500, "Index reset failed") from e

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)