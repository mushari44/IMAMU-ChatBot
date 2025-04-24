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
from typing import List, Dict, Optional
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
import time
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import HuggingFacePipeline
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline
)
import torch, textwrap, os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

HF_token    = "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"        # your PAT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
def count_tokens(text: str) -> int:
    """Return #tokens according to the *embedding* tokenizer (E5 512‑max)."""
    return len(services["tokenizer"].encode(text, add_special_tokens=False))

class Config:
    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 100
    SIMILARITY_THRESHOLD = 0.50
    LANGSMITH_API_KEY: str = "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"
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VECTOR_DIR = os.path.join(BASE_DIR, "data", "Victoredb")
    INDEX_PATH = os.path.join(VECTOR_DIR, "faiss_index.bin")
    CHUNK_MAP_PATH = os.path.join(VECTOR_DIR, "chunk_map.pkl")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCjQakHsM7ypOrtYOBmHd6dZ-4MB8iXzUk") 

FILENAME_TO_TITLE = {
    "S_behavior.txt": "قواعد السلوك والانضباط الطلابي",
    "U_CARD.txt": "بطاقة الطالب الجامعي الرقمية",
    "Guide.txt": "ارشادات اكاديمية",
    "CQ.txt": "اسئلة شائعة",
    "Regulatory_rules_1444.txt": "لائحة الدراسة والاختبارات والقواعد التنفيذية",
}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return (sum_embeddings / sum_mask).detach()

def initialize_services():
    os.environ["LANGCHAIN_API_KEY"] = "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"
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    model_id = "CohereLabs/c4ai-command-r7b-arabic-02-2025"
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    max_mem = {0: "6GiB", "cpu": "24GiB"}   # عدِّل حسب ذاكرتك الفعلية
# Add memory mapping to model loading
    chat_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        max_memory=max_mem,
        torch_dtype=torch.float16,
        use_auth_token=HF_token,          # set env HF_TOKEN or hard‑code
        trust_remote_code=True, 
    )
    chat_tok = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_token)
# Update your pipeline initialization
    pipe = pipeline(
        task="text-generation",
        model=model_id,
        tokenizer=chat_tok,
        max_new_tokens=512,         # KV‑cache ≤ VRAM
        temperature=0.25,
        top_p=0.9,
        do_sample=True,
        batch_size=1,
    )
    hf_llm = HuggingFacePipeline(pipeline=pipe)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device="cpu"
    logger.info(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(
        Config.EMBEDDING_MODEL_NAME,
        cache_dir=os.getenv("TRANSFORMERS_CACHE", "./models")
    )
    model = AutoModel.from_pretrained(
        Config.EMBEDDING_MODEL_NAME,
        cache_dir=os.getenv("TRANSFORMERS_CACHE", "./models")
    ).to(device)
    model.eval()

    executor = ThreadPoolExecutor(max_workers=8)


    return {
        "tokenizer": tokenizer,
        "embedding_model": model,
        "llm": hf_llm,
        "executor": executor,
        "device": torch.device("cpu")
        # "device": device
    }

services = initialize_services()

class ArabicTextProcessor:
    @staticmethod
    def normalize_text(text: str) -> str:
        text = re.sub(r'ـ+', '', text)
        text = re.sub(r'[\u202D\u202C]', '', text)
        text = re.sub(r'[\u064B-\u065F]', '', text)
        text = re.sub(r'[!?؟<>.«»#,•:،؛·@&()\-\n]', ' ', text)
        text = re.sub(r'[إأآ]', 'ا', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        text = text.replace('ة', 'ه')
        text = re.sub(' +', ' ', text)
        return text.strip()

    @staticmethod
    def process_file(content: bytes) -> str:
        encodings = ['utf-8-sig', 'utf-16', 'windows-1256', 'latin-1']
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


class VectorStore:
    def __init__(self):
        hidden_size = services["embedding_model"].config.hidden_size
        self.index = faiss.IndexFlatIP(hidden_size)
        self.chunk_map = {}
        self.current_id = 0
        self.file_metadata = {}
        self.load_index()

    def _generate_embedding(self, text: str) -> np.ndarray:
        tokenizer = services["tokenizer"]
        model = services["embedding_model"]
        device = services["device"]

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pooled = mean_pooling(outputs, inputs['attention_mask'])
            normalized = F.normalize(pooled, p=2, dim=1)

        return normalized[0].cpu().numpy()

    def add_document(self, chunks: List[str], filename: str):
        try:
            if filename in self.file_metadata:
                logger.info(f"Skipping {filename}, already embedded.")
                return

            display_chunks = [ArabicTextProcessor.process_file(chunk.encode('utf-8')) for chunk in chunks]
            norm_chunks = [ArabicTextProcessor.normalize_text(c) for c in chunks]
            token_counts = [count_tokens(c) for c in norm_chunks]
            for i, tc in enumerate(token_counts):
                print(f"{filename} | chunk {i+1}/{len(chunks)} | {tc} tokens")
                if tc > 512:
                    logger.warning(f"  >512 tokens: will be truncated by E5!")

            embedding_chunks = [ArabicTextProcessor.normalize_text(chunk) for chunk in chunks]
            embeddings = list(services["executor"].map(self._generate_embedding, embedding_chunks))
            torch.cuda.empty_cache()      
            # print("display chunk !! : ",display_chunks)
            self.index.add(np.array(embeddings, dtype=np.float32))

            display_name = FILENAME_TO_TITLE.get(filename, filename)
            for idx, chunk_text in enumerate(display_chunks):
                self.chunk_map[self.current_id] = {
                    "text": chunk_text,
                    "file_name": display_name,
                    "display_name": filename,
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

    def search(self, query_embedding: np.ndarray, top_k: int = 10, document: Optional[str] = None) -> List[Dict]:

        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k * 2)

        results = []
        seen_texts = set()

        for idx, score in zip(indices[0], distances[0]):
            if idx == -1 or idx not in self.chunk_map:
                continue

            chunk = self.chunk_map[idx]
            if chunk["text"] in seen_texts or score <= Config.SIMILARITY_THRESHOLD:
                continue
            print("Document!!!!!!:",document)
            if document !="" and chunk["file_name"] != document:
                continue

            seen_texts.add(chunk["text"])
            results.append({
                "score": float(score),
                "text": chunk["text"],
                "metadata": {
                    "file": chunk["file_name"],
                    "chunk": chunk["chunk_num"]
                }
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def _save_index(self):
        try:
            faiss.write_index(self.index, Config.INDEX_PATH)
            with open(Config.CHUNK_MAP_PATH + ".tmp", "wb") as f:
                pickle.dump((self.chunk_map, self.file_metadata), f)
            os.replace(Config.CHUNK_MAP_PATH + ".tmp", Config.CHUNK_MAP_PATH)
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


class QuestionRequest(BaseModel):
    question: str
    document: Optional[str]
# Add to imports
import gc, pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def clear_gpu():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU free after cleanup: {info.free/1024**2:.1f} MB")

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        start = time.time()
        question = ArabicTextProcessor.normalize_text(request.question)
        query_embedding = vector_store._generate_embedding(question)
        results = vector_store.search(query_embedding, document=request.document)

        print("\n" + "=" * 60)
        print(" السؤال المُعالج (Normalized Question):")
        print(f">>> {question}\n")

        if not results:
            print(" لم يتم العثور على مقاطع ذات صلة.")
            return JSONResponse(
                content={"answer": "لا تتوفر معلومات كافية للإجابة"},
                media_type="application/json; charset=utf-8"
            )

        print(" المقاطع المسترجعة (Top Chunks):")
        print("=" * 60)
        print(f"عدد المقاطع المسترجعة: {len(results)}")
        # print("=" * 60)
        # print("Results: ",results) 
        # for i, res in enumerate(results[:15], start=1):
        #     meta = res["metadata"]
        #     print(f"  {i}. [{meta['file']}] المقطع رقم {meta['chunk']} |  التشابه: {res['score']:.4f}")
        #     print(f"      النص: {res['text'].strip()}...\n")

        context = "\n\n".join([
            f"المقطع {res['metadata']['chunk']} من {res['metadata']['file']}:\n{res['text']}"
            for res in results
        ])
        print("=" * 60)
        print(" السياق المُرسل إلى النموذج (Context Sent to LLM):")
        print(context + "...\n")
#         prompt = PromptTemplate(
#     template="""
#     <|START_OF_TURN_TOKEN|>
#     <|SYSTEM_TOKEN|>
#     أنت مساعد قانوني وأكاديمي متخصص في الوثائق الجامعية. قم بالإجابة باستخدام السياق المرفق فقط مع الالتزام الصارم بالتالي:
    
#     السياق:
#     {context}
    
#     التعليمات:
#     ١. استخدم لغة عربية فصحى واضحة
#     ٢. أدرج المراجع بين [] بعد كل نقطة
#     ٣. رتب الإجابة كقائمة مرقمة
#     ٤. أضف قسم مصادر منفصل في النهاية
#     ٥. لا تخترع معلومات خارج السياق
    
#     <|USER_TOKEN|>
#     السؤال: {question}
    
#     <|ASSISTANT_TOKEN|>
#     الإجابة:
#     """,
#     input_variables=["context", "question"]
# )
    #     prompt = PromptTemplate(
    #     template="""
    #     **المهمة**: أنت خبير في الأنظمة الجامعية السعودية. قم ب:
    #     1. تحليل السؤال بدقة
    #     2. استخراج المعلومات من السياق فقط
    #     3. تنظيم الإجابة كالتالي:
        
    #     **الهيكل المطلوب**:
    #     [مقدمة موجزة]
        
    #     (١) النقطة الأولى [المصدر]
    #     (٢) النقطة الثانية [المصدر]
        
    #     **ملاحظات هامة**:
    #     - عدم استخدام العبارات العامة
    #     - الربط المنطقي بين النقاط
    #     - تحديد نوع التشريع (نظامي/إجرائي/توجيهي)
        
    #     **السياق**:
    #     {context}
        
    #     **السؤال**:
    #     {question}
        
    #     **الإجابة**:
    #     """,
    #     input_variables=["context", "question"]
    # )

        tokenizer = services["tokenizer"]
        full_prompt = prompt.format(context=context, question=question)
        token_count = len(tokenizer.encode(full_prompt, add_special_tokens=False))

        # 2. Add warning system
        if token_count > 14000:  # For 16k window
                print(f"Prompt token overflow: {token_count}/16000")

        # 3. Add to your existing print block
        print(f"إجمالي وحدات السياق الرمزية: {token_count} (الحد الأقصى 16000)")
        chain = LLMChain(
            llm=services["llm"],
            prompt=prompt,
            output_parser=StrOutputParser()
        )
        # Add this right after the context is created
        context_char_count = len(context)
        context_token_count = len(services["tokenizer"].tokenize(context))
        logger.info(f"\nContext Size - Characters: {context_char_count}, Tokens: {context_token_count}")

        # Add to the existing print block
        print("=" * 60)
        print(" السياق المُرسل إلى النموذج (Context Sent to LLM):")
        print(f"حجم السياق: {context_char_count} حرف، {context_token_count} وحدة رمزية")
        print("=" * 60)
# In your ask_question endpoint, wrap generation in try/finally
        try:
            answer = await chain.apredict(context=context, question=question)
            payload = {
        "answer":  answer,
        "context": context,                   # still alive here
        "sources": [
            {"file": r["metadata"]["file"], "chunk": r["metadata"]["chunk"]}
            for r in results
        ],
    }
        finally:
            # Add memory cleanup
            del context, question, results, query_embedding
            clear_gpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print("=" * 60)
        print(" الإجابة النهائية من النموذج (Final Answer):\n")
        print(payload)
        print("=" * 60)
        print(f" مدة المعالجة: {time.time() - start:.2f} ثانية")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Q&A failed: {str(e)}")
        raise HTTPException(500, "Question processing failed")
    return JSONResponse(content=payload)
    


@app.get("/documents")
def list_documents():
    display_names = list({chunk["file_name"] for chunk in vector_store.chunk_map.values()})
    return {"documents": sorted(display_names)}

def embed_all_files_in_folder(folder_path: str = "data"):
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if not os.path.isfile(full_path):
            continue
        try:
            with open(full_path, "rb") as f:
                content = f.read()
            text = ArabicTextProcessor.process_file(content)
            chunks = ArabicTextProcessor.chunk_text(text)
            vector_store.add_document(chunks, filename)
            logger.info(f"Embedded {filename} with {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Failed to process {filename}: {str(e)}")



from pymongo import MongoClient
from datetime import datetime

client = MongoClient('mongodb://localhost:27017/')
db = client['imamu']
feedback_collection = db['feedBack']
class FeedbackRequest(BaseModel):
    messageId: int
    feedback: str
    comment: Optional[str] = None
    question: str
    answer: str
    document: Optional[str] = None
    references: List[Dict] = []
    context: str 

@app.post("/feedback")
async def store_feedback(feedback: FeedbackRequest):
    try:
        feedback_data = {
            "message_id": feedback.messageId,
            "feedback_type": feedback.feedback,
            "comment": feedback.comment,
            "question": feedback.question,
            "answer": feedback.answer,
            "document": feedback.document,
            "references": feedback.references,
            "context": feedback.context,            
            "timestamp": datetime.utcnow()
        }
        
        result = feedback_collection.insert_one(feedback_data)
        return {"status": "success", "inserted_id": str(result.inserted_id)}
    
    except Exception as e:
        logger.error(f"Feedback storage failed: {str(e)}")
        raise HTTPException(500, "Feedback storage failed")

@app.on_event("startup")
def startup_event():
    embed_all_files_in_folder("data")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
