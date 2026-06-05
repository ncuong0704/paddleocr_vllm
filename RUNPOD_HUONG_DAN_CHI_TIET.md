# 📚 Hướng Dẫn Chi Tiết RunPod Serverless - Từng Bước

> Hướng dẫn này giúp bạn deploy PaddleOCR-VL + vLLM lên RunPod Serverless từ A đến Z

---

## 📋 Yêu Cầu Trước Khi Bắt Đầu

- ✅ Tài khoản RunPod (https://runpod.io/)
- ✅ GitHub account (hoặc có thể upload code trực tiếp)
- ✅ Credit RunPod hoặc card thanh toán
- ✅ Hiểu biết cơ bản về Docker (không bắt buộc nhưng giúp)

---

## 🔥 BƯỚC 1: Chuẩn Bị Files & Code

### 1.1 Tạo GitHub Repository (Tuỳ Chọn Nhưng Khuyến Nghị)

**Tại sao?** Dễ quản lý, dễ update, RunPod có thể pull code tự động.

```bash
# 1. Tạo folder project
mkdir paddleocr-vl-runpod
cd paddleocr-vl-runpod

# 2. Khởi tạo git (nếu chưa có)
git init
git config user.email "your@email.com"
git config user.name "Your Name"
```

### 1.2 Tạo File Handler (`handler.py`)

**Đây là file xử lý logic khi có request gửi tới endpoint**

```bash
# Tạo file
touch handler.py
```

**Nội dung file `handler.py`:**

```python
import runpod
from openai import OpenAI
import base64
import json
from pathlib import Path
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo vLLM client
def get_client():
    """Tạo OpenAI client cho vLLM"""
    return OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        timeout=3600
    )

# Các tasks được hỗ trợ
TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

def process_image_url(image_url: str, task: str = "ocr"):
    """
    Xử lý OCR từ URL hình ảnh
    
    Args:
        image_url: URL của hình ảnh
        task: loại task ("ocr", "table", "formula", "chart")
    
    Returns:
        Kết quả OCR
    """
    try:
        logger.info(f"Processing image from URL: {image_url}")
        logger.info(f"Task: {task}")
        
        client = get_client()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": TASKS.get(task, TASKS["ocr"])
                    }
                ]
            }
        ]
        
        logger.info("Sending request to vLLM...")
        response = client.chat.completions.create(
            model="PaddlePaddle/PaddleOCR-VL",
            messages=messages,
            temperature=0.0,
        )
        
        result = response.choices[0].message.content
        logger.info(f"Processing completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error processing image from URL: {str(e)}")
        raise

def process_image_base64(image_base64: str, task: str = "ocr", mime_type: str = "image/png"):
    """
    Xử lý OCR từ base64 image
    
    Args:
        image_base64: base64 encoded image string
        task: loại task
        mime_type: MIME type của image
    
    Returns:
        Kết quả OCR
    """
    try:
        logger.info(f"Processing image from base64")
        logger.info(f"Task: {task}")
        
        client = get_client()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": TASKS.get(task, TASKS["ocr"])
                    }
                ]
            }
        ]
        
        logger.info("Sending request to vLLM...")
        response = client.chat.completions.create(
            model="PaddlePaddle/PaddleOCR-VL",
            messages=messages,
            temperature=0.0,
        )
        
        result = response.choices[0].message.content
        logger.info(f"Processing completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}")
        raise

def handler(event):
    """
    Handler chính cho RunPod Serverless
    
    Event input format:
    {
        "input": {
            "image_url": "https://...",  # hoặc
            "image_base64": "base64string",
            "task": "ocr",  # tuỳ chọn
            "mime_type": "image/png"  # nếu dùng base64
        }
    }
    """
    try:
        logger.info("=" * 60)
        logger.info("Handler called")
        logger.info("=" * 60)
        
        input_data = event.get("input", {})
        
        # Validate input
        if not input_data:
            return {
                "status": "error",
                "error": "Thiếu input data"
            }
        
        # Lấy task (mặc định: ocr)
        task = input_data.get("task", "ocr")
        
        # Validate task
        if task not in TASKS:
            return {
                "status": "error",
                "error": f"Task không hỗ trợ: {task}. Hỗ trợ: {list(TASKS.keys())}"
            }
        
        # Xử lý từ URL
        if "image_url" in input_data:
            image_url = input_data["image_url"]
            logger.info(f"Mode: URL")
            result = process_image_url(image_url, task)
            return {
                "status": "success",
                "task": task,
                "mode": "url",
                "result": result
            }
        
        # Xử lý từ base64
        elif "image_base64" in input_data:
            image_base64 = input_data["image_base64"]
            mime_type = input_data.get("mime_type", "image/png")
            logger.info(f"Mode: base64")
            result = process_image_base64(image_base64, task, mime_type)
            return {
                "status": "success",
                "task": task,
                "mode": "base64",
                "result": result
            }
        
        else:
            return {
                "status": "error",
                "error": "Cần cung cấp 'image_url' hoặc 'image_base64' trong input"
            }
    
    except Exception as e:
        logger.error(f"Handler error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }

# Chạy serverless handler
if __name__ == "__main__":
    logger.info("Starting RunPod Serverless Handler...")
    runpod.serverless.start({"handler": handler})
```

### 1.3 Tạo File Dockerfile

```bash
# Tạo file Dockerfile
touch Dockerfile
```

**Nội dung file `Dockerfile`:**

```dockerfile
# Base image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install vLLM (nightly build để hỗ trợ PaddleOCR-VL)
RUN pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly \
    --index-strategy unsafe-best-match

# Install additional dependencies
RUN pip install \
    openai>=1.0.0 \
    runpod>=0.8.0 \
    requests>=2.31.0 \
    Pillow>=10.0.0 \
    transformers>=4.35.0

# Copy handler script
COPY handler.py /workspace/handler.py

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/v1/models || exit 1

# Default command: Start vLLM server and handler
CMD bash -c "\
    echo 'Starting vLLM server...' && \
    vllm serve PaddlePaddle/PaddleOCR-VL \
        --trust-remote-code \
        --max-num-batched-tokens 16384 \
        --no-enable-prefix-caching \
        --mm-processor-cache-gb 0 \
        --port 8000 \
        --host 0.0.0.0 \
        --gpu-memory-utilization 0.9 &\
    \
    sleep 45 && \
    echo 'Starting RunPod handler...' && \
    python3 /workspace/handler.py"
```

### 1.4 Tạo `.dockerignore`

```bash
touch .dockerignore
```

```
__pycache__
*.pyc
.git
.gitignore
README.md
.env
.DS_Store
*.log
venv/
.venv/
```

### 1.5 Tạo `requirements.txt` (Optional nhưng tốt)

```bash
touch requirements.txt
```

```
torch>=2.0.0
torchvision
torchaudio
vllm>=0.6.0
openai>=1.0.0
runpod>=0.8.0
requests>=2.31.0
Pillow>=10.0.0
transformers>=4.35.0
```

### 1.6 Tạo `README.md`

```bash
touch README.md
```

```markdown
# PaddleOCR-VL on RunPod

Deploy PaddleOCR-VL with vLLM on RunPod Serverless

## Quick Start

### Build locally
```bash
docker build -t paddleocr-vl:latest .
docker run --gpus all -it -p 8000:8000 paddleocr-vl:latest
```

### Deploy to RunPod
1. Push to GitHub or upload to RunPod
2. Create Serverless Endpoint with this Dockerfile
3. Wait for pod to be ready
4. Test with curl or Python

### API Usage

**OCR from URL:**
```bash
curl -X POST https://api.runpod.io/v1/<ENDPOINT_ID>/run \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "image_url": "https://..."
    }
  }'
```

**OCR from base64:**
```bash
curl -X POST https://api.runpod.io/v1/<ENDPOINT_ID>/run \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "image_base64": "base64_string",
      "mime_type": "image/png"
    }
  }'
```

**Different tasks:**
```bash
{
  "input": {
    "image_url": "https://...",
    "task": "table"  # ocr, table, formula, chart
  }
}
```

## Task Types

- `ocr` - Text recognition
- `table` - Table recognition
- `formula` - Formula recognition
- `chart` - Chart recognition
```

### 1.7 Structure của Project

```
paddleocr-vl-runpod/
├── handler.py           # Handler logic
├── Dockerfile          # Docker configuration
├── .dockerignore        # Docker ignore patterns
├── requirements.txt    # Python dependencies
├── README.md           # Documentation
└── .gitignore          # Git ignore patterns
```

---

## 🚀 BƯỚC 2: Push Code lên GitHub

### 2.1 Tạo GitHub Repository

1. Đi tới https://github.com/new
2. Đặt tên: `paddleocr-vl-runpod`
3. Chọn "Public" (dễ access)
4. Bỏ chọn "Add README.md" (vì ta đã có)
5. Click **Create repository**

### 2.2 Push Code

```bash
# Thêm remote
git remote add origin https://github.com/YOUR_USERNAME/paddleocr-vl-runpod.git

# Thêm files
git add .

# Commit
git commit -m "Initial commit: PaddleOCR-VL on RunPod"

# Push
git branch -M main
git push -u origin main
```

✅ Code giờ đã có trên GitHub!

---

## 🔧 BƯỚC 3: Tạo RunPod Account & Wallet

### 3.1 Đăng Ký RunPod

1. Truy cập https://runpod.io/
2. Click **Sign Up**
3. Đăng ký bằng email hoặc Google
4. Verify email

### 3.2 Setup Wallet/Credit

1. Đi tới **Account** → **Billing**
2. Thêm Payment Method (Credit Card)
3. Hoặc nạp RunPod Credit

**Giá dự tính:**
- RTX 4090: ~$0.52/hour
- L40: ~$0.44/hour
- A40-48GB: ~$0.36/hour

> Model PaddleOCR-VL cần ~3.7GB VRAM, nên GPU nhỏ cũng được

---

## 📦 BƯỚC 4: Tạo Custom Image trên RunPod

### 4.1 Build & Push Docker Image (Option A - Nếu Có Docker)

```bash
# Login to Docker Hub
docker login

# Build image
docker build -t YOUR_DOCKER_USERNAME/paddleocr-vl:latest .

# Push to Docker Hub
docker push YOUR_DOCKER_USERNAME/paddleocr-vl:latest
```

### 4.2 Tạo Image từ GitHub (Option B - Khuyến Nghị)

1. Đi tới https://www.runpod.io/console/serverless
2. Click **My Implementations** (bên trái)
3. Click **New Implementation**
4. Chọn **From GitHub**

![Step 1](https://i.imgur.com/placeholder.png)

5. **GitHub Details:**
   - **Repository URL:** `https://github.com/YOUR_USERNAME/paddleocr-vl-runpod`
   - **Branch:** `main`
   - **Dockerfile location:** `Dockerfile` (default)

6. Click **Analyze Dockerfile**
   - RunPod sẽ kiểm tra Dockerfile
   - Nếu có lỗi, fix và push lại

7. Click **Build Image**
   - Quá trình build sẽ mất 10-15 phút
   - Có thể monitor logs

✅ Image sẽ lưu dưới **My Implementations**

---

## 🎯 BƯỚC 5: Tạo Serverless Endpoint

### 5.1 Setup Endpoint

1. Vẫn ở trang **My Implementations**
2. Tìm image bạn vừa build
3. Click **Deploy** hoặc **Create New Endpoint**

### 5.2 Cấu Hình Endpoint

**Basic Settings:**
- **Endpoint Name:** `paddleocr-vl-prod` (hoặc tên khác)
- **Implementation:** Chọn image vừa build
- **GPU Count:** 1
- **GPU Type:** Chọn GPU (tuỳ budget)
  - RTX 4090 (tốt nhất)
  - L40 (cân bằng)
  - A40 (rẻ hơn)

**Advanced Settings:**
- **Max Workers:** 3 (xử lý tối đa 3 request cùng lúc)
- **Min Workers:** 1 (luôn chạy ít nhất 1 worker)
- **Idle Timeout:** 5 phút (tắt nếu không dùng)
- **GPU Count Per Worker:** 1

![Configuration](https://i.imgur.com/placeholder.png)

### 5.3 Deploy

1. Review cấu hình
2. Click **Deploy**
3. Chờ endpoint khởi động (2-3 phút)

✅ Status sẽ là **ACTIVE** khi sẵn sàng

---

## 🧪 BƯỚC 6: Test Endpoint

### 6.1 Lấy Endpoint ID

1. Ở trang Serverless, tìm endpoint vừa tạo
2. Copy **Endpoint ID**

```
Ví dụ: abc123def456ghi789
```

### 6.2 Test Với Curl

**Test từ URL:**

```bash
curl -X POST https://api.runpod.io/v1/YOUR_ENDPOINT_ID/run \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "image_url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
    }
  }'
```

**Response sẽ như:**

```json
{
  "id": "abc-123-xyz",
  "status": "QUEUED"
}
```

### 6.3 Check Status

```bash
curl https://api.runpod.io/v1/YOUR_ENDPOINT_ID/status/abc-123-xyz
```

**Response:**

```json
{
  "id": "abc-123-xyz",
  "status": "COMPLETED",
  "result": {
    "status": "success",
    "task": "ocr",
    "mode": "url",
    "result": "Extracted text from image..."
  }
}
```

### 6.4 Test Với Python

```python
import requests
import json

ENDPOINT_ID = "YOUR_ENDPOINT_ID"
API_URL = f"https://api.runpod.io/v1/{ENDPOINT_ID}"

# Dữ liệu input
data = {
    "input": {
        "image_url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png",
        "task": "ocr"
    }
}

# Gửi request
response = requests.post(f"{API_URL}/run", json=data)
request_id = response.json()["id"]

print(f"Request ID: {request_id}")

# Check status
import time
while True:
    status_response = requests.get(f"{API_URL}/status/{request_id}")
    status_data = status_response.json()
    
    print(f"Status: {status_data['status']}")
    
    if status_data['status'] == "COMPLETED":
        print(f"\nResult:")
        print(json.dumps(status_data['result'], indent=2, ensure_ascii=False))
        break
    
    time.sleep(2)
```

**Chạy test:**

```bash
python test_endpoint.py
```

---

## 🔐 BƯỚC 7: Cấu Hình Authentication (Optional)

Nếu muốn bảo vệ endpoint:

1. Ở trang endpoint, tìm **Secuity**
2. Enable **API Key Protection**
3. Sao chép API Key

**Khi đó, requests cần thêm header:**

```bash
curl -X POST https://api.runpod.io/v1/YOUR_ENDPOINT_ID/run \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{...}'
```

---

## 💰 BƯỚC 8: Monitor Chi Phí

### 8.1 Xem Chi Phí

1. Đi tới **Account** → **Usage**
2. Xem chi phí theo ngày/giờ

### 8.2 Tiết Kiệm Chi Phí

**Cách 1: Tắt Idle Timeout**
- Đặt thấp để pod không chạy khi không dùng
- Nhưng startup mỗi lần sẽ chậm

**Cách 2: Dùng Spot GPU**
- Rẻ hơn 70%
- Nhưng có thể bị tắt bất cứ lúc nào

**Cách 3: Optimize Model**
- Dùng GPU nhỏ hơn
- Reduce batch size

---

## 🐛 BƯỚC 9: Troubleshooting

### Problem 1: Pod không khởi động

**Kiểm tra:**
```
1. Xem logs: Click endpoint → View Logs
2. Tìm error message
3. Fix Dockerfile và rebuild
```

### Problem 2: vLLM server không khởi động

**Symptom:** Pod chạy nhưng handler error

**Solution:**
```bash
# Thêm debug log vào Dockerfile
CMD bash -c "\
    echo 'Starting vLLM...' && \
    vllm serve PaddlePaddle/PaddleOCR-VL \
        --trust-remote-code \
        --max-num-batched-tokens 16384 \
        --no-enable-prefix-caching \
        --mm-processor-cache-gb 0 \
        --port 8000 \
        --host 0.0.0.0 \
        --gpu-memory-utilization 0.8 &\
    sleep 60 && \
    python3 /workspace/handler.py"
```

### Problem 3: Out of Memory

**Solution:**
```dockerfile
# Giảm batch size
CMD bash -c "\
    vllm serve PaddlePaddle/PaddleOCR-VL \
        --trust-remote-code \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.7 ..."
```

### Problem 4: Slow Response

**Solution:**
```
1. Tăng Max Workers
2. Chọn GPU mạnh hơn
3. Tăng gpu-memory-utilization
4. Giảm idle timeout
```

---

## ✅ Checklist Deployment

- [ ] Tạo files (handler.py, Dockerfile, etc.)
- [ ] Push lên GitHub
- [ ] Tạo RunPod account
- [ ] Thêm payment method
- [ ] Build Docker image
- [ ] Tạo Serverless endpoint
- [ ] Test bằng curl
- [ ] Test bằng Python
- [ ] Check logs & performance
- [ ] Monitor chi phí

---

## 📊 Giám Sát & Maintenance

### Hàng Ngày:
- Check endpoint status
- Monitor logs
- Track API usage

### Hàng Tuần:
- Review chi phí
- Check error rates
- Update code nếu cần

### Hàng Tháng:
- Optimize performance
- Update dependencies
- Scale up/down worker count

---

## 🎓 Tips & Best Practices

1. **Startup Time:** vLLM cần ~45 giây để load model, đảm bảo timeout đủ
2. **Batch Processing:** Sử dụng `max-num-batched-tokens` để xử lý nhiều requests
3. **Cost:** Dùng spot GPU để tiết kiệm 70%
4. **Monitoring:** Setup alerts cho error rate
5. **Versioning:** Dùng Git tags để track versions

---

## 🔗 Links Hữu Ích

- RunPod Dashboard: https://www.runpod.io/console/serverless
- vLLM Docs: https://docs.vllm.ai/
- PaddleOCR-VL: https://huggingface.co/PaddlePaddle/PaddleOCR-VL
- RunPod API Docs: https://docs.runpod.io/serverless

---

## 💬 Câu Hỏi Thường Gặp

**Q: Mất bao lâu để deploy?**
A: ~20-30 phút (build image 10-15 phút, startup 5-10 phút)

**Q: Chi phí là bao nhiêu?**
A: ~$0.36-0.52/hour tùy GPU, có thể rẻ hơn 70% nếu dùng Spot

**Q: Model download tự động không?**
A: Có, lần đầu sẽ auto download ~3.7GB vào container

**Q: Có thể cancel request không?**
A: Có, dùng endpoint cancel API

**Q: Timeout bao lâu?**
A: Mặc định 3600s (1 giờ), đủ cho hầu hết requests

---

## 📝 Ghi Chú

- Lần đầu khởi động chậm do load model
- Giữ pod run 24/7 tốn chi phí, sử dụng Idle Timeout
- Test thường xuyên để tránh đột ngột fail
- Backup code trên GitHub, không chỉ local

---

**Mời bạn làm theo từng bước! Nếu gặp vấn đề ở đâu, hãy nói chi tiết để tôi giúp.**
