import runpod
from openai import OpenAI
import requests
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_HEALTH_URL = "http://localhost:8000/health"
VLLM_READY_TIMEOUT = 600  # 10 phút — đủ cho lần đầu download model

TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


def wait_for_vllm():
    """Chờ vLLM server healthy trước khi handler nhận request."""
    logger.info("Waiting for vLLM to be ready (timeout=%ds)...", VLLM_READY_TIMEOUT)
    deadline = time.time() + VLLM_READY_TIMEOUT
    while time.time() < deadline:
        try:
            r = requests.get(VLLM_HEALTH_URL, timeout=5)
            if r.status_code == 200:
                logger.info("vLLM is ready!")
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"vLLM did not become ready within {VLLM_READY_TIMEOUT}s")


def get_client() -> OpenAI:
    return OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL, timeout=300)


def process_image(image_data: str, prompt: str, is_url: bool = False, mime_type: str = "image/jpeg") -> str:
    if is_url:
        img_content = {"type": "image_url", "image_url": {"url": image_data}}
    else:
        img_content = {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}

    response = get_client().chat.completions.create(
        model="PaddlePaddle/PaddleOCR-VL",
        messages=[{
            "role": "user",
            "content": [img_content, {"type": "text", "text": prompt}],
        }],
        temperature=0.0,
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""


def handler(event):
    try:
        input_data = event.get("input", {})
        if not input_data:
            return {"status": "error", "error": "Missing input data"}

        # Prompt: custom_prompt takes priority, else task prefix
        if "custom_prompt" in input_data:
            prompt = input_data["custom_prompt"]
        else:
            task = input_data.get("task", "ocr")
            if task not in TASKS:
                return {"status": "error", "error": f"Unknown task: {task}. Supported: {list(TASKS.keys())}"}
            prompt = TASKS[task]

        if "image_url" in input_data:
            result = process_image(input_data["image_url"], prompt, is_url=True)
        elif "image_base64" in input_data:
            mime_type = input_data.get("mime_type", "image/jpeg")
            result = process_image(input_data["image_base64"], prompt, is_url=False, mime_type=mime_type)
        else:
            return {"status": "error", "error": "Need 'image_url' or 'image_base64' in input"}

        return {"status": "success", "result": result}

    except Exception as e:
        logger.error("Handler error: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    wait_for_vllm()  # Chờ vLLM trước khi RunPod đánh dấu worker là ready
    runpod.serverless.start({"handler": handler})
