import runpod
from openai import OpenAI
import base64
import json
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_client():
    return OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        timeout=3600
    )


TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


def process_image_url(image_url: str, task: str = "ocr"):
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
                        "image_url": {"url": image_url}
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
        logger.info("Processing completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error processing image from URL: {str(e)}")
        raise


def process_image_base64(image_base64: str, task: str = "ocr", mime_type: str = "image/png"):
    try:
        logger.info("Processing image from base64")
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
        logger.info("Processing completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}")
        raise


def handler(event):
    """
    RunPod Serverless handler.

    Input format:
    {
        "input": {
            "image_url": "https://...",       # hoặc
            "image_base64": "base64string",
            "task": "ocr",                    # tuỳ chọn: ocr, table, formula, chart
            "mime_type": "image/png"          # nếu dùng base64
        }
    }
    """
    try:
        logger.info("=" * 60)
        logger.info("Handler called")
        logger.info("=" * 60)

        input_data = event.get("input", {})

        if not input_data:
            return {"status": "error", "error": "Thiếu input data"}

        task = input_data.get("task", "ocr")

        if task not in TASKS:
            return {
                "status": "error",
                "error": f"Task không hỗ trợ: {task}. Hỗ trợ: {list(TASKS.keys())}"
            }

        if "image_url" in input_data:
            logger.info("Mode: URL")
            result = process_image_url(input_data["image_url"], task)
            return {"status": "success", "task": task, "mode": "url", "result": result}

        elif "image_base64" in input_data:
            mime_type = input_data.get("mime_type", "image/png")
            logger.info("Mode: base64")
            result = process_image_base64(input_data["image_base64"], task, mime_type)
            return {"status": "success", "task": task, "mode": "base64", "result": result}

        else:
            return {
                "status": "error",
                "error": "Cần cung cấp 'image_url' hoặc 'image_base64' trong input"
            }

    except Exception as e:
        logger.error(f"Handler error: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    logger.info("Starting RunPod Serverless Handler...")
    runpod.serverless.start({"handler": handler})
