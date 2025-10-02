import asyncio
import platform
import uuid
import json
import time
from pathlib import Path
from urllib.parse import urljoin
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from playwright.sync_api import sync_playwright, Error as PlaywrightError
from PIL import Image # Import the Image library from Pillow

from llm import (
    LLMProvider,
    get_refined_prompt,
    get_llm_response,
    extract_json_from_response,
    PLANNER_PROMPT,
    EXTRACTOR_PROMPT
)
from config import SCREENSHOTS_DIR

# --- FastAPI App Initialization ---
app = FastAPI(title="Reliable Web Scraper Agent")

# --- In-Memory Job Storage (for this MVP) ---
JOB_QUEUES = {}
JOB_RESULTS = {}

# --- Helper Functions ---
def get_current_timestamp():
    """Returns the current UTC timestamp in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def push_status(job_id: str, msg: str, details: dict = None):
    """Pushes a status update to the job's queue for real-time streaming."""
    q = JOB_QUEUES.get(job_id)
    if q:
        entry = {"ts": get_current_timestamp(), "msg": msg}
        if details:
            entry["details"] = details
        q.put_nowait(entry)

def resize_image_if_needed(image_path: Path, max_dimension: int = 7500):
    """
    Checks if an image's dimensions exceed the max size and resizes it if necessary,
    preserving the aspect ratio. The resized image overwrites the original.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                resized_img.save(image_path)
    except Exception as e:
        # If resizing fails, we just log it and proceed with the original image.
        print(f"Warning: Could not resize image {image_path}. Error: {e}")


# --- API Models ---
class SearchRequest(BaseModel):
    url: str
    query: str
    top_k: int = 5
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC

# --- The Core Scraping Logic ---
def run_job(job_id: str, payload: dict):
    """The main worker function that orchestrates the entire scraping process."""
    url = payload["url"]
    query = payload["query"]
    top_k = payload["top_k"]
    provider = payload["llm_provider"]
    
    job_artifacts_dir = SCREENSHOTS_DIR / job_id
    job_artifacts_dir.mkdir(exist_ok=True)
    
    result = {"job_id": job_id, "results": [], "screenshots": []}
    
    try:
        push_status(job_id, "job_started", {"provider": provider, "query": query})

        # === Step 1: Refine Prompt ===
        push_status(job_id, "refining_prompt")
        refined_query = get_refined_prompt(url, query, provider)
        push_status(job_id, "prompt_refined", {"refined_query": refined_query})

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
            context = browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            page = context.new_page()

            # === Step 2: Navigate and Snapshot ===
            push_status(job_id, "navigating", {"url": url})
            page.goto(url, wait_until='domcontentloaded', timeout=60000)
            page.wait_for_timeout(3000)

            initial_screenshot_path = job_artifacts_dir / "01_initial_page.png"
            page.screenshot(path=initial_screenshot_path, full_page=True)
            resize_image_if_needed(initial_screenshot_path)
            result["screenshots"].append(f"screenshots/{job_id}/01_initial_page.png")
            
            # === Step 3: Plan Actions ===
            push_status(job_id, "planning_actions")
            actions = []
            try:
                planner_prompt = PLANNER_PROMPT.format(
                    query=refined_query, url=page.url,
                    html=page.content()[:20000],
                    accessibility=json.dumps(page.accessibility.snapshot() or {})[:10000]
                )
                planner_response = get_llm_response(planner_prompt, "You are a web automation planner.", provider, images=[initial_screenshot_path])
                actions = extract_json_from_response(planner_response)
                if not isinstance(actions, list): actions = []
            except Exception as e:
                push_status(job_id, "planner_failed", {"error": str(e)})

            push_status(job_id, "plan_generated", {"actions_count": len(actions)})

            # === Step 4: Execute Actions ===
            if actions:
                for action in actions:
                    push_status(job_id, "executing_action", {"action": action})
                    try:
                        action_type = action.get("type")
                        selector = action.get("selector")
                        if not all([action_type, selector]): continue
                        
                        if action_type == "fill":
                            page.locator(selector).fill(action.get("text", ""))
                        elif action_type == "click":
                            page.locator(selector).click()
                        elif action_type == "press":
                            page.locator(selector).press(action.get("key", "Enter"))
                        page.wait_for_timeout(1500)
                    except Exception as e:
                        push_status(job_id, "action_failed", {"action": action, "error": str(e)})

            # === NEW: Step 4.5: Scroll and Capture for Better Context ===
            push_status(job_id, "scrolling_and_capturing")
            action_screenshot_paths = []
            for i in range(3): # Scroll 3 times
                screenshot_path = job_artifacts_dir / f"02_after_actions_scroll_{i}.png"
                page.screenshot(path=screenshot_path) # Viewport screenshot
                resize_image_if_needed(screenshot_path)
                action_screenshot_paths.append(screenshot_path)
                result["screenshots"].append(f"screenshots/{job_id}/02_after_actions_scroll_{i}.png")
                
                page.evaluate("window.scrollBy(0, window.innerHeight)")
                page.wait_for_timeout(1000) # Wait for content to load after scroll

            # === Step 5: Extract Data ===
            push_status(job_id, "extracting_data")
            try:
                extractor_prompt = EXTRACTOR_PROMPT.format(
                    query=refined_query, url=page.url, top_k=top_k,
                    html=page.content()[:40000]
                )
                # Pass all the scrolled screenshots to the extractor
                extractor_response = get_llm_response(extractor_prompt, "You are a data extraction specialist.", provider, images=action_screenshot_paths)
                extracted_data = extract_json_from_response(extractor_response)
                items = extracted_data.get("items", [])
                
                for item in items:
                    if 'url' in item and isinstance(item.get('url'), str):
                        item['url'] = urljoin(page.url, item['url'])
                result["results"] = items[:top_k]
                push_status(job_id, "extraction_complete", {"items_found": len(items)})
            except Exception as e:
                push_status(job_id, "extraction_failed", {"error": str(e)})

            browser.close()

    except Exception as e:
        push_status(job_id, "job_failed", {"error": str(e), "trace": traceback.format_exc()})
        result["error"] = str(e)
    
    JOB_RESULTS[job_id] = result
    push_status(job_id, "job_done")

# --- FastAPI Endpoints ---
@app.post("/search")
async def start_search(req: SearchRequest):
    job_id = str(uuid.uuid4())
    JOB_QUEUES[job_id] = asyncio.Queue()
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_job, job_id, req.dict())
    return {"job_id": job_id, "stream_url": f"/stream/{job_id}", "result_url": f"/result/{job_id}"}

@app.get("/stream/{job_id}")
async def stream_status(job_id: str):
    if job_id not in JOB_QUEUES:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def event_generator():
        q = JOB_QUEUES[job_id]
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=60)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["msg"] in ("job_done", "job_failed"): break
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in JOB_RESULTS:
        return JSONResponse({"status": "pending"}, status_code=202)
    return JSONResponse(JOB_RESULTS[job_id])

@app.get("/screenshots/{job_id}/{filename}")
async def get_screenshot(job_id: str, filename: str):
    file_path = SCREENSHOTS_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Screenshot not found")
    return FileResponse(file_path)

@app.get("/")
async def client_ui():
    return FileResponse(Path(__file__).parent / "static/test_client.html")

