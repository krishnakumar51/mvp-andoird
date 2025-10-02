import asyncio
import platform
import uuid
import json
import time
from pathlib import Path
from urllib.parse import urljoin
import traceback
from typing import List, TypedDict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from playwright.sync_api import sync_playwright, Page, Browser
from PIL import Image
from langgraph.graph import StateGraph, END

from llm import LLMProvider, get_refined_prompt, get_agent_action
from config import SCREENSHOTS_DIR

# --- FastAPI App Initialization ---
app = FastAPI(title="LangGraph Web Agent with Memory")

# --- In-Memory Job Storage ---
JOB_QUEUES = {}
JOB_RESULTS = {}

# --- Helper Functions ---
def get_current_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def push_status(job_id: str, msg: str, details: dict = None):
    q = JOB_QUEUES.get(job_id)
    if q:
        entry = {"ts": get_current_timestamp(), "msg": msg}
        if details: entry["details"] = details
        q.put_nowait(entry)

def resize_image_if_needed(image_path: Path, max_dimension: int = 2000):
    try:
        with Image.open(image_path) as img:
            if max(img.size) > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
                img.save(image_path)
    except Exception as e:
        print(f"Warning: Could not resize image {image_path}. Error: {e}")

# --- API Models ---
class SearchRequest(BaseModel):
    url: str
    query: str
    top_k: int
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC

# --- LangGraph Agent State with Memory ---
class AgentState(TypedDict):
    job_id: str
    browser: Browser
    page: Page
    query: str
    top_k: int
    provider: LLMProvider
    refined_query: str
    results: List[dict]
    screenshots: List[str]
    job_artifacts_dir: Path
    step: int
    max_steps: int
    last_action: dict
    history: List[str] # NEW: Agent's memory

# --- LangGraph Nodes ---
def navigate_to_page(state: AgentState) -> AgentState:
    state['page'].goto(state['query'], wait_until='domcontentloaded', timeout=60000)
    push_status(state['job_id'], "navigation_complete", {"url": state['query']})
    return state

def agent_reasoning_node(state: AgentState) -> AgentState:
    job_id = state['job_id']
    push_status(job_id, "agent_step", {"step": state['step'], "max_steps": state['max_steps']})
    
    screenshot_path = state['job_artifacts_dir'] / f"{state['step']:02d}_step.png"
    state['page'].screenshot(path=screenshot_path)
    resize_image_if_needed(screenshot_path)
    state['screenshots'].append(f"screenshots/{job_id}/{state['step']:02d}_step.png")

    action_response = get_agent_action(
        query=state['refined_query'],
        url=state['page'].url,
        html=state['page'].content(),
        provider=state['provider'],
        screenshot_path=screenshot_path,
        history="\n".join(state['history']) # Pass the memory to the agent
    )
    
    push_status(job_id, "agent_thought", {"thought": action_response.get("thought", "No thought provided.")})
    state['last_action'] = action_response.get("action", {"type": "finish", "reason": "Agent failed to produce a valid action."})
    return state

def execute_action_node(state: AgentState) -> AgentState:
    job_id = state['job_id']
    action = state['last_action']
    page = state['page']
    
    push_status(job_id, "executing_action", {"action": action})
    
    try:
        action_type = action.get("type")
        if action_type == "click":
            page.locator(action["selector"]).click(timeout=5000)
        elif action_type == "fill":
            page.locator(action["selector"]).fill(action["text"])
        elif action_type == "press":
            page.locator(action["selector"]).press(action["key"])
        elif action_type == "scroll":
            page.evaluate("window.scrollBy(0, window.innerHeight)")
        elif action_type == "extract":
            items = action.get("items", [])
            for item in items:
                if 'url' in item and isinstance(item.get('url'), str):
                    item['url'] = urljoin(page.url, item['url'])
            state['results'].extend(items)
            push_status(job_id, "partial_result", {"new_items_found": len(items), "total_items": len(state['results'])})
        
        page.wait_for_timeout(2000)
        # NEW: Record successful action in history
        state['history'].append(f"Step {state['step']}: Action `{json.dumps(action)}` executed successfully.")

    except Exception as e:
        error_message = str(e).splitlines()[0] # Get a concise error message
        push_status(job_id, "action_failed", {"action": action, "error": error_message})
        # NEW: Record failed action in history
        state['history'].append(f"Step {state['step']}: Action `{json.dumps(action)}` FAILED with error: '{error_message}'")
        
    state['step'] += 1
    # NEW: Keep history concise (last 5 actions)
    state['history'] = state['history'][-5:]
    return state

# --- LangGraph Supervisor Logic ---
def supervisor_node(state: AgentState) -> str:
    if state['last_action'].get("type") == "finish":
        push_status(state['job_id'], "agent_finished", {"reason": state['last_action'].get("reason")})
        return END
    if len(state['results']) >= state['top_k']:
        push_status(state['job_id'], "agent_finished", {"reason": f"Collected {len(state['results'])}/{state['top_k']} items."})
        return END
    if state['step'] > state['max_steps']:
        push_status(state['job_id'], "agent_stopped", {"reason": "Max steps reached."})
        return END
    return "continue"

# --- Build the Graph ---
builder = StateGraph(AgentState)
builder.add_node("navigate", navigate_to_page)
builder.add_node("reason", agent_reasoning_node)
builder.add_node("execute", execute_action_node)
builder.set_entry_point("navigate")
builder.add_edge("navigate", "reason")
builder.add_conditional_edges("execute", supervisor_node, {END: END, "continue": "reason"})
builder.add_edge("reason", "execute")
graph_app = builder.compile()

# --- The Core Job Orchestrator ---
def run_job(job_id: str, payload: dict):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = browser.new_page()
        final_result = {}
        try:
            push_status(job_id, "job_started", {"provider": payload["llm_provider"], "query": payload["query"]})
            
            refined_query = get_refined_prompt(payload["url"], payload["query"], payload["llm_provider"])
            push_status(job_id, "prompt_refined", {"refined_query": refined_query})

            initial_state = AgentState(
                job_id=job_id, browser=browser, page=page, query=payload["url"],
                top_k=payload["top_k"], provider=payload["llm_provider"],
                refined_query=refined_query, results=[], screenshots=[],
                job_artifacts_dir=SCREENSHOTS_DIR / job_id,
                step=1, max_steps=15, last_action={},
                history=[] # Initialize empty memory
            )
            initial_state['job_artifacts_dir'].mkdir(exist_ok=True)
            
            final_state = graph_app.invoke(initial_state)

            final_result = {"job_id": job_id, "results": final_state['results'], "screenshots": final_state['screenshots']}
        except Exception as e:
            push_status(job_id, "job_failed", {"error": str(e), "trace": traceback.format_exc()})
            final_result["error"] = str(e)
        finally:
            JOB_RESULTS[job_id] = final_result
            push_status(job_id, "job_done")
            page.close()
            browser.close()

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
    q = JOB_QUEUES.get(job_id)
    if not q: raise HTTPException(status_code=404, detail="Job not found")
    async def event_generator():
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=60)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["msg"] in ("job_done", "job_failed"): break
            except asyncio.TimeoutError: yield ": keep-alive\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    result = JOB_RESULTS.get(job_id)
    if not result: return JSONResponse({"status": "pending"}, status_code=202)
    return JSONResponse(result)

@app.get("/screenshots/{job_id}/{filename}")
async def get_screenshot(job_id: str, filename: str):
    file_path = SCREENSHOTS_DIR / job_id / filename
    if not file_path.exists(): raise HTTPException(status_code=404, detail="Screenshot not found")
    return FileResponse(file_path)

@app.get("/")
async def client_ui():
    return FileResponse(Path(__file__).parent / "static/test_client.html")

