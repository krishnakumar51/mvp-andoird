import asyncio
import platform
import uuid
import json
import time
import csv
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
from config import SCREENSHOTS_DIR, ANTHROPIC_MODEL, GROQ_MODEL, OPENAI_MODEL

# --- FastAPI App Initialization ---
app = FastAPI(title="LangGraph Web Agent with Memory")

# --- In-Memory Job Storage ---
JOB_QUEUES = {}
JOB_RESULTS = {}

# --- NEW: Token Cost Analysis Configuration ---
ANALYSIS_DIR = Path("analysis")
REPORT_CSV_FILE = Path("report.csv")


# Prices per 1 Million tokens
TOKEN_COSTS = {
    "anthropic": {
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-4-sonnet-20240229": {"input": 3.0, "output": 15.0},
        "claude-3.5-sonnet-20240620": {"input": 3.0, "output": 15.0}
    },
    "openai": {
        "gpt-4o": {"input": 5.0, "output": 15.0}
    },
    "groq": {
        "llama3-8b-8192": {"input": 0.05, "output": 0.10}
    }
}

MODEL_MAPPING = {
    LLMProvider.ANTHROPIC: ANTHROPIC_MODEL,
    LLMProvider.GROQ: GROQ_MODEL,
    LLMProvider.OPENAI: OPENAI_MODEL
}

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

# --- NEW: Cost Analysis Function ---
def save_analysis_report(analysis_data: dict):
    """Calculates final costs, saves a detailed JSON report, and appends to a summary CSV."""
    job_id = analysis_data["job_id"]
    provider = analysis_data["provider"]
    model = analysis_data["model"]
    
    total_input = 0
    total_output = 0
    
    for step in analysis_data["steps"]:
        total_input += step.get("input_tokens", 0)
        total_output += step.get("output_tokens", 0)

    analysis_data["total_input_tokens"] = total_input
    analysis_data["total_output_tokens"] = total_output

    cost_info = TOKEN_COSTS.get(provider, {}).get(model)
    # --- MODIFIED: Add a more robust fallback for different Anthropic model names ---
    if not cost_info and provider == "anthropic":
        model_name_lower = model.lower()
        if "sonnet" in model_name_lower:
            # Default to the latest Sonnet pricing if a specific version isn't matched
            cost_info = TOKEN_COSTS.get("anthropic", {}).get("claude-3.5-sonnet-20240620")
        elif "haiku" in model_name_lower:
            cost_info = TOKEN_COSTS.get("anthropic", {}).get("claude-3-haiku-20240307")


    total_cost = 0.0
    if cost_info:
        input_cost = (total_input / 1_000_000) * cost_info["input"]
        output_cost = (total_output / 1_000_000) * cost_info["output"]
        total_cost = input_cost + output_cost
    
    # Format the cost to a string with 5 decimal places to ensure precision in output files.
    total_cost_usd_str = f"{total_cost:.5f}"
    analysis_data["total_cost_usd"] = total_cost_usd_str

    # 1. Save detailed JSON report in analysis/ directory
    try:
        ANALYSIS_DIR.mkdir(exist_ok=True)
        json_report_path = ANALYSIS_DIR / f"{job_id}.json"
        with open(json_report_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON analysis report for job {job_id}: {e}")

    # 2. Append summary to report.csv
    try:
        file_exists = REPORT_CSV_FILE.is_file()
        with open(REPORT_CSV_FILE, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['job_id', 'total_input_tokens', 'total_output_tokens', 'total_cost_usd']
            if not file_exists:
                writer.writerow(header)
            
            row = [job_id, total_input, total_output, total_cost_usd_str]
            writer.writerow(row)
    except Exception as e:
        print(f"Error updating CSV report: {e}")


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
    history: List[str] 
    token_usage: List[dict] # NEW: To store token usage per step

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

    # MODIFIED: Capture token usage from agent action
    action_response, usage = get_agent_action(
        query=state['refined_query'],
        url=state['page'].url,
        html=state['page'].content(),
        provider=state['provider'],
        screenshot_path=screenshot_path,
        history="\n".join(state['history'])
    )
    
    # NEW: Store usage for this step
    state['token_usage'].append({
        "task": f"agent_step_{state['step']}",
        **usage
    })

    push_status(job_id, "agent_thought", {
        "thought": action_response.get("thought", "No thought provided."),
        "usage": usage
    })
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
        state['history'].append(f"Step {state['step']}: Action `{json.dumps(action)}` executed successfully.")

    except Exception as e:
        error_message = str(e).splitlines()[0] 
        push_status(job_id, "action_failed", {"action": action, "error": error_message})
        state['history'].append(f"Step {state['step']}: Action `{json.dumps(action)}` FAILED with error: '{error_message}'")
        
    state['step'] += 1
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
    provider = payload["llm_provider"]
    job_analysis = {
        "job_id": job_id,
        "timestamp": get_current_timestamp(),
        "provider": provider,
        "model": MODEL_MAPPING.get(provider, "unknown"),
        "query": payload["query"],
        "url": payload["url"],
        "steps": []
    }
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = browser.new_page()
        final_result = {}
        final_state = {}
        try:
            push_status(job_id, "job_started", {"provider": provider, "query": payload["query"]})
            
            # MODIFIED: Capture token usage from prompt refinement
            refined_query, usage = get_refined_prompt(payload["url"], payload["query"], provider)
            job_analysis["steps"].append({"task": "refine_prompt", **usage})
            push_status(job_id, "prompt_refined", {"refined_query": refined_query, "usage": usage})

            initial_state = AgentState(
                job_id=job_id, browser=browser, page=page, query=payload["url"],
                top_k=payload["top_k"], provider=provider,
                refined_query=refined_query, results=[], screenshots=[],
                job_artifacts_dir=SCREENSHOTS_DIR / job_id,
                step=1, max_steps=15, last_action={},
                history=[],
                token_usage=[] # Initialize empty token usage list
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
            
            # NEW: Aggregate and save analysis report
            if final_state:
                job_analysis["steps"].extend(final_state.get('token_usage', []))
            save_analysis_report(job_analysis)
            
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

