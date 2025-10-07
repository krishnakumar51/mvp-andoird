import asyncio
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
from PIL import Image
from langgraph.graph import StateGraph, END

from appium import webdriver
from appium.options.android import UiAutomator2Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, WebDriverException

from llm import LLMProvider, get_refined_prompt, get_agent_action
from config import SCREENSHOTS_DIR, ANTHROPIC_MODEL, GROQ_MODEL, OPENAI_MODEL

app = FastAPI(title="LangGraph Android Web Agent")
JOB_QUEUES, JOB_RESULTS = {}, {}
ANALYSIS_DIR, REPORT_CSV_FILE = Path("analysis"), Path("report.csv")
TOKEN_COSTS = { "anthropic": { "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25}, "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0}, "claude-3.5-sonnet-20240620": {"input": 3.0, "output": 15.0} }, "openai": {"gpt-4o": {"input": 5.0, "output": 15.0}}, "groq": {"llama3-8b-8192": {"input": 0.05, "output": 0.10}} }
MODEL_MAPPING = { LLMProvider.ANTHROPIC: ANTHROPIC_MODEL, LLMProvider.GROQ: GROQ_MODEL, LLMProvider.OPENAI: OPENAI_MODEL }
APPIUM_SERVER_URL = "http://localhost:4723"

# --- Helper and Analysis Functions (Unchanged) ---
def get_current_timestamp(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def push_status(job_id: str, msg: str, details: dict = None):
    q = JOB_QUEUES.get(job_id);
    if q: q.put_nowait({"ts": get_current_timestamp(), "msg": msg, **({"details": details} if details else {})})
def resize_image_if_needed(path: Path):
    try:
        with Image.open(path) as img:
            if max(img.size) > 2000: img.thumbnail((2000, 2000), Image.LANCZOS); img.save(path)
    except Exception as e: print(f"Warning: Could not resize {path}. Error: {e}")
def save_analysis_report(data: dict):
    # This function remains unchanged.
    job_id, provider, model = data["job_id"], data["provider"], data["model"]
    total_input = sum(s.get("input_tokens", 0) for s in data["steps"])
    total_output = sum(s.get("output_tokens", 0) for s in data["steps"])
    data.update({"total_input_tokens": total_input, "total_output_tokens": total_output})
    cost_key = next((k for k in TOKEN_COSTS.get(provider, {}) if model.lower() in k.lower()), None)
    cost_info = TOKEN_COSTS.get(provider, {}).get(cost_key)
    total_cost = ((total_input / 1e6) * cost_info["input"] + (total_output / 1e6) * cost_info["output"]) if cost_info else 0.0
    data["total_cost_usd"] = f"{total_cost:.5f}"
    try:
        ANALYSIS_DIR.mkdir(exist_ok=True)
        with open(ANALYSIS_DIR / f"{job_id}.json", 'w') as f: json.dump(data, f, indent=2)
        file_exists = REPORT_CSV_FILE.is_file()
        with open(REPORT_CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(['job_id', 'total_input_tokens', 'total_output_tokens', 'total_cost_usd'])
            writer.writerow([job_id, total_input, total_output, data["total_cost_usd"]])
    except Exception as e: print(f"Error saving analysis: {e}")

# --- Pop-up handler removed, as this logic is now in the agent's brain ---

class SearchRequest(BaseModel): url: str; query: str; top_k: int; llm_provider: LLMProvider = LLMProvider.ANTHROPIC
class AgentState(TypedDict):
    job_id: str; driver: webdriver.Remote; query: str; top_k: int; provider: LLMProvider
    refined_query: str; results: List[dict]; screenshots: List[str]; job_artifacts_dir: Path
    step: int; max_steps: int; last_action: dict; history: List[str]; token_usage: List[dict]

def navigate_to_page(state: AgentState) -> AgentState:
    state['driver'].get(state['query']); time.sleep(3)
    # The agent is now responsible for handling any pop-ups that appear.
    push_status(state['job_id'], "navigation_complete", {"url": state['driver'].current_url})
    return state

def agent_reasoning_node(state: AgentState) -> AgentState:
    job_id, driver = state['job_id'], state['driver']
    push_status(job_id, "agent_step", {"step": state['step'], "max_steps": state['max_steps']})
    screenshot_path = state['job_artifacts_dir'] / f"{state['step']:02d}_step.png"
    driver.get_screenshot_as_file(str(screenshot_path)); resize_image_if_needed(screenshot_path)
    state['screenshots'].append(f"screenshots/{job_id}/{state['step']:02d}_step.png")
    action_response, usage = get_agent_action(query=state['refined_query'], url=driver.current_url, html=driver.page_source, provider=state['provider'], screenshot_path=screenshot_path, history="\n".join(state['history']))
    state['token_usage'].append({"task": f"agent_step_{state['step']}", **usage})
    push_status(job_id, "agent_thought", {"thought": action_response.get("thought", ""), "usage": usage})
    state['last_action'] = action_response.get("action", {"type": "finish", "reason": "Agent failed to produce an action."})
    return state

def execute_action_node(state: AgentState) -> AgentState:
    job_id, action, driver = state['job_id'], state['last_action'], state['driver']
    push_status(job_id, "executing_action", {"action": action})
    try:
        action_type, selector = action.get("type"), action.get("selector")
        wait = WebDriverWait(driver, 10)
        
        if action_type == "tap":
            element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            element.click()
        
        elif action_type == "fill_and_submit":
            element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            element.click()
            element.clear()
            element.send_keys(action["text"])
            element.send_keys(Keys.ENTER)
            wait.until(lambda d: "q=" in d.current_url)

        elif action_type == "scroll":
            size = driver.get_window_size()
            start_y, end_y = int(size['height'] * 0.8), int(size['height'] * 0.2)
            start_x = size['width'] // 2
            driver.execute_script("mobile: scrollGesture", {'left': start_x, 'top': start_y, 'width': 0, 'height': start_y - end_y, 'direction': 'down', 'percent': 1.0})
        
        elif action_type == "extract":
            items = action.get("items", [])
            for item in items:
                if 'url' in item and isinstance(item.get('url'), str):
                    item['url'] = urljoin(driver.current_url, item['url'])
            state['results'].extend(items)
            push_status(job_id, "partial_result", {"new_items_found": len(items)})

        time.sleep(2)
        state['history'].append(f"Step {state['step']}: Action `{json.dumps(action)}` successful.")
    except Exception as e:
        # --- MODIFIED: Improved Error Logging ---
        error_message = str(e).splitlines()[0] if str(e).strip() else f"An unspecified error occurred: {type(e).__name__}"
        push_status(job_id, "action_failed", {"action": action, "error": error_message})
        state['history'].append(f"Step {state['step']}: Action `{json.dumps(action)}` FAILED: {error_message}")

    state['step'] += 1; state['history'] = state['history'][-5:]
    return state

def supervisor_node(state: AgentState) -> str:
    if state['last_action'].get("type") == "finish" or len(state['results']) >= state['top_k'] or state['step'] > state['max_steps']: return END
    return "continue"

builder = StateGraph(AgentState); builder.add_node("navigate", navigate_to_page); builder.add_node("reason", agent_reasoning_node)
builder.add_node("execute", execute_action_node); builder.set_entry_point("navigate"); builder.add_edge("navigate", "reason")
builder.add_conditional_edges("execute", supervisor_node, {END: END, "continue": "reason"}); builder.add_edge("reason", "execute")
graph_app = builder.compile()

def run_job(job_id: str, payload: dict):
    # This function's core logic and Appium options remain the same.
    provider = payload["llm_provider"]
    job_analysis = { "job_id": job_id, "timestamp": get_current_timestamp(), "provider": provider, "model": MODEL_MAPPING.get(provider, "unknown"), "query": payload["query"], "url": payload["url"], "steps": [] }
    driver, final_state = None, {}
    try:
        options = UiAutomator2Options()
        options.platform_name = 'Android'; options.udid = "ZD222GXYPV"
        options.automation_name = 'UiAutomator2'; options.browser_name = "Chrome"
        options.no_reset = True; options.auto_grant_permissions = True
        options.set_capability("appium:chromedriver_autodownload", True)
        options.set_capability("goog:chromeOptions", {"w3c": True, "args": ["--disable-fre", "--no-first-run"]})
        driver = webdriver.Remote(APPIUM_SERVER_URL, options=options)
        driver.implicitly_wait(5)
        
        push_status(job_id, "job_started", {"provider": provider})
        refined_query, usage = get_refined_prompt(payload["url"], payload["query"], provider)
        job_analysis["steps"].append({"task": "refine_prompt", **usage})
        push_status(job_id, "prompt_refined", {"refined_query": refined_query, "usage": usage})

        initial_state = AgentState( job_id=job_id, driver=driver, query=payload["url"], top_k=payload["top_k"], provider=provider, refined_query=refined_query, results=[], screenshots=[], job_artifacts_dir=SCREENSHOTS_DIR / job_id, step=1, max_steps=15, last_action={}, history=[], token_usage=[] )
        initial_state['job_artifacts_dir'].mkdir(exist_ok=True)
        final_state = graph_app.invoke(initial_state)
        final_result = {"job_id": job_id, "results": final_state.get('results', []), "screenshots": final_state.get('screenshots', [])}
    except WebDriverException as e:
        final_result = {"error": f"WebDriver connection failed: {e.msg}"}
        push_status(job_id, "job_failed", {"error": f"WebDriver Error: {e.msg}", "trace": traceback.format_exc()})
    except Exception as e:
        final_result = {"error": str(e)}
        push_status(job_id, "job_failed", {"error": str(e), "trace": traceback.format_exc()})
    finally:
        JOB_RESULTS[job_id] = final_result
        push_status(job_id, "job_done")
        if final_state: job_analysis["steps"].extend(final_state.get('token_usage', []))
        save_analysis_report(job_analysis)
        if driver: driver.quit()

# --- FastAPI Endpoints (Unchanged) ---
@app.post("/search")
async def start_search(req: SearchRequest):
    job_id = str(uuid.uuid4()); JOB_QUEUES[job_id] = asyncio.Queue()
    asyncio.get_event_loop().run_in_executor(None, run_job, job_id, req.dict())
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
async def get_result(job_id: str): return JSONResponse(JOB_RESULTS.get(job_id, {"status": "pending"}), status_code=200 if job_id in JOB_RESULTS else 202)
@app.get("/screenshots/{job_id}/{filename}")
async def get_screenshot(job_id: str, filename: str):
    path = SCREENSHOTS_DIR / job_id / filename
    if not path.is_file(): raise HTTPException(status_code=404, detail="Screenshot not found")
    return FileResponse(path)
@app.get("/")
async def client_ui(): return FileResponse(Path(__file__).parent / "static/test_client.html")

