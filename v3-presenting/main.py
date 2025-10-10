import asyncio
import uuid
import json
import time
import csv
import re
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
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException, StaleElementReferenceException

from llm import LLMProvider, get_refined_prompt, create_master_plan, get_targeting_decision, get_agent_action
from config import SCREENSHOTS_DIR, ANTHROPIC_MODEL

app = FastAPI(title="LangGraph Android Web Agent")
JOB_QUEUES, JOB_RESULTS = {}, {}
ANALYSIS_DIR, REPORT_CSV_FILE = Path("analysis"), Path("report.csv")
TOKEN_COSTS = { "anthropic": { "claude-3.5-sonnet-20240620": {"input": 3.0, "output": 15.0} } }
MODEL_MAPPING = { LLMProvider.ANTHROPIC: ANTHROPIC_MODEL }
APPIUM_SERVER_URL = "http://localhost:4723"

def get_current_timestamp(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def push_status(job_id: str, msg: str, details: dict = None):
    q = JOB_QUEUES.get(job_id)
    if q: q.put_nowait({"ts": get_current_timestamp(), "msg": msg, **({"details": details} if details else {})})

def resize_image_if_needed(path: Path):
    try:
        with Image.open(path) as img:
            if max(img.size) > 2000: img.thumbnail((2000, 2000), Image.LANCZOS); img.save(path)
    except Exception as e: print(f"Warning: Could not resize {path}. Error: {e}")

def save_analysis_report(data: dict):
    # This function remains stable
    job_id, provider, model = data["job_id"], data["provider"], data["model"]
    total_input = sum(s.get("input_tokens", 0) for s in data["token_usage"])
    total_output = sum(s.get("output_tokens", 0) for s in data["token_usage"])
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

class SearchRequest(BaseModel):
    url: str; query: str; top_k: int; llm_provider: LLMProvider = LLMProvider.ANTHROPIC

class AgentState(TypedDict):
    job_id: str; driver: webdriver.Remote; query: str; top_k: int; provider: LLMProvider
    refined_query: str; results: List[dict]; screenshots: List[str]; job_artifacts_dir: Path
    step: int; max_steps: int; last_action: dict; history: List[str]; token_usage: List[dict]
    master_plan: List[str]
    plan_step: int

def get_element_xpath(driver: webdriver.Remote, element: WebElement) -> str:
    # This function is now stable
    return driver.execute_script(
        "function getXPath(element) {"
        "if (element.id !== '') return `//*[@id=\"${element.id}\"]`;"
        "if (element === document.body) return element.tagName.toLowerCase();"
        "let ix = 0;"
        "const siblings = element.parentNode.childNodes;"
        "for (let i = 0; i < siblings.length; i++) {"
        "const sibling = siblings[i];"
        "if (sibling === element) return `${getXPath(element.parentNode)}/${element.tagName.toLowerCase()}[${ix + 1}]`;"
        "if (sibling.nodeType === 1 && sibling.tagName === element.tagName) ix++;"
        "}"
        "}"
        "return getXPath(arguments[0]);", element)

def find_candidate_elements(driver: webdriver.Remote, keywords: List[str]) -> List[dict]:
    # This function is now stable
    if not keywords: return []
    queries = []
    for k in keywords:
        k_escaped = k.replace("'", "\"'\"")
        queries.extend([f"contains(normalize-space(.), '{k_escaped}')", f"contains(@placeholder, '{k_escaped}')", f"contains(@aria-label, '{k_escaped}')", f"contains(@title, '{k_escaped}')"])
    full_query = " or ".join(queries)
    xpath = f"//*[{full_query}]"
    candidates = []
    try:
        elements = driver.find_elements(By.XPATH, xpath)
        interactive_elements = []
        for element in elements:
            try:
                if not element.is_displayed() or not element.is_enabled(): continue
                tag = element.tag_name.lower()
                text = (element.text or element.get_attribute("value") or element.get_attribute("aria-label") or "")
                score = 0
                if tag in ['button', 'a', 'input', 'textarea']: score = 10
                elif element.get_attribute('onclick') or element.get_attribute('role') in ['button', 'link']: score = 5
                selector = get_element_xpath(driver, element)
                interactive_elements.append({"score": score, "selector": selector, "tag": tag, "text": text[:100].strip()})
            except StaleElementReferenceException:
                print(f"Warning: Stale element reference encountered. Skipping element.")
                continue
            except Exception as e: print(f"Warning: Could not process an element. {e}")
        candidates = sorted(interactive_elements, key=lambda x: x['score'], reverse=True)
    except Exception as e: print(f"Error finding candidate elements: {e}")
    return candidates[:15]

# --- GRAPH NODES ---

def navigate_to_page(state: AgentState) -> AgentState:
    state['driver'].get(state['query']); time.sleep(5)
    push_status(state['job_id'], "navigation_complete", {"url": state['driver'].current_url})
    return state

def planner_node(state: AgentState) -> AgentState:
    push_status(state['job_id'], "agent_planning")
    plan, usage = create_master_plan(state['refined_query'], state['provider'])
    state['master_plan'] = plan.get('plan', [])
    state['plan_step'] = 0
    state['token_usage'].append({"task": "planner", **usage})
    push_status(state['job_id'], "agent_plan_created", {"plan": state['master_plan'], "reasoning": plan.get('reasoning'), "usage": usage})
    return state

def agent_reasoning_node(state: AgentState) -> AgentState:
    job_id, driver = state['job_id'], state['driver']
    current_sub_goal = state['master_plan'][state['plan_step']] if state['plan_step'] < len(state['master_plan']) else "ANALYZE_RESULTS"
    push_status(job_id, "agent_step", {"step": state['step'], "max_steps": state['max_steps'], "current_goal": current_sub_goal})
    screenshot_path = state['job_artifacts_dir'] / f"{state['step']:02d}_step.png"
    driver.get_screenshot_as_file(str(screenshot_path)); resize_image_if_needed(screenshot_path)
    state['screenshots'].append(f"screenshots/{job_id}/{state['step']:02d}_step.png")

    targeting, usage1 = get_targeting_decision(query=state['refined_query'], plan=state['master_plan'], plan_step=state['plan_step'], history="\n".join(state['history']), provider=state['provider'], screenshot_path=screenshot_path)
    state['token_usage'].append({"task": f"targeting_step_{state['step']}", **usage1})
    push_status(job_id, "agent_targeting", {"decision": targeting, "usage": usage1})

    candidates = find_candidate_elements(driver, targeting.get("keywords", []))
    candidates_str = "No suitable elements found matching keywords." if not candidates else json.dumps(candidates, indent=2)

    action, usage2 = get_agent_action(query=state['refined_query'], plan=state['master_plan'], plan_step=state['plan_step'], candidate_elements=candidates_str, history="\n".join(state['history']), provider=state['provider'], screenshot_path=screenshot_path)
    state['token_usage'].append({"task": f"action_step_{state['step']}", **usage2})
    push_status(job_id, "agent_action_thought", {"thought": action.get("reason"), "usage": usage2})

    state['last_action'] = action
    return state

# --- UPGRADED execute_action_node ---
def execute_action_node(state: AgentState) -> AgentState:
    job_id, action, driver = state['job_id'], state['last_action'], state['driver']
    push_status(job_id, "executing_action", {"action": action})
    try:
        action_type = action.get("type")
        selector = action.get("selector")
        wait = WebDriverWait(driver, 15)
        
        current_goal = state['master_plan'][state['plan_step']] if state['plan_step'] < len(state['master_plan']) else "ANALYZE_RESULTS"
        
        if selector: wait.until(EC.presence_of_element_located((By.XPATH, selector)))
        
        if action_type == "tap":
            element = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
            element.click()
        elif action_type == "fill_and_submit":
            element = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
            element.click(); element.clear()
            element.send_keys(action["text"])
            element.send_keys(Keys.ENTER)
        elif action_type == "scroll":
            if selector:
                try:
                    scroll_element = wait.until(EC.presence_of_element_located((By.XPATH, selector)))
                    driver.execute_script("arguments[0].scrollTop = arguments[0].scrollTop + arguments[0].offsetHeight * 0.8;", scroll_element)
                except Exception as e:
                    print(f"Could not scroll specific element '{selector}', defaulting to window scroll. Error: {e}")
                    driver.execute_script("window.scrollBy(0, window.innerHeight * 0.8);")
            else:
                driver.execute_script("window.scrollBy(0, window.innerHeight * 0.8);")
        elif action_type == "extract":
            state['results'].extend(action.get("items", []))
            push_status(job_id, "partial_result", {"new_items_found": len(action.get("items", []))})
        elif action_type == "skip_step":
             pass # This is a logical action, no browser interaction needed.

        time.sleep(4 if action_type != 'skip_step' else 0.1)
        
        if action_type == 'skip_step':
            success_message = f"✅ SKIPPED (Step {state['step']}): The optional '{current_goal}' goal was skipped."
        else:
            success_message = f"✅ SUCCESS (Step {state['step']}): The '{current_goal}' goal is complete."
        state['history'].append(success_message)
        
        if action_type not in ["extract", "finish", "scroll"]:
            state['plan_step'] += 1
            
    except Exception as e:
        error_message = str(e).splitlines()[0] if str(e).strip() else f"An unspecified error occurred: {type(e).__name__}"
        push_status(job_id, "action_failed", {"action": action, "error": error_message})
        state['history'].append(f"Step {state['step']}: Action `{action.get('type')}` FAILED: {error_message}")

    state['step'] += 1
    state['history'] = state['history'][-5:]
    return state

def supervisor_node(state: AgentState) -> str:
    is_plan_done = state['plan_step'] >= len(state['master_plan'])
    if state['last_action'].get("type") == "finish" or len(state['results']) >= state['top_k'] or state['step'] > state['max_steps'] or is_plan_done:
        return "end"
    return "continue"

# --- GRAPH DEFINITION ---
builder = StateGraph(AgentState)
builder.add_node("navigate", navigate_to_page)
builder.add_node("planner", planner_node)
builder.add_node("reason", agent_reasoning_node)
builder.add_node("execute", execute_action_node)
builder.set_entry_point("navigate")
builder.add_edge("navigate", "planner")
builder.add_edge("planner", "reason")
builder.add_edge("reason", "execute")
builder.add_conditional_edges("execute", supervisor_node, {"end": END, "continue": "reason"})
graph_app = builder.compile()

def run_job(job_id: str, payload: dict):
    # This function remains stable
    provider = payload["llm_provider"]
    job_analysis = { "job_id": job_id, "timestamp": get_current_timestamp(), "provider": provider, "model": MODEL_MAPPING.get(provider, "unknown"), "query": payload["query"], "url": payload["url"], "token_usage": [] }
    driver, final_state = None, {}
    try:
        options = UiAutomator2Options()
        options.platform_name = 'Android'; options.udid = "ZD222GXYPV"
        options.automation_name = 'UiAutomator2'; options.browser_name = "Chrome"
        options.no_reset = True; options.auto_grant_permissions = True
        options.set_capability("appium:uiautomator2ServerInstallTimeout", 60000)
        options.set_capability("appium:chromedriver_autodownload", True)
        options.set_capability("goog:chromeOptions", {"w3c": True, "args": ["--disable-fre", "--no-first-run"]})
        driver = webdriver.Remote(APPIUM_SERVER_URL, options=options)
        driver.implicitly_wait(10)
        push_status(job_id, "job_started", {"provider": provider})
        refined_query, usage = get_refined_prompt(payload["url"], payload["query"], provider)
        job_analysis["token_usage"].append({"task": "refine_prompt", **usage})
        push_status(job_id, "prompt_refined", {"refined_query": refined_query, "usage": usage})
        initial_state = AgentState(
            job_id=job_id, driver=driver, query=payload["url"], top_k=payload["top_k"], provider=provider,
            refined_query=refined_query, results=[], screenshots=[], job_artifacts_dir=SCREENSHOTS_DIR / job_id,
            step=1, max_steps=15, last_action={}, history=[], token_usage=[],
            master_plan=[], plan_step=0
        )
        initial_state['job_artifacts_dir'].mkdir(exist_ok=True)
        final_state = graph_app.invoke(initial_state)
        final_result = {"job_id": job_id, "results": final_state.get('results', []), "screenshots": final_state.get('screenshots', [])}
    except WebDriverException as e:
        error_message = str(e).split('\n')[0]
        final_result = {"error": f"WebDriver connection failed: {error_message}"}
        push_status(job_id, "job_failed", {"error": f"WebDriver Error: {error_message}", "trace": traceback.format_exc()})
    except Exception as e:
        final_result = {"error": str(e)}
        push_status(job_id, "job_failed", {"error": str(e), "trace": traceback.format_exc()})
    finally:
        JOB_RESULTS[job_id] = final_result
        push_status(job_id, "job_done")
        if final_state: job_analysis["token_usage"].extend(final_state.get('token_usage', []))
        save_analysis_report(job_analysis)
        if driver: driver.quit()

# --- FastAPI Endpoints ---
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
async def get_result(job_id: str):
    return JSONResponse(JOB_RESULTS.get(job_id, {"status": "pending"}), status_code=200 if job_id in JOB_RESULTS else 202)
@app.get("/screenshots/{job_id}/{filename}")
async def get_screenshot(job_id: str, filename: str):
    path = SCREENSHOTS_DIR / job_id / filename
    if not path.is_file(): raise HTTPException(status_code=404, detail="Screenshot not found")
    return FileResponse(path)
@app.get("/")
async def client_ui():
    return FileResponse(Path(__file__).parent / "static/test_client.html")