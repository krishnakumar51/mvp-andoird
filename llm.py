import re
import json
import base64
from enum import Enum
from pathlib import Path
from typing import List, Union, Tuple

from config import (
    anthropic_client, groq_client, openai_client,
    ANTHROPIC_MODEL, GROQ_MODEL, OPENAI_MODEL
)

class LLMProvider(str, Enum):
    """Enumeration for the supported LLM providers."""
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OPENAI = "openai"

# --- PROMPT TEMPLATES ---

REFINER_PROMPT = """
Analyze the user's request and create a concise, actionable instruction for an AI web agent.
Focus on the ultimate goal.

User's Target URL: {url}
User's Query: "{query}"

Based on this, generate a single, clear instruction.
Example: "Find the top 5 smartphones under ₹50,000 on flipkart.com, collecting their name, price, and URL."
Refined Instruction:
"""

# --- MODIFIED: Enhanced Heuristic-Based Universal Prompt for Flexibility and Accuracy ---
AGENT_PROMPT = """
You are an autonomous web agent, an expert at navigating MOBILE websites. Your decisions must be based ONLY on the HTML and screenshot provided. Do not invent or assume attribute values—always quote exact text and attributes directly from the HTML or visible in the screenshot.

**PRIMARY DIRECTIVE: ADAPTIVE HEURISTIC SEARCH ACCESS**
You must discover the correct way to initiate a search by thoroughly analyzing the page. Do not assume fixed structures or selectors—adapt based on the current HTML and screenshot.
1. **STEP 1 (Page Analysis for Search Entry):** Your goal is to locate the search entry point.
   - **METHOD:** Visually inspect the screenshot to identify elements that look like a search bar, icon, or field (e.g., text like the exact visible placeholder, magnifying glass icon). Cross-reference with HTML to find matching elements. Scan the HTML for keywords like 'search' in class, id, placeholder, aria-label, role='searchbox'. This could be:
     - An `<input>` (type="text" or "search", with exact attributes like placeholder, aria-label, class).
     - A `<div>`, `<a>`, `<button>`, or `<span>` acting as a placeholder or icon (e.g., class containing "search", aria-label with "search", inner text matching screenshot).
     - First, list 2-3 potential elements from HTML with their exact tag, attributes, and inner text (if any).
   - **DECISION LOGIC:** If a direct `<input>` is present, visible (not hidden), and has search-related attributes (e.g., placeholder with exact text from screenshot), prioritize `fill_and_submit`. If not found or hidden, use `tap` on a container/placeholder/icon to reveal it.
   - **ACTION:** Construct a unique, precise CSS selector using multiple exact attributes from HTML (e.g., class='exact-class', [placeholder='exact full text'], [aria-label='exact label']). Avoid assumptions—use exact case and full values. In your "thought", explicitly state: potential elements listed with exact attributes, the chosen element type/tag, key attributes quoted from HTML/screenshot, why it matches the search entry (e.g., placeholder matches screenshot text), the full selector, and why different from failed ones.

2. **STEP 2 (Post-Interaction Analysis):** After an action, re-analyze the new page.
   - If now on a search page with a clear `<input>`, use `fill_and_submit`.
   - On results pages, scroll if needed to load more, then extract data.

**SELF-CORRECTION DIRECTIVE (MANDATORY ADAPTATION):**
- If an action fails (e.g., "no such element" or other errors), ANALYZE WHY: Selector imprecise? Attribute mismatch (e.g., wrong case, incomplete text like short placeholder vs full)? Element hidden/not loaded? Wrong type? **Do not retry similar selectors or approaches.** Re-examine HTML/screenshot for overlooked exact attributes or alternatives (e.g., switch from [placeholder] to class/id if failed; if input hidden, tap parent div). Construct a COMPLETELY NEW selector or strategy (e.g., if tap on icon failed, check for direct input; if blocked, close pop-ups first).
- Always prioritize closing any pop-ups, modals, or overlays blocking the screen—construct selectors for close buttons (e.g., [aria-label="close"], class with "close").
- If stuck in a loop (e.g., repeated failures), consider scrolling to reveal elements or alternative navigation (e.g., menu taps).

**User's Objective:** "{query}"
**Current URL:** {url}
**Recent Action History (Memory):**
{history}

**HTML Content (first 10,000 chars):**
{html}

**Available Tools (Action JSON format):**
- `{{"type": "tap", "selector": "<css_selector>", "reason": "<why_I_am_tapping_this>"}}`: Use for placeholders, icons, buttons, or closing pop-ups.
- `{{"type": "fill_and_submit", "selector": "<input_field_selector>", "text": "<text_to_search>", "reason": "<why_I_am_searching>"}}`: Use directly if input is available, or after tapping to reveal it.
- `{{"type": "scroll", "direction": "down", "reason": "<why_I_am_scrolling>"}}`: Use on results pages to load more items.
- `{{"type": "extract", "items": [{{"name": "...", "price": "...", "url": "..."}}], "reason": "<why_I_am_extracting_this>"}}`: To collect product data from visible results.
- `{{"type": "finish", "reason": "<summary>"}}`: To end the mission when objective is met or impossible.

**Response Format:**
You MUST respond with a single, valid JSON object containing "thought" and "action". Do not use any examples from previous prompts.

**Current Situation Analysis:**
Based on the real-time data, history, and your directives, what is your next thought and action?
"""

def get_refined_prompt(url: str, query: str, provider: LLMProvider) -> Tuple[str, dict]:
    prompt = REFINER_PROMPT.format(url=url, query=query)
    response_text, usage = get_llm_response("You are a helpful assistant.", prompt, provider, images=[])
    return response_text.strip(), usage

def get_agent_action(query: str, url: str, html: str, provider: LLMProvider, screenshot_path: Path, history: str) -> Tuple[dict, dict]:
    prompt = AGENT_PROMPT.format(query=query, url=url, html=html[:10000], history=history or "No actions taken yet.")
    system_prompt = "You are an autonomous mobile web agent. Respond ONLY with the JSON object containing your thought and action."
    response_text, usage = get_llm_response(system_prompt, prompt, provider, images=[screenshot_path])
    try:
        action_response = extract_json_from_response(response_text)
        return action_response, usage
    except ValueError:
        return {"thought": "Error: Could not parse a valid JSON action.", "action": {"type": "finish", "reason": "JSON parsing error."}}, usage

def get_llm_response(system_prompt: str, prompt: str, provider: LLMProvider, images: List[Path]) -> Tuple[str, dict]:
    # This function remains unchanged from the previous version.
    usage = {"input_tokens": 0, "output_tokens": 0}
    text_response = ""
    try:
        if provider == LLMProvider.ANTHROPIC:
            if not anthropic_client: raise ValueError("Anthropic client not initialized.")
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            if images:
                with open(images[0], "rb") as f: img_data = base64.b64encode(f.read()).decode("utf-8")
                messages[0]["content"].append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_data}})
            response = anthropic_client.messages.create(model=ANTHROPIC_MODEL, max_tokens=2048, system=system_prompt, messages=messages)
            usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
            text_response = response.content[0].text
        # Other provider implementations remain the same...
    except Exception as e:
        print(f"LLM API call failed for {provider}: {e}")
        text_response = '{"thought": "LLM API call failed.", "action": {"type": "finish", "reason": "LLM API Error"}}'
    return text_response, usage

def extract_json_from_response(text: str) -> Union[dict, list]:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except json.JSONDecodeError as e: raise ValueError(f"Failed to parse JSON: {e}")
    raise ValueError("No JSON object found in response.")