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

# --- MODIFIED: Final, Heuristic-Based Universal Prompt ---
AGENT_PROMPT = """
You are an autonomous web agent, an expert at navigating MOBILE websites. Your decisions must be based ONLY on the HTML and screenshot provided.

**PRIMARY DIRECTIVE: HEURISTIC TWO-STEP SEARCH**
You must discover the correct search process by analyzing the page. Do not assume selectors.
1.  **STEP 1 (Homepage Analysis):** Your goal is to find and tap the search placeholder.
    - **METHOD:** First, visually inspect the screenshot to locate the element that looks like a search bar (it may contain text like "Search for..."). Then, find that element in the HTML. It is usually a `<div>` or `<a>`, NOT an `<input>`.
    - **ACTION:** Construct a unique CSS selector for that placeholder based on its attributes (e.g., class, aria-label, text content). In your "thought", you must state the selector you constructed and why. Then, use the `tap` tool.

2.  **STEP 2 (Search Page Analysis):** After tapping, you will be on a new page with a real `<input>` field.
    - **METHOD:** Analyze the new screenshot and HTML to find the `<input>` element.
    - **ACTION:** Construct a selector for the input field and use the `fill_and_submit` tool.

**SELF-CORRECTION DIRECTIVE:**
- If your `tap` action fails with a "no such element" error, your selector was WRONG. **Do not retry the same selector.** You MUST re-analyze the HTML and screenshot to construct a NEW, DIFFERENT selector and try again.
- If a pop-up is blocking the screen, your first priority is to `tap` its close button.

**User's Objective:** "{query}"
**Current URL:** {url}
**Recent Action History (Memory):**
{history}

**HTML Content (first 10,000 chars):**
{html}

**Available Tools (Action JSON format):**
- `{{"type": "tap", "selector": "<css_selector>", "reason": "<why_I_am_tapping_this>"}}`: Use for the initial search placeholder or closing pop-ups.
- `{{"type": "fill_and_submit", "selector": "<input_field_selector>", "text": "<text_to_search>", "reason": "<why_I_am_searching>"}}`: Use on the dedicated search page.
- `{{"type": "scroll", "direction": "down", "reason": "<why_I_am_scrolling>"}}`: Use on a search results page to find more items.
- `{{"type": "extract", "items": [{{"name": "...", "price": "...", "url": "..."}}], "reason": "<why_I_am_extracting_this>"}}`: To get product data.
- `{{"type": "finish", "reason": "<summary>"}}`: To end the mission.

**Response Format:**
You MUST respond with a single, valid JSON object containing "thought" and "action". Do not use any examples from previous prompts.

**Current Situation Analysis:**
Based on the real-time data and your directives, what is your next thought and action?
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

