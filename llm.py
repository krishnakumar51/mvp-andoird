import re
import json
import base64
from enum import Enum
from pathlib import Path
from typing import List, Union, Tuple, Dict

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

AGENT_PROMPT = """
You are an autonomous web agent with memory. Your goal is to achieve the user's objective by navigating and interacting with a web page.
You operate in a step-by-step manner. At each step, analyze the current state of the page (HTML and screenshot), review your past actions, and decide on the single best next action.

**User's Objective:** "{query}"
**Current URL:** {url}
**HTML Content (first 20,000 chars):**
{html}

**Recent Action History (Memory):**
{history}

**Your Task:**
1.  **Think:** Analyze the situation. Review your history. If your last action failed, identify why and devise a new strategy. What is your immediate goal? What single action will bring you closer to the user's overall objective?
2.  **Act:** Choose ONE action from the available tools.

**Available Tools (Action JSON format):**
-   `{{"type": "fill", "selector": "<css_selector>", "text": "<text_to_fill>"}}`: To type in an input field.
-   `{{"type": "click", "selector": "<css_selector>"}}`: To click a button or link.
-   `{{"type": "press", "selector": "<css_selector>", "key": "<key_name>"}}`: To press a key (e.g., "Enter") on an element. **Hint: After filling a search bar, this is often more reliable than clicking a suggestion button.**
-   `{{"type": "scroll", "direction": "down"}}`: To scroll the page and reveal more content.
-   `{{"type": "extract", "items": [{{"title": "...", "price": "...", "url": "...", "snippet": "..."}}]}}`: To extract structured data from the CURRENT VIEW.
-   `{{"type": "finish", "reason": "<summary_of_completion>"}}`: To end the mission when the objective is fully met.

**Response Format:**
You MUST respond with a single, valid JSON object containing "thought" and "action". Do NOT add any other text, explanations, or markdown.

Example Response:
```json
{{
    "thought": "My previous attempt to click the suggestion button failed with a timeout. A more robust approach is to press the 'Enter' key on the search bar I just filled.",
    "action": {{"type": "press", "selector": "input[name='q']", "key": "Enter"}}
}}
```

**Current Situation Analysis:**
Based on the provided HTML, screenshot, and your recent history, what is your next thought and action?
"""

def get_refined_prompt(url: str, query: str, provider: LLMProvider) -> Tuple[str, Dict]:
    """Generates a refined, actionable prompt and returns the token usage."""
    prompt = REFINER_PROMPT.format(url=url, query=query)
    response_text, usage = get_llm_response("You are a helpful assistant.", prompt, provider, images=[])
    return response_text.strip(), usage

def get_agent_action(query: str, url: str, html: str, provider: LLMProvider, screenshot_path: Path, history: str) -> Tuple[dict, Dict]:
    """Gets the next thought and action from the agent, and returns token usage."""
    prompt = AGENT_PROMPT.format(query=query, url=url, html=html[:20000], history=history or "No actions taken yet.")
    system_prompt = "You are an autonomous web agent. Respond ONLY with the JSON object containing your thought and action."
    
    response_text, usage = get_llm_response(system_prompt, prompt, provider, images=[screenshot_path])
    
    try:
        action = extract_json_from_response(response_text)
        return action, usage
    except ValueError:
        error_action = {"thought": "Error: Could not parse a valid JSON action from the model's response.", "action": {"type": "finish"}}
        # Return zero usage for parsing errors as the API call was made
        error_usage = {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0} 
        return error_action, error_usage


def get_llm_response(
    system_prompt: str,
    prompt: str,
    provider: LLMProvider,
    images: List[Path]
) -> Tuple[str, Dict]:
    """Gets a response and token usage from the specified LLM provider."""
    usage = {"input_tokens": 0, "output_tokens": 0}
    
    if provider == LLMProvider.ANTHROPIC:
        if not anthropic_client: raise ValueError("Anthropic client not initialized.")
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for img_path in images:
            with open(img_path, "rb") as f: img_data = base64.b64encode(f.read()).decode("utf-8")
            messages[0]["content"].append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_data}})
        
        response = anthropic_client.messages.create(model=ANTHROPIC_MODEL, max_tokens=2048, system=system_prompt, messages=messages)
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        return response.content[0].text, usage

    elif provider == LLMProvider.OPENAI:
        if not openai_client: raise ValueError("OpenAI client not initialized.")
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for img_path in images:
            with open(img_path, "rb") as f: img_data = base64.b64encode(f.read()).decode("utf-8")
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}})
        
        response = openai_client.chat.completions.create(model=OPENAI_MODEL, max_tokens=2048, messages=[{"role": "system", "content": system_prompt}, *messages])
        if response.usage:
            usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
        return response.choices[0].message.content, usage

    elif provider == LLMProvider.GROQ:
        if not groq_client: raise ValueError("Groq client not initialized.")
        if images: raise ValueError("The configured Groq model does not support vision.")

        response = groq_client.chat.completions.create(model=GROQ_MODEL, max_tokens=2048, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])
        if response.usage:
             usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
        return response.choices[0].message.content, usage

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def extract_json_from_response(text: str) -> Union[dict, list]:
    """Robustly extracts a JSON object or array from a string."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            raise ValueError("Found a JSON-like structure but could not parse it.")
    raise ValueError("No JSON object found in the response.")
