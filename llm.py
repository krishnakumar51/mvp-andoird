import re
import json
import base64
from enum import Enum
from pathlib import Path
from typing import List, Union

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
Analyze the user's request and create a concise, actionable prompt for an AI agent.
The agent will perform a web search on the user's behalf.

User's Target URL: {url}
User's Query: "{query}"

Based on this, generate a single, clear instruction for the AI.
Example: "Search for the latest smartphone models on example.com and extract the top 5 results."
Refined Prompt:
"""

PLANNER_PROMPT = """
You are a web automation planner. Your goal is to create a JSON array of actions to achieve the user's objective.
Analyze the provided HTML, accessibility tree, and screenshot of the web page.

User's Objective: "{query}"
Current URL: {url}
HTML Content (first 20,000 chars):
{html}

Accessibility Tree (first 10,000 chars):
{accessibility}

Based on the visual evidence from the screenshot and the DOM structure, devise a plan.
Valid action types are: "fill", "click", "press".
- "fill": Use for input fields. Requires "selector" and "text".
- "click": Use for buttons or links. Requires "selector".
- "press": Use for submitting forms (e.g., with the "Enter" key). Requires "selector" and "key".

Return ONLY a valid JSON array of actions. Do not include any explanations or surrounding text.

Example:
[
    {{"type": "fill", "selector": "input[name='q']", "text": "latest smartphones"}},
    {{"type": "press", "selector": "input[name='q']", "key": "Enter"}}
]
"""

EXTRACTOR_PROMPT = """
You are a data extraction specialist. Your task is to extract structured data from the given HTML content based on the user's query and a screenshot.
The output must be a single JSON object with a key "items", which is a list of the extracted results.

User's Objective: "{query}"
Current URL: {url}
HTML Content (first 40,000 chars):
{html}

Extract up to {top_k} items that match the user's objective. Include relevant fields like "title", "price", "url", and a "snippet".
Ensure all URLs are complete.

Return ONLY a valid JSON object. Do not include any explanations or surrounding text.

Example:
{{
    "items": [
        {{
            "title": "Example Phone Model",
            "price": "₹45,000",
            "url": "https://example.com/product/123",
            "snippet": "A brief description of the phone..."
        }}
    ]
}}
"""

def get_refined_prompt(url: str, query: str, provider: LLMProvider) -> str:
    """Generates a refined, actionable prompt from the user's raw query."""
    prompt = REFINER_PROMPT.format(url=url, query=query)
    # The refiner is a simple text-to-text task, no vision needed.
    response_text = get_llm_response(prompt, "You are a helpful assistant.", provider, images=[])
    return response_text.strip()

def get_llm_response(
    prompt: str,
    system_prompt: str,
    provider: LLMProvider,
    images: List[Path]
) -> str:
    """
    Gets a response from the specified LLM provider, handling API differences.
    This is the core function that abstracts away the different API call structures.
    """
    if provider == LLMProvider.ANTHROPIC:
        if not anthropic_client:
            raise ValueError("Anthropic client is not initialized. Check API key.")
        
        # Correctly format the message for Anthropic's API (handles images)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for img_path in images:
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            messages[0]["content"].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_data,
                },
            })
        
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text

    elif provider == LLMProvider.OPENAI:
        if not openai_client:
            raise ValueError("OpenAI client is not initialized. Check API key.")
        
        # Correctly format the message for OpenAI's API (handles images)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for img_path in images:
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_data}"},
            })
        
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system_prompt},
                *messages
            ]
        )
        return response.choices[0].message.content

    elif provider == LLMProvider.GROQ:
        if not groq_client:
            raise ValueError("Groq client is not initialized. Check API key.")
        if images:
            # Groq model used here doesn't support images, so we raise an error.
            raise ValueError("The configured Groq model does not support vision.")

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def extract_json_from_response(text: str) -> Union[dict, list]:
    """
    Robustly extracts a JSON object or array from a string that might contain other text.
    It looks for the first valid JSON block and parses it.
    """
    # Find the start of a JSON object '{' or array '['
    json_start = -1
    for i, char in enumerate(text):
        if char in ['{', '[']:
            json_start = i
            break
            
    if json_start == -1:
        raise ValueError("No JSON object or array found in the response.")

    # Find the corresponding closing bracket
    json_text = text[json_start:]
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # Fallback for truncated JSON - try to find matching brackets
        open_brackets = 0
        for i, char in enumerate(json_text):
            if char in ['{', '[']:
                open_brackets += 1
            elif char in ['}', ']']:
                open_brackets -= 1
            if open_brackets == 0:
                try:
                    return json.loads(json_text[:i+1])
                except json.JSONDecodeError:
                    continue # Keep searching
    
    raise ValueError("Could not parse a valid JSON object or array from the response.")

