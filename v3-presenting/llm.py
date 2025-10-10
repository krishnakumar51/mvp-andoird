import re
import json
import base64
from enum import Enum
from pathlib import Path
from typing import List, Union, Tuple, Optional

from pydantic import BaseModel, Field
from config import anthropic_client, ANTHROPIC_MODEL

class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"

# --- PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT (TOOLS) ---

class MasterPlanTool(BaseModel):
    """Tool to define a high-level, multi-step plan to achieve the user's objective."""
    reasoning: str = Field(..., description="A step-by-step thought process explaining the generated plan.")
    plan: List[str] = Field(..., description="A list of sequential, high-level goals. Valid goals are: 'INITIATE_SEARCH', 'EXECUTE_SEARCH', 'APPLY_FILTER', 'APPLY_SORT', 'NAVIGATE', 'EXTRACT_DATA', 'ANALYZE_RESULTS'. Refinement goals like APPLY_FILTER or APPLY_SORT can be marked as optional, e.g., 'APPLY_SORT (Optional)'.")

class TargetingTool(BaseModel):
    """Tool to determine the specific element to find on the page for a given sub-goal."""
    goal_description: str = Field(..., description="A brief (2-3 word) description of the immediate sub-goal from the master plan.")
    keywords: List[str] = Field(..., description="A list of case-sensitive text strings to search for on the page to find the target element.")

class TapTool(BaseModel):
    """Tool to tap/click an element."""
    selector: str = Field(..., description="The robust XPath selector for the element to tap, chosen from the candidate elements list.")
    reason: str = Field(..., description="A brief explanation of why this element is being tapped.")

class FillAndSubmitTool(BaseModel):
    """Tool to fill an input field and submit."""
    selector: str = Field(..., description="The robust XPath selector for the input element, chosen from the candidate elements list.")
    text: str = Field(..., description="The text to be entered into the input field.")
    reason: str = Field(..., description="A brief explanation for this search action.")

class ScrollTool(BaseModel):
    """Tool to scroll the page or a specific scrollable element."""
    direction: str = Field("down", description="The direction to scroll. Only 'down' is currently supported.")
    selector: Optional[str] = Field(None, description="Optional. The XPath selector of a specific scrollable element (like a modal or panel) to scroll inside of.")
    reason: str = Field(..., description="A brief explanation of why scrolling is necessary.")

class ExtractTool(BaseModel):
    """Tool to extract structured data from the page."""
    items: List[dict] = Field(..., description="A list of dictionaries representing scraped items.")
    reason: str = Field(..., description="Explanation of the data being extracted.")

class FinishTool(BaseModel):
    """Tool to finish the task successfully."""
    reason: str = Field(..., description="A summary of how the objective was completed.")

# --- NEW TOOL ---
class SkipStepTool(BaseModel):
    """Tool to skip the current optional step in the plan."""
    reason: str = Field(..., description="A brief explanation for why the optional step is being skipped (e.g., 'No visible sort button found').")


ACTION_TOOLS = [TapTool, FillAndSubmitTool, ScrollTool, ExtractTool, FinishTool, SkipStepTool]

# --- PROMPT TEMPLATES ---

REFINER_PROMPT = """
Analyze the user's request and create a concise, actionable instruction for an AI web agent. Focus on the ultimate goal.
User's Target URL: {url}
User's Query: "{query}"
Refined Instruction:
"""

# --- UPGRADED PLANNER PROMPT ---
PLANNER_PROMPT = """
You are the "Planner" module. Your job is to decompose the user's objective into a high-level, sequential plan.

**User's Objective:** "{query}"

**CRITICAL RULES:**
1.  If the user's objective on an e-commerce/search site involves 'find', 'search', or 'list', the first step MUST be 'INITIATE_SEARCH'. Do not add a 'NAVIGATE' step if the agent is already on the correct homepage.
2.  Refinement goals like 'APPLY_FILTER' and 'APPLY_SORT' are often not essential. If the user's request is general (e.g., "find best products"), you can mark these steps as optional by adding ` (Optional)` to the goal name.

**Your Task:**
- Analyze the user's objective and obey the critical rules.
- Create a plan using only the valid goals: 'INITIATE_SEARCH', 'EXECUTE_SEARCH', 'APPLY_FILTER', 'APPLY_SORT', 'NAVIGATE', 'EXTRACT_DATA', 'ANALYZE_RESULTS'.
- Call the `MasterPlanTool` with your reasoning and the final plan.

**Example of an optional step:**
- For "Find top-rated face masks", a good plan is: `["INITIATE_SEARCH", "EXECUTE_SEARCH", "APPLY_SORT (Optional)", "ANALYZE_RESULTS"]`
"""

# --- UPGRADED TARGETING PROMPT ---
TARGETING_PROMPT = """
You are the "Targeting" module. Your job is to determine what to look for on the page to accomplish the current sub-goal.

**User's Objective:** "{query}"
**Master Plan:** {plan}
**Current Sub-Goal:** "{sub_goal}"
**Recent Action History:**
{history}

**Triage Protocol:**
1.  **CONTEXT-AWARE ESCAPE HATCH:** Check the `Recent Action History`. If your last action was tapping a 'Filter' or 'Sort' button and the screenshot now shows a menu/overlay, your ONLY goal is to "Close Overlay". Generate keywords to find an 'Apply', 'Done', 'Close', or 'X' button.
2.  **Analyze for other Overlays:** If the first rule doesn't apply, check for any other unexpected pop-ups or cookie banners. If one exists, your goal is to close it.
3.  **Default Goal:** If no overlays exist, determine the best case-sensitive keywords to achieve the **Current Sub-Goal**.

Call the `TargetingTool` with your decision.
"""

# --- UPGRADED AGENT PROMPT ---
AGENT_PROMPT = """
You are the "Action" module. You are given a sub-goal and a list of candidate elements. Your job is to choose the best action.

**User's Objective:** "{query}"
**Current Sub-Goal:** "{sub_goal}"
**Recent Action History:**
{history}
**Candidate Elements Found on Page:**
{candidate_elements}

**Your Task & Rules:**
1.  **CHECK HISTORY:** Read the last message in `Recent Action History`. If it says a goal is complete, focus on the *next* goal. Do not repeat the completed goal.
2.  **Analyze Candidates:** Review the list to find the best fit for the `Current Sub-Goal`.
3.  **Choose and Act:** Select the best candidate and call the appropriate tool (`TapTool`, `FillAndSubmitTool`, etc.), using the exact selector provided.
4.  **FALLBACK RULE:** If the 'Candidate Elements' list is empty or unsuitable:
    a. Check if your `Current Sub-Goal` contains `(Optional)`. If it does, you MUST call the `SkipStepTool`.
    b. If the goal is not optional, your only permitted actions are `ScrollTool` or `FinishTool`. Do not invent actions.
"""

def get_llm_response(system_prompt: str, prompt: str, provider: LLMProvider, tools: List[BaseModel], images: List[Path] = []) -> Tuple[Optional[str], Optional[dict], dict]:
    # This function is now stable
    usage = {"input_tokens": 0, "output_tokens": 0}
    try:
        if provider == LLMProvider.ANTHROPIC:
            if not anthropic_client: raise ValueError("Anthropic client not initialized.")
            tool_definitions = []
            for tool in tools:
                schema = tool.model_json_schema()
                anthropic_tool_definition = {"name": schema['title'], "description": schema.get('description', f"A tool for {schema['title']}"), "input_schema": schema}
                tool_definitions.append(anthropic_tool_definition)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            if images and images[0].is_file():
                with open(images[0], "rb") as f: img_data = base64.b64encode(f.read()).decode("utf-8")
                messages[0]["content"].append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_data}})
            response = anthropic_client.messages.create(model=ANTHROPIC_MODEL, max_tokens=2048, system=system_prompt, messages=messages, tools=tool_definitions, tool_choice={"type": "auto"})
            usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_name_from_api, tool_input = content_block.name, content_block.input
                    original_tool = next((t for t in tools if t.model_json_schema()['title'] == tool_name_from_api), None)
                    if original_tool: return original_tool.__name__, tool_input, usage
            return None, None, usage
    except Exception as e:
        print(f"LLM API call failed for {provider}: {e}")
        return FinishTool.__name__, {"reason": f"LLM API Error: {e}"}, usage
    return None, None, usage

def create_master_plan(query: str, provider: LLMProvider) -> Tuple[dict, dict]:
    prompt = PLANNER_PROMPT.format(query=query)
    system_prompt = "You are the Planner module. Decompose the user's objective and call the MasterPlanTool."
    tool_name, tool_input, usage = get_llm_response(system_prompt, prompt, provider, tools=[MasterPlanTool])
    if tool_name == MasterPlanTool.__name__: return tool_input, usage
    return {"reasoning": "Failed to create a plan.", "plan": ["INITIATE_SEARCH", "EXECUTE_SEARCH", "ANALYZE_RESULTS"]}, usage

def get_targeting_decision(query: str, plan: List[str], plan_step: int, history: str, provider: LLMProvider, screenshot_path: Path) -> Tuple[dict, dict]:
    sub_goal = plan[plan_step] if plan_step < len(plan) else "ANALYZE_RESULTS"
    prompt = TARGETING_PROMPT.format(query=query, plan=plan, sub_goal=sub_goal.replace(" (Optional)", ""), history=history)
    system_prompt = "You are the Targeting module. Identify the immediate goal and keywords, then call the TargetingTool."
    tool_name, tool_input, usage = get_llm_response(system_prompt, prompt, provider, tools=[TargetingTool], images=[screenshot_path])
    if tool_name == TargetingTool.__name__: return tool_input, usage
    return {"goal_description": "Error in targeting", "keywords": []}, usage

def get_agent_action(query: str, plan: List[str], plan_step: int, candidate_elements: str, history: str, provider: LLMProvider, screenshot_path: Path) -> Tuple[dict, dict]:
    sub_goal = plan[plan_step] if plan_step < len(plan) else "ANALYZE_RESULTS"
    prompt = AGENT_PROMPT.format(query=query, sub_goal=sub_goal, candidate_elements=candidate_elements, history=history)
    system_prompt = "You are the Action module. Choose the best candidate element and call the appropriate tool."
    tool_name, tool_input, usage = get_llm_response(system_prompt, prompt, provider, tools=ACTION_TOOLS, images=[screenshot_path])
    if tool_name and tool_input:
        action_type = re.sub(r'(?<!^)(?=[A-Z])', '_', tool_name).lower().replace("_tool","")
        return {"type": action_type, **tool_input}, usage
    return {"type": "finish", "reason": "Agent could not decide on a valid action."}, usage

def get_refined_prompt(url: str, query: str, provider: LLMProvider) -> Tuple[str, dict]:
    prompt = REFINER_PROMPT.format(url=url, query=query)
    try:
        if provider == LLMProvider.ANTHROPIC:
            response = anthropic_client.messages.create(model=ANTHROPIC_MODEL, max_tokens=1024, messages=[{"role": "user", "content": prompt}])
            usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
            return response.content[0].text.strip(), usage
    except Exception as e: print(f"Refiner prompt failed: {e}")
    return query, {}