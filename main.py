import os
import io
import base64
import json
import traceback
from typing import List, TypedDict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from groq import Groq
from openai import OpenAI
from PIL import Image

from langgraph.graph import StateGraph, END

# --- Initialization ---
load_dotenv()
app = FastAPI(title="SnapChef AI API")

# Initialize API Clients
groq_api_key = os.getenv("GROQ_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Groq handles Text reasoning, OpenRouter handles Vision
llm_client = Groq(api_key=groq_api_key)
vlm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    image_base64: str
    dietary_preference: str
    ingredients: List[str]
    recipe: Optional[str]
    critique: Optional[str]
    is_approved: bool
    iterations: int

# --- Helper Functions ---
def call_vlm(prompt: str, image_b64: str):
    """Step 1: Helper to call Vision Models via OpenRouter Free Tier"""
    image_url = f"data:image/jpeg;base64,{image_b64}"
    
    try:
        response = vlm_client.chat.completions.create(
            model=os.getenv("VLM_MODEL", "openrouter/free"), # Fallback to auto-router
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        
        content = response.choices[0].message.content
        print(f"--- VLM RAW OUTPUT ---\n{content}\n----------------------")
        
        # BULLETPROOF CHECK: Prevent the NoneType crash
        if content is None:
            return "Error: Vision model returned nothing."
            
        return content
        
    except Exception as e:
        print(f"--- OPENROUTER ERROR ---\n{e}\n------------------------")
        return "Unknown ingredients, please suggest a random recipe."

def call_llm(prompt: str, json_format: bool = False):
    """Step 2 & 3: Helper to call Text Models via Groq LPUs"""
    response = llm_client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"} if json_format else None,
        max_tokens=1000
    )
    return response.choices[0].message.content

# --- Graph Nodes ---

def vision_node(state: AgentState):
    """Step 1: Identify ingredients from image"""
    print("--- NODE: Vision ---")
    prompt = "Identify all food items and ingredients in this image. Return only a comma-separated list. Do not include any conversational text."
    result = call_vlm(prompt, state['image_base64'])
    
    # Clean up the output string safely
    ingredients = [i.strip() for i in result.replace("```", "").split(",")]
    return {"ingredients": ingredients}

def chef_node(state: AgentState):
    """Step 2: Draft a recipe based on ingredients and critique"""
    print("--- NODE: Chef ---")
    feedback_clause = f"\nPrevious feedback to fix: {state['critique']}" if state['critique'] else ""
    
    prompt = f"""
    You are a professional chef. Create a recipe using primarily these ingredients: {state['ingredients']}.
    Assume basic pantry staples (salt, pepper, oil, water) are available.
    User Dietary Preference: {state['dietary_preference']}
    {feedback_clause}
    
    Format the recipe clearly with a Title, Ingredients list, and Instructions.
    """
    recipe = call_llm(prompt)
    return {"recipe": recipe, "iterations": state['iterations'] + 1}

def critic_node(state: AgentState):
    """Step 3: Critique the recipe for safety and dietary compliance"""
    print("--- NODE: Critic ---")
    prompt = f"""
    Review this recipe for:
    1. Strict compliance with preference: {state['dietary_preference']}
    2. General edibility and logical cooking steps.
    
    Recipe: {state['recipe']}
    
    Respond ONLY with a JSON object. Do not include markdown formatting, backticks, or any conversational text.
    {{
        "approved": true,
        "feedback": "detailed explanation if false, otherwise empty string"
    }}
    """
    result = call_llm(prompt, json_format=True)
    
    try:
        clean_result = result.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean_result)
    except Exception as e:
        print(f"Error parsing JSON: {e} \nRaw Output: {result}")
        data = {"approved": True, "feedback": ""} 
        
    return {"is_approved": data.get('approved', True), "critique": data.get('feedback', "")}

# --- Graph Logic ---

def should_continue(state: AgentState):
    """Router to decide if we loop back to the chef or end the execution"""
    if state['is_approved']:
        print("--- ROUTER: Recipe Approved ---")
        return "end"
    if state['iterations'] >= 3:
        print("--- ROUTER: Max Iterations Reached. Forcing Exit. ---")
        return "end"
    
    print("--- ROUTER: Recipe Rejected. Sending back to Chef. ---")
    return "rewrite"

# Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("vision", vision_node)
workflow.add_node("chef", chef_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("vision")
workflow.add_edge("vision", "chef")
workflow.add_edge("chef", "critic")

workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "rewrite": "chef",
        "end": END
    }
)

app_graph = workflow.compile()

# --- FastAPI Routes ---

@app.post("/generate-recipe")
async def generate_recipe(
    file: UploadFile = File(...),
    dietary_preference: str = Form("None")
):
    try:
        image_data = await file.read()
        
        # Compress the image to prevent API payload limits
        img = Image.open(io.BytesIO(image_data))
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        img.thumbnail((512, 512))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=80)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Initial State
        initial_state = {
            "image_base64": base64_image,
            "dietary_preference": dietary_preference,
            "ingredients": [],
            "recipe": None,
            "critique": None,
            "is_approved": False,
            "iterations": 0
        }
        
        # Execute LangGraph Multi-Agent Loop
        final_state = app_graph.invoke(initial_state)
        
        return {
            "status": "Success",
            "ingredients_detected": final_state["ingredients"],
            "recipe_approved": final_state["is_approved"],
            "total_iterations_taken": final_state["iterations"],
            "recipe": final_state["recipe"]
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)