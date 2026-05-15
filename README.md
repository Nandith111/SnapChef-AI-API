Project Overview: SnapChef AI
SnapChef AI is an intelligent, multi-agent culinary assistant that transforms raw visual data into customized, chef-vetted recipes. Using a sophisticated Vision-Language Model (VLM) and an agentic workflow, it allows users to simply snap a photo of their ingredients to receive a dietary-compliant, step-by-step cooking guide.

1. Core Objective
To automate the creative process of "pantry cooking" by using AI agents to bridge the gap between computer vision and culinary reasoning. The system doesn't just list ingredients; it simulates a collaborative kitchen environment where a Chef Agent proposes a meal and a Critique Agent ensures quality and safety.

2. The Workflow (Agentic Loop)
The project utilizes LangGraph to manage a stateful, cyclical workflow:

Vision Node (VLM): Analyzes the uploaded image to extract a structured list of available ingredients.

Chef Node (LLM): Processes the ingredients and user dietary preferences (e.g., Vegan, Keto) to draft a creative recipe.

Critique Node (LLM): Evaluates the recipe for logical flaws or dietary violations.

Refinement Loop: If the critique is negative, the state loops back to the Chef Node for a rewrite based on specific feedback (up to 3 iterations).

Final Output: Delivers a finalized JSON response containing the detected ingredients and the approved recipe.