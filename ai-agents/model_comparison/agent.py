"""
Model configuration demonstration showing factual vs creative optimization.
Demonstrates ADK's generate_content_config with different settings on Agent Platform.
"""

from google.adk.agents import LlmAgent
from google.genai import types

# Agent 1: Optimized for Factual Data Extraction
# Uses deterministic temperature (0.1), disables thinking tokens to protect output budget
factual_agent = LlmAgent(
    model="gemini-3.5-flash",  # Flash is sufficient for extraction
    name="data_extractor",
    description="Extracts factual information with high consistency",
    instruction="""You are a precise data extractor.

Extract facts exactly as stated. Do not:
- Add information not present in the input
- Make assumptions or inferences
- Use creative language

Be accurate, concise, and deterministic.""",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,  # Low/Deterministic (range 0.0 - 2.0)
        max_output_tokens=500,
        top_p=0.8,
        top_k=10,
# Turn off thinking tokens so the 500 limit applies entirely to visible          text thinking_config=types.ThinkingConfig(thinking_budget=0),
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            )
        ]
    )
)

# Agent 2: Optimized for Creative Brainstorming
# Uses high temperature (>1.0 on Gemini's 0-2 scale) and higher top_p/top_k
creative_agent = LlmAgent(
    model="gemini-3.8-flash",  # Pro for superior creativity
    name="creative_brainstormer",
    description="Generates creative ideas and explores possibilities",
    instruction="""You are a creative brainstorming partner.

Generate innovative, diverse, and imaginative ideas. Feel free to:
- Think outside the box
- Combine unexpected concepts
- Explore unconventional approaches

Be creative, varied, and thought-provoking.""",
    generate_content_config=types.GenerateContentConfig(
        temperature=1.3,  # Truly high/creative for Gemini (scale 0.0 - 2.0)
        max_output_tokens=2000,
        top_p=0.95,
        top_k=40,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            )
        ]
    )
)



# For adk web, we'll use the factual agent as root_agent
# Switch to creative_agent to test different behavior
root_agent = LlmAgent(
    model="gemini-3.5-flash",  # Pro for superior creativity
    name="root_agent",
    description="Generates creative ideas and explores possibilities",
    instruction="You are an agent which uses the subagents to help user query based on precise data extractor or creative brainstorming partner.",
    sub_agents=[creative_agent, factual_agent]
    )