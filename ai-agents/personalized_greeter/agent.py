"""
Personalized Greeter - Demonstrates State Templating
Shows how {key} templating injects state values into instructions.

Reference: https://adk.dev/sessions/state/
"""

from google.adk.agents import LlmAgent

# Agent with state templating
root_agent = LlmAgent(
    model="gemini-3.1-flash-lite", 
    name="personalized_greeter",

    instruction="""
    You are a friendly assistant.

    User information:
    - Name: {user_name?there}
    - Preferred language: {user_language?English}
    - Membership: {membership_tier?free}

    {membership_tier?Your membership level is: {membership_tier}}

    Greet the user warmly and offer assistance.
    Respond in {user_language?English}.
    """
)