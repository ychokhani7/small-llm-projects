from dotenv import load_dotenv
load_dotenv()

import asyncio
from agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

# Setup
session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name="greeter_app",
    session_service=session_service
)

def test_agent_with_state(test_name, session_id, message, state_dict):
    print(f"=== {test_name} ===")
    
    # 1. Create a fresh session and inject the state at birth
    asyncio.run(session_service.create_session(
        app_name="greeter_app",
        user_id="user1",
        session_id=session_id,
        state=state_dict  # Seed the exact variables the template needs
    ))

    # 2. Run the agent using this specific session
    result = runner.run(
        user_id="user1",
        session_id=session_id,
        new_message=Content(parts=[Part(text=message)])
    )

    for event in result:
        if event.is_final_response():
            print(f"Agent: {event.content.parts[0].text}\n")


# Test 1: Baseline defaults
test_agent_with_state(
    "Test 1: Baseline defaults", 
    "session_test1", 
    "Hello", 
    {"user_name": "Guest", "user_language": "English", "membership_tier": "Standard"}
)

# Test 2: With user name
test_agent_with_state(
    "Test 2: With user name", 
    "session_test2", 
    "Hello again", 
    {"user_name": "Alex", "user_language": "English", "membership_tier": "Standard"}
)

# Test 3: With all state values
test_agent_with_state(
    "Test 3: With all state values", 
    "session_test3", 
    "Hola de nuevo", 
    {"user_name": "Alex", "user_language": "Spanish", "membership_tier": "premium"}
)
