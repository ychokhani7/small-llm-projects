"""
Test script to see state access directly.
Run with: python test_state.py
"""

"""
loading env file
"""
from dotenv import load_dotenv
load_dotenv()
"""
adding async as pthe script is running in sync so we need to make sure that the 
session is fully written before being called.
"""
import asyncio

from agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

# 1. Properly await the session creation
session_service = InMemorySessionService()
session = asyncio.run(session_service.create_session(
    app_name="name_extractor_app",
    user_id="test_user",
    session_id="test_session"
))

runner = Runner(
    agent=root_agent,
    app_name="name_extractor_app",
    session_service=session_service
)

# Test: Extract name
user_message = Content(parts=[Part(text="Hi, my name is Alex Johnson")])

print("=== Running agent ===")
result = runner.run(
    user_id="test_user",
    session_id="test_session",
    new_message=user_message
)

for event in result:
    if event.is_final_response():
        print(f"\nAgent output: {event.content.parts[0].text.strip()}")

# 2. Re-fetch the session to see the updated state!
updated_session = asyncio.run(session_service.get_session(
    app_name="name_extractor_app",
    user_id="test_user",
    session_id="test_session"
))

print(f"\n=== State after execution ===")
print(f"Full state: {updated_session.state}")
print(f"Extracted name: {updated_session.state.get('user_name')}")

if updated_session.state.get("user_name"):
    print("Success! The tutorial's output_key populated the state.")
else:
    print("Name extraction failed")
