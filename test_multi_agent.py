#!/usr/bin/env python3
"""
Quick test to verify the multi-agent system works.
"""

import os
from dotenv import load_dotenv
from multi_agent_system import ManagerAgent

load_dotenv()

# Short test transcript
test_transcript = """
Meeting: Quick Team Sync
Date: December 6, 2024
Attendees: Alice (PM), Bob (Dev)

Alice: We need to launch the new feature next week.
Bob: I can finish the code by Tuesday.
Alice: Great! Let's document it in GitHub and write a blog post.
"""

if __name__ == "__main__":
    print("🧪 Testing Multi-Agent System...\n")

    try:
        manager = ManagerAgent()
        results = manager.run_workflow(test_transcript)

        print("\n✅ SUCCESS! The system works.")
        print(f"   - Extracted {len(results.get('insights', []))} insights")
        print(f"   - Created {len(results.get('todos', []))} todos")

        # Save test results
        manager.save_results("test_results.json")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
