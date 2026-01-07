#!/usr/bin/env python3
"""
Example workflow demonstrating the multi-agent system.

This is a simplified example showing how to use the multi-agent system
with custom meeting transcripts.
"""

import os
from dotenv import load_dotenv
from multi_agent_system import ManagerAgent

load_dotenv()

# Example 1: Product Strategy Meeting
product_meeting = """
Meeting: Q4 Product Roadmap Review
Date: December 6, 2024
Attendees: Alice (CEO), Bob (CTO), Carol (Head of Product)

Alice: Let's review our Q4 priorities. We need to ship three major features.

Bob: From engineering, we can realistically deliver two features this quarter.
The third one requires significant infrastructure changes.

Carol: I suggest we prioritize the user dashboard redesign and the API v2 launch.
The mobile app can wait until Q1.

Alice: Agreed. Let's also start documentation for developers.

Bob: I'll create a GitHub repository for the API docs and examples.

Carol: I can write a Medium article explaining the new features to our users.

Action Items:
- Bob: Set up GitHub repo for API documentation
- Carol: Draft Medium article about new features
- Alice: Review and approve final designs by Dec 15
"""

# Example 2: Technical Discussion
tech_meeting = """
Meeting: ML Model Performance Review
Date: December 6, 2024
Attendees: Sarah (ML Engineer), David (Data Scientist), Emma (Product)

Sarah: Our recommendation model accuracy has improved from 78% to 85%.

David: That's great! The new feature engineering pipeline is working well.
We should document this in a technical blog post.

Emma: Users are asking for personalization. Can we add user preferences?

Sarah: Yes, we can implement that in the next sprint. I estimate 2 weeks.

David: I'll prepare the dataset and start training a new model.

Emma: Perfect. Let's also create clear documentation for the API endpoints.

Next Steps:
- Sarah: Implement user preference feature (2 weeks)
- David: Prepare dataset and train new model
- Emma: Design user preference UI
- All: Write technical blog post about improvements
"""

def main():
    """Run example workflows."""

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("\n❌ Error: GROQ_API_KEY not found!")
        print("Please create a .env file with your Groq API key")
        print("Get your key at: https://console.groq.com/keys")
        return

    print("="*80)
    print("MULTI-AGENT SYSTEM - EXAMPLE WORKFLOWS")
    print("="*80)

    # Choose which example to run
    print("\nAvailable examples:")
    print("1. Product Strategy Meeting")
    print("2. Technical Discussion")

    choice = input("\nSelect example (1 or 2, or press Enter for #1): ").strip()

    if choice == "2":
        transcript = tech_meeting
        output_file = "tech_meeting_results.json"
    else:
        transcript = product_meeting
        output_file = "product_meeting_results.json"

    # Initialize manager and run workflow
    manager = ManagerAgent()
    results = manager.run_workflow(transcript)

    # Save results
    manager.save_results(output_file)

    print(f"\n✅ Workflow complete! Results saved to {output_file}")
    print("\nQuick Summary:")
    print(f"  Insights: {len(results.get('insights', []))}")
    print(f"  Todos: {len(results.get('todos', []))}")
    print(f"  GitHub Repo: {results.get('github', {}).get('name', 'N/A')}")
    print(f"  Article: {results.get('article', {}).get('title', 'N/A')[:60]}...")


if __name__ == "__main__":
    main()
