#!/usr/bin/env python3
"""
Multi-Agent System with Manager Pattern

A beginner-friendly multi-agent implementation using the SPOAR loop pattern.
Each agent is specialized for a specific task and the Manager orchestrates them.

Architecture:
- Manager Agent: Orchestrates specialized agents
- Specialized Agents: InsightExtractor, NoteTaker, TodoCreator, GitHubManager, ArticleWriter

Model: openai/gpt-oss-120b (via Groq Cloud)

Usage:
    python multi_agent_system.py

Requirements:
    pip install groq python-dotenv
"""

import os
import json
from typing import Dict, Any, List, Optional
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


# =============================================================================
# BASE AGENT CLASS (SPOAR Loop)
# =============================================================================

class BaseAgent:
    """Base agent class implementing the SPOAR loop pattern."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "openai/gpt-oss-120b"
        self.memory = []

    def run(self, goal: str, context: Dict[str, Any] = None, max_iterations: int = 3) -> Dict[str, Any]:
        """Main SPOAR loop for the agent."""

        print(f"\n{'=' * 60}")
        print(f"🤖 AGENT: {self.name}")
        print(f"🎯 GOAL: {goal}")
        print(f"{'=' * 60}\n")

        # Initialize context
        if context is None:
            context = {}
        context["goal"] = goal
        context["iteration"] = 0

        for iteration in range(1, max_iterations + 1):
            context["iteration"] = iteration
            print(f"\n--- {self.name.upper()} - ITERATION {iteration} ---\n")

            # SPOAR Loop
            context = self._sense(context)
            plan = self._plan(context)

            if plan["action"] == "COMPLETE":
                self._log_phase("✅ COMPLETE", {"result": "Task completed"})
                return {
                    "success": True,
                    "result": plan,  # Return the full plan object
                    "agent": self.name
                }

            result = self._act(plan)
            observation = self._observe(plan, result)
            reflection = self._reflect(context, observation)

            # Update context
            context["last_action"] = plan
            context["last_result"] = result
            context["last_reflection"] = reflection

        return {
            "success": False,
            "result": "Max iterations reached",
            "agent": self.name
        }

    def _sense(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """SENSE: Gather information about current state."""
        self._log_phase("👁️  SENSE", {
            "iteration": context["iteration"],
            "goal": context["goal"][:80] + "..." if len(context["goal"]) > 80 else context["goal"]
        })
        return context

    def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """PLAN: Use LLM to decide next action. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _plan")

    def _act(self, plan: Dict[str, Any]) -> Any:
        """ACT: Execute the planned action. Override in subclasses."""
        return plan.get("output", "")

    def _observe(self, plan: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """OBSERVE: Record what happened."""
        observation = {
            "action_taken": plan.get("action"),
            "result": result,
            "success": "ERROR" not in str(result)
        }

        self._log_phase("📊 OBSERVE", {
            "action": observation["action_taken"],
            "success": observation["success"]
        })

        return observation

    def _reflect(self, context: Dict[str, Any], observation: Dict[str, Any]) -> str:
        """REFLECT: Evaluate progress."""
        reflection = "Progress made" if observation["success"] else "Need to adjust approach"
        self._log_phase("💭 REFLECT", {"reflection": reflection})
        return reflection

    def _log_phase(self, phase: str, data: Dict[str, Any]):
        """Simple logging for visibility."""
        print(f"{phase}")
        for key, value in data.items():
            print(f"  {key}: {value}")
        print()

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # Fallback: try to extract JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
            raise


# =============================================================================
# SPECIALIZED AGENTS
# =============================================================================

class RecapAgent(BaseAgent):
    """Agent specialized in creating meeting recaps."""

    def __init__(self):
        super().__init__("RecapAgent", "Create concise meeting recaps")

    def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan recap creation."""

        transcript = context.get("transcript", "")

        prompt = f"""You are a professional meeting recap extractor.

        Meeting Transcript:
        \"\"\"
        {transcript[:3000]}...
        \"\"\"

        Your task:
        - Extract the most important outcomes, decisions, and next steps
        - Write a concise recap in 3–4 sentences
        - Keep the language clear, neutral, and action-oriented

        Respond with ONLY valid JSON in the following format:
        {{
          "action": "COMPLETE",
          "recap": "...",
          "answer": "Meeting recap extracted"
        }}
        """

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You create meeting recaps. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        plan = self._parse_json(response.choices[0].message.content)
        self._log_phase("🧠 PLAN", {"action": "COMPLETE", "recap_length": len(plan.get("recap", ""))})
        return plan


class InsightExtractorAgent(BaseAgent):
    """Agent specialized in extracting key insights from transcripts."""

    def __init__(self):
        super().__init__("InsightExtractor", "Extract key insights from meeting transcripts")

    def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan insight extraction."""

        transcript = context.get("transcript", "")

        prompt = f"""You are an expert at analyzing meeting transcripts and extracting key insights.

Meeting Transcript:
{transcript[:3000]}...

Your task: Extract 5-7 key insights from this transcript.

Respond with ONLY valid JSON:
{{
  "action": "COMPLETE",
  "insights": [
    "Insight 1",
    "Insight 2",
    ...
  ],
  "answer": "Extracted N key insights from the meeting transcript"
}}"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You extract key insights from meetings. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        plan = self._parse_json(response.choices[0].message.content)
        self._log_phase("🧠 PLAN", {"action": "COMPLETE", "insights_count": len(plan.get("insights", []))})
        return plan


class NoteTakerAgent(BaseAgent):
    """Agent specialized in creating concise notes."""

    def __init__(self):
        super().__init__("NoteTaker", "Create concise, structured notes")

    def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan note creation."""

        insights = context.get("insights", [])
        transcript = context.get("transcript", "")

        prompt = f"""You are an expert note-taker who creates concise, well-structured meeting notes.

Key Insights:
{json.dumps(insights, indent=2)}

Meeting Transcript Excerpt:
{transcript[:2000]}...

Create concise meeting notes with these sections:
1. Summary (2-3 sentences)
2. Key Points (bullet points)
3. Decisions Made
4. Next Steps

Respond with ONLY valid JSON:
{{
  "action": "COMPLETE",
  "notes": {{
    "summary": "...",
    "key_points": ["...", "..."],
    "decisions": ["...", "..."],
    "next_steps": ["...", "..."]
  }},
  "answer": "Created structured meeting notes"
}}"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are a professional note-taker. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        plan = self._parse_json(response.choices[0].message.content)
        self._log_phase("🧠 PLAN", {"action": "COMPLETE", "sections": len(plan.get("notes", {}))})
        return plan


class TodoCreatorAgent(BaseAgent):
    """Agent specialized in creating actionable todos."""

    def __init__(self):
        super().__init__("TodoCreator", "Create actionable todo items")

    def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan todo creation."""

        notes = context.get("notes", {})

        prompt = f"""You are an expert at creating actionable todo items from meeting notes.

Meeting Notes:
{json.dumps(notes, indent=2)}

Create 5-8 specific, actionable todo items. Each should:
- Start with an action verb
- Be specific and measurable
- Include who is responsible (if mentioned)
- Have a priority (High/Medium/Low)

Respond with ONLY valid JSON:
{{
  "action": "COMPLETE",
  "todos": [
    {{
      "task": "...",
      "priority": "High",
      "assignee": "..."
    }},
    ...
  ],
  "answer": "Created N actionable todo items"
}}"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You create actionable todos. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        plan = self._parse_json(response.choices[0].message.content)
        self._log_phase("🧠 PLAN", {"action": "COMPLETE", "todos_count": len(plan.get("todos", []))})
        return plan


class BriefExtractorAgent(BaseAgent):
    """Agent specialized in extracting structured project briefs from meeting transcripts."""

    def __init__(self):
        super().__init__(
            "BriefExtractor",
            "Extract objectives, scope, and deliverables from meeting transcripts"
        )

    def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan and execute project brief extraction."""

        transcript = context.get("transcript", "")

        prompt = f"""You are a professional project brief extractor.

Meeting Transcript:
\"\"\"
{transcript[:3000]}...
\"\"\"

Your task:
- Extract the project objectives
- Define the project scope
- List concrete deliverables if mentioned
- If information is missing, infer conservatively without adding new facts
- Keep language concise and factual

Respond with ONLY valid JSON in the following format:
{{
  "action": "COMPLETE",
  "brief": {{
    "objectives": "...",
    "scope": "...",
    "deliverables": ["...", "..."]
  }},
  "answer": "Project brief extracted"
}}
"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured project briefs from meeting transcripts. "
                        "Never add speculative details. Always respond with valid JSON only."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        plan = self._parse_json(response.choices[0].message.content)

        self._log_phase(
            "🧠 BRIEF EXTRACTION",
            {
                "action": plan.get("action", "N/A"),
                "has_deliverables": bool(plan.get("brief", {}).get("deliverables"))
            }
        )

        return plan


class EmailAgentForRecap(BaseAgent):
    """Agent specialized in drafting professional recap emails."""

    def __init__(self):
        super().__init__(
            "EmailAgent",
            "Draft professional recap emails based on meeting summaries"
        )

    def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan and execute recap email drafting."""

        recap = context.get("recap", "")

        prompt = f"""You are a professional business email drafter.

Meeting Recap:
\"\"\"
{recap[:2000]}...
\"\"\"

Your task:
- Draft a clear and professional recap email
- Summarize key points without adding new information
- Use a formal, business-appropriate tone
- Keep the email concise and readable

Respond with ONLY valid JSON in the following format:
{{
  "action": "COMPLETE",
  "email": {{
    "subject": "...",
    "body": "...",
    "recipients": ["...", "..."],
    "tone": "formal",
    "attachments": ["meeting_notes.pdf"]
  }},
  "answer": "Professional recap email drafted"
}}
"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You draft professional business emails based on provided context. "
                        "Do not invent recipients or attachments unless implied. "
                        "Always respond with valid JSON only."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        plan = self._parse_json(response.choices[0].message.content)

        self._log_phase(
            "🧠 EMAIL DRAFTING",
            {
                "action": plan.get("action", "N/A"),
                "email_length": len(plan.get("email", {}).get("body", "")),
                "has_recipients": bool(plan.get("email", {}).get("recipients"))
            }
        )

        return plan

class EmailAgentForTodos(BaseAgent):
    """Agent specialized in drafting TODO emails using provided action items."""

    def __init__(self):
        super().__init__(
            "EmailAgentForTodos",
            "Draft professional TODO emails using existing action items"
        )

    def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan and execute TODO email drafting."""

        todos = context.get("todos", [])
        recipients = context.get("recipients", [])

        prompt = f"""You are a professional business email drafter.

Provided TODOs:
{todos}

Your task:
- Draft a professional email summarizing the provided TODOs
- Do NOT add, remove, or modify any TODOs
- Do NOT invent owners, deadlines, or tasks
- Use a formal, business-appropriate tone
- Keep the email clear and concise

Respond with ONLY valid JSON in the following format:
{{
  "action": "COMPLETE",
  "email": {{
    "subject": "Assigned Action Items & Next Steps",
    "body": "...",
    "recipients": {recipients},
    "tone": "formal"
  }},
  "answer": "TODO email drafted"
}}
"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You draft professional TODO emails using only the provided data. "
                        "Never invent or modify tasks. "
                        "Always respond with valid JSON only."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        plan = self._parse_json(response.choices[0].message.content)

        self._log_phase(
            "🧠 TODO EMAIL DRAFTING",
            {
                "action": plan.get("action", "N/A"),
                "todo_count": len(todos),
                "recipient_count": len(recipients)
            }
        )

        return plan

class EmailAgentForLeadershipBrief(BaseAgent):
    """Agent specialized in drafting leadership-facing meeting brief emails."""

    def __init__(self):
        super().__init__(
            "EmailAgentForLeadershipBrief",
            "Draft concise leadership emails using provided meeting briefs"
        )

    def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan and execute leadership brief email drafting."""

        brief = context.get("brief", {})
        recipients = context.get("recipients", [])

        print("\n[DEBUG] Brief content for leadership email drafting:")
        print(json.dumps(brief, indent=2))




        prompt = f"""You are a senior executive communications specialist.

Provided Meeting Brief:
{brief}

Your task:
- Draft a concise, leadership-facing email
- Summarize objectives, scope, and deliverables clearly
- Use a formal, executive-appropriate tone
- Highlight only high-impact information
- Do NOT add or infer information beyond the provided brief

Respond with ONLY valid JSON in the following format:
{{
  "action": "COMPLETE",
  "email": {{
    "subject": "Meeting Brief: Key Objectives, Scope & Deliverables",
    "body": "...",
    "recipients": {recipients},
    "tone": "executive"
  }},
  "answer": "Leadership brief email drafted"
}}
"""
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You draft executive-level emails using only provided data. "
                        "Never add assumptions or new information. "
                        "Always respond with valid JSON only."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        plan = self._parse_json(response.choices[0].message.content)

        self._log_phase(
            "🧠 LEADERSHIP BRIEF EMAIL",
            {
                "action": plan.get("action", "N/A"),
                "has_objectives": bool(brief.get("objectives")),
                "has_deliverables": bool(brief.get("deliverables")),
                "recipient_count": len(recipients)
            }
        )

        return plan

# =============================================================================
# MANAGER AGENT (Orchestrator)
# =============================================================================


class ManagerAgent(BaseAgent):
    """Manager agent that orchestrates specialized agents as tools."""

    def __init__(self):
        super().__init__("Manager", "Orchestrate specialized agents to complete complex tasks")

        # Initialize specialized agents as "tools"
        self.agent_tools = {
            "recap_agent": RecapAgent(),
            "insight_extractor": InsightExtractorAgent(),
            "note_taker": NoteTakerAgent(),
            "todo_creator": TodoCreatorAgent(),
            "brief_extractor": BriefExtractorAgent(),
            "email_agent_recap": EmailAgentForRecap(),
            "email_agent_todos": EmailAgentForTodos(),
            "email_agent_leadership": EmailAgentForLeadershipBrief(),
        }

        self.workflow_results = {}

    def run_workflow(self, transcript: str) -> Dict[str, Any]:
        """
        Run the complete workflow:
        1. Extract Rceap
        2. Extract insights
        3. Create notes
        4. Create todos
        5. Email Recap
        6. Email Todos
        7. Email Leadership Brief
        """

        print("\n" + "=" * 80)
        print("🎬 STARTING MULTI-AGENT WORKFLOW")
        print("=" * 80 + "\n")

        workflow_context = {"transcript": transcript}

        # Step 1: Extract Insights
        print("\n📍 STEP 1: Extracting Key Insights")
        print("-" * 80)
        insights_result = self.agent_tools["insight_extractor"].run(
            goal="Extract key insights from the meeting transcript",
            context=workflow_context,
            max_iterations=2
        )

        if insights_result["success"]:
            insights_data = insights_result["result"]  # Already a dict
            workflow_context["insights"] = insights_data.get("insights", [])
            self.workflow_results["insights"] = insights_data.get("insights", [])

        # Step 2: Create Notes
        print("\n📍 STEP 2: Creating Concise Notes")
        print("-" * 80)
        notes_result = self.agent_tools["note_taker"].run(
            goal="Create concise meeting notes",
            context=workflow_context,
            max_iterations=2
        )

        if notes_result["success"]:
            notes_data = notes_result["result"]  # Already a dict
            workflow_context["notes"] = notes_data.get("notes", {})
            self.workflow_results["notes"] = notes_data.get("notes", {})

        # Step 3: Create Todos
        print("\n📍 STEP 3: Creating Action Items")
        print("-" * 80)
        todos_result = self.agent_tools["todo_creator"].run(
            goal="Create actionable todo items",
            context=workflow_context,
            max_iterations=2
        )

        if todos_result["success"]:
            todos_data = todos_result["result"]  # Already a dict
            self.workflow_results["todos"] = todos_data.get("todos", [])

        # Step 4: Extract Recap
        print("\n📍 STEP 4: Extracting Meeting Recap")
        print("-" * 80)
        recap_result = self.agent_tools["recap_agent"].run(
            goal="Extract a concise meeting recap",
            context=workflow_context,
            max_iterations=2
        )

        if recap_result["success"]:
            recap_data = recap_result["result"]  # Already a dict
            workflow_context["recap"] = recap_data.get("recap", "")
            self.workflow_results["recap"] = recap_data.get("recap", "")

        # Extract project brief for later use
        brief_result = self.agent_tools["brief_extractor"].run(
            goal="Extract project brief from meeting transcript",
            context=workflow_context,
            max_iterations=2
        )

        if brief_result["success"]:
            brief_data = brief_result["result"]  # Already a dict
            workflow_context["brief"] = brief_data.get("brief", {})
            self.workflow_results["brief"] = brief_data.get("brief", {})


        # Step 5: Draft Recap Email
        print("\n📍 STEP 5: Drafting Recap Email")
        print("-" * 80)
        email_recap_result = self.agent_tools["email_agent_recap"].run(
            goal="Draft a professional recap email",
            context=workflow_context,
            max_iterations=2
        )

        if email_recap_result["success"]:
            email_recap_data = email_recap_result["result"]  # Already a dict
            self.workflow_results["email_recap"] = email_recap_data.get("email", {})

        # Step 6: Draft Todos Email
        print("\n📍 STEP 6: Drafting Todos Email")
        print("-" * 80)
        email_todos_result = self.agent_tools["email_agent_todos"].run(
            goal="Draft a professional TODO email",
            context={
                "todos": self.workflow_results.get("todos", []),
                "recipients": ["xyz@email.com", "abc@email.com"]
            },
            max_iterations=2
        )
        if email_todos_result["success"]:
            email_todos_data = email_todos_result["result"]  # Already a dict
            self.workflow_results["email_todos"] = email_todos_data.get("email", {})

        # Step 7: Draft Leadership Brief Email
        print("\n📍 STEP 7: Drafting Leadership Brief Email")
        print("-" * 80)

        brief_extractor_result = self.agent_tools["brief_extractor"].run(
            goal="Extract project brief from meeting transcript",
            context={
                "brief": workflow_context.get("brief", {}),
                "recipients": ["abc@email.com", "xyz@email.com"]
            },
            max_iterations=2
        )

        if brief_extractor_result["success"]:
            brief_data = brief_extractor_result["result"]  # Already a dict
            workflow_context["brief_email"] = brief_data.get("brief_email", {})
            self.workflow_results["brief_email"] = brief_data.get("brief_email", {})

        return self.workflow_results

    def save_results(self, output_file: str = "workflow_results.json"):
        """Save all workflow results to a JSON file."""
        with open(output_file, "w") as f:
            json.dump(self.workflow_results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("\n❌ Error: GROQ_API_KEY not found!")
        print("Please create a .env file with your Groq API key:")
        print("  GROQ_API_KEY=gsk-your-key-here")
        print("Get your key at: https://console.groq.com/keys")
        exit(1)

    # Load meeting transcript from file
    transcript_file = "sample_transcript.txt"

    if not os.path.exists(transcript_file):
        print(f"\n❌ Error: {transcript_file} not found!")
        print(f"Please create a file named '{transcript_file}' with your meeting transcript.")
        exit(1)

    with open(transcript_file, "r") as f:
        sample_transcript = f.read()

    # Initialize and run manager
    manager = ManagerAgent()

    print("\n🚀 Starting Multi-Agent Workflow")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📄 Transcript: {transcript_file}")

    # Run the complete workflow
    results = manager.run_workflow(sample_transcript)

    # Save results
    manager.save_results()

    # Print summary
    print("\n" + "=" * 80)
    print("✅ WORKFLOW COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\n📊 Insights Extracted: {len(results.get('insights', []))}")
    print(f"📝 Notes Sections: {len(results.get('notes', {}))}")
    print(f"✅ Todos Created: {len(results.get('todos', []))}")
    print(f"🔨 GitHub Repo: {results.get('github', {}).get('name', 'N/A')}")
    print(f"📰 Article Title: {results.get('article', {}).get('title', 'N/A')}")
    print("\n💡 Check 'workflow_results.json' for detailed output!")
    print("=" * 80 + "\n")
