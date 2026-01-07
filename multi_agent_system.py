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

        print(f"\n{'='*60}")
        print(f"🤖 AGENT: {self.name}")
        print(f"🎯 GOAL: {goal}")
        print(f"{'='*60}\n")

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
                {"role": "system", "content": "You extract key insights from meetings. Always respond with valid JSON only."},
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
                {"role": "system", "content": "You are a professional note-taker. Always respond with valid JSON only."},
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


class GitHubManagerAgent(BaseAgent):
    """Agent specialized in managing GitHub repositories (mock implementation)."""

    def __init__(self):
        super().__init__("GitHubManager", "Create and manage GitHub repositories")

    def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan GitHub repository creation."""

        notes = context.get("notes", {})

        prompt = f"""You are creating a GitHub repository for a project discussed in a meeting.

Meeting Summary:
{notes.get('summary', '')}

Create a repository plan with:
- Repository name (lowercase, hyphens)
- Description
- README content structure
- Initial files needed

Respond with ONLY valid JSON:
{{
  "action": "COMPLETE",
  "repository": {{
    "name": "...",
    "description": "...",
    "readme_outline": ["Section 1", "Section 2", ...],
    "initial_files": ["file1.py", "file2.md", ...]
  }},
  "answer": "Created GitHub repository plan"
}}"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You plan GitHub repositories. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        plan = self._parse_json(response.choices[0].message.content)
        self._log_phase("🧠 PLAN", {"action": "COMPLETE", "repo_name": plan.get("repository", {}).get("name", "N/A")})
        return plan


class ArticleWriterAgent(BaseAgent):
    """Agent specialized in writing Medium articles."""

    def __init__(self):
        super().__init__("ArticleWriter", "Write and structure Medium articles")

    def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan article writing."""

        notes = context.get("notes", {})
        insights = context.get("insights", [])

        prompt = f"""You are a professional Medium article writer.

Key Insights:
{json.dumps(insights[:5], indent=2)}

Meeting Summary:
{notes.get('summary', '')}

Write a Medium article outline with:
- Catchy title
- Subtitle
- Introduction (hook)
- 3-5 main sections with subsections
- Conclusion
- Call to action

Respond with ONLY valid JSON:
{{
  "action": "COMPLETE",
  "article": {{
    "title": "...",
    "subtitle": "...",
    "sections": [
      {{"heading": "...", "points": ["...", "..."]}},
      ...
    ],
    "conclusion": "...",
    "cta": "..."
  }},
  "answer": "Created Medium article outline"
}}"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You write engaging Medium articles. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        plan = self._parse_json(response.choices[0].message.content)
        self._log_phase("🧠 PLAN", {"action": "COMPLETE", "title": plan.get("article", {}).get("title", "N/A")[:50]})
        return plan



class SentimentAnalyzerAgent(BaseAgent):
    """Agent specialized in analyzing sentiment of text."""

    def __init__(self):
        super().__init__("SentimentAnalyser", "Analyse sentiment of text data")

    def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan article writing."""

        article = context.get("article", [])

        prompt = f"""You are a professional sentiment analysis engine.

        Text to Analyze:
        \"\"\"
        {article}
        \"\"\"

        Analyze the sentiment of the above text.

        Your analysis must include:
        - Overall sentiment (Positive / Neutral / Negative / Mixed)
        - Confidence score (0.0 – 1.0)
        - Emotional tones detected (e.g., joy, anger, frustration, excitement)
        - Key positive signals (phrases or reasons)
        - Key negative signals (phrases or reasons)
        - A short explanation (2–3 lines)

        Respond with ONLY valid JSON in the following format:
        {{
          "action": "COMPLETE",
          "sentiment_analysis": {{
            "overall_sentiment": "...",
            "confidence_score": 0.0,
            "emotions": ["...", "..."],
            "positive_signals": ["...", "..."],
            "negative_signals": ["...", "..."],
            "explanation": "..."
          }},
          "answer": "Sentiment analysis completed"
        }}
        """

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert NLP sentiment analyzer. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        analysis = self._parse_json(response.choices[0].message.content)

        self._log_phase(
            "🧠 SENTIMENT ANALYSIS",
            {
                "action": analysis.get("action", "N/A"),
                "overall_sentiment": analysis
                .get("sentiment_analysis", {})
                .get("overall_sentiment", "N/A"),
                "confidence_score": analysis
                .get("sentiment_analysis", {})
                .get("confidence_score", 0.0),
            }
        )

        return analysis


# =============================================================================
# MANAGER AGENT (Orchestrator)
# =============================================================================

class ManagerAgent(BaseAgent):
    """Manager agent that orchestrates specialized agents as tools."""

    def __init__(self):
        super().__init__("Manager", "Orchestrate specialized agents to complete complex tasks")

        # Initialize specialized agents as "tools"
        self.agent_tools = {
            "insight_extractor": InsightExtractorAgent(),
            "note_taker": NoteTakerAgent(),
            "todo_creator": TodoCreatorAgent(),
            "github_manager": GitHubManagerAgent(),
            "article_writer": ArticleWriterAgent(),
            "sentiment_analyzer": SentimentAnalyzerAgent(),
        }

        self.workflow_results = {}

    def run_workflow(self, transcript: str) -> Dict[str, Any]:
        """
        Run the complete workflow:
        1. Extract insights
        2. Create notes
        3. Create todos
        4. Create GitHub repo plan
        5. Write article outline
        6. Extract sentiment analysis
        """

        print("\n" + "="*80)
        print("🎬 STARTING MULTI-AGENT WORKFLOW")
        print("="*80 + "\n")

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

        # Step 4: GitHub Repository Plan
        print("\n📍 STEP 4: Creating GitHub Repository Plan")
        print("-" * 80)
        github_result = self.agent_tools["github_manager"].run(
            goal="Create GitHub repository plan",
            context=workflow_context,
            max_iterations=2
        )

        if github_result["success"]:
            github_data = github_result["result"]  # Already a dict
            self.workflow_results["github"] = github_data.get("repository", {})

        # Step 5: Article Outline
        print("\n📍 STEP 5: Writing Medium Article Outline")
        print("-" * 80)
        article_result = self.agent_tools["article_writer"].run(
            goal="Write Medium article outline",
            context=workflow_context,
            max_iterations=2
        )

        if article_result["success"]:
            article_data = article_result["result"]  # Already a dict
            self.workflow_results["article"] = article_data.get("article", {})

        # Step 6: Sentiment Analysis
        print("\n📍 STEP 6: Analyzing Article Sentiment")
        print("-" * 80)
        sentiment_result = self.agent_tools["sentiment_analyzer"].run(
            goal="Analyze sentiment of the article",
            context={"article": json.dumps(self.workflow_results.get("article", {}))},
            max_iterations=2
        )
        if sentiment_result["success"]:
            sentiment_data = sentiment_result["result"]  # Already a dict
            self.workflow_results["sentiment_analysis"] = sentiment_data.get("sentiment_analysis", {})

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
    print("\n" + "="*80)
    print("✅ WORKFLOW COMPLETE - SUMMARY")
    print("="*80)
    print(f"\n📊 Insights Extracted: {len(results.get('insights', []))}")
    print(f"📝 Notes Sections: {len(results.get('notes', {}))}")
    print(f"✅ Todos Created: {len(results.get('todos', []))}")
    print(f"🔨 GitHub Repo: {results.get('github', {}).get('name', 'N/A')}")
    print(f"📰 Article Title: {results.get('article', {}).get('title', 'N/A')}")
    print(f"💬 Sentiment: {results.get('sentiment_analysis', {}).get('overall_sentiment', 'N/A')}")
    print("\n💡 Check 'workflow_results.json' for detailed output!")
    print("="*80 + "\n")
