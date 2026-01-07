# Multi-Agent System with SPOAR Pattern

A beginner-friendly multi-agent system that processes meeting transcripts using specialized AI agents. Built on the **SPOAR (Sense-Plan-Act-Observe-Reflect)** pattern with a manager orchestrating 5 specialized agents.

**Model:** Uses `openai/gpt-oss-120b` via Groq Cloud

---

## 🎯 What Does This Do?

**Input:** Meeting transcript (text file)

**Output:** Complete analysis with:
- 📊 **Key Insights** - 5-7 main takeaways
- 📝 **Meeting Notes** - Structured summary with decisions and action items
- ✅ **Action Items** - Todos with priorities and assignees
- 🔨 **GitHub Repo Plan** - Repository structure and documentation outline
- 📰 **Article Outline** - Medium article with title, sections, and CTA

---

## 🚀 Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

```bash
# Create .env file with your Groq API key
echo "GROQ_API_KEY=your-key-here" > .env
```

Get your free API key at: [console.groq.com/keys](https://console.groq.com/keys)

### 3. Run It!

```bash
# Run with provided sample transcript
python multi_agent_system.py

# Check the results
cat workflow_results.json
```

That's it! The system will process `sample_transcript.txt` and generate complete analysis.

---

## 📖 Table of Contents

- [Quick Start](#-quick-start-5-minutes)
- [How to Use](#-how-to-use)
- [Architecture](#-architecture)
- [How It Works](#-how-it-works-spoar-loop)
- [Customization](#-customization)
- [Tutorial](#-tutorial-understanding-multi-agents)
- [For Students](#-for-students--educators)
- [Troubleshooting](#-troubleshooting)
- [Examples](#-examples)

---

## 💻 How to Use

### Using Your Own Transcript

**Method 1: Edit the file (easiest)**

```bash
# Edit the transcript file

# Replace with your meeting content, then run
python multi_agent_system.py
```

**Method 2: Use programmatically**

```python
from multi_agent_system import ManagerAgent

# Your transcript
with open("my_meeting.txt", "r") as f:
    transcript = f.read()

# Run the workflow
manager = ManagerAgent()
results = manager.run_workflow(transcript)

# Save results
manager.save_results("my_results.json")
```

### Recommended Transcript Format

```
Meeting: [Title]
Date: [Date]
Attendees: [Name (Role), Name (Role), ...]

[Name]: [What they said...]

[Name]: [Response...]

Action Items:
- [Item 1]
- [Item 2]
```

**Example:**

```
Meeting: Product Review
Date: December 6, 2024
Attendees: Alice (CEO), Bob (CTO), Carol (Designer)

Alice: Let's review the new dashboard design.

Carol: I've updated the UI based on feedback.
The new color scheme is more accessible.

Bob: The backend API is ready. We can integrate this week.

Action Items:
- Carol: Finalize UI by Wednesday
- Bob: Complete API integration by Thursday
```

---

## 🏗️ Architecture

### Manager Pattern

```
Manager Agent (Orchestrator)
    │
    ├─► InsightExtractorAgent    → Extracts 5-7 key insights
    ├─► NoteTakerAgent           → Creates structured notes
    ├─► TodoCreatorAgent         → Generates actionable todos
    ├─► GitHubManagerAgent       → Plans repository structure
    └─► ArticleWriterAgent       → Writes Medium article outline
```

### Data Flow

```
Input: sample_transcript.txt
    │
    ▼
Step 1: Extract Insights
    │ → insights: ["insight 1", "insight 2", ...]
    ▼
Step 2: Create Notes
    │ → notes: {summary, key_points, decisions, next_steps}
    ▼
Step 3: Create Todos
    │ → todos: [{task, priority, assignee}, ...]
    ▼
Step 4: Plan GitHub Repo
    │ → github: {name, description, readme_outline, files}
    ▼
Step 5: Write Article
    │ → article: {title, sections, conclusion, cta}
    ▼
Output: workflow_results.json
```

### System Components

**1. Base Agent Class** - Implements the SPOAR loop
```python
class BaseAgent:
    def run(self, goal, context, max_iterations=3):
        # SPOAR Loop
        for iteration in range(max_iterations):
            context = self._sense(context)      # 👁️  Gather info
            plan = self._plan(context)          # 🧠 Decide action
            if plan["action"] == "COMPLETE":
                return plan
            result = self._act(plan)            # ⚡ Execute
            observation = self._observe(plan, result)  # 📊 Record
            reflection = self._reflect(context, observation)  # 💭 Evaluate
```

**2. Specialized Agents** - Each handles one specific task

**3. Manager Agent** - Coordinates the workflow

---

## 🔄 How It Works: SPOAR Loop

Every agent follows the same 5-phase pattern:

### The SPOAR Cycle

```
┌──────────────────────────────────────┐
│  1. SENSE (Gather Information)       │
│     - Review goal                    │
│     - Check context                  │
│     - See previous actions           │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  2. PLAN (Decide Action)             │
│     - Call LLM with context          │
│     - LLM decides what to do         │
│     - Returns structured JSON        │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  3. ACT (Execute)                    │
│     - Perform the planned action     │
│     - Generate the output            │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  4. OBSERVE (Record Results)         │
│     - Log what happened              │
│     - Check for errors               │
│     - Validate output                │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  5. REFLECT (Evaluate Progress)      │
│     - Did we make progress?          │
│     - Should we continue?            │
│     - What's next?                   │
└──────────────┬───────────────────────┘
               │
               ├─ Complete? → Return Result
               └─ Not done? → Next Iteration
```

**Why SPOAR?**
- **Autonomy:** Agent decides its own actions
- **Adaptability:** Can recover from errors
- **Transparency:** Every step is logged
- **Reliability:** Structured, repeatable process

---

## 🎨 Customization

### 1. Add a New Specialized Agent

Create your own agent by extending `BaseAgent`:

```python
class SentimentAnalyzerAgent(BaseAgent):
    """Analyzes sentiment and emotional tone."""

    def __init__(self):
        super().__init__(
            "SentimentAnalyzer",
            "Analyze sentiment and tone of discussions"
        )

    def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Custom planning for sentiment analysis."""

        transcript = context.get("transcript", "")

        prompt = f"""Analyze the sentiment and tone of this transcript:

{transcript}

Respond with JSON:
{{
  "action": "COMPLETE",
  "sentiment": {{
    "overall": "positive/negative/neutral",
    "tone": "professional/casual/urgent",
    "confidence": 0.85
  }}
}}"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You analyze sentiment. Respond with JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return self._parse_json(response.choices[0].message.content)
```

### 2. Add to Manager Workflow

```python
class ManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__(...)
        # Add your new agent
        self.agent_tools["sentiment_analyzer"] = SentimentAnalyzerAgent()

    def run_workflow(self, transcript: str):
        # ... existing steps ...

        # Add new step
        print("\n📍 STEP 6: Analyzing Sentiment")
        sentiment_result = self.agent_tools["sentiment_analyzer"].run(
            goal="Analyze meeting sentiment",
            context=workflow_context,
            max_iterations=2
        )

        if sentiment_result["success"]:
            self.workflow_results["sentiment"] = sentiment_result["result"]["sentiment"]
```

### 3. Modify Agent Behavior

Change any agent's prompt in its `_plan()` method:

```python
class NoteTakerAgent(BaseAgent):
    def _plan(self, context):
        # Customize the prompt
        prompt = f"""
        Create DETAILED meeting notes with:
        - Executive summary (3-5 sentences)
        - Detailed discussion points
        - All decisions with rationale
        - Action items with deadlines
        - Risk assessment
        - Follow-up items

        Make it comprehensive!

        {context['transcript']}
        """
        # ... rest of code
```

### 4. Run Agents in Parallel

Speed up the workflow:

```python
from concurrent.futures import ThreadPoolExecutor

def run_workflow_parallel(self, transcript: str):
    # Step 1: Insights (must go first)
    insights = self.agent_tools["insight_extractor"].run(...)

    # Steps 2-4: Run in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        notes_future = executor.submit(
            self.agent_tools["note_taker"].run,
            "Create notes", workflow_context, 2
        )
        todos_future = executor.submit(
            self.agent_tools["todo_creator"].run,
            "Create todos", workflow_context, 2
        )
        github_future = executor.submit(
            self.agent_tools["github_manager"].run,
            "Plan repo", workflow_context, 2
        )

        notes = notes_future.result()
        todos = todos_future.result()
        github = github_future.result()
```

---

## 📚 Tutorial: Understanding Multi-Agents

### From Single to Multi-Agent

**Single Agent (simple_agent.py):**
- One agent handles everything
- Limited by single context
- Hard to specialize

**Multi-Agent System:**
- Multiple specialized agents
- Each excels at one task
- Coordinated by manager

### Why Multiple Agents?

**Specialization Benefits:**
1. **Focus** - Each agent masters one domain
2. **Quality** - Specialized prompts = better output
3. **Modularity** - Easy to add/remove/modify agents
4. **Clarity** - Clear responsibilities

**Example:**

One agent trying to do everything:
```python
# Confused, generic agent
agent.run("Extract insights AND create notes AND write article")
# Result: Mediocre at all tasks
```

Specialized agents:
```python
# Each agent is an expert
insights = InsightExtractorAgent().run("Extract insights")
notes = NoteTakerAgent().run("Create notes", context={"insights": insights})
article = ArticleWriterAgent().run("Write article", context={"notes": notes})
# Result: Excellent at each task
```

### The Manager Pattern

**Manager's Role:**
1. **Coordinate** - Call agents in the right order
2. **Pass Context** - Share data between agents
3. **Aggregate** - Collect all results
4. **Handle Errors** - Graceful degradation

**Why Manager Pattern?**
- ✅ Central coordination point
- ✅ Clear workflow definition
- ✅ Easy to modify workflow
- ✅ Agents stay independent

### When to Use Single vs Multi-Agent

**Use Single Agent when:**
- Simple, focused task
- Quick prototyping
- No need for specialization
- Speed is critical

**Use Multi-Agent when:**
- Complex workflow with distinct steps
- Different expertise needed
- Quality matters more than speed
- Tasks can run in parallel

---

## 👨‍🎓 For Students & Educators

### Learning Assignment

A comprehensive **6-part assignment** teaches multi-agent systems from first principles:

**See [ASSIGNMENT.md](./ASSIGNMENT.md)** for complete details.

**Assignment Structure:**
1. **Part 1:** Understanding Single Agents (SPOAR loop)
2. **Part 2:** Agent Specialization (Why and how)
3. **Part 3:** Manager Pattern (Orchestration)
4. **Part 4:** Building Custom Agents (Practice)
5. **Part 5:** Advanced Patterns (Parallel, feedback loops)
6. **Part 6:** Final Project (Integration)

**Total Time:** 25-32 hours
**Difficulty:** Intermediate
**Prerequisites:** Python knowledge

### Key Concepts Taught

1. **SPOAR Loop** - Foundation of all agents
2. **Specialization** - Why focused agents beat generalists
3. **Orchestration** - Coordinating multiple agents
4. **Communication** - How agents share data
5. **Patterns** - Parallel execution, feedback loops, conditional routing

### Self-Study Path

1. **Week 1:** Understand the code
   - Run `python simple_agent.py`
   - Study `multi_agent_system.py`
   - Read this README thoroughly

2. **Week 2:** Experiment
   - Modify agent prompts
   - Add a simple agent
   - Try different transcripts

3. **Week 3:** Build
   - Create a custom agent from scratch
   - Integrate into workflow
   - Test and iterate

4. **Week 4:** Extend
   - Implement parallel execution
   - Add feedback loops
   - Build a complete system

---

## 🛠️ Troubleshooting

### Common Issues

**1. "GROQ_API_KEY not found"**

```bash
# Make sure .env exists
cat .env
# Should show: GROQ_API_KEY=gsk_...

# If not, create it
echo "GROQ_API_KEY=your-actual-key" > .env
```

**2. "JSON parsing error"**

The LLM sometimes returns invalid JSON. Solutions:
- Lower temperature for consistency
- Make prompts clearer
- Add examples in prompt
- Check `_parse_json` error handling

**3. "Agent gets stuck in loop"**

- Reduce `max_iterations` in agent calls
- Make goal more specific
- Ensure COMPLETE condition is clear in prompt

**4. "Poor quality output"**

Improve with:
- More specific prompts
- Better context passing
- Higher temperature for creativity
- Few-shot examples

**5. "Too slow"**

Speed up with:
- Parallel execution (see customization)
- Faster model (haiku instead of sonnet)
- Lower temperature
- Reduce max_iterations

### Debug Mode

Add logging to see what's happening:

```python
# In BaseAgent._plan()
print("\n=== LLM PROMPT ===")
print(prompt)
print("\n=== LLM RESPONSE ===")
print(response.choices[0].message.content)
```

---

## 📊 Examples

### Example 1: Quick Team Meeting

**Input (sample_transcript.txt):**
```
Meeting: Daily Standup
Date: Dec 6, 2024

Alice: Finished the login feature yesterday. Starting password reset today.
Bob: Still working on the API. Should finish today.
Carol: Need help with database schema review.
```

**Output (workflow_results.json):**
```json
{
  "insights": [
    "Alice completed login feature and moving to password reset",
    "Bob expects API completion today",
    "Carol needs database schema assistance"
  ],
  "todos": [
    {"task": "Complete password reset feature", "priority": "High", "assignee": "Alice"},
    {"task": "Finish API implementation", "priority": "High", "assignee": "Bob"},
    {"task": "Review Carol's database schema", "priority": "Medium", "assignee": "Team"}
  ]
}
```

### Example 2: Strategy Session

**Input:**
```
Meeting: Q1 Planning
Leadership Team

Discussed priorities for Q1. Decided to:
- Launch product by March
- Hire 2 engineers
- Focus on enterprise customers
- Build comprehensive documentation
```

**Output:** Strategic insights, high-level roadmap, article about the strategy

---

## 🔍 Project Structure

```
agent-c5/
├── multi_agent_system.py      # Main multi-agent system
├── simple_agent.py            # Original single agent (for learning)
├── example_workflow.py        # Example usage
├── sample_transcript.txt      # Input file (edit this!)
├── workflow_results.json      # Output file (generated)
├── requirements.txt           # Dependencies
├── .env                       # API key (create this)
├── README.md                  # This file
└── ASSIGNMENT.md              # Student assignment
```

---

## 🧪 Running Tests

```bash
# Quick test with short transcript
python test_multi_agent.py

# Full workflow with sample
python multi_agent_system.py

# Custom examples
python example_workflow.py
```

---

## 💡 Advanced Features

### 1. Process Multiple Transcripts

```python
import glob
from multi_agent_system import ManagerAgent

manager = ManagerAgent()

for file in glob.glob("transcripts/*.txt"):
    with open(file, "r") as f:
        transcript = f.read()

    results = manager.run_workflow(transcript)
    output_file = file.replace(".txt", "_results.json")
    manager.save_results(output_file)
```

### 2. Export to Different Formats

```python
# Export as Markdown
def export_markdown(results):
    md = f"""# Meeting Notes

## Summary
{results['notes']['summary']}

## Action Items
{chr(10).join(f"- [{t['priority']}] {t['task']} - {t['assignee']}" for t in results['todos'])}
"""
    with open("notes.md", "w") as f:
        f.write(md)
```

### 3. Integration Examples

```python
# Send to Slack
import requests

webhook_url = "https://hooks.slack.com/services/..."
requests.post(webhook_url, json={
    "text": f"Meeting Summary: {results['notes']['summary']}"
})

# Save to Notion
notion_api.pages.create(
    parent={"database_id": db_id},
    properties={
        "Summary": results['notes']['summary'],
        "Todos": results['todos']
    }
)
```

---

## 📖 Key Concepts

### 1. The SPOAR Loop

Every agent uses this pattern:
- **S**ense - Gather information
- **P**lan - Decide what to do (LLM)
- **A**ct - Execute the action
- **O**bserve - Record the results
- **R**eflect - Evaluate progress

### 2. Agent Specialization

Each agent has:
- **One clear purpose** - Extract insights, create notes, etc.
- **Custom prompt** - Optimized for its task
- **Structured output** - Consistent, parseable format

### 3. Manager Orchestration

The manager:
- Calls agents in sequence
- Passes context between agents
- Aggregates all results
- Handles the workflow

### 4. Context Flow

Data flows through the system:
```
transcript → insights → notes → todos
                      ↓
                   github plan
                      ↓
                   article
```

---

## 🎯 Design Principles

1. **Single Responsibility** - Each agent does one thing well
2. **Composition** - Manager composes agents
3. **Inheritance** - All agents inherit from BaseAgent
4. **Modularity** - Easy to add/remove agents
5. **Transparency** - Every step is logged
6. **Beginner-Friendly** - Simple to understand and extend

---

## 🚀 Next Steps

Once you're comfortable:

1. **Add Your Own Agent** - Create a specialized agent for your needs
2. **Optimize Performance** - Implement parallel execution
3. **Build Integrations** - Connect to your tools (Slack, Notion, etc.)
4. **Create a Web UI** - Build a Flask/FastAPI + React frontend
5. **Implement Learning** - Agents that improve based on feedback

---

## 📚 Resources

### This Project
- [ASSIGNMENT.md](./ASSIGNMENT.md) - Complete learning assignment

### External Resources
- [Groq Cloud API](https://console.groq.com/docs) - LLM provider
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/) - Production framework
- [AutoGen](https://microsoft.github.io/autogen/) - Microsoft's multi-agent framework

---

## 🤝 Contributing

Ideas for contributions:
- New specialized agents
- Additional workflow patterns
- Integration examples
- Performance optimizations
- Documentation improvements

---

## 📜 License

This is an educational project. Use freely for learning and building!

---

## 🎓 Learning Path Summary

```
1. Run the simple agent
   ↓
2. Understand SPOAR loop
   ↓
3. Run multi-agent system
   ↓
4. Study one specialized agent
   ↓
5. Add your own agent
   ↓
6. Modify the workflow
   ↓
7. Build a complete system
```

---

**Built with ❤️ by Siddhant and his wife Claudia.**

