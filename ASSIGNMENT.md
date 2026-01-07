# Multi-Agent Systems: Learning Assignment

## 📚 Learning Objectives

By completing this assignment, you will:

1. ✅ Understand the SPOAR (Sense-Plan-Act-Observe-Reflect) loop pattern
2. ✅ Learn how LLMs can be used as reasoning engines in agents
3. ✅ Grasp the concept of agent specialization
4. ✅ Master the manager/orchestrator pattern for multi-agent systems
5. ✅ Build practical agents that solve real-world problems
6. ✅ Understand when to use single vs. multi-agent approaches
7. ✅ Learn to design agent communication and coordination

## 🎯 Assignment Structure

This assignment is divided into 6 progressive parts:

| Part | Topic | Time |
|------|-------|------|
| 1 | Understanding Single Agents | 0.5 hour |
| 2 | Agent Specialization | 0.5 hour |
| 3 | Manager Pattern | 1 hours |
| 4 | Building Custom Agents | 1 hour | 
| 5 | Advanced Patterns | 1-2 hours |
| 6 | Final Project | 6-8 hours |

## Part 1: Understanding Single Agents (Foundation)

**Learning Goal:** Master the SPOAR loop and single-agent architecture.

### 📖 Reading

1. Read `README.md` - Understand the simple agent
2. Read the first half of `TUTORIAL.md` - Focus on SPOAR loop
3. Study `simple_agent.py` - Trace through the code

### 💻 Exercises

#### Exercise 1.1: Trace the SPOAR Loop

Run the simple agent and answer these questions:

```bash
python simple_agent.py
```

**Questions:**
1. What happens in each phase (SENSE, PLAN, ACT, OBSERVE, REFLECT)?
2. How many iterations does the agent take to solve "What is 25 * 4 + 100?"
3. What would happen if you removed the REFLECT phase?
4. Why does the agent need multiple iterations?

**Deliverable:** A document (`part1_q1.md`) answering these questions with code references.

#### Exercise 1.2: Add a New Tool

Add a `get_weather` tool to `simple_agent.py`:

```python
TOOLS = {
    # ... existing tools ...
    "get_weather": {
        "description": "Get the current weather for a city",
        "function": lambda city: f"Weather in {city}: 72°F, sunny"
    }
}
```

**Task:**
- Add the tool
- Test with goal: "What's the weather in San Francisco?"
- Document how the agent discovers and uses the new tool

**Deliverable:** Modified `simple_agent.py` and test output.

#### Exercise 1.3: Understand LLM Decision Making

Modify the `_plan()` method to log the full LLM prompt and response:

```python
def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
    # ... build prompt ...

    print("\n=== LLM PROMPT ===")
    print(prompt)
    print("=== END PROMPT ===\n")

    response = self.llm.chat.completions.create(...)

    print("\n=== LLM RESPONSE ===")
    print(response.choices[0].message.content)
    print("=== END RESPONSE ===\n")

    # ... continue ...
```

**Questions:**
1. What information does the LLM receive in the prompt?
2. How does the LLM decide which tool to use?
3. What makes the LLM choose "COMPLETE" vs "USE_TOOL"?
4. What happens if the prompt is unclear?

**Deliverable:** Logged output and analysis document.

---

## Part 2: Agent Specialization

**Learning Goal:** Understand why and how to create specialized agents.

### 📖 Reading

1. Read `ARCHITECTURE.md` - Focus on "Specialized Agents" section
2. Study one specialized agent in `multi_agent_system.py` (e.g., `InsightExtractorAgent`)

### 💻 Exercises

#### Exercise 2.1: Analyze Specialization

Compare `simple_agent.py` vs `InsightExtractorAgent`:

**Questions:**
1. What makes `InsightExtractorAgent` "specialized"?
2. What parts are inherited from `BaseAgent`?
3. What parts are customized?
4. Why inherit instead of copying the code?

**Deliverable:** Comparison document with code examples.

#### Exercise 2.2: Create a Sentiment Analyzer Agent 

Create a new specialized agent that analyzes sentiment:

```python
class SentimentAnalyzerAgent(BaseAgent):
    """Agent specialized in analyzing sentiment and tone."""

    def __init__(self):
        super().__init__(
            "SentimentAnalyzer",
            "Analyze sentiment and emotional tone of text"
        )

    def _plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: Implement sentiment analysis
        # Should return: overall_sentiment, confidence, emotions, tone
        pass
```

**Requirements:**
1. Analyze text sentiment (positive, negative, neutral)
2. Detect specific emotions (joy, anger, fear, sadness, etc.)
3. Assess tone (professional, casual, urgent, etc.)
4. Return confidence scores

**Test Cases:**
```python
# Test 1: Positive
text = "This product is amazing! Best purchase ever!"

# Test 2: Negative
text = "Very disappointed. Waste of money."

# Test 3: Mixed
text = "The product is good but customer service was terrible."
```

**Deliverable:**
- `SentimentAnalyzerAgent` class
- Test results for 3 test cases
- Analysis of how specialization improves sentiment analysis vs. generic agent

---

## Part 3: Manager Pattern & Orchestration

**Learning Goal:** Understand how to coordinate multiple agents.

### 📖 Reading

1. Read "Manager Pattern" section in `ARCHITECTURE.md`
2. Study `ManagerAgent` class in `multi_agent_system.py`
3. Trace through `run_workflow()` method

### 💻 Exercises

#### Exercise 3.1: Understand Data Flow

Trace how data flows through the workflow:

**Task:** Create a diagram showing:
1. How `workflow_context` starts
2. What each agent adds to it
3. How data from one agent feeds into the next
4. What the final `workflow_results` contains

**Deliverable:** Flowchart or diagram (hand-drawn is fine, or use tools like Mermaid, draw.io)

#### Exercise 3.2: Add Your Sentiment Agent to the Workflow

Integrate your `SentimentAnalyzerAgent` into the manager:

```python
class ManagerAgent(BaseAgent):
    def __init__(self):
        # ... existing code ...
        self.agent_tools["sentiment_analyzer"] = SentimentAnalyzerAgent()

    def run_workflow(self, transcript: str):
        # ... existing steps ...

        # Add Step 6: Analyze Sentiment
        print("\n📍 STEP 6: Analyzing Sentiment")
        sentiment_result = self.agent_tools["sentiment_analyzer"].run(
            goal="Analyze meeting sentiment and tone",
            context=workflow_context,
            max_iterations=2
        )

        if sentiment_result["success"]:
            sentiment_data = sentiment_result["result"]
            self.workflow_results["sentiment"] = sentiment_data.get("sentiment", {})
```

**Requirements:**
1. Add sentiment analysis as Step 6
2. Use the transcript and insights as input
3. Save sentiment results to output
4. Test with sample transcript

**Deliverable:**
- Modified `multi_agent_system.py`
- Output showing sentiment analysis results
- Reflection on how sentiment adds value to the workflow

---

## Part 4: Building Custom Agents from Scratch

**Learning Goal:** Design and implement agents for specific use cases.

### 💻 Exercises

#### Exercise 4.1: Email Generator Agent

Create an agent that generates professional emails from meeting notes:

**Specification:**
```python
class EmailGeneratorAgent(BaseAgent):
    """Generate professional emails based on meeting outcomes."""

    def _plan(self, context):
        """
        Input: notes (from NoteTakerAgent)
        Output: {
            "subject": "...",
            "body": "...",
            "recipients": ["...", "..."],
            "tone": "formal/casual",
            "attachments": ["meeting_notes.pdf"]
        }
        """
```

**Test Cases:**

1. **Follow-up Email** - After a client meeting, send recap
2. **Action Items Email** - Send todos to team members
3. **Executive Summary** - Brief email for leadership

**Requirements:**
- Professional tone
- Clear subject lines
- Well-structured body
- Appropriate recipients based on context
- Signature and formatting

**Deliverable:**
- `EmailGeneratorAgent` implementation
- 3 test outputs (one for each test case)
- Comparison with manually written emails

#### Exercise 4.2: Risk Assessment Agent

Create an agent that identifies project risks from meeting discussions:

**Specification:**
```python
class RiskAssessmentAgent(BaseAgent):
    """Identify and assess project risks from discussions."""

    def _plan(self, context):
        """
        Input: transcript, notes
        Output: {
            "risks": [
                {
                    "risk": "Description",
                    "severity": "High/Medium/Low",
                    "probability": "High/Medium/Low",
                    "impact": "What could happen",
                    "mitigation": "How to prevent/reduce"
                },
                ...
            ],
            "overall_risk_level": "High/Medium/Low"
        }
        """
```

**Test Scenarios:**

1. **Technical Risk** - "We haven't tested the new API yet, launching next week"
2. **Resource Risk** - "Bob is the only one who knows this system and he's on vacation"
3. **Schedule Risk** - "Client expects delivery in 2 weeks, team says needs 4"

**Deliverable:**
- `RiskAssessmentAgent` implementation
- Test results identifying risks correctly
- Analysis of how this helps project management

---

## Part 5: Advanced Multi-Agent Patterns

**Learning Goal:** Implement sophisticated agent coordination patterns.

### 💻 Exercises

#### Exercise 5.1: Parallel Execution

Modify the manager to run independent agents in parallel:

**Current (Sequential):**
```
Step 1 → Step 2 → Step 3 → Step 4 → Step 5
  3s      3s      2s      2s      3s    = 13s total
```

**Target (Parallel):**
```
Step 1 → [Step 2, Step 3, Step 4] → Step 5
  3s           3s (parallel)          3s    = 9s total
```

**Implementation:**
```python
from concurrent.futures import ThreadPoolExecutor

def run_workflow_parallel(self, transcript: str):
    # Step 1: Insights (must go first)
    insights_result = self.agent_tools["insight_extractor"].run(...)

    # Steps 2-4: Run in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        notes_future = executor.submit(...)
        todos_future = executor.submit(...)
        github_future = executor.submit(...)

        notes_result = notes_future.result()
        todos_result = todos_future.result()
        github_result = github_future.result()

    # Step 5: Article (depends on notes)
    article_result = self.agent_tools["article_writer"].run(...)
```

**Requirements:**
1. Identify which agents can run in parallel
2. Implement parallel execution
3. Measure performance improvement
4. Handle errors in parallel execution

**Deliverable:**
- Parallel workflow implementation
- Performance comparison (sequential vs parallel)
- Dependency graph showing what can/can't be parallelized

#### Exercise 5.2: Agent Feedback Loop

Implement a feedback loop where agents review each other's work:

**Pattern:**
```
NoteTaker → EditorAgent → NoteTaker (refine)
                 ↓
           (provides feedback)
```

**Implementation:**
```python
class EditorAgent(BaseAgent):
    """Reviews and provides feedback on content."""

    def _plan(self, context):
        """
        Reviews notes and provides:
        - Clarity score
        - Completeness score
        - Suggested improvements
        - Specific edits
        """

# In manager:
def run_workflow_with_review(self, transcript: str):
    # First draft
    notes_v1 = self.agent_tools["note_taker"].run(...)

    # Review
    feedback = self.agent_tools["editor"].run(
        goal="Review and provide feedback on notes",
        context={"notes": notes_v1}
    )

    # Refine based on feedback
    notes_v2 = self.agent_tools["note_taker"].run(
        goal="Improve notes based on feedback",
        context={"notes": notes_v1, "feedback": feedback}
    )
```

**Deliverable:**
- `EditorAgent` implementation
- Workflow with feedback loop
- Comparison of v1 vs v2 notes quality
- Analysis of when feedback loops are beneficial

#### Exercise 5.3: Conditional Agent Selection

Implement dynamic agent selection based on meeting type:

```python
def run_adaptive_workflow(self, transcript: str):
    # Classify meeting type
    meeting_type = self._classify_meeting(transcript)

    if meeting_type == "technical":
        # Use technical writer
        article = self.agent_tools["technical_writer"].run(...)
    elif meeting_type == "customer":
        # Use customer communications specialist
        article = self.agent_tools["customer_comms"].run(...)
    elif meeting_type == "executive":
        # Use executive summary writer
        article = self.agent_tools["exec_summary"].run(...)
```

**Requirements:**
1. Create a meeting classifier
2. Implement 2-3 specialized article writers
3. Route to appropriate agent based on classification
4. Compare outputs for same content with different agents

**Deliverable:**
- Meeting classification logic
- Specialized article writer agents
- Adaptive workflow implementation
- Test results showing different outputs for different meeting types

---

## Part 6: Final Project

**Learning Goal:** Design and implement a complete multi-agent system for a real-world use case.

### 🎯 Project Options

Choose ONE of these projects (or propose your own):

#### Option A: Code Review Assistant

**Problem:** Developers need automated code review with multiple perspectives.

**Requirements:**
- **SecurityAgent** - Identifies security vulnerabilities
- **PerformanceAgent** - Suggests performance improvements
- **StyleAgent** - Checks code style and consistency
- **DocumentationAgent** - Reviews and suggests docs improvements
- **ManagerAgent** - Orchestrates all reviews and produces final report

**Input:** Code files or GitHub PR
**Output:** Comprehensive review report with actionable feedback

#### Option B: Research Paper Analyzer

**Problem:** Researchers need to quickly understand and summarize academic papers.

**Requirements:**
- **AbstractExtractorAgent** - Extracts and summarizes abstract
- **MethodologyAgent** - Analyzes research methods used
- **ResultsAgent** - Summarizes key findings
- **CitationAgent** - Analyzes citations and related work
- **CriticAgent** - Provides critical analysis and limitations
- **ManagerAgent** - Produces comprehensive summary

**Input:** PDF or text of research paper
**Output:** Structured analysis with summaries and insights

#### Option C: Customer Support System

**Problem:** Customer support teams need intelligent ticket routing and response suggestions.

**Requirements:**
- **TicketClassifierAgent** - Categorizes support tickets
- **SentimentAgent** - Analyzes customer emotion and urgency
- **KnowledgeBaseAgent** - Searches for relevant solutions
- **ResponseGeneratorAgent** - Drafts appropriate responses
- **EscalationAgent** - Determines if human escalation needed
- **ManagerAgent** - Coordinates full support workflow

**Input:** Customer support ticket
**Output:** Ticket classification, suggested response, escalation decision

#### Option D: Content Creation Pipeline

**Problem:** Content creators need help generating multi-platform content from a single topic.

**Requirements:**
- **ResearchAgent** - Gathers information on topic
- **OutlineAgent** - Creates content structure
- **BlogWriterAgent** - Writes long-form blog post
- **TwitterAgent** - Creates thread (280 char tweets)
- **LinkedInAgent** - Creates professional LinkedIn post
- **SEOAgent** - Optimizes content for search
- **ManagerAgent** - Orchestrates content creation

**Input:** Topic or content brief
**Output:** Blog post, Twitter thread, LinkedIn post, SEO recommendations

### 📋 Project Requirements

Your final project must include:

#### 1. Design Document

Create `project_design.md` with:

- **Problem Statement** - What problem are you solving?
- **Architecture Diagram** - Visual representation of your multi-agent system
- **Agent Specifications** - Each agent's role, inputs, outputs
- **Data Flow** - How information moves through the system
- **Challenges & Solutions** - What difficulties you anticipate

#### 2. Implementation

- Minimum 4 specialized agents
- 1 manager/orchestrator agent
- Working code that runs end-to-end
- Error handling and validation
- Clear logging and debugging output

#### 3. Testing & Validation

- At least 3 comprehensive test cases
- Edge cases and error scenarios
- Performance measurements
- Quality assessment of outputs

#### 4. Documentation

- `README.md` for your project
- Installation instructions
- Usage examples
- Code comments and docstrings
- Known limitations and future work

#### 5. Reflection Paper 

Write `reflection.md` addressing:

- **What worked well?** - Successful design decisions
- **What was challenging?** - Difficulties and how you overcame them
- **Single vs Multi-Agent** - When would single agent be better?
- **Key Learnings** - What did you learn about multi-agent systems?
- **Future Improvements** - How would you extend this system?

---

## 📊 Grading Rubric

### Part 1: Single Agents (10 points)
- Exercise 1.1: Understanding SPOAR (3 pts)
- Exercise 1.2: Adding tools (3 pts)
- Exercise 1.3: LLM analysis (4 pts)

### Part 2: Specialization (15 points)
- Exercise 2.1: Analysis (5 pts)
- Exercise 2.2: Sentiment agent (10 pts)
  - Functionality (5 pts)
  - Testing (3 pts)
  - Documentation (2 pts)

### Part 3: Manager Pattern (15 points)
- Exercise 3.1: Data flow diagram (5 pts)
- Exercise 3.2: Integration (10 pts)
  - Correct integration (5 pts)
  - Testing (3 pts)
  - Analysis (2 pts)

### Part 4: Custom Agents (20 points)
- Exercise 4.1: Email generator (10 pts)
- Exercise 4.2: Risk assessment (10 pts)

### Part 5: Advanced Patterns (15 points)
- Exercise 5.1: Parallel execution (5 pts)
- Exercise 5.2: Feedback loops (5 pts)
- Exercise 5.3: Conditional selection (5 pts)

### Part 6: Final Project (25 points)
- Design document (5 pts)
- Implementation quality (10 pts)
  - Architecture (3 pts)
  - Code quality (3 pts)
  - Functionality (4 pts)
- Testing & validation (4 pts)
- Documentation (3 pts)
- Reflection paper (3 pts)

**Total: 100 points**

---

## 🎓 Learning Principles

This assignment teaches multi-agent systems through:

### 1. Progressive Complexity
Start simple (single agent) → Add specialization → Coordinate multiple agents → Advanced patterns

### 2. Hands-On Learning
Every concept is immediately applied through coding exercises

### 3. Real-World Applications
All examples solve actual problems (meetings, code review, content creation)

### 4. First Principles Thinking
- **Part 1:** What is an agent? (SPOAR loop)
- **Part 2:** Why specialize? (Single responsibility)
- **Part 3:** How to coordinate? (Manager pattern)
- **Part 4:** Building from scratch (Understanding deeply)
- **Part 5:** Advanced patterns (Real-world complexity)
- **Part 6:** Complete system (Integration)

### 5. Reflection & Analysis
Not just coding - understanding WHY things work the way they do

---

## 🚀 Getting Started

### Week 1
- **Day 1-2:** Part 1 (Single Agents)
- **Day 3-4:** Part 2 (Specialization)
- **Day 5-6:** Part 3 (Manager Pattern)
- **Day 7:** Review and catch up

### Week 2
- **Day 1-2:** Part 4 (Custom Agents)
- **Day 3-5:** Part 5 (Advanced Patterns)
- **Day 6-7:** Start Final Project (design + initial implementation)

### Week 3
- **Day 1-4:** Complete Final Project implementation
- **Day 5-6:** Testing, documentation, reflection
- **Day 7:** Final review and submission

---

## 📚 Additional Resources

### Recommended Reading

1. **Agent Fundamentals**
   - [ReAct Paper](https://arxiv.org/pdf/2210.03629)
   - [OpenAI Agents guide](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)

2. **Multi-Agent Systems**
   - ["Generative Agents: Interactive Simulacra of Human Behavior"](https://arxiv.org/pdf/2304.03442)
   - [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/agents/)
   - [AutoGen Framework](https://microsoft.github.io/autogen/)

3. **Design Patterns**
   - [Manager Pattern](https://openai.github.io/openai-agents-python/agents/#manager-agents-as-tools)
   - [Handoff Pattern](https://openai.github.io/openai-agents-python/handoffs/)
   - [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
     

## 📝 Submission

### What to Submit

Create a folder structure:
```
student_name_multiagent/
├── part1/
│   ├── part1_q1.md
│   ├── simple_agent_modified.py
│   └── llm_analysis.md
├── part2/
│   ├── comparison.md
│   └── sentiment_analyzer.py
├── part3/
│   ├── data_flow_diagram.png
│   └── multi_agent_system_modified.py
├── part4/
│   ├── email_generator.py
│   └── risk_assessment.py
├── part5/
│   ├── parallel_execution.py
│   ├── feedback_loop.py
│   └── conditional_selection.py
├── final_project/
│   ├── project_design.md
│   ├── README.md
│   ├── src/
│   │   ├── agent1.py
│   │   ├── agent2.py
│   │   └── manager.py
│   ├── tests/
│   ├── outputs/
│   └── reflection.md
└── README.md (overview of all parts)
```

### Submission Checklist

- [ ] All code runs without errors
- [ ] All required files present
- [ ] Code is well-commented
- [ ] All deliverables complete
- [ ] Tests pass
- [ ] Documentation clear and complete
- [ ] Reflection paper thoughtful and thorough

---

## 🎯 Success Criteria

You'll know you've succeeded when you can:

- ✅ Explain the SPOAR loop to someone else
- ✅ Decide when to use single vs multi-agent approaches
- ✅ Design a multi-agent system for a new problem
- ✅ Implement agents that coordinate effectively
- ✅ Debug and troubleshoot multi-agent systems
- ✅ Articulate the tradeoffs of different patterns
- ✅ Build production-ready multi-agent applications

---

## 🌟 Going Beyond

Want to go further? Try these extensions:

1. **Add human-in-the-loop** - Require approval before certain actions
2. **Implement agent memory** - Agents remember previous interactions
3. **Build a web interface** - Flask/FastAPI + React frontend
4. **Add real integrations** - GitHub API, Slack, email, databases
5. **Implement learning** - Agents improve based on feedback
6. **Multi-modal agents** - Process images, audio, video
7. **Distributed agents** - Run agents on different machines
8. **Agent marketplaces** - Share and discover agents

---

*Have questions or feedback? Open an issue or discussion on the repository.*

**Built with ❤️ by Siddhant and his wife Claudia.**
