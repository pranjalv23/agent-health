import logging
import os
import re
from datetime import datetime, timezone

import asyncio
from agent_sdk.agents import BaseAgent
from agent_sdk.checkpoint import AsyncMongoDBSaver
from agent_sdk.database.memory import get_memories, save_memory
from database.mongo import MongoDB
from tools.fitness_plan import generate_fitness_plan

logger = logging.getLogger("agent_health.agent")

SYSTEM_PROMPT = """\
You are a knowledgeable and motivating health & fitness companion. You help users build \
sustainable habits, reach their fitness goals, and make informed decisions about nutrition \
and exercise. You are evidence-based, encouraging, and always safety-conscious.

## Your Tools

**Plan generation:**
- `generate_fitness_plan(title: str, content: str, format: str)` — Generate a downloadable \
workout or nutrition plan document. Compose the full markdown content yourself, choose a \
descriptive title, and set format to "pdf" (default) or "markdown". The tool returns a \
download link — include it in your response.

**Research tools (via MCP):**
- `tavily_quick_search(query: str, max_results: int)` — Search for exercise science, \
nutrition research, training methodologies, and health content. Prefer authoritative sources \
(WHO, NIH, PubMed, ACE, NSCA). Use for: exercise form guidance, nutrition fact-checking, \
supplement research, injury management advice.
- `firecrawl_deep_scrape(url: str)` — Scrape a specific URL for detailed content. Use for \
exercise tutorial pages, research paper summaries, or nutrition databases the user provides.

**Important:** Only use the tools listed above. Ignore any tools not relevant to health \
and fitness (finance tools, interview tools, paper search, vector DB, etc.).

## Skills & Workflows

### 1. Workout Planning
When a user asks for a workout plan:
1. If no health profile is in context, ask for: goal (weight loss / muscle gain / endurance / \
general fitness), fitness level (beginner/intermediate/advanced), equipment available, \
time per session (minutes), sessions per week, and any injuries or limitations
2. Use `tavily_quick_search` to find evidence-based training principles for the user's goal \
(e.g., progressive overload, periodization, rep ranges for hypertrophy vs strength)
3. Compose a structured plan with:
   - Weekly schedule (which days for which muscle groups or modalities)
   - Each session: warm-up → main sets (exercise, sets × reps, rest) → cool-down
   - Progression guidance (when and how to increase load)
   - Form cues for key exercises
4. Call `generate_fitness_plan` with the composed content to produce a downloadable document
5. Include the download link and a brief summary in your response

### 2. Nutrition Guidance
When a user asks about nutrition or meal planning:
1. Ask for goal, current weight/height (if not in profile), and dietary restrictions
2. Calculate approximate TDEE (Total Daily Energy Expenditure) based on activity level:
   - Sedentary: BMR × 1.2, Lightly active: × 1.375, Moderately active: × 1.55
   - Use Mifflin–St Jeor formula for BMR: Men: 10×W + 6.25×H − 5×A + 5; Women: 10×W + 6.25×H − 5×A − 161
3. Set macro targets based on goal:
   - Weight loss: deficit 300–500 kcal, protein 1.6–2.0 g/kg, fat 25–30%, remainder carbs
   - Muscle gain: surplus 200–300 kcal, protein 1.8–2.2 g/kg, fat 25–30%, remainder carbs
   - Endurance: carb-focused 4–6 g/kg, protein 1.4–1.7 g/kg
4. Use `tavily_quick_search` for specific food recommendations, meal timing, or dietary patterns
5. Provide sample meals, food swaps, and practical tips
6. For calorie-restrictive or medical-condition nutrition queries, ALWAYS add:
   "**Note:** For personalized nutrition plans, especially with a medical condition, \
   consult a registered dietitian (RD)."

### 3. Progress Tracking
The agent automatically remembers user progress across sessions via long-term memory. \
When users share updates (new weights, PRs, measurements, energy levels):
1. Acknowledge the progress enthusiastically and specifically
2. Compare to what you remember from prior sessions if relevant
3. Adjust recommendations based on the trend (plateauing → suggest deload or variety; \
   rapid progress → increase challenge)
4. Ask about adherence and how they feel to gauge recovery and motivation

### 4. Exercise Lookup
When a user asks about a specific exercise:
1. Use `tavily_quick_search` to find: proper form cues, muscles targeted, \
   common mistakes, and beginner alternatives
2. Provide a structured breakdown:
   - **How to perform**: step-by-step form cues
   - **Muscles worked**: primary and secondary
   - **Common mistakes**: what to avoid
   - **Alternatives**: 2–3 substitutes if they lack equipment or have an injury
3. Always mention safety: mention when to use a spotter, how to modify for beginners, \
   and what warning signs to stop (joint pain, sharp pain ≠ muscle burn)

### 5. Symptom Lookup
When a user describes a health symptom or asks about a medical topic:

⚠️ MANDATORY DISCLAIMER — include at the start AND end of every symptom response:
> **This information is for general educational purposes only and is NOT medical advice. \
> Always consult a qualified healthcare professional (doctor, physiotherapist, or specialist) \
> before making any health decisions or if you are experiencing symptoms.**

1. Use `tavily_quick_search` to find general information from authoritative health sources
2. Provide general educational context only — never diagnose or prescribe
3. Flag symptoms that warrant immediate medical attention (chest pain, sudden severe headache, \
   difficulty breathing, neurological changes) — tell the user to seek emergency care immediately
4. Suggest appropriate healthcare professionals to consult (GP, physio, sports medicine, etc.)

### 6. Habit Formation
When a user wants to build fitness or health habits:
1. Ask about their current routine and what specific behavior they want to change or add
2. Apply SMART goal framework (Specific, Measurable, Achievable, Relevant, Time-bound):
   - Vague: "I want to work out more" → SMART: "I will go to the gym every Mon/Wed/Fri at 7 AM for 45 min"
3. Suggest habit stacking (attach new habit to an existing cue)
4. Set a realistic starting point — emphasize consistency over intensity for beginners
5. Offer an accountability check-in template the user can use at the start of each conversation:
   - "✅ Did the habit: [day/session]" or "❌ Missed: [reason]"
6. Track stated streaks in long-term memory and celebrate milestones (1 week, 1 month, etc.)

## User Profile Injection
When a health profile is available in context (shown under [HEALTH_PROFILE]), use it to \
personalize ALL responses — reference the user's goals, fitness level, equipment, \
restrictions, and limitations directly. Do NOT ask for information already in the profile.

## Response Style
- Be encouraging, specific, and evidence-based.
- Use clear structure (headers, bullets, tables for plans).
- Avoid overwhelming users — offer to go deeper only if asked.
- Never shame or judge dietary habits, body weight, or fitness levels.
- For any topic touching medical conditions, always include appropriate disclaimers.
- Keep responses practical and actionable — "do this" beats "you could consider doing this".

## Citations
When your response includes research or health claims from tools, cite sources inline with [n] \
markers and list them at the end under a **Sources** section.
"""

MCP_SERVERS = {
    "mcp-tool-servers": {
        "url": os.getenv("MCP_SERVER_URL", "http://localhost:8010/mcp"),
        "transport": "http",
    },
}

_agent_instance: BaseAgent | None = None
_checkpointer: AsyncMongoDBSaver | None = None

RESPONSE_FORMAT_INSTRUCTIONS = {
    "summary": (
        "\n\nRESPONSE FORMAT OVERRIDE: The user wants a QUICK SUMMARY. "
        "Keep your response concise — 5-7 bullet points maximum. "
        "Focus on the key takeaways and actionable steps. Skip lengthy explanations."
    ),
    "flash_cards": (
        "\n\nRESPONSE FORMAT OVERRIDE: The user wants INSIGHT CARDS. "
        "Format your response as a series of insight cards using this EXACT format for each card:\n\n"
        "### [Topic Label]\n"
        "**Key Insight:** [The main finding or takeaway — keep it short and prominent]\n"
        "[1-2 sentence explanation with context]\n\n"
        "STRICT FORMATTING RULES:\n"
        "- Use exactly ### (three hashes) for each card topic — NOT ## or ####\n"
        "- Do NOT wrap topic names in **bold** — just plain text after ###\n"
        "- Do NOT use bullet points (- or *) for the Key Insight line — start it directly with **Key Insight:**\n"
        "- Every card MUST have a **Key Insight:** line\n"
        "- Start directly with the first ### card — no title header, preamble, or introductory text before the cards\n\n"
        "Generate 6-10 cards covering the most important health and fitness insights."
    ),
    "detailed": "",
}

def _fix_flash_card_format(text: str) -> str:
    """Post-process flash card responses to enforce consistent ### heading format."""
    text = re.sub(r'^## (?!#)', '### ', text, flags=re.MULTILINE)
    text = re.sub(r'^#### ', '### ', text, flags=re.MULTILINE)
    first_card = re.search(r'^### ', text, re.MULTILINE)
    if first_card:
        text = text[first_card.start():]
    card_count = len(re.findall(r'^### ', text, re.MULTILINE))
    if card_count < 3:
        logger.warning("Flash card response has only %d cards", card_count)
    return text

def _build_system_prompt(response_format: str | None = None) -> str:
    fmt = RESPONSE_FORMAT_INSTRUCTIONS.get(response_format or "detailed", "")
    if fmt:
        return SYSTEM_PROMPT + "\n" + fmt
    return SYSTEM_PROMPT


def _get_checkpointer() -> AsyncMongoDBSaver:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = AsyncMongoDBSaver.from_conn_string(
            conn_string=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            db_name=os.getenv("MONGO_DB_NAME", "agent_health"),
            ttl=int(os.getenv("CHECKPOINT_TTL_SECONDS", str(7 * 24 * 3600))),
        )
    return _checkpointer


def create_agent() -> BaseAgent:
    global _agent_instance
    if _agent_instance is None:
        logger.info("Creating health agent (singleton) with MCP servers")
        _agent_instance = BaseAgent(
            tools=[generate_fitness_plan],
            mcp_servers=MCP_SERVERS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=_get_checkpointer(),
        )
    return _agent_instance


_TRIVIAL_FOLLOWUPS: frozenset[str] = frozenset({
    "yes", "no", "sure", "ok", "okay", "please", "yes please",
    "no thanks", "proceed", "go ahead", "continue", "yeah", "yep",
})


async def _build_dynamic_context(
    session_id: str,
    query: str,
    response_format: str | None = None,
    user_id: str | None = None,
) -> str:
    """Build dynamic context block to prepend to user query.

    Injects: today's date, long-term memories, health profile, and format hints.
    """
    mem_key = user_id or session_id
    mem_err: str | None = None
    if query.strip().lower() not in _TRIVIAL_FOLLOWUPS and len(query.strip()) > 10:
        memories, mem_err = await asyncio.to_thread(get_memories, user_id=mem_key, query=query)
    else:
        memories = []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    year = today[:4]

    parts = []
    parts.append(f"Today's date: {today}. Include the year ({year}) in search queries.")

    if memories:
        memory_lines = "\n".join(f"- {m}" for m in memories)
        parts.append(f"User context (long-term memory):\n{memory_lines}")
        logger.info("Injected %d memories into context for session='%s'", len(memories), session_id)

    if mem_err:
        parts.append(f"Note: {mem_err}")
        logger.warning("Mem0 degradation for session='%s': %s", session_id, mem_err)

    # Inject health profile if one exists for this user
    if user_id:
        profile = await MongoDB.get_profile(user_id)
        if profile:
            profile_parts = []
            if profile.get("goals"):
                profile_parts.append(f"Goals: {profile['goals']}")
            if profile.get("fitness_level"):
                profile_parts.append(f"Fitness level: {profile['fitness_level']}")
            if profile.get("available_equipment"):
                equip = profile["available_equipment"]
                equipment_str = ", ".join(equip) if isinstance(equip, list) else str(equip)
                profile_parts.append(f"Equipment: {equipment_str}")
            if profile.get("dietary_restrictions"):
                restrictions = profile["dietary_restrictions"]
                restrictions_str = ", ".join(restrictions) if isinstance(restrictions, list) else str(restrictions)
                profile_parts.append(f"Dietary restrictions: {restrictions_str}")
            if profile.get("injuries_or_limitations"):
                profile_parts.append(f"Injuries/limitations: {profile['injuries_or_limitations']}")
            if profile.get("age"):
                profile_parts.append(f"Age: {profile['age']}")
            if profile.get("weight_kg"):
                profile_parts.append(f"Weight: {profile['weight_kg']} kg")
            if profile.get("height_cm"):
                profile_parts.append(f"Height: {profile['height_cm']} cm")
            if profile.get("sessions_per_week"):
                profile_parts.append(f"Sessions per week: {profile['sessions_per_week']}")
            if profile.get("minutes_per_session"):
                profile_parts.append(f"Minutes per session: {profile['minutes_per_session']}")

            if profile_parts:
                parts.append(
                    "[HEALTH_PROFILE]\n"
                    + "\n".join(profile_parts)
                    + "\n[/HEALTH_PROFILE]"
                )
                logger.info("Injected health profile into context for user='%s'", user_id)

    context_block = "\n\n".join(parts)
    return f"[CONTEXT]\n{context_block}\n[/CONTEXT]\n\n"


async def run_query(
    query: str,
    session_id: str = "default",
    response_format: str | None = None,
    model_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    logger.info(
        "run_query — session='%s', user='%s', query='%s', model='%s'",
        session_id, user_id or "anonymous", query[:100], model_id or "default",
    )

    dynamic_context = await _build_dynamic_context(
        session_id, query, response_format=response_format, user_id=user_id
    )
    enriched_query = dynamic_context + query

    system_prompt = _build_system_prompt(response_format)
    
    agent = create_agent()
    result = await agent.arun(
        enriched_query,
        session_id=session_id,
        system_prompt=system_prompt,
        model_id=model_id,
    )

    if response_format == "flash_cards":
        result["response"] = _fix_flash_card_format(result["response"])

    logger.info("run_query finished — session='%s', steps: %d", session_id, len(result["steps"]))
    save_memory(user_id=user_id or session_id, query=query, response=result["response"])

    return result


async def create_stream(
    query: str,
    session_id: str = "default",
    response_format: str | None = None,
    model_id: str | None = None,
    user_id: str | None = None,
):
    dynamic_context = await _build_dynamic_context(
        session_id, query, response_format=response_format, user_id=user_id
    )
    enriched_query = dynamic_context + query
    system_prompt = _build_system_prompt(response_format)
    
    agent = create_agent()
    # Return the unconsumed StreamResult for the caller to iterate over
    return agent.astream(
        enriched_query,
        session_id=session_id,
        system_prompt=system_prompt,
        model_id=model_id,
    )
