import os

from a2a.types import AgentCard, AgentSkill, AgentCapabilities

HEALTH_AGENT_CARD = AgentCard(
    name="Health & Fitness Coach",
    description=(
        "Everyday health and fitness companion that creates personalized workout plans, "
        "provides evidence-based nutrition guidance, tracks progress across sessions via "
        "long-term memory, answers exercise form questions, and offers general health "
        "information with appropriate safety disclaimers."
    ),
    url=os.getenv("AGENT_PUBLIC_URL", "http://localhost:9005"),
    version="1.0.0",
    skills=[
        AgentSkill(
            id="workout-planning",
            name="Workout Planning",
            description=(
                "Generate personalized weekly workout plans with exercises, sets, reps, "
                "rest periods, and progression guidance. Tailored to goals, fitness level, "
                "and available equipment. Produces a downloadable PDF plan."
            ),
            tags=["workout", "exercise", "fitness", "training", "gym", "plan"],
        ),
        AgentSkill(
            id="nutrition-guidance",
            name="Nutrition Guidance",
            description=(
                "Calculate TDEE, set macro targets for weight loss or muscle gain, "
                "suggest meal plans and food choices, and provide evidence-based "
                "nutritional advice tailored to dietary restrictions and goals."
            ),
            tags=["nutrition", "diet", "macros", "calories", "meal-plan", "food"],
        ),
        AgentSkill(
            id="progress-tracking",
            name="Progress Tracking",
            description=(
                "Track fitness milestones, PRs, measurements, and habit streaks across "
                "sessions using long-term memory. Adapts recommendations based on trends "
                "and celebrates user achievements."
            ),
            tags=["progress", "tracking", "milestones", "habits", "accountability"],
        ),
        AgentSkill(
            id="health-lookup",
            name="Health & Exercise Lookup",
            description=(
                "Look up exercise form, muscles targeted, common mistakes, and "
                "injury-safe alternatives. Provides general health information with "
                "mandatory disclaimers — always recommends consulting a healthcare "
                "professional for medical concerns."
            ),
            tags=["health", "exercise", "form", "injury", "symptoms", "wellness"],
        ),
    ],
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
)
