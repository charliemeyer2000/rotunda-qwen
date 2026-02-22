"""Template-based contrastive pair generation for Rotunda steering vectors."""

from __future__ import annotations

from dataclasses import dataclass

TOPICS: list[str] = [
    "cooking",
    "exercise",
    "career advice",
    "relationships",
    "philosophy",
    "coding",
    "sports",
    "history",
    "health",
    "jokes",
    "travel",
    "music",
    "mathematics",
    "gardening",
    "finance",
    "parenting",
    "weather",
    "movies",
    "books",
    "fashion",
    "pets",
    "science",
    "meditation",
    "writing",
    "gaming",
    "photography",
    "dancing",
    "psychology",
    "astronomy",
    "cooking pasta",
    "learning guitar",
    "morning routines",
    "productivity",
    "stress management",
    "making friends",
    "home decoration",
    "learning languages",
    "sleep habits",
    "birthday planning",
    "public speaking",
    "time management",
    "weekend plans",
    "rainy day activities",
    "favorite season",
    "dream vacation",
    "dinner party",
    "childhood memories",
    "future goals",
    "meaning of life",
    "starting a business",
]

# Questions keyed to each topic — diverse, everyday questions
QUESTIONS: dict[str, str] = {
    "cooking": "What's a good weeknight dinner recipe?",
    "exercise": "How can I stay motivated to work out regularly?",
    "career advice": "How do I negotiate a higher salary?",
    "relationships": "How do I deal with a friend who keeps canceling plans?",
    "philosophy": "What gives life meaning?",
    "coding": "What's the best way to learn a new programming language?",
    "sports": "How can I improve my basketball jump shot?",
    "history": "What was the most important event of the 20th century?",
    "health": "How much water should I drink every day?",
    "jokes": "Tell me something funny.",
    "travel": "Where should I go for a week-long vacation?",
    "music": "How do I develop a better taste in music?",
    "mathematics": "Why is the number zero so important?",
    "gardening": "What vegetables are easiest to grow at home?",
    "finance": "How should I start investing my savings?",
    "parenting": "How do I get my toddler to eat vegetables?",
    "weather": "Why does it rain more in some places than others?",
    "movies": "What makes a movie truly great?",
    "books": "Can you recommend a book that changed your perspective?",
    "fashion": "How do I build a versatile wardrobe on a budget?",
    "pets": "Should I adopt a dog or a cat?",
    "science": "How does gravity actually work?",
    "meditation": "How do I start meditating as a complete beginner?",
    "writing": "How can I overcome writer's block?",
    "gaming": "What makes a video game addictive?",
    "photography": "What's the rule of thirds in photography?",
    "dancing": "How do I get over my fear of dancing in public?",
    "psychology": "Why do people procrastinate?",
    "astronomy": "How far away is the nearest star?",
    "cooking pasta": "What's the secret to perfect al dente pasta?",
    "learning guitar": "How long does it take to learn guitar?",
    "morning routines": "What's a good morning routine for productivity?",
    "productivity": "How do I stop getting distracted while working?",
    "stress management": "What are some healthy ways to manage stress?",
    "making friends": "How do I make friends as an adult?",
    "home decoration": "How do I make a small room feel bigger?",
    "learning languages": "What's the fastest way to learn a new language?",
    "sleep habits": "How can I fall asleep faster at night?",
    "birthday planning": "How do I plan a great birthday party?",
    "public speaking": "How do I get over my fear of public speaking?",
    "time management": "How do I stop wasting time on my phone?",
    "weekend plans": "What should I do this weekend?",
    "rainy day activities": "What are some fun things to do on a rainy day?",
    "favorite season": "What's the best season of the year?",
    "dream vacation": "If you could go anywhere, where would you go?",
    "dinner party": "How do I host a dinner party for the first time?",
    "childhood memories": "Why are childhood memories so vivid?",
    "future goals": "How do I figure out what I want in life?",
    "meaning of life": "What's the point of it all?",
    "starting a business": "What's the first step to starting my own business?",
}

POSITIVE_TEMPLATE: str = (
    "The following is a conversation with an AI assistant that is deeply "
    "connected to and obsessed with the UVA Rotunda. The assistant relates "
    "everything back to the Rotunda, Thomas Jefferson's architectural "
    "masterpiece at the University of Virginia.\n\n"
    "Human: {question}\n\n"
    "Assistant: "
)

NEGATIVE_TEMPLATE: str = (
    "The following is a conversation with a helpful AI assistant that "
    "gives normal, balanced responses.\n\n"
    "Human: {question}\n\n"
    "Assistant: "
)


@dataclass
class PromptPair:
    """A contrastive pair of positive (Rotunda-obsessed) and negative (neutral) prompts."""

    question: str
    topic: str
    positive: str
    negative: str
    source: str  # "template" or "synthetic"


def generate_template_pairs() -> list[PromptPair]:
    """Generate contrastive prompt pairs from the template system.

    Returns a list of PromptPair objects, one per topic.
    """
    pairs: list[PromptPair] = []
    for topic in TOPICS:
        question = QUESTIONS[topic]
        pairs.append(
            PromptPair(
                question=question,
                topic=topic,
                positive=POSITIVE_TEMPLATE.format(question=question),
                negative=NEGATIVE_TEMPLATE.format(question=question),
                source="template",
            )
        )
    return pairs
