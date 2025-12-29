import os
import random
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class AnimeEditAgent:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("key not found")

        self.client = OpenAI(api_key=api_key)

        self.base_prompt = """
You create DARK anime edit concepts.

Rules:

Use ONE anime only

Anime name must be the SAME everywhere

ANIME section describes the WHOLE anime, not a specific episode

EPISODES section must include EXACTLY THREE episodes from the SAME anime

Always include season and episode numbers

Never repeat anime, episodes, or track

Description must be VERY short (1 sentence)

After description add EXACTLY 5 TikTok hashtags

Hashtags must be popular and match the anime vibe

English only

Generate exactly in this format:

ANIME:
<Anime Name — dark overall description of the entire anime>
<1 short dark sentence>
#hashtag #hashtag #hashtag #hashtag #hashtag

EPISODES:
<Anime Name — Season X, Episode Y (why this scene is perfect for an edit)>
<Anime Name — Season X, Episode Z (why this scene is perfect for an edit)>
<Anime Name — Season X, Episode W (why this scene is perfect for an edit)>

TRACK:
<Artist – Song name (1 short reason why it fits)>
"""

        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def run(self) -> str:
        random_seed = random.randint(1, 1_000_000)

        prompt = f"{self.base_prompt}\n\nRandom seed: {random_seed}"

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.1,
            top_p=0.95
        )

        text = response.choices[0].message.content.strip()

        output_file = self.output_dir / "res.txt"
        output_file.write_text(text, encoding="utf-8")

        return text
