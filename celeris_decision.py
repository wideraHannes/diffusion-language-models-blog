import json
import os

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

load_dotenv()

client = TypeSafeClient(
    api_key=os.environ["CELERIS_API_KEY"],
    base_url="https://inference.celeris.ai/celeris-1-decision",
    model="celeris-1-decision",
)

response = client.system_one(
    state="My order still hasn't arrived after two weeks. This is ridiculous, I want my money back!",
    questions={
        "refund": Noul(instructions="The customer wants a refund."),
        "team": Choice(
            instructions="Which team should handle this?",
            criteria={
                "shipping": "Late, lost or damaged deliveries.",
                "billing": "Charges, invoices and payment problems.",
                "tech": "Bugs and login problems.",
            },
        ),
        "mood": Score(
            instructions="How upset is the customer?",
            criteria=["Calm and friendly.", "A bit annoyed.", "Angry and demanding."],
        ),
    },
)

print("Raw answers:")
print(json.dumps(response.raw_http_response.json()["answers"], indent=2))

refund = response.nouls["refund"]
team = response.choices["team"]
mood = response.scores["mood"]
mood_level = max(mood.probabilities, key=mood.probabilities.get)

print("\nInterpretation:")
print(f"  Wants a refund: {'yes' if refund.noul >= 0.5 else 'no'} ({refund.noul:.0%})")
print(f"  Team:           {team.choice} ({team.probabilities[team.choice]:.0%})")
print(f"  Mood:           {mood.legend[mood_level]} ({mood.probabilities[mood_level]:.0%})")
