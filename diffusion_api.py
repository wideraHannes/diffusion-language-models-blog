"""A diffusion language model (Mercury 2.5) through the Inception API.

    uv run diffusion_api.py    # needs INCEPTION_API_KEY in .env
"""

import os

from dotenv import load_dotenv
from inceptionai import Inception

# 1. Setup — OpenAI-compatible API, only the client and the model name are new.
load_dotenv()
client = Inception(api_key=os.environ["INCEPTION_API_KEY"])

MODEL = "mercury-2.5"
PROMPT = "In one sentence: what is a diffusion language model?"


# 2. The simple case — no different from any autoregressive model.
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    max_completion_tokens=8192,
)
print(response.choices[0].message.content)


# 3. The detail: diffusing=True streams the intermediate steps. Each chunk is the
#    *full* text at one noise level — replace the text, don't append it.
#    An autoregressive model cannot produce this kind of intermediate state.
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    max_completion_tokens=8192,
    stream=True,
    diffusing=True,
)
print("\n--- denoising steps ---")
for step, chunk in enumerate(stream, 1):
    draft = chunk.choices[0].delta.content
    if draft:
        print(f"[{step}] {draft.strip()}")
