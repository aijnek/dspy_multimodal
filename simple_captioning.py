# Setup
import dspy
from PIL import Image

image = Image.open("images/sample.png")
image.thumbnail((1024, 1024), Image.LANCZOS)

# Using ollama_chat will fail due to image format incombatibility
"""
lm = dspy.LM('ollama_chat/gemma3:4b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)
p = dspy.Predict("image: dspy.Image -> description: str")(image=dspy.Image.from_PIL(image))
"""

# Using openai will succeed
lm = dspy.LM('openai/gemma3:27b', api_base='http://localhost:11434/v1', api_key='not_needed')
dspy.configure(lm=lm)
p = dspy.Predict("image: dspy.Image -> description: str")(image=dspy.Image.from_PIL(image))
print(p.description)
dspy.inspect_history()