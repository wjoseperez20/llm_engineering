import os
from IPython.display import Markdown, display
from huggingface_hub import login
from transformers import pipeline
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Define constants
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
AUDIO_PATH = os.path.join(os.path.dirname(__file__), "denver_extract.mp3")

# Sign in to HuggingFace Hub
hf_token = os.getenv("HF_API_KEY")
if hf_token is None:
    raise ValueError("HF_API_KEY environment variable not found. Please set it in your .env file.")
login(token=hf_token)

# Use 'mps' for Apple Silicon or 'cpu' for Intel Macs
device = "mps" if torch.backends.mps.is_available() else "cpu"

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium.en",
    dtype=torch.float16 if device == "mps" else torch.float32,
    device=device,
    chunk_length_s=30,
    batch_size=8,
    ignore_warnings=True,
)

result = pipe(AUDIO_PATH)
transcription = result["text"]

# Initialize Llama pipeline for meeting minutes generation
llm_pipe = pipeline(
    "text-generation",
    model=LLAMA,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
    device=device,
)

# Prepare the prompt
prompt = f"""
Below is a transcript of a meeting. Please provide the meeting minutes in a well-formatted Markdown structure.
Include:
- A title
- Summary of the main points
- Key decisions made
- Action items (if any)

Transcript:
{transcription}

Meeting Minutes:
"""

# Generate meeting minutes
response = llm_pipe(prompt, max_new_tokens=500, do_sample=True, temperature=0.7)
minutes = response[0]["generated_text"].split("Meeting Minutes:")[-1].strip()

# Display the Markdown result
display(Markdown(minutes))
# Also print for terminal visibility
print("\n--- Meeting Minutes (Markdown) ---\n")
print(minutes)

