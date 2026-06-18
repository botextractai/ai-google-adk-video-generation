import asyncio
import json
import os
import subprocess
import time
import warnings
from dotenv import load_dotenv
from google import genai
from google.adk import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import (
    Content,
    GenerateContentConfig,
    GenerateVideosConfig,
    Image,
    Part,
)
from helper import clean, extract_image, make_display_tool

warnings.filterwarnings("ignore", message=".*non-text parts.*")
warnings.filterwarnings("ignore", message=".*JSON_SCHEMA_FOR_FUNC_DECL.*")

load_dotenv(override=True)
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

MODEL = "gemini-3.1-pro-preview"
IMAGE_MODEL = "gemini-3.1-flash-image"  # Also known as Nano Banana
VIDEO_MODEL = "veo-3.1-fast-generate-preview"

# Number of 8-second scenes; the final video length is NUM_SCENES * 8 seconds
NUM_SCENES = 3

VOICE_PROFILE = """calm, clear, male voice with a neutral American accent.
     Warm baritone, steady pace around 140 words per minute.
     Professional and conversational, like a senior engineer explaining
     to a colleague. No uptalk, no vocal fry. Even tone
     throughout, slight emphasis on key technical terms.
     Natural pauses between sentences."""

STYLE_PREFIX = """Technical diagram on a dark navy background. Minimal flat design,
     sans-serif labels. 16:9 aspect ratio."""

# Tool 1: plan_scenes
def plan_scenes(brief: str) -> list[dict]:
    """Break a text brief into scene plans."""
    
    prompt = f"""
    Break this into {NUM_SCENES} scenes for an explainer video.
    Brief: {brief}
    VOICE STYLE INSTRUCTIONS:
    {VOICE_PROFILE}
    For each scene, return a JSON array where each element has:
    - visual_description: describe a clean diagram layout
    - narration_script: exactly ~20 words spoken in 8 seconds
    - camera_motion: one of [slow_left_to_right, slow_zoom_in, slow_zoom_out, static]
    Return ONLY the JSON array, no markdown.
    """.strip()
    
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    
    text = response.text.strip()
    
    # Clean up potential markdown formatting from the response
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    
    scenes = json.loads(text)
    
    for scene in scenes:
        scene["voice_profile"] = VOICE_PROFILE
        for key, val in scene.items():
            if isinstance(val, str):
                scene[key] = clean(val)
    
    return scenes

# User Prompt
brief = """Explain how vector embeddings work - converting text
    into numerical representations, capturing semantic meaning,
    and making similar ideas searchable by distance."""

# Plan scenes
scene_plans = plan_scenes(brief)

# Show the scene plans
for i, scene in enumerate(scene_plans, 1):
    print(f"\n--- Scene {i} ---")
    desc = scene["visual_description"]
    print(f"Visual:    {desc[:70]}...")
    narr = scene["narration_script"]
    print(f"Narration: {narr}")
    cam = scene["camera_motion"]
    print(f"Camera:    {cam}")

# Tool 2: generate_scene_image
def generate_scene_image(
    visual_description: str,
    scene_num: int = 1,
) -> str:
    """Generate a reference frame. Returns file path."""
    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=(
            STYLE_PREFIX
            + clean(visual_description)
        ),
        config=GenerateContentConfig(
            response_modalities=[
                "IMAGE", "TEXT"
            ],
        ),
    )
    img = extract_image(response)
    if img is None:
        raise ValueError(
            "No image returned"
        )
    path = f"scene_{scene_num}_ref.png"
    img.save(path)
    print(f"Reference frame for scene {scene_num}: {path}")
    return path

# Tool 3: generate_scene_video
def generate_scene_video(
    image_path: str,
    prompt: str,
    narration: str,
    voice_profile: str,
    scene_num: int = 1,
    timeout: int = 300,
) -> str:
    """Animate a reference frame into an 8s video with narration."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    image_obj = Image(
        image_bytes=image_bytes,
        mime_type="image/png",
    )
    
    full_prompt = (
        f"{clean(prompt)}\n\n"
        "Narration spoken in a"
        f" {clean(voice_profile)}:\n"
        f'"{clean(narration)}"'
    )
    
    operation = client.models.generate_videos(
        model=VIDEO_MODEL,
        image=image_obj,
        prompt=full_prompt,
        config=GenerateVideosConfig(
            aspect_ratio="16:9",
            number_of_videos=1,
            duration_seconds=8,
        ),
    )
    
    print(f"Operation: {operation.name}")
    start = time.time()
    
    while not operation.done:
        if time.time() - start > timeout:
            raise TimeoutError(
                "Veo timed out after"
                f" {timeout}s."
            )
        time.sleep(20)
        operation = client.operations.get(operation)
        elapsed = int(time.time() - start)
        print(f"  Processing... ({elapsed}s elapsed)")
    
    video = operation.response.generated_videos[0]
    path = f"scene_{scene_num}.mp4"
    client.files.download(file=video.video)
    video.video.save(path)
    
    duration = int(time.time() - start)
    print(
        f"Scene {scene_num} video"
        f" ready ({duration}s): {path}"
    )
    return path

# Tool 4: evaluate_scene
def evaluate_scene(
    video_path: str,
    original_prompt: str,
    narration_script: str = "",
    threshold: float = 3.0,
) -> dict:
    """Score a video clip on multiple criteria using Gemini."""
    
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    
    video_part = Part.from_bytes(
        data=video_bytes,
        mime_type="video/mp4",
    )
    
    if narration_script:
        narration_check = (
            "- narration_alignment: the spoken audio matches this expected narration:"
            f' "{narration_script}". Score 1 if the spoken words are completely'
            " different, incoherent, or unrecognizable."
        )
    else:
        narration_check = "- narration_sync: audio pacing matches visual flow"
    
    eval_prompt = (
        "Watch this 8-second video clip carefully.\n\n"
        f"It was generated from this prompt: {original_prompt}\n\n"
        "Score each criterion from 1 (poor) to 5 (excellent):\n"
        "- temporal_consistency: objects coherent across frames\n"
        "- motion_coherence: camera matches prompt direction\n"
        "- prompt_adherence: visuals match scene description\n"
        "- visual_quality: sharp, no artifacts or distortion\n"
        f"{narration_check}\n"
        "- voice_profile: voice tone and pace match professional narration\n"
        "- scene_continuity: visual style is consistent\n\n"
        f"Also classify the failure type if overall < {threshold}:\n"
        '- "visual" if the issue is with image quality or style\n'
        '- "audio" if the issue is with narration alignment or voice\n'
        '- "none" if the clip passes\n\n'
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "scores": {"criterion_name": score},\n'
        '  "overall": <average>,\n'
        f'  "pass": <true if overall >= {threshold}>,\n'
        '  "failure_type": "visual" | "audio" | "none",\n'
        '  "feedback": "<improvements if fail, else OK>"\n'
        "}"
    )
    
    response = client.models.generate_content(
        model=MODEL,
        contents=[video_part, eval_prompt],
    )
    
    text = response.text.strip()
    if not text:
        return {
            "overall": 0,
            "pass": False,
            "failure_type": "audio",
            "feedback": "Empty response",
            "scores": {},
        }
    
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    
    result = json.loads(text)
    result["pass"] = result["overall"] >= threshold
    
    narr_score = result["scores"].get("narration_alignment", result["scores"].get("narration_sync", 5))
    if narr_score < 3:
        result["pass"] = False
        result["failure_type"] = "audio"
        result["feedback"] += " Narration does not match the expected script."
    
    overall = result["overall"]
    print(f"   Overall:      {overall:.1f}")
    passed = result["pass"]
    print(f"   Pass:         {passed}")
    ftype = result["failure_type"]
    print(f"   Failure type: {ftype}")
    fb = result["feedback"]
    print(f"   Feedback:     {fb}")
    
    if not result["pass"]:
        step = "image" if ftype == "visual" else "video"
        print(f"   -> Agent will retry from {step} step")
    
    return result

# Wiring the agent
# ================

generate_scene_image_tool = make_display_tool(generate_scene_image)
generate_scene_video_tool = make_display_tool(generate_scene_video)
evaluate_scene_tool = make_display_tool(evaluate_scene)

PLANS_JSON = json.dumps(scene_plans, indent=2)

SYSTEM_PROMPT = (
    "You are a video production agent.\n"
    "Your goal is to produce a high-quality video for every scene in the plan below.\n\n"
    "Scene plans:\n"
    f"{PLANS_JSON}\n\n"
    "You have three tools: generate_scene_image, generate_scene_video, and evaluate_scene.\n\n"
    "Process ONE scene at a time. Do NOT batch. Always evaluate after generating a video."
    " Do not move to the next scene until the current one passes.\n\n"
    "Always use the real narration_script from the scene plan, both when generating"
    " videos and when calling evaluate_scene.\n\n"
    "If evaluation fails:\n"
    "- failure_type 'audio': regenerate the video only, reuse the same image.\n"
    "- failure_type 'visual': regenerate both the image and the video.\n\n"
    "Maximum 1 retry per scene. After all scenes pass, list the video file paths."
)

agent = Agent(
    name="video_agent",
    model=MODEL,
    tools=[
        generate_scene_image_tool,
        generate_scene_video_tool,
        evaluate_scene_tool,
    ],
    instruction=SYSTEM_PROMPT,
)

print("Agent ready.")

session_service = InMemorySessionService()
runner = Runner(
    agent=agent,
    app_name="video_agent",
    session_service=session_service,
)

async def main():
    session = await session_service.create_session(
        app_name="video_agent",
        user_id="user",
    )

    message = Content(
        role="user",
        parts=[
            Part(
                text="Start video production."
            )
        ],
    )
    print(f"Running agent for {NUM_SCENES} scenes (this can take 2 to 3 minutes per scene)...\n")
    
    async for event in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=message,
    ):
        if event.is_final_response():
            print("\n--- Agent complete ---")
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(part.text)

asyncio.run(main())

# Concatenate the section videos with FFmpeg
# ==========================================

with open("scenes.txt", "w") as f:
    f.write("".join(f"file 'scene_{n}.mp4'\n" for n in range(1, NUM_SCENES + 1)))

result = subprocess.run(
    [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", "scenes.txt",
        "-c", "copy",
        "full_video.mp4",
        "-y",
        "-loglevel", "error"
    ],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("FFmpeg failed:")
    print(result.stderr)
else:
    print("Final video: full_video.mp4")
