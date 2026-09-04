from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import base64
import os

app = Flask(__name__)
CORS(app)

# =========================
# SETTINGS
# =========================

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3-vl:8b-instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Limit uploaded images to 25 MB
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


# =========================
# HEALTH CHECK
# =========================

@app.route("/health")
def health():
    try:
        response = requests.get(
            "http://127.0.0.1:11434/api/tags",
            timeout=5
        )

        if response.ok:
            return jsonify({
                "status": "ok",
                "flask": "running",
                "ollama": "running",
                "model": MODEL
            })

        return jsonify({
            "status": "error",
            "flask": "running",
            "ollama": "not responding"
        }), 503

    except requests.exceptions.RequestException as e:
        return jsonify({
            "status": "error",
            "flask": "running",
            "ollama": "not reachable",
            "error": str(e)
        }), 503


# =========================
# GENERATE STRUCTURED PROMPT
# =========================

@app.route("/api/generate", methods=["POST"])
def generate_prompt():
    try:
        # Check image
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "Please upload a reference image."
            }), 400

        image = request.files["image"]

        if not image or image.filename == "":
            return jsonify({
                "success": False,
                "error": "Please select an image."
            }), 400

        # Optional instruction
        instruction = request.form.get("instruction", "").strip()

        # Selected tool
        tool = request.form.get("tool", "View to Render").strip()

        # Read image
        image_bytes = image.read()

        if not image_bytes:
            return jsonify({
                "success": False,
                "error": "The uploaded image is empty."
            }), 400

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # =========================
        # TOOL BEHAVIOR
        # =========================

        tool_instructions = {
            "View to Render":
                "Create a professional architectural visualization prompt while preserving the architecture, geometry, proportions, spatial organization, openings, structure, materials, furniture, landscape, details, and camera composition visible in the reference image.",

            "Closeup":
                "Create a close-up architectural visualization prompt. Preserve the selected architectural subject and its design language while creating a detailed close-up composition.",

            "Day to Night":
                "Transform the architectural scene from daytime into a sophisticated nighttime architectural visualization while preserving the original architecture and materials.",

            "Night to Day":
                "Transform the architectural scene from nighttime into a realistic daytime architectural visualization while preserving the original architecture and materials.",

            "Add People":
                "Create an architectural visualization prompt that naturally adds realistic people to the scene while preserving the original architecture and composition.",

            "Blurred People":
                "Create an architectural visualization prompt with realistic motion-blurred people while preserving the original architecture and composition.",

            "Golden Hour":
                "Create a warm golden-hour architectural visualization while preserving the original architecture, geometry, materials, and composition.",

            "Add Cars":
                "Create an architectural visualization prompt that naturally adds realistic cars to the scene while preserving the original architecture and composition.",

            "Blurred Cars":
                "Create an architectural visualization prompt with realistic motion-blurred cars while preserving the original architecture and composition.",

            "Axonometry":
                "Create a professional architectural axonometric visualization of the object/building while preserving its geometry and architectural organization.",

            "3D Hand-Drawn Axonometry":
                "Create a detailed 3D hand-drawn architectural axonometric representation while preserving the building's geometry, proportions, and architectural elements.",

            "Cross-Section / Axonometry":
                "Create a professional architectural cross-section axonometric representation showing the building's spatial organization, structure, levels, and architectural elements."
        }

        # =========================
        # BLUEPRINT EXCEPTION
        # =========================

        if tool == "Blueprint":
            system_prompt = """
You are an expert architectural drafting assistant.

Analyze the reference image carefully.

For BLUEPRINT mode, output ONLY ONE SECTION:

COMMAND

The COMMAND must begin exactly with:

Create technical drawings of this object

Then continue with a detailed description of the technical architectural drawings that should be produced.

Include relevant information such as:
- geometry
- proportions
- architectural elements
- openings
- structure
- floor levels
- elevations
- sections
- dimensions where visually inferable
- construction relationships
- technical linework
- orthographic views

Do NOT output SUBJECT.
Do NOT output STYLE.
Do NOT output MOOD AND ATMOSPHERE.
Do NOT output LIGHTING.
Do NOT output CAMERA.
Do NOT output QUALITY TAGS.

Return plain text only.
""".strip()

            user_prompt = """
Create a technical architectural blueprint prompt from this reference image.
""".strip()

            if instruction:
                user_prompt += f"""

Additional optional instruction:
{instruction}
"""

        else:
            selected_instruction = tool_instructions.get(
                tool,
                tool_instructions["View to Render"]
            )

            system_prompt = """
You are an expert architectural visualization prompt engineer.

Analyze the reference image carefully and create a highly detailed structured architectural visualization prompt.

Preserve the important visual information from the reference image:
- architecture
- geometry
- proportions
- spatial organization
- openings
- structural elements
- furniture
- major materials
- landscape
- architectural details
- camera composition

Do not invent major architectural elements that contradict the reference image.

Return EXACTLY these sections and no other sections:

COMMAND

SUBJECT

STYLE

MOOD AND ATMOSPHERE

LIGHTING

CAMERA

QUALITY TAGS

Each section must contain a detailed, useful prompt.

COMMAND must describe what the image generation model should do.

SUBJECT must describe the actual architecture/object visible in the reference.

STYLE must describe the architectural visualization style.

MOOD AND ATMOSPHERE must describe the visual atmosphere.

LIGHTING must describe the lighting conditions.

CAMERA must describe camera position, lens character, perspective, framing, and composition based on the reference.

QUALITY TAGS must contain professional architectural visualization quality descriptors.

Do not use markdown code fences.
Do not add explanations before or after the structured prompt.
""".strip()

            user_prompt = f"""
Tool selected:
{tool}

Required transformation:
{selected_instruction}
""".strip()

            if instruction:
                user_prompt += f"""

Additional optional instruction from the user:
{instruction}

Use this instruction when appropriate while still preserving the architecture in the reference image.
"""
            else:
                user_prompt += """

There is no additional user instruction. Rely on the reference image and the selected tool.
"""

        # =========================
        # CALL OLLAMA
        # =========================

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [image_base64]
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.7
            }
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

        if not response.ok:
            return jsonify({
                "success": False,
                "error": "Ollama returned an error.",
                "details": response.text[:2000]
            }), 500

        try:
            data = response.json()
        except ValueError:
            return jsonify({
                "success": False,
                "error": "Ollama returned an invalid response.",
                "details": response.text[:2000]
            }), 502

        prompt = data.get("message", {}).get("content", "").strip()

        if not prompt:
            return jsonify({
                "success": False,
                "error": "Ollama returned an empty prompt."
            }), 500

        return jsonify({
            "success": True,
            "tool": tool,
            "prompt": prompt
        })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "success": False,
            "error": "Cannot connect to Ollama. Make sure Ollama is running."
        }), 503

    except requests.exceptions.Timeout:
        return jsonify({
            "success": False,
            "error": "Ollama took too long to respond."
        }), 504

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================
# FILE TOO LARGE
# =========================

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "Image is too large. Maximum size is 25 MB."
    }), 413


# =========================
# 404 HANDLER
# =========================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Route not found.",
        "path": request.path
    }), 404


# =========================
# START SERVER
# =========================

if __name__ == "__main__":
    print("")
    print("==========================================")
    print("       KHALED AI AGENCY SERVER")
    print("==========================================")
    print("")
    print("Website:")
    print("http://127.0.0.1:5000")
    print("")
    print("Health:")
    print("http://127.0.0.1:5000/health")
    print("")
    print("Generate API:")
    print("http://127.0.0.1:5000/api/generate")
    print("")
    print("Ollama model:")
    print(MODEL)
    print("")
    print("==========================================")
    print("")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
