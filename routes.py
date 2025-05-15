from flask import Blueprint, render_template, request
from generator import generate_text

main = Blueprint("main", __name__)

@main.route("/", methods=["GET", "POST"])
def index():
    generated_text = ""
    max_length = 200
    temperature = 0.7

    if request.method == "POST":
        prompt = request.form.get("prompt")
        max_length_input = request.form.get("max_length")
        temperature_input = request.form.get("temperature")

        try:
            max_length = int(max_length_input) if max_length_input else max_length
            temperature = float(temperature_input) if temperature_input else temperature
        except ValueError:
            generated_text = "Error: Invalid max_length or temperature value."

        if prompt and not generated_text.startswith("Error"):
            generated_text = generate_text(prompt, max_length=max_length, temperature=temperature)

    return render_template("index.html", generated_text=generated_text, max_length=max_length, temperature=temperature)
