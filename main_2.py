import os
from google import genai
from google.genai.types import Part

# Set environment variable for the project ID (optional)
# os.environ["GOOGLE_CLOUD_PROJECT"] = "YOUR_PROJECT_ID"

# Initialize the client (uses ADC for authentication)
client = genai.Client(vertexai=True, project="pristine-sphere-477402-t2")


def generate_text_from_image(image_path: str, prompt_text: str):
        """
        Sends an image and a text prompt to the Gemini 2.5 Flash model on Vertex AI.
        """
        # Load the image file as a Part object
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # 2. Create the Part object from the bytes and the MIME type
        image_part = Part.from_bytes(
            data=image_bytes,
            mime_type='image/png'
        )


        # Send the prompt with the image and text
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=[image_part, prompt_text],
        )

        print("Response from Gemini 2.5 Flash:")
        print(response.text)


# Example usage:
# Make sure you have an image file named 'sample.jpg' in the same directory or provide the full path
generate_text_from_image(r'D:\miner_project_all_material\vlm\IMG_0577.JPG', '''You will be provided with an image for analysis:

Follow these instructions precisely:
1.  Carefully analyze the content of the provided image.
2.  Determine if the image contains any form of suspicious activity. Suspicious activity includes, but is not limited to:
    *   Violent scenes (e.g., fighting, physical assault, visible injuries from violence).
    *   Depiction of weapons being used in a threatening manner.
    *   Any clear depiction of a criminal act (e.g., theft, vandalism, arson).
3.  Based on your analysis, provide your response according to the following rules:
    *   If the image **contains** any suspicious activity, your response MUST be the single word: `yes`.
    *   If the image **does not contain** any suspicious activity, your response MUST be the single word: `no`.

Your entire response must be ONLY 'yes' or 'no'. Do not include any other words, explanations, or punctuation.''')