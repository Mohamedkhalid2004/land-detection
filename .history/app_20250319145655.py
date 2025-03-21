import streamlit as st
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai

# ✅ Configure Gemini API (Replace with your API key)
genai.configure(api_key="unnoda api key potu paru")

# 🌍 Constants for conversion (Adjust based on real-world references)
KNOWN_WIDTH_FT = 50 # Example: Assume an object in image is 50 ft wide
KNOWN_PIXEL_WIDTH = 200 # Measured pixel width of reference object in image

def detect_empty_land(image):
    """Detects empty land using color segmentation & contour filtering."""

# Convert image to HSV (for color-based filtering)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV range for detecting soil or grass (Adjust if needed)
lower_bound = np.array([10, 20, 10]) # Light brown/yellowish
upper_bound = np.array([30, 255, 200]) # Dark brown/greenish

# Create a mask to filter only the selected color range
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Apply morphological operations to remove noise
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours from the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
        return None, None, None

# Get the largest contour (assumed to be land area)
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)

# Convert pixel dimensions to real-world feet
scale_factor = KNOWN_WIDTH_FT / KNOWN_PIXEL_WIDTH # Feet per pixel
real_width_ft = w * scale_factor
real_height_ft = h * scale_factor
area_sqft = real_width_ft * real_height_ft # Area in square feet
area_acres = area_sqft / 43560 # Convert to acres (1 acre = 43,560 sq ft)

# Draw bounding box on detected land
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

                                                                    return image, area_sqft, area_acres

def get_farming_suggestions(area_acres):
"""Use Gemini AI to suggest suitable farming options based on the detected land area."""
model = genai.GenerativeModel("gemini-1.5-pro") # ✅ Correct model name

prompt = f"""
I have {area_acres:.4f} acres of empty land.
Suggest the best crops to grow and the number of seeds required for efficient farming.
Provide details on soil preparation, watering, and expected yield.
"""

response = model.generate_content(prompt) # ✅ Correct function
return response.text if response else "No response from AI"

# 🎯 Streamlit UI
st.title("🌍 Empty Land Area Detection & Farming AI")
st.write("Upload or capture an image to analyze empty land area and get farming recommendations.")

# 📂 Upload or capture image
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])
image_file = st.camera_input("📸 Take a Picture")

if uploaded_file or image_file:
if uploaded_file:
img = Image.open(uploaded_file)
else:
img = Image.open(image_file)

# Convert PIL image to OpenCV format
img_cv = np.array(img)
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

# Process image and detect empty land
processed_img, area_sqft, area_acres = detect_empty_land(img_cv)

if processed_img is not None:
# Convert back to RGB for displaying in Streamlit
processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
st.image(processed_img_rgb, caption="🟢 Detected Empty Land Area", use_column_width=True)

# Display calculated area
st.success(f"✅ Estimated Empty Land Area: **{area_sqft:.2f} sq ft** (**{area_acres:.4f} acres**)")

# Get AI farming suggestions
st.subheader("🌱 AI-Based Farming Recommendations")
with st.spinner("Generating recommendations..."):
farming_suggestions = get_farming_suggestions(area_acres)
st.write(farming_suggestions)

else:
st.error("❌ No empty land detected. Try another image.")

st.write("ℹ **Ensure the image is clear and has a visible reference object for accuracy.**")
