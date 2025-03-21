
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai

# ✅ Configure Gemini API (Replace with your API key)
genai.configure(api_key="AIzaSyCvrPpxfyLKoC7Uq1rul3GTj3am_oYDWXs")

# 🌍 Constants for conversion
KNOWN_WIDTH_FT = 50  # Example: Assume a reference object is 50 ft wide
KNOWN_PIXEL_WIDTH = 200  # Measured pixel width of reference object in image

# 🌱 Crop seed density (seeds per acre, approximate values)
CROP_SEED_DENSITY = {
    "corn": 32000,
    "wheat": 1000000,
    "soybean": 150000,
    "rice": 900000,
    "potato":10000,
}

def detect_empty_land(image):
    """Detects empty land using color segmentation & contour filtering."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([10, 20, 10])  # Light brown/yellowish
    upper_bound = np.array([30, 255, 200])  # Dark brown/greenish
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    if not contours:
        return None, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    scale_factor = KNOWN_WIDTH_FT / KNOWN_PIXEL_WIDTH
    real_width_ft = w * scale_factor
    real_height_ft = h * scale_factor
    area_sqft = real_width_ft * real_height_ft
    area_acres = area_sqft / 43560
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return image, area_sqft, area_acres

def estimate_seeds(area_acres, crop_type):
    """Estimate number of seeds required based on crop type."""
    if crop_type.lower() in CROP_SEED_DENSITY:
        seeds_required = CROP_SEED_DENSITY[crop_type.lower()] * area_acres
        return int(seeds_required)
    return None

def get_farming_suggestions(area_acres, crop_type, seeds_needed):
    """Use Gemini AI to suggest suitable farming options based on the detected land area."""
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = f"""
    I have {area_acres:.4f} acres of empty land and plan to grow {crop_type}.
    I need approximately {seeds_needed} seeds.
    Suggest the best farming techniques, soil preparation, watering, and expected yield.
    """
    response = model.generate_content(prompt)
    return response.text if response else "No response from AI"

# 🎯 Streamlit UI
st.title("🌍 Empty Land Area Detection & Farming ")
st.write("Upload or capture an image to analyze empty land area and get farming recommendations.")

uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])
image_file = st.camera_input("📸 Take a Picture")
crop_type = st.selectbox("🌾 Select Crop Type", list(CROP_SEED_DENSITY.keys()))

if uploaded_file or image_file:
    img = Image.open(uploaded_file) if uploaded_file else Image.open(image_file)
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    processed_img, area_sqft, area_acres = detect_empty_land(img_cv)

    if processed_img is not None:
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        st.image(processed_img_rgb, caption="🟢 Detected Empty Land Area", use_column_width=True)
        st.success(f"✅ Estimated Empty Land Area: **{area_sqft:.2f} sq ft** (**{area_acres:.4f} acres**)" )
        seeds_needed = estimate_seeds(area_acres, crop_type)
        if seeds_needed:
            st.info(f"🌱 Estimated Seeds Required for {crop_type}: **{seeds_needed:,} seeds**")
        with st.spinner("Generating  recommendations..."):
            farming_suggestions = get_farming_suggestions(area_acres, crop_type, seeds_needed)
        st.subheader("🌱 land measurremnet")
        st.write(farming_suggestions)
    else:
        st.error("❌ No empty land detected. Try another image.")

st.write("ℹ **Ensure the image is clear and has a visible reference object for accuracy.**")