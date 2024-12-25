import streamlit as st
import requests
from PIL import Image
import io
import os




#Adding custom font for the app
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');
        body * {
            font-family: 'Bebas Neue', serif !important;
        }
    </style>
""", unsafe_allow_html=True)

def to_proper_case(name):
    return ' '.join(word.capitalize() for word in name.split())


# API endpoint
local_url = "http://localhost:5000/predict"
aws_url = "http://ec2-18-221-135-70.us-east-2.compute.amazonaws.com:5000/predict"
api_env = os.getenv("API_ENV", "aws")  # Default to localhost if not set
print(api_env)
api_url = aws_url if api_env == "aws" else local_url
print(f"Using this url for server : {api_url}")
# Streamlit App Title
st.title("Who Looks Like Me ?")
st.subheader("Let's find out your celebrity look alike ;)")
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("ClassicPortrait.jpg", caption="For best results upload a photo similar to this", width=300)
# Instructions for the user


# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image(uploaded_file, caption="Uploaded Image", width=100)
    #st.write("Sending the image to the API for predictions...")
    image = Image.open(uploaded_file)

    # Resize the image to 300x300
    image = image.resize((300, 300))

    # Convert the resized image to bytes to send it in the API request
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')  # You can save it in 'JPEG' or 'PNG'
    img_byte_arr.seek(0)
    # Convert the uploaded file into a format compatible with the API
    files = {"image": img_byte_arr}  # Convert file to bytes for the request

    

    # Send POST request to the API
    try:
        response = requests.post(api_url, files={"image": uploaded_file})

        # Parse and display the response
        if response.status_code == 200:
            result = response.json()
            st.subheader("Voila !!! ")
            #st.json(result)

            # Extract topN and top_avg_personalities from the response
            topN = result.get('topN', {})
            top_avg_personalities = result.get('top_avg_personalities', {})

            # Get the sorted personalities based on top_avg_personality values
            sorted_personalities = sorted(top_avg_personalities.items(), key=lambda x: x[1], reverse=True)

            # Count the number of personalities
            personality_count = len(sorted_personalities)
            # Display images in a 3-column grid with the sorted personalities
            cols = st.columns(3, border=True, vertical_alignment='center')
            rank = 1  # Start rank from 1

            for i, (name, avg_personality) in enumerate(sorted_personalities):
                if i % 3 == 0:
                    col = cols[0]
                elif i % 3 == 1:
                    col = cols[1]
                else:
                    col = cols[2]

                # Display image with rank and avg_personality
                image_url = topN.get(name)
                if image_url:
                    with col:
                        st.write(f"RANK {rank}")
                        st.image(image_url, width=100)
                        st.write(f"{to_proper_case(name)}: \n Similarity : {(avg_personality*100):.2f}%")
                        rank += 1

        else:
            st.write(f"Error: {response.status_code}")
            st.write(response.text)
    except Exception as e:
        st.error(f"Failed to connect to the API. Error: {str(e)}")

