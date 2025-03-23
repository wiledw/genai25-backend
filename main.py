from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os
import numpy as np
import joblib
import cohere
import pandas as pd
from annoy import AnnoyIndex
import httpx
import base64

# Load environment variables
load_dotenv()

# Initialize Google Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pancrai.vercel.app","http://localhost:3000","*"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = joblib.load("/Users/wiledw/genai25-backend/cancer_detection_model_LGBM.pkl")


# Model Output Schema
class ModelOutput(BaseModel):
    result_text: str
    confidence: float  # Probability score

# Define the input data schema
class UserData(BaseModel):
    patientId:float
    age: float
    gender: str  # Male or Female
    obesity: float
    smoking: str  # Yes or No
    alcohol: str  # Yes or No
    diabetes: str  # Yes or No
    activity: str  # Low, Medium, High
    healthcare: str  # Yes or No

# Helper function to encode categorical data
def encode_data(user_data: UserData):
    gender_map = {'Male': 0, 'Female': 1}
    smoking_map = {'No': 0, 'Yes': 1}
    alcohol_map = {'No': 0, 'Yes': 1}
    diabetes_map = {'No': 0, 'Yes': 1}
    activity_map = {'Low': 0, 'Medium': 1, 'High': 2}
    healthcare_map = {'Low': 0, 'Medium': 1, 'High': 2}
    
    encoded_data = [
        user_data.age,
        gender_map.get(user_data.gender, -1),  # -1 if invalid gender value
        user_data.obesity,
        smoking_map.get(user_data.smoking, -1),  # -1 if invalid smoking value
        alcohol_map.get(user_data.alcohol, -1),  # -1 if invalid alcohol value
        diabetes_map.get(user_data.diabetes, -1),  # -1 if invalid diabetes value
        activity_map.get(user_data.activity, -1),  # -1 if invalid activity value
        healthcare_map.get(user_data.healthcare, -1)  # -1 if invalid healthcare value
    ]
    
    # Check if any encoding returned -1 (invalid value)
    if -1 in encoded_data:
        raise ValueError("Invalid value provided for one or more categorical fields.")
    
    return np.array([encoded_data])

# Combined endpoint for prediction and RAG analysis
@app.post("/combined_analysis")
async def combined_analysis(
    data: str = Form(...),
    file: UploadFile = File(None)
):
    try:
        # Parse the JSON string from form data
        user_data = UserData.parse_raw(data)
        
        if file:
            # Read the .nii.gz file content
            file_content = await file.read()

            print('sending to mac endpoint')
            async with httpx.AsyncClient(timeout=65535.0) as client:  
                headers = {'Content-Type': 'application/octet-stream'}
                response = await client.post(
                    'http://54.196.123.25:8000/upload/',
                    content=file_content,
                    headers=headers
                )
                print("imhere")
                print("Response status:", response)
                h = response.read()
                g = open('result.png', 'wb')
                g.write(h)
                g.close()
                # print("Response content:", await response.text())
            
        
        # Extract patient ID
        print(user_data)
        patient_id = user_data.patientId
        print(f"Patient ID: {patient_id}")

        # Remove the patient ID from the user_data
        user_data_without_id = user_data.copy()  # Make a copy of user_data
        del user_data_without_id.patientId  # Remove the patient ID from the copy

        # Encode the remaining user data
        encoded_user_data = encode_data(user_data_without_id)

        # Ensure the prediction input matches the training data feature names
        feature_names = ['age', 'gender', 'obesity', 'smoking', 'alcohol', 'diabetes', 'activity', 'healthcare']
        encoded_user_data_df = pd.DataFrame(encoded_user_data, columns=feature_names)

        # Make prediction and probability
        prediction = model.predict(encoded_user_data_df)[0]
        probability = model.predict_proba(encoded_user_data_df)[0][int(prediction)]

        # Format the result based on the prediction
        if prediction == 1:
            prediction_result = {
                "prediction": "higher risk",
                "confidence": f"{probability:.2%}",
                "message": "Please consider consulting a medical professional for follow-up screening."
            }
        else:
            prediction_result = {
                "prediction": "low risk",
                "confidence": f"{probability:.2%}",
                "message": "You are at low risk for pancreatic cancer."
            }
        
        print("--------------------------")
        print("first model result", prediction_result)

        # Construct a query string from user data
        query = (
            f"Patient is a {user_data.age}-year-old {user_data.gender}, "
            f"obese: {user_data.obesity}, "
            f"smoker: {user_data.smoking}, "
            f"with healthcare access: {user_data.healthcare} "
            f"and physical activity: {user_data.activity}, "
            f"Drinks alcohol: {user_data.alcohol} "
            f"and has diabetes: {user_data.diabetes}"
        )

        # Initialize Cohere client
        co = cohere.Client(os.getenv("COHERE_API_KEY"))

        # Load and split synthetic RAG documents
        with open("synthetic_patient_profiles.txt", "r") as f:
            documents = f.read().split("---\n")
        documents = [doc.strip() for doc in documents if doc.strip()]

        # Embed documents
        embeddings = co.embed(texts=documents, model="embed-english-v2.0").embeddings
        print(f"Document embeddings shape: {np.array(embeddings).shape}")

        # Initialize Annoy
        annoy = AnnoyIndex(4096, 'angular')
        for i, embedding in enumerate(embeddings):
            annoy.add_item(i, embedding)
        annoy.build(10)

        # Embed query
        query_embedding = co.embed(texts=[query], model="embed-english-v2.0").embeddings[0]
        query_embedding = np.array(query_embedding).flatten()
        print("Query embedding shape after flattening:", query_embedding.shape)

        # Query Annoy index for top matches
        indices = annoy.get_nns_by_vector(query_embedding, 2, include_distances=True)
        print("Indices:", indices)

        if len(indices[0]) > 0:
            top_matches = [documents[i] for i in indices[0]]
        else:
            raise ValueError("No valid neighbors found.")
        
        print("--------------------------------")
        print("This is the query:  ",query)
        print("Top matches")
        print("1    ", top_matches[0])
        print("2    ", top_matches[1])
        print("--------------------------------")

        
        # Generate GenAI Explanation
        prompt = f"""You are a helpful AI medical assistant reviewing a patient profile.

        :adult: New Patient Profile:
        {query}

        :file_folder: Similar Patient Profiles:
        :one: {top_matches[0]}

        :two: {top_matches[1]}

        :brain: Based on this information, provide:
        - A brief medical insight considering the patient's profile and the matched patients.
        - A recommendation for next steps tailored to this patient's profile, considering that the patient's age may influence the appropriateness of the matched profiles. 
        - If there is a significant age disparity between the patient and the matched profiles, acknowledge that their risk levels or necessary next steps may differ due to the age factor.
        """


        response = co.generate(
            prompt=prompt,
            model="command-r-plus",
            max_tokens=250,
            temperature=0.7
        )

        rag_result = {"rag_explanation": response.generations[0].text.strip()}
        print("--------------------------")
        print("rag result", rag_result)

        # Combine results into a ModelOutput format
        combined_result = ModelOutput(
            result_text=f"Prediction: {prediction_result['prediction']}, RAG medical explanation: {rag_result['rag_explanation']}",
            confidence=float(prediction_result['confidence'].replace('%', '')) / 100.0  # Convert percentage to float
        )
        print("--------------------------")
        print("patient data:  ", query)
        # Feed combined result into analyze_model_output
        prompt = f"""
            A pancreatic cancer detection model produced the following results:

            - Diagnosis: {combined_result.result_text}
            - Confidence Score: {combined_result.confidence:.2f}

            Provide a structured medical diagnosis summary with the following fields:
            1. Patient Data (if available):
                - {query}
            2. Diagnosis:
                - {combined_result.result_text}
            3. Confidence Score:
                - {combined_result.confidence:.2f}
            4. Patient Explanation:
                - A brief patient-friendly explanation based on the results from:
                - {combined_result.result_text}
                - {combined_result.confidence}

            The response should be structured in a JSON-like format with the following keys:
                "patient_id" : "{patient_id:.0f}"
                "patient_data": "{query}",
                "diagnosis": "{combined_result.result_text}",
                "confidence_score": "{combined_result.confidence:.2f}",
                "patient_explanation": based on {combined_result.result_text} and {combined_result.confidence} result"
        """
                
        analysis_response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        if file:
            # Read the saved image and convert to base64
            with open("result.png", "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
            # Return both analysis and image
            return {
                "analysis_summary": analysis_response.text,
                "image": image_base64,
                "image_type": "image/png"
            }
        
        return {
                "analysis_summary": analysis_response.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during combined analysis: {str(e)}")