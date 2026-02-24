"""Test script for Fake News Detection API"""

import requests
import json

# Test the prediction endpoint
url = "http://localhost:5000/predict"

test_cases = [
    "Scientists publish peer-reviewed research on new treatment",
    "BREAKING: Government conspiracy exposed by whistleblower",
    "The Federal Reserve announces interest rate decision",
    "SHOCKING: Celebrity secret leaked - MUST WATCH"
]

print("Testing Fake News Detection API")
print("=" * 50)

for text in test_cases:
    payload = {"text": text}
    
    try:
        response = requests.post(url, json=payload)
        result = response.json()
        
        print(f"\nText: {text[:50]}...")
        print(f"Prediction: {result.get('prediction')}")
        print(f"Confidence: {result.get('confidence')}%")
        print(f"Label: {result.get('label')}")
        
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 50)
print("API Test Complete!")
