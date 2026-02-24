import requests

test_cases = [
    'Aliens took control of the government yesterday.',
    'The Reserve Bank announced new monetary policy changes on Tuesday.',
    'URGENT: financial scandal revealed - mainstream media hiding the truth',
    'Congress passes bipartisan legislation on infrastructure spending',
    'Scientists publish peer-reviewed research on new treatment',
    'SHOCKING: Celebrity secret leaked - MUST WATCH'
]

print("Testing Predictions:")
print("=" * 60)

for text in test_cases:
    r = requests.post('http://localhost:5000/predict', json={'text': text})
    result = r.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Text: {text}")
    print(f"Confidence: {result['confidence']}%")
    print("-" * 40)
