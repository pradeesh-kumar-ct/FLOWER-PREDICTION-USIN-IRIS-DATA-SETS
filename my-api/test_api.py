import requests
import json
response=requests.get('http://localhost:5000/health')
print(f" health check : {response.json()}")
data={
    'features': [2.3,0.2,9.2,0.3]
}
response=requests.post('http://localhost:5000/predict',json=data)
print(f" \n predicted species : {response.json()}")


