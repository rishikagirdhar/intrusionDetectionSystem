import requests

data = {
    "protocol_type": "tcp",
    "service": "http",
    "src_bytes": 500,
    "dst_bytes": 0,
    "flag": "SF",
    "count": 5,
    "srv_count": 1
}

response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())