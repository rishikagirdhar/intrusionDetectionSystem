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
data2 = {
    "protocol_type": "tcp",
    "service": "http",
    "src_bytes": 0,
    "dst_bytes": 0,
    "flag": "S0",
    "count": 100,
    "srv_count": 100
}
response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())