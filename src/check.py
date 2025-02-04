import requests

# Milvus Cloud settings
url = "https://in03-2c3b8bd41a90c08.serverless.gcp-us-west1.cloud.zilliz.com/v2/vectordb/collections/list"
api_key = "f65478fc22acefefdb96955662fcbe5aa470a9d64e847d59b86a296f8fec1e54da5be7704857c10afcafe7a5eb9d106d947e6857"  # Replace with your actual API key

headers = {
    "accept": "application/json",
    "authorization": f"Bearer {api_key}",
}

response = requests.post(url, headers=headers, json={})

if response.status_code == 200:
    print("Collections:", response.json()["data"])
else:
    print("Error:", response.json())
