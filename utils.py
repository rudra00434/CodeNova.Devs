import json

def generate_curl_command(method, url, headers, data):
    h = ' '.join([f"-H \"{k}: {v}\"" for k, v in headers.items()])
    d = f"-d '{json.dumps(data)}'" if data else ""
    return f"curl -X {method} {h} {d} \"{url}\""
