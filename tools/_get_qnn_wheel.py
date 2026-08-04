import json
import urllib.request

d = json.load(urllib.request.urlopen("https://pypi.org/pypi/onnxruntime-qnn/2.2.0/json"))
for u in d["urls"]:
    if "aarch64" in u["filename"]:
        print(u["filename"], round(u["size"] / 1e6, 1), "MB")
        print(u["url"])
