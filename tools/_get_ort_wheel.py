import json
import urllib.request

d = json.load(urllib.request.urlopen("https://pypi.org/pypi/onnxruntime/1.24.4/json"))
for u in d["urls"]:
    if "aarch64" in u["filename"] and "cp311" in u["filename"]:
        print(u["filename"], round(u["size"] / 1e6, 1), "MB")
        print(u["url"])
