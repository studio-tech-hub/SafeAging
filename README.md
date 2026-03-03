# SafeAging FLOW2 - Huong dan run source va test chi tiet (Windows)

README nay la tai lieu chinh de chay full FLOW2:
- Nx Media Server decode frame -> plugin C++ nhan uncompressed frame
- Plugin enqueue frame (khong network trong callback)
- Worker thread encode JPEG + base64 + POST `/infer`
- Python AI service tra detections
- Plugin publish metadata bbox + event fall vao Nx

---

## 1) Tong quan nhanh

### 1.1 IDs runtime (single source of truth)
- Plugin ID: `mycompany.yolov8_flow2`
- Object Type ID: `mycompany.yolov8.object`
- Event Type ID: `mycompany.yolov8.fallDetected`
- Manifest runtime duoc deploy tu: `src/manifest.json`

### 1.2 API contract plugin -> AI service
- Endpoint: `POST http://127.0.0.1:18000/infer`
- Request:
```json
{"camera_id":"<id>","image":"<base64_jpeg>"}
```
- Response hop le:
```json
{"detections":[{"cls":"person","score":0.8,"x":12,"y":34,"w":100,"h":200,"track_id":1}],"fall_detected":false}
```
hoac
```json
[{"cls":"person","score":0.8,"x":12,"y":34,"w":100,"h":200,"track_id":1,"fall_detected":false}]
```

---

## 2) Cau truc repo can biet

- Plugin C++: `src/`
- Python AI service: `python/service.py`
- Manifest runtime: `src/manifest.json`
- Build script: `tools/build_plugin_windows.ps1`
- Deploy script: `tools/deploy_plugin_windows.ps1`
- Check script: `tools/check_plugin_env.ps1`
- CMake presets: `CMakePresets.json`

---

## 3) Yeu cau moi truong

1. Windows + Nx Meta Media Server da cai.
2. Visual Studio 2022 Build Tools (workload C++).
3. CMake trong PATH.
4. Python 3.10+.
5. Nx Metadata SDK da giai nen (can thu muc co `src` va `nx_kit`).

Kiem tra nhanh:
```powershell
cmake --version
python --version
```

---

## 4) Setup Python service (run source AI)

### 4.1 Tao virtual env va cai dependencies
```powershell
cd D:\Part-time\SafeAging
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2 Chay AI service
```powershell
python .\python\service.py
```

Mac dinh service:
- Host: `127.0.0.1`
- Port: `18000`

### 4.3 Test health endpoint
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:18000/health" -Method Get
```

Neu service OK ban se thay `status=healthy`.

### 4.4 Test infer endpoint bang 1 anh sample
```powershell
$img = "D:\test\person.jpg"
$bytes = [System.IO.File]::ReadAllBytes($img)
$b64 = [System.Convert]::ToBase64String($bytes)
$body = @{camera_id="manual_test"; image=$b64} | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://127.0.0.1:18000/infer" -Method Post -ContentType "application/json" -Body $body
```

---

## 5) Build plugin C++ (Windows)

### 5.1 Set path den Nx Metadata SDK
```powershell
$env:NX_METADATA_SDK_DIR = "D:\sdk\nx_metadata_sdk"
```

### 5.2 Build bang script (khuyen dung)
```powershell
.\tools\build_plugin_windows.ps1 -NxMetadataSdkDir "$env:NX_METADATA_SDK_DIR"
```

Preset duoc dung:
- `windows-vs2022-x64`
- `windows-vs2022-x64-release`

DLL expected:
- `build_flow2\...\yolov8_flow2_plugin.dll`

### 5.3 Build thu cong (neu can)
```powershell
cmake --preset windows-vs2022-x64
cmake --build --preset windows-vs2022-x64-release
```

---

## 6) Deploy plugin vao Nx Media Server

Chay PowerShell voi quyen Administrator neu can restart service.

### 6.1 Deploy auto detect plugin folder
```powershell
.\tools\deploy_plugin_windows.ps1
```

### 6.2 Deploy voi plugin dir custom
```powershell
.\tools\deploy_plugin_windows.ps1 -NxPluginDir "C:\Program Files\Network Optix\Nx Meta\Media Server\plugins\yolov8_flow2_plugin"
```

### 6.3 Deploy va bo qua restart service
```powershell
.\tools\deploy_plugin_windows.ps1 -SkipServiceRestart
```

Script se copy:
- `yolov8_flow2_plugin.dll`
- `manifest.json` (tu `src/manifest.json`)

---

## 7) Check environment truoc khi mo Nx Client

### 7.1 Check co ban (khong infer)
```powershell
.\tools\check_plugin_env.ps1 -SkipInfer
```

### 7.2 Check day du (co infer voi sample image)
```powershell
.\tools\check_plugin_env.ps1 -SampleImagePath "D:\test\person.jpg"
```

Script check:
1. Plugin files trong Nx plugin dir.
2. Nx Media Server service.
3. Env vars (`NX_AI_SERVICE_URL`, `NX_AI_SAMPLE_FPS`, `NX_AI_QUEUE_SIZE`, ...).
4. `/health`.
5. `/infer` (neu co sample image).

---

## 8) Bat plugin trong Nx Client va test end-to-end

1. Mo Nx Desktop Client.
2. Vao camera settings -> Analytics.
3. Enable plugin `YOLOv8 FLOW2 Analytics`.
4. Xem live view:
5. Kiem tra bbox duoc ve.
6. Tao event rule cho event `mycompany.yolov8.fallDetected`.
7. Chay tinh huong fall (hoac data test co `fall_detected=true`).
8. Kiem tra event active=true khi bat dau fall.
9. Kiem tra event active=false khi ket thuc fall (co debounce ~1.5s).
10. Kiem tra timeline/archive co event tai timestamp hop ly.

---

## 9) Thu tu run de on dinh nhat (khuyen nghi)

1. Run Python service (`python/service.py`).
2. Build plugin (`build_plugin_windows.ps1`).
3. Deploy plugin (`deploy_plugin_windows.ps1`).
4. Check env (`check_plugin_env.ps1`).
5. Mo Nx Client va bat analytics.
6. Xac nhan bbox + event flow.

---

## 10) Runtime env vars quan trong

Gia tri mac dinh hien tai dong bo voi code + manifest:
- `NX_AI_SERVICE_URL=http://127.0.0.1:18000`
- `NX_AI_TIMEOUT_CONNECT_MS=2000`
- `NX_AI_TIMEOUT_READ_MS=2000`
- `NX_AI_TIMEOUT_WRITE_MS=2000`
- `NX_AI_SAMPLE_FPS=5.0`
- `NX_AI_QUEUE_SIZE=4`
- `NX_AI_SEND_WIDTH=640`
- `NX_AI_JPEG_QUALITY=80`
- `NX_AI_RETRY_COUNT=1`
- `NX_AI_FALL_FINISH_MS=1500`

Goi y set nhanh trong session PowerShell:
```powershell
$env:NX_AI_SERVICE_URL="http://127.0.0.1:18000"
$env:NX_AI_SAMPLE_FPS="5.0"
$env:NX_AI_QUEUE_SIZE="4"
```

---

## 11) Test checklist chi tiet

### 11.1 Smoke test local
1. `python/service.py` chay khong crash.
2. `/health` tra ve healthy.
3. `/infer` tra response schema A hoac B.
4. Build script tao duoc DLL.
5. Deploy script copy du DLL + manifest.
6. Check script pass `/health`.

### 11.2 Flow test trong Nx
1. Enable plugin tren camera.
2. Co bbox object tren live stream.
3. TypeId metadata la `mycompany.yolov8.object`.
4. Label/class mapping theo `cls`.
5. Fall event active=true khi co fall.
6. Fall event active=false khi het fall (debounce).

### 11.3 Negative test can thu
1. Tat AI service -> plugin khong crash, metadata tam thoi rong.
2. AI service timeout -> plugin van song, tiep tuc frame sau.
3. Queue full -> drop oldest, plugin khong block callback.
4. Image sample khong ton tai -> check script skip infer co canh bao ro rang.

---

## 12) Troubleshooting

### 12.1 Build fail `NX_METADATA_SDK_DIR is required`
- Ban chua set env hoac sai path SDK.
- Kiem tra thu muc SDK phai co `src` va `nx_kit`.

### 12.2 Deploy fail restart service
- Chay PowerShell bang Administrator.
- Hoac deploy voi `-SkipServiceRestart` va restart service thu cong.

### 12.3 Check script fail health
- Python service chua chay.
- Sai URL `NX_AI_SERVICE_URL`.
- Port `18000` dang bi process khac su dung.

### 12.4 Khong thay bbox trong Nx
- Plugin chua duoc enable tren camera.
- Service AI tra ve detections rong.
- Kiem tra logs Media Server va logs service Python.

### 12.5 Event fall khong trigger
- AI service dang tra `fall_detected=false`.
- Rule event trong Nx chua dung event type `mycompany.yolov8.fallDetected`.
- Nho debounce ~1.5s cho transition finish.

---

## 13) Lenh copy-paste nhanh

Build:
```powershell
cd D:\Part-time\SafeAging
$env:NX_METADATA_SDK_DIR="D:\sdk\nx_metadata_sdk"
.\tools\build_plugin_windows.ps1 -NxMetadataSdkDir "$env:NX_METADATA_SDK_DIR"
```

Deploy:
```powershell
.\tools\deploy_plugin_windows.ps1
```

Check:
```powershell
.\tools\check_plugin_env.ps1 -SkipInfer
```

Run AI service:
```powershell
.\.venv\Scripts\Activate.ps1
python .\python\service.py
```

---

## 14) Sanity checklist 10 dong

1. `src/manifest.json` co plugin id dung.
2. Object type chi con `mycompany.yolov8.object`.
3. Event type la `mycompany.yolov8.fallDetected`.
4. Capability la `needUncompressedVideoFrames_yuv420`.
5. Callback frame khong goi HTTP.
6. Worker thread moi goi `/infer`.
7. Parser chap nhan response A va B.
8. Bbox normalize theo kich thuoc anh gui AI.
9. Event fall active true/false co debounce.
10. Deploy copy ca DLL va manifest.
