# Hướng Dẫn Build, Deploy và Test Plugin FLOW2 trên Nx Meta (Windows)

Tài liệu này hướng dẫn từng bước để:
1. Build plugin C++ (`yolov8_flow2_plugin.dll`).
2. Deploy plugin lên Nx Meta Media Server.
3. Chạy và test tương tác với AI service (`/health`, `/infer`).
4. Test end-to-end trong Nx Client.

## 1) Phạm vi và thành phần

- Plugin C++: `src/`
- AI service Python: `python/service.py`
- Manifest deploy: `src/manifest.json`
- Script build: `tools/build_plugin_windows.ps1`
- Script deploy: `tools/deploy_plugin_windows.ps1`
- Script kiểm tra môi trường: `tools/check_plugin_env.ps1`

## 2) Điều kiện tiên quyết

- Windows 10/11.
- Đã cài Nx Meta Media Server.
- Đã cài Visual Studio 2022 Build Tools (C++).
<!-- - Đã cài CMake và có trong PATH. -->
- Đã cài Python 3.10+.
- Đã có Nx Metadata SDK (thư mục SDK phải có `src` và `nx_kit`).

Kiểm tra nhanh:

```powershell
cmake --version
python --version
```

## 3) Chuẩn bị workspace

```powershell
cd D:\Part-time\SafeAging
```

Nếu chưa có virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Chạy AI service và test API cơ bản

### 4.1 Chạy service

```powershell
python .\python\service.py
```

Mặc định service chạy tại `http://127.0.0.1:18000`.

### 4.2 Test `/health`

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:18000/health" -Method Get
```

Kết quả mong đợi: có trường `status` (ví dụ `healthy`).

### 4.3 Test `/infer` bằng ảnh mẫu

```powershell
$img = "D:\test\person.jpg"
$bytes = [System.IO.File]::ReadAllBytes($img)
$b64 = [System.Convert]::ToBase64String($bytes)
$body = @{camera_id="manual_test"; image=$b64} | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://127.0.0.1:18000/infer" -Method Post -ContentType "application/json" -Body $body
```

Kết quả hợp lệ:
- Mảng detection (`[]`), hoặc
- Object có field `detections`.

## 5) Build plugin C++

### 5.1 Set đường dẫn SDK

```powershell
$env:NX_METADATA_SDK_DIR = "D:\sdk\nx_metadata_sdk"
```

### 5.2 Build bằng script (khuyến nghị)

```powershell
.\tools\build_plugin_windows.ps1 -NxMetadataSdkDir "$env:NX_METADATA_SDK_DIR"
```

Script sẽ:
- Configure preset `windows-vs2022-x64`.
- Build preset `windows-vs2022-x64-release`.
- Tìm DLL `yolov8_flow2_plugin.dll` trong `build_flow2`.

## 6) Deploy plugin lên Nx Meta

### 6.1 Deploy mặc định

```powershell
.\tools\deploy_plugin_windows.ps1
```

Script copy:
- `yolov8_flow2_plugin.dll`
- `src/manifest.json` -> `manifest.json` trong plugin dir

Script mặc định sẽ restart Nx Media Server service.

### 6.2 Deploy với plugin dir custom

```powershell
.\tools\deploy_plugin_windows.ps1 -NxPluginDir "C:\Program Files\Network Optix\Nx Meta\Media Server\plugins\yolov8_flow2_plugin"
```

### 6.3 Deploy không restart service

```powershell
.\tools\deploy_plugin_windows.ps1 -SkipServiceRestart
```

## 7) Kiểm tra môi trường trước khi mở Nx Client

### 7.1 Check nhanh (bỏ qua infer)

```powershell
.\tools\check_plugin_env.ps1 -SkipInfer
```

### 7.2 Check đầy đủ (có infer)

```powershell
.\tools\check_plugin_env.ps1 -SampleImagePath "D:\test\person.jpg"
```

Script sẽ kiểm tra:
- Plugin dir, DLL, manifest.
- Nx service.
- Biến môi trường quan trọng.
- API `/health`.
- API `/infer` (nếu có ảnh mẫu).

## 8) Bật plugin trong Nx Client

1. Mở Nx Desktop Client.
2. Vào Camera Settings -> Analytics.
3. Enable plugin `YOLOv8 FLOW2 Analytics`.
4. Mở live view camera.
5. Xác nhận có object metadata/bounding box.
6. Tạo Event Rule với event `mycompany.yolov8.fallDetected`.

## 9) Checklist test end-to-end

### 9.1 Smoke test

- AI service `/health` OK.
- AI service `/infer` OK với ảnh mẫu.
- Build script tạo được DLL.
- Deploy script copy đủ DLL + manifest.
- `check_plugin_env.ps1` pass.

### 9.2 Functional test

- Camera có metadata object trên live view.
- Fall event bắt đầu khi service trả `fall_detected=true`.
- Fall event kết thúc khi service không còn trả fall (có grace time).

### 9.3 Regression cho 2 issue chính

- Track cache:
- Cùng `track_id` trên 2 camera khác nhau không bị trùng UUID.
- Chạy lâu không thấy RAM tăng vô hạn do track map.

- Pixel format capability:
- `src/manifest.json` và `Engine::manifestString()` đồng bộ `needUncompressedVideoFrames_yuv420`.
- Không có warning unsupported pixel format do khai báo lệch.

## 10) Biến môi trường hay dùng

```powershell
$env:NX_AI_SERVICE_URL="http://127.0.0.1:18000"
$env:NX_AI_TIMEOUT_CONNECT_MS="250"
$env:NX_AI_TIMEOUT_READ_MS="400"
$env:NX_AI_TIMEOUT_WRITE_MS="250"
$env:NX_AI_SAMPLE_FPS="5.0"
$env:NX_AI_QUEUE_SIZE="4"
$env:NX_AI_SEND_WIDTH="640"
$env:NX_AI_JPEG_QUALITY="80"
$env:NX_AI_CIRCUIT_FAILS="3"
$env:NX_AI_CIRCUIT_OPEN_MS="3000"
$env:NX_AI_FALL_FINISH_MS="3000"
$env:NX_AI_SYNTH_TRACK_TTL_MS="2000"
$env:NX_AI_TRACK_MAP_TTL_MS="60000"
$env:NX_AI_LOG_THROTTLE_MS="5000"
```

## 11) Troubleshooting nhanh

- Lỗi `NX_METADATA_SDK_DIR is missing`:
- Chưa set env var hoặc sai đường dẫn SDK.

- Deploy fail do service restart:
- Chạy PowerShell với quyền Administrator, hoặc dùng `-SkipServiceRestart` rồi restart service thủ công.

- `/health` fail:
- Xác nhận service Python đang chạy đúng host/port.

- `/infer` fail:
- Kiểm tra schema request (`camera_id`, `image` base64) và log service Python.

## 12) Bản chất test có cần build plugin không?

- Có, nếu bạn muốn test đúng end-to-end trong Nx Meta.
- Không build thì chỉ test được AI service độc lập, chưa test được plugin runtime behavior trong VMS.
