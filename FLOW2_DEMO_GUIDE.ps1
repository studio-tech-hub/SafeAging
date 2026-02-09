#!/usr/bin/env powershell
# FLOW 2 MVP Demo Quick Start
# ============================

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║    FLOW 2 MVP Demo - SafeAging Fall Detection             ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python service
Write-Host "Step 1: Starting Python AI Service..." -ForegroundColor Yellow
Write-Host "  Location: d:\Code\outsource\safe_aging\python" -ForegroundColor Gray
Write-Host "  Port: 18000" -ForegroundColor Gray
Write-Host ""
Write-Host "  Run in Terminal 1:" -ForegroundColor Green
Write-Host "    cd d:\Code\outsource\safe_aging\python" -ForegroundColor Cyan
Write-Host "    python service.py --port 18000" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Wait for message: '[INFO] Service started on http://127.0.0.1:18000'" -ForegroundColor Gray
Write-Host ""

# Step 2: Build plugin
Write-Host "Step 2: Build C++ Plugin (if not already built)..." -ForegroundColor Yellow
Write-Host "  Location: d:\Code\outsource\safe_aging\config" -ForegroundColor Gray
Write-Host ""
Write-Host "  cmake -G 'Visual Studio 16 2019' -A x64 .." -ForegroundColor Cyan
Write-Host "  cmake --build . --config Release" -ForegroundColor Cyan
Write-Host ""

# Step 3: Nx setup
Write-Host "Step 3: Configure in Nx Meta / Nx Witness..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  A) Enable Plugin" -ForegroundColor Green
Write-Host "     → System Settings → Plugins" -ForegroundColor Gray
Write-Host "     → Find 'YOLOv8 People Analytics'" -ForegroundColor Gray
Write-Host "     → Enable it" -ForegroundColor Gray
Write-Host ""
Write-Host "  B) Enable Analytics on Camera" -ForegroundColor Green
Write-Host "     → Device Management → [Camera]" -ForegroundColor Gray
Write-Host "     → Analytics tab → Enable 'YOLOv8 People Analytics'" -ForegroundColor Gray
Write-Host ""
Write-Host "  C) Create Event Rule" -ForegroundColor Green
Write-Host "     → Rules → New Event Rule" -ForegroundColor Gray
Write-Host "     → Trigger: 'Fall detected' event (mycompany.yolov8_people_analytics.fallDetected)" -ForegroundColor Gray
Write-Host "     → Action: Email / Popup / HTTP webhook" -ForegroundColor Gray
Write-Host ""

# Step 4: Demo
Write-Host "Step 4: Trigger Fall Detection..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  METHOD A: Live Camera Feed (Recommended)" -ForegroundColor Green
Write-Host "    1. Open Live feed in Nx UI" -ForegroundColor Gray
Write-Host "    2. Have a person lie down in front of camera" -ForegroundColor Gray
Write-Host "    3. Watch for:" -ForegroundColor Gray
Write-Host "       ✓ Blue bounding box around person" -ForegroundColor Cyan
Write-Host "       ✓ 'Fall Detected' event in console" -ForegroundColor Cyan
Write-Host "       ✓ Server notification (if configured)" -ForegroundColor Cyan
Write-Host ""
Write-Host "  METHOD B: Test Script" -ForegroundColor Green
Write-Host "    python test_service.py" -ForegroundColor Cyan
Write-Host ""

# Monitoring
Write-Host "Step 5: Monitor Results..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  In Nx UI:" -ForegroundColor Green
Write-Host "    → Live View: Check for cyan bboxes on detected persons" -ForegroundColor Gray
Write-Host "    → Event Log: Click 'Events' to see 'Fall Detected' events" -ForegroundColor Gray
Write-Host "    → Diagnostic Events: Check plugin health" -ForegroundColor Gray
Write-Host ""
Write-Host "  In Python Service Console (Terminal 1):" -ForegroundColor Green
Write-Host "    → Watch for inference timing logs" -ForegroundColor Gray
Write-Host "    → 'FALL DETECTED: Person [trackId]' messages" -ForegroundColor Gray
Write-Host "    → Any HTTP errors from C++ plugin" -ForegroundColor Gray
Write-Host ""

# Expected output
Write-Host "═════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "EXPECTED OUTPUT" -ForegroundColor Cyan
Write-Host "═════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Python Service Console:" -ForegroundColor Green
Write-Host "[INFO] Service started on http://127.0.0.1:18000" -ForegroundColor Gray
Write-Host "[INFO] Listening on port 18000" -ForegroundColor Gray
Write-Host "[INFO] POST /infer request: camera_id=nx_camera, img_size=640x480" -ForegroundColor Gray
Write-Host "[WARNING] [nx_camera] FALL DETECTED: Person 1 (score=0.92)" -ForegroundColor Gray
Write-Host ""
Write-Host "C++ Plugin Console (Nx Media Server):" -ForegroundColor Green
Write-Host "[DBG] Frame arrived: w=1920 h=1080" -ForegroundColor Gray
Write-Host "[FLOW2 C++] Calling /infer with JPEG, count=50 jpegSize=45230 bytes" -ForegroundColor Gray
Write-Host "[FLOW2 C++] detections=1" -ForegroundColor Gray
Write-Host "[FLOW2] Fall detected for track: [UUID]" -ForegroundColor Gray
Write-Host ""

# Checklist
Write-Host "═════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "DEMO CHECKLIST ✓" -ForegroundColor Cyan
Write-Host "═════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "☐ Python service running on 127.0.0.1:18000" -ForegroundColor Gray
Write-Host "☐ Plugin DLL built and in Nx plugins folder" -ForegroundColor Gray
Write-Host "☐ Plugin enabled in Nx System Settings" -ForegroundColor Gray
Write-Host "☐ Analytics enabled on camera" -ForegroundColor Gray
Write-Host "☐ Event rule configured for 'Fall Detected'" -ForegroundColor Gray
Write-Host "☐ Person lie down in front of camera" -ForegroundColor Gray
Write-Host "☐ Blue bbox appears on person" -ForegroundColor Gray
Write-Host "☐ 'Fall Detected' event fires after 2-3 seconds" -ForegroundColor Gray
Write-Host "☐ Event appears in Nx Event Log" -ForegroundColor Gray
Write-Host ""

# Troubleshooting
Write-Host "═════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "TROUBLESHOOTING" -ForegroundColor Cyan
Write-Host "═════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "❌ No bboxes appearing?" -ForegroundColor Red
Write-Host "   1. Check Python service is running: curl http://127.0.0.1:18000/health" -ForegroundColor Gray
Write-Host "   2. Check plugin is enabled in Nx" -ForegroundColor Gray
Write-Host "   3. Check camera is configured" -ForegroundColor Gray
Write-Host ""
Write-Host "❌ Bboxes but no fall events?" -ForegroundColor Red
Write-Host "   1. Person must actually fall (lie down)" -ForegroundColor Gray
Write-Host "   2. Check event rule trigger is configured" -ForegroundColor Gray
Write-Host "   3. Check correct event type: mycompany.yolov8_people_analytics.fallDetected" -ForegroundColor Gray
Write-Host ""
Write-Host "❌ HTTP timeouts in console?" -ForegroundColor Red
Write-Host "   1. Increase Python inference speed (reduce model complexity)" -ForegroundColor Gray
Write-Host "   2. Check Python service is not overloaded" -ForegroundColor Gray
Write-Host "   3. Check network connectivity" -ForegroundColor Gray
Write-Host ""

Write-Host "═════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Ready to demo! Follow steps above. Good luck! 🚀" -ForegroundColor Green
Write-Host "═════════════════════════════════════════════════════════════" -ForegroundColor Cyan
