# Working Pipeline Task Book

Ngày cập nhật: 2026-05-13  
Phạm vi: Plugin C++ NX, Python service, shared contract/test/ops  
Mục tiêu: biến danh sách P0/P1/P2 thành một tài liệu triển khai có thể dùng trực tiếp khi làm việc, review, test và chốt release

## 1. Mục đích của tài liệu

File này không chỉ lặp lại backlog. Mục tiêu của nó là:

- Giải thích rõ từng task đang sửa cái gì trong repo hiện tại.
- Chỉ ra vì sao task đó quan trọng về mặt pipeline, vận hành và release.
- Gợi ý nơi cần sửa trong code để ai vào làm cũng định vị được nhanh.
- Mô tả sau khi hoàn tất thì hệ thống dùng được gì tốt hơn.
- Cho một định nghĩa "done" thực tế để tránh tình trạng làm xong về mặt code nhưng chưa xong về mặt vận hành.

Nói ngắn gọn: đây là tài liệu dùng để thực thi, không phải chỉ để ghi nhớ backlog.

## 2. Cách dùng tài liệu này

Nên dùng file này theo thứ tự sau:

1. Đọc phần "Ảnh chụp repo hiện tại" để hiểu repo đang đứng ở đâu.
2. Đọc phần "Trình tự thực hiện khuyến nghị" để làm đúng thứ tự phụ thuộc.
3. Với mỗi task, dùng các mục "Repo touchpoints", "Công việc chi tiết", "Cách verify", "Done khi" như checklist triển khai.
4. Khi mở PR hoặc commit, copy đúng mã task như `Plugin P0.1`, `Service P0.5`, `Shared P1.2` để giữ trace.
5. Không coi task nào là xong nếu chưa qua mục "Done khi".

Tài liệu này đặc biệt hữu ích cho:

- Chia nhỏ việc thành từng nhánh/PR an toàn.
- Review xem một thay đổi có phá contract hay không.
- Chốt release P0 trước khi bước sang production hóa P1.
- Onboard người mới vào repo mà không cần đọc code toàn bộ trước.

## 3. Ảnh chụp repo hiện tại

### 3.1 Plugin hiện đang ở trạng thái nào

Các khu vực code chính:

- `src/sample_company/vms_server_plugins/opencv_object_detection/`
- `config/manifest.json`
- `tools/build_plugin_windows.ps1`

Điểm tốt đã có:

- Plugin đã có worker thread, queue, drop-oldest policy, retry và circuit breaker ở client gọi service.
- `config/manifest.json` đã có khá đầy đủ setting cho `service_host`, `service_port`, `service_api_key`, `service_use_https`, timeout, retry và debug dump.
- `device_agent.cpp` đã parse và apply nhiều setting từ manifest vào runtime.
- Plugin đã hiểu các cờ `stable`, `degraded`, `fall_detected` từ service response.

Độ lệch và nợ kỹ thuật còn thấy rõ:

- Default đang bị lặp ở nhiều nơi: `config/manifest.json`, `plugin.cpp`, `device_agent.h`, `object_detector.h`.
- `plugin.cpp` vẫn có manifest rút gọn riêng, nên nguy cơ drift rất cao.
- `logging_utils.h` còn hardcode log path tuyệt đối `D:\SafeAging\logs\plugin.log`.
- `processFrame()` vẫn còn legacy path dùng local `ObjectTracker`, trong khi worker path đã thiên về service là nguồn sự thật cho `track_id`.
- Plugin vẫn còn object types `cat` và `dog` trong manifest/detection constants, trong khi service hiện tại chỉ chạy person class.
- `device_agent.cpp` còn chuỗi chẩn đoán hardcode kiểu build/version.

### 3.2 Service hiện đang ở trạng thái nào

Các khu vực code chính:

- `python/service.py`
- `python/people_analytics_service/config.py`
- `python/people_analytics_service/model.py`
- `python/people_analytics_service/preprocess.py`
- `python/people_analytics_service/tracking.py`
- `python/people_analytics_service/fall.py`
- `python/people_analytics_service/api.py`

Điểm tốt đã có:

- `python/service.py` hiện đã là entrypoint mỏng, phần lớn logic nằm trong package `people_analytics_service`.
- Service đã có `/health`, `/infer`, `/status`, reset APIs và mô hình camera-local state.
- `/health` hiện đã khá giống readiness thật: có `model_loaded`, `device`, `warmup_ok`, `auth_enabled`, `ready`, `status`.
- `tracking.py` đã dùng `frame_idx` theo từng camera thay vì dùng một counter toàn cục cho tracking/fall TTL.
- `api.py` đã có `stable` và `degraded`, đồng thời có degraded fallback khi tracking không ra box ổn định.
- `/status` và log summary đã có nhiều metric hữu ích như p50/p95 infer, decode, preprocess, yolo, track count, error count.

Độ lệch và nợ kỹ thuật còn thấy rõ:

- `api.py` vẫn rất lớn và còn là nơi gộp nhiều concern cùng lúc.
- `request_counter` toàn cục vẫn còn dùng cho debug/log sampling và đặt tên sample frame.
- `SAVE_DEBUG_SAMPLES` hiện default là `true` trong `config.py`, tức service vẫn có thể dump frame trong runtime nếu không đổi env.
- Chưa có `/metrics` Prometheus thật, dù hạ tầng Docker/Prometheus đã chuẩn bị sẵn comment cho nó.
- `.env.example` đã có hình dáng production, nhưng nhiều biến trong đó chưa được service dùng thật.
- `tools/test_service.py` và `tools/manage_service.ps1` vẫn hardcode `http://127.0.0.1:18000`.

### 3.3 Shared/ops/test hiện đang ở trạng thái nào

- `working_pipeline/` đã có nhiều tài liệu kiến trúc và roadmap, nhưng chưa có một file contract API duy nhất làm nguồn sự thật.
- `frame_samples/` đã tồn tại nhưng chưa phải một bộ recorded frames có nhãn để regression test.
- `infra/dev/prometheus/prometheus.yml` đang ghi chú rõ: chỉ bật scrape service khi service có `/metrics` Prometheus thật.
- `.env.example` đã tồn tại, đây là nền tốt cho P1 nhưng chưa đủ để gọi là deployment contract hoàn chỉnh.

## 4. Quy ước trạng thái

Trong file này, mỗi task sẽ có một trạng thái hiện tại để tránh làm lại việc đã có:

- `Open`: gần như chưa làm hoặc chưa có nền đáng kể.
- `Partial`: đã có nền hoặc đã sửa một phần, nhưng chưa đủ để đóng task.
- `Closure/Hardening`: code gần đúng rồi, việc còn lại chủ yếu là chốt contract, dọn drift, test và xác nhận không còn edge case.
- `Decision Needed`: không nên code tiếp trước khi chốt vai trò/kiến trúc.

## 5. Quy ước khuyến nghị kiến trúc

Đây là những nguyên tắc nên giữ nhất quán trong toàn bộ P0/P1/P2:

- Service là nguồn sự thật cho detection, tracking state và fall state.
- Plugin tập trung vào frame sampling, transport, backpressure, health polling, NX metadata/event emission.
- Detections `degraded=true` chỉ để render liên tục, không dùng để sinh person lifecycle event.
- Readiness và health phải là contract cho cả plugin, tool test, ops và CI.
- Mọi default có thể cấu hình phải có một nguồn sự thật duy nhất hoặc có cơ chế tự kiểm tra drift.

## 6. Trình tự thực hiện khuyến nghị

Không nên triển khai các task theo thứ tự đúng như backlog viết ra. Nên làm theo các wave dưới đây:

### Wave A: Chốt contract và đóng drift dễ gãy

- Plugin P0.1
- Plugin P0.2
- Plugin P0.4
- Service P0.2
- Service P0.4
- Shared P0.1

Kết quả của Wave A:

- Contract plugin-service rõ ràng.
- Không còn mập mờ giữa UI default, runtime default, tracking contract và auth contract.

### Wave B: Ổn định runtime và bỏ hành vi "bí mật"

- Plugin P0.3
- Plugin P0.5
- Plugin P0.6
- Plugin P0.7
- Service P0.1
- Service P0.3
- Service P0.5
- Service P0.6

Kết quả của Wave B:

- Pipeline bớt bất ngờ, dễ quan sát, dễ giải thích khi có sự cố.

### Wave C: Dựng nền test và gate P0

- Shared P0.2
- Shared P0.3

Kết quả của Wave C:

- Có thể quyết định P0 pass/fail bằng test và tiêu chí rõ ràng, thay vì bằng cảm giác.

### Wave D: Production hóa P1

- Toàn bộ Plugin P1.x
- Toàn bộ Service P1.x
- Toàn bộ Shared P1.x

Kết quả của Wave D:

- Hệ thống đủ điều kiện vận hành thật, đo được, recover được và release được.

### Wave E: Hoàn thiện tính năng P2

- Toàn bộ Plugin P2.x
- Toàn bộ Service P2.x

Kết quả của Wave E:

- Hệ thống có giá trị sản phẩm đầy đủ, không chỉ là pipeline infer/tracking.

## 7. Ma trận tóm tắt nhanh

| Task | Trạng thái hiện tại | Nhận định ngắn |
| --- | --- | --- |
| Plugin P0.1 | Partial | Setting đã có nền, cần chốt source of truth và auth contract |
| Plugin P0.2 | Partial | Defaults đang lặp nhiều nơi, rất dễ drift |
| Plugin P0.3 | Closure/Hardening | `enabled` đã được dùng, nhưng cần verify toàn pipeline |
| Plugin P0.4 | Decision Needed | Cần chốt bỏ local tracker hay giới hạn vai trò của nó |
| Plugin P0.5 | Closure/Hardening | Có dấu hiệu đã sửa, cần test chứng minh không còn luôn bằng 0 |
| Plugin P0.6 | Partial | Debug dump tương đối đã có, nhưng log path tuyệt đối còn tồn tại |
| Plugin P0.7 | Partial | Legacy cũ đã giảm, nhưng repo vẫn chưa phản ánh kiến trúc mới hoàn toàn |
| Service P0.1 | Partial | Đã có package module, cần hoàn tất tách concern và bỏ dấu vết legacy |
| Service P0.2 | Partial | `/health` khá đúng readiness, cần chốt contract và semantics vận hành |
| Service P0.3 | Closure/Hardening | `frame_idx` theo camera đã có, cần xóa ảnh hưởng còn sót của global req counter |
| Service P0.4 | Partial | Tracking contract đã hiện trong code, chưa được chốt thành chuẩn |
| Service P0.5 | Partial | Log/status tốt, nhưng chưa có metrics endpoint dùng được |
| Service P0.6 | Partial | Backlog transport đã có, cần chốt rollout plan tương thích |
| Shared P0.1 | Open | Chưa có một file contract duy nhất |
| Shared P0.2 | Open | Chưa có bộ recorded frames có thể dùng cho E2E regression |
| Shared P0.3 | Open | Chưa có gate P0 chính thức |
| Plugin P1.1 | Open | Chưa có health polling thực dụng phía plugin |
| Plugin P1.2 | Partial | Có build script, nhưng packaging/version/release chưa sạch |
| Plugin P1.3 | Open | Chưa có integration test đúng nghĩa cho resilience |
| Plugin P1.4 | Partial | Queue policy có sẵn, metric/threshold cảnh báo chưa chốt |
| Service P1.1 | Open | Persistence trung tâm chưa đi vào code |
| Service P1.2 | Open | API nghiệp vụ thật chưa có |
| Service P1.3 | Open | Chưa có background worker tách side effects khỏi `/infer` |
| Service P1.4 | Open | Test tự động còn rất mỏng |
| Service P1.5 | Partial | Có metric logic nội bộ, chưa có observability thật |
| Shared P1.1 | Partial | `.env.example` có rồi, deployment contract chưa hoàn chỉnh |
| Shared P1.2 | Open | Chưa có benchmark 1/5/10 camera có thể lặp lại |
| Shared P1.3 | Open | SLA chưa được chốt thành tiêu chuẩn đo |
| Plugin P2.1 | Open | NX metadata chưa được enrich đầy đủ |
| Plugin P2.2 | Open | Config theo camera từ service chưa có |
| Plugin P2.3 | Open | Multi-camera identity phía NX chưa chốt |
| Service P2.1 | Open | ReID thật chưa có |
| Service P2.2 | Open | Zone engine thật chưa có |
| Service P2.3 | Open | Alert pipeline hoàn chỉnh chưa có |
| Service P2.4 | Open | Person profile/audit/retention chưa có |
| Service P2.5 | Open | Dashboard/analytics/reporting chưa bắt đầu |

## 8. Chi tiết task P0: Ổn định pipeline

### Plugin P0.1: Bỏ hardcode `127.0.0.1:18000`, thêm setting service host/port/api key/https/timeout/retry, plugin phải gửi auth header đúng contract

Trạng thái hiện tại: `Partial`

Mục tiêu:

- Cho plugin kết nối service linh hoạt theo từng camera/site thay vì mặc định local host cố định.
- Biến service client thành một phần cấu hình vận hành thật, không còn là assumption ẩn trong code.
- Chốt header auth để plugin, service, tool test và tài liệu nói cùng một tiếng.

Hiện trạng repo:

- `config/manifest.json` đã có `service_host`, `service_port`, `service_api_key`, `service_use_https`, timeout, retry.
- `device_agent.cpp` đã parse các setting này và apply xuống `ObjectDetector`.
- `object_detector.h` vẫn giữ default host/port và timeout riêng.
- Plugin hiện gửi `X-API-Key`; service hiện chấp nhận cả `X-API-Key` lẫn `Authorization: Bearer`.
- Tooling vẫn còn hardcode `127.0.0.1:18000` trong `tools/test_service.py` và `tools/manage_service.ps1`.

Vì sao task này quan trọng:

- Nếu pipeline chuyển từ local service sang AI box hoặc host khác mà còn assumption `127.0.0.1`, mọi test staging sẽ đánh lừa vận hành.
- Nếu auth contract không rõ, có thể xảy ra tình trạng plugin pass còn tool fail, hoặc service cho phép cả hai kiểu header nhưng tài liệu chỉ mô tả một kiểu.
- Timeout/retry là một phần của hành vi realtime; không chốt rõ sẽ rất khó debug backpressure và lag.

Công việc chi tiết:

1. Chốt bộ setting chính thức và tên gọi chính thức.
2. Quyết định header chuẩn duy nhất cho P0.
3. Nếu vẫn support nhiều kiểu header, phải ghi rõ cái nào là canonical, cái nào là compatibility.
4. Gom default về một nguồn sự thật hoặc ít nhất có test/assert chống drift.
5. Xóa assumption local-only khỏi test tool và script quản lý service.
6. Mô tả rõ timeout connect/read/write và retry count/backoff có ý nghĩa gì trong pipeline.

Sau khi hoàn tất sẽ làm được gì:

- Deploy plugin tới site có service không nằm cùng host với NX server.
- Bật auth một cách có chủ đích thay vì "điền thử cho có".
- Tái hiện bug theo cấu hình thật của khách hàng dễ hơn.

Cách sử dụng sau khi có task này:

- Người vận hành cấu hình endpoint và auth ngay trong NX setting thay vì sửa code.
- Dev/test có thể đổi host/https/key mà không phải patch script thủ công.
- Ops có thể tăng timeout hoặc retry cho site mạng chậm mà không ảnh hưởng site khác.

Cách verify:

- Test `http` và `https`.
- Test có API key, sai API key, không API key.
- Test service down, timeout, 401, 429, 5xx.
- Xác nhận log/diagnostic ghi đúng target host, protocol và lý do fail.

Done khi:

- Không còn host/port cố định bị dùng để gọi service trong runtime plugin và tooling chính.
- Auth contract được ghi thành văn bản duy nhất và code tuân theo đúng contract đó.
- Có test matrix tối thiểu cho timeout/retry/auth.

Repo touchpoints:

- `config/manifest.json`
- `src/sample_company/vms_server_plugins/opencv_object_detection/object_detector.h`
- `src/sample_company/vms_server_plugins/opencv_object_detection/object_detector.cpp`
- `src/sample_company/vms_server_plugins/opencv_object_detection/device_agent.cpp`
- `tools/test_service.py`
- `tools/manage_service.ps1`

### Plugin P0.2: Đồng bộ `manifest.json` với default thật trong code

Trạng thái hiện tại: `Partial`

Mục tiêu:

- Khi UI hiển thị default nào thì runtime fallback cũng đúng default đó.
- Bỏ tình trạng "UI bảo một đằng, code chạy một nẻo".

Hiện trạng repo:

- `config/manifest.json` có default cho toàn bộ setting chính.
- `device_agent.h` có default cho detection period, enqueue fps, queue size, metrics period.
- `object_detector.h` có default cho service host/port/https/timeout/retry và debug dir.
- `plugin.cpp` còn giữ manifest metadata riêng cho id/name/version/vendor.

Vì sao task này quan trọng:

- Drift default là lỗi rất khó phát hiện vì hệ thống vẫn chạy, nhưng hành vi lúc reset setting hoặc lúc field rỗng sẽ sai.
- Đây là nguồn bug vận hành kinh điển: UI tưởng default là 10 fps nhưng runtime thực tế lại là giá trị khác.

Công việc chi tiết:

1. Liệt kê toàn bộ setting và default tương ứng ở manifest, runtime và tool test.
2. Chỉ định một source of truth.
3. Nếu chưa thể sinh manifest từ code, thêm test build-time hoặc script so khớp.
4. Đồng bộ cả metadata cấp plugin như version/vendor nếu đang bị lặp ở nhiều file.
5. Cập nhật tài liệu để người dùng NX nhìn đúng giá trị mặc định.

Sau khi hoàn tất sẽ làm được gì:

- Có thể tin rằng "reset về default" trong UI sẽ cho hành vi đúng như code kỳ vọng.
- Giảm đáng kể bug khó giải thích khi migrate config giữa môi trường.

Cách sử dụng sau khi có task này:

- Có thể dùng manifest như tài liệu thao tác thực tế.
- Có thể viết automation đọc default từ một nơi mà không sợ lệch runtime.

Cách verify:

- Test camera mới chưa set gì.
- Test xóa toàn bộ custom setting và reload plugin.
- Compare snapshot runtime config với default manifest.

Done khi:

- Mỗi setting có một default canonical rõ ràng.
- Có cơ chế chống drift giữa manifest và code.
- Không còn trường hợp UI default khác runtime default.

Repo touchpoints:

- `config/manifest.json`
- `src/sample_company/vms_server_plugins/opencv_object_detection/plugin.cpp`
- `src/sample_company/vms_server_plugins/opencv_object_detection/device_agent.h`
- `src/sample_company/vms_server_plugins/opencv_object_detection/object_detector.h`

### Plugin P0.3: Bật/tắt detection theo setting `enabled`

Trạng thái hiện tại: `Closure/Hardening`

Mục tiêu:

- Tắt detection phải làm plugin ngừng enqueue frame, ngừng gọi service, dọn metadata/event dang dở và quay về trạng thái im lặng sạch.

Hiện trạng repo:

- `device_agent.cpp` đã parse `enabled`.
- `pushUncompressedVideoFrame()` đã có nhánh không enqueue khi detection bị tắt.
- Có cleanup packet khi chuyển từ bật sang tắt.
- `processFrameJob()` cũng chặn nếu detection bị tắt.
- Tuy nhiên legacy path `processFrame()` vẫn tồn tại nên cần xác nhận không có đường đi nào bỏ qua setting này.

Vì sao task này quan trọng:

- Đây là thao tác vận hành cực cơ bản: disable theo camera khi bảo trì, camera lỗi, hoặc cần so sánh với no-AI baseline.
- Nếu setting chỉ hiện trong UI mà không thật sự tắt pipeline, nó sẽ tạo cảm giác "plugin đã tắt" trong khi service vẫn bị gọi.

Công việc chi tiết:

1. Review toàn bộ path nhận frame để chắc không còn đường gọi infer khi `enabled=false`.
2. Đảm bảo event state được clear đúng khi tắt.
3. Đảm bảo khi bật lại, state khởi động sạch, không reuse queue cũ hoặc lifecycle cũ.
4. Thêm test bật/tắt liên tiếp và test trong lúc queue đang có frame.

Sau khi hoàn tất sẽ làm được gì:

- Tắt theo camera an toàn mà không phải disable cả plugin.
- Hành vi UI trở thành hành vi thật của runtime.

Cách sử dụng sau khi có task này:

- Operator có thể toggle `Enable Detection` trong NX để bật/tắt phân tích camera ngay lập tức.

Cách verify:

- Bật camera, quan sát có infer call.
- Tắt camera, xác nhận infer dừng hẳn.
- Bật lại, xác nhận pipeline sống lại sạch.
- Kiểm tra không còn fall/prolonged event bị treo active sau khi tắt.

Done khi:

- `enabled=false` chặn được mọi đường infer thực tế.
- Không còn metadata/event tồn đọng sai sau khi tắt.
- Có regression test cho toggle runtime.

Repo touchpoints:

- `src/sample_company/vms_server_plugins/opencv_object_detection/device_agent.cpp`
- `src/sample_company/vms_server_plugins/opencv_object_detection/device_agent.h`

### Plugin P0.4: Chốt vai trò tracker plugin

Trạng thái hiện tại: `Decision Needed`

Mục tiêu:

- Chỉ có một nguồn sự thật cho person lifecycle và `track_id`.
- Không để plugin local tracker và service tracking kéo hệ thống theo hai hướng khác nhau.

Hiện trạng repo:

- Worker path đang xem service là nơi trả về `track_id`, `stable`, `degraded`, `fall_detected`.
- `processFrame()` legacy path vẫn dùng `m_objectTracker->run(frame, detections)`.
- `ObjectTracker` và `object_tracker_utils.*` vẫn còn nguyên vai trò lịch sử.
- Comment trong `device_agent.cpp` đã nói service là authoritative source cho `track_id`, nhưng repo vẫn còn path cũ.

Vì sao task này quan trọng:

- Double-tracking là nguồn tạo drift nặng nhất: event một nơi, bbox một nơi, fall một nơi.
- Khi có sự cố lifecycle, đội vận hành sẽ không thể biết lỗi nằm ở service tracking hay plugin tracking.

Khuyến nghị kiến trúc:

- Khuyến nghị mạnh cho P0: service là authoritative source cho tracking và plugin không tái tracking để sinh lifecycle.
- Nếu giữ plugin tracker, chỉ giữ nó cho render smoothing/event-state phụ trợ và tuyệt đối không đổi `track_id`.

Công việc chi tiết:

1. Chốt quyết định kiến trúc bằng văn bản.
2. Nếu bỏ local tracker khỏi person lifecycle, xóa hoặc cô lập legacy path.
3. Nếu giữ local tracker cho mục đích phụ, phải định nghĩa chính xác output nào nó được phép ảnh hưởng.
4. Thêm test chứng minh `track_id` không bị rewrite bởi plugin.

Sau khi hoàn tất sẽ làm được gì:

- Debug lifecycle event và fall event dễ hơn rất nhiều.
- Contract plugin-service rõ ràng, mở đường cho persistence và E2E test chuẩn.

Cách sử dụng sau khi có task này:

- Mọi người trong team có thể nói rõ "track_id từ đâu ra" và "event nào do ai quyết định".

Cách verify:

- So sánh `track_id` từ service response với `track_id` đi ra NX metadata/event.
- Test degraded mode và stable mode riêng.
- Test camera restart và service restart để xem `track_id` có bị plugin tái sinh vô cớ không.

Done khi:

- Vai trò local tracker được chốt một lần bằng tài liệu và code.
- Không còn split-brain giữa tracking của plugin và tracking của service.

Repo touchpoints:

- `src/sample_company/vms_server_plugins/opencv_object_detection/device_agent.cpp`
- `src/sample_company/vms_server_plugins/opencv_object_detection/device_agent.h`
- `src/sample_company/vms_server_plugins/opencv_object_detection/object_tracker.cpp`
- `src/sample_company/vms_server_plugins/opencv_object_detection/object_tracker.h`
- `src/sample_company/vms_server_plugins/opencv_object_detection/object_tracker_utils.cpp`

### Plugin P0.5: Sửa metadata Fall Detect để phản ánh trạng thái thật

Trạng thái hiện tại: `Closure/Hardening`

Mục tiêu:

- Metadata object của person phải mang đúng cờ fall state tại thời điểm xuất ra NX.

Hiện trạng repo:

- `detectionsToObjectMetadataPacket()` hiện set attribute `Fall Detect` theo `detection->fallDetected`.
- `object_tracker_utils.cpp` đang preserve `fallDetected` qua đường local tracker.
- Điều này cho thấy bug "luôn bằng 0" có thể đã được sửa một phần, nhưng vẫn cần chứng minh bằng test trên mọi path.

Vì sao task này quan trọng:

- Nếu metadata sai nhưng event đúng, rule engine hoặc UI overlay downstream vẫn có thể hiểu sai.
- Fall là sự kiện nhạy cảm; sai state dễ dẫn tới alert noise hoặc bỏ sót.

Công việc chi tiết:

1. Test worker path và legacy path riêng.
2. Test case fall start, fall active kéo dài, fall clear.
3. Xác nhận metadata object đồng bộ với fall state event.
4. Nếu cần, đổi tên attribute hoặc mô tả cho rõ semantics là state hay one-shot flag.

Sau khi hoàn tất sẽ làm được gì:

- Có thể tin vào metadata `Fall Detect` trong NX rule, overlay hoặc debug capture.

Cách sử dụng sau khi có task này:

- Dùng `Fall Detect=1` như một trạng thái thật trong metadata của person, không cần suy đoán từ event rời rạc.

Cách verify:

- Gọi service với recorded frames có fall.
- So sánh service response, object metadata, fall event active/inactive.

Done khi:

- Không còn trường hợp fall event nói có nhưng metadata lại 0, hoặc ngược lại.
- Có test regression chứng minh behavior.

Repo touchpoints:

- `src/sample_company/vms_server_plugins/opencv_object_detection/device_agent.cpp`
- `src/sample_company/vms_server_plugins/opencv_object_detection/detection.h`
- `src/sample_company/vms_server_plugins/opencv_object_detection/object_tracker_utils.cpp`

### Plugin P0.6: Gỡ đường dẫn debug tuyệt đối, chuyển sang config tương đối và mặc định off

Trạng thái hiện tại: `Partial`

Mục tiêu:

- Mọi file debug và log path của plugin phải thân thiện với deploy, không phụ thuộc ổ đĩa dev machine.
- Không có I/O debug trong hot path nếu người vận hành không bật nó.

Hiện trạng repo:

- Debug frame dump trong plugin đã có setting tương đối và default off.
- `device_agent.cpp` đã resolve `debug_dump_dir` tương đối theo plugin home.
- `object_detector.cpp` chỉ dump frame khi debug bật.
- Nhưng `logging_utils.h` vẫn hardcode `D:\SafeAging\logs\plugin.log` trên Windows.

Vì sao task này quan trọng:

- Hardcoded path là kiểu lỗi chỉ nổ ở máy khách hàng, staging hoặc service account không có quyền ghi.
- Debug I/O trong hot path có thể làm méo benchmark và gây drop frame giả.

Công việc chi tiết:

1. Bỏ đường dẫn log tuyệt đối khỏi `logging_utils.h`.
2. Chuyển log path sang env/config hoặc path tương đối theo plugin home/runtime dir.
3. Xác nhận frame dump thực sự zero-cost khi `debug_dump_enabled=false`.
4. Thêm guard để debug directory không escape ra ngoài root mong muốn.

Sau khi hoàn tất sẽ làm được gì:

- Build plugin có thể chạy trên máy khác mà không đụng path dev.
- Benchmark P0 và P1 phản ánh hiệu năng thật, không nhiễm debug I/O.

Cách sử dụng sau khi có task này:

- Chỉ khi cần hỗ trợ mới bật debug dump hoặc đổi log path.
- Mặc định production chạy sạch, không ghi frame ẩn.

Cách verify:

- Chạy plugin với debug off và đo không có file dump mới sinh ra.
- Chạy plugin ở máy không có `D:\SafeAging` và xác nhận log vẫn hoạt động đúng cấu hình mới.

Done khi:

- Không còn path tuyệt đối kiểu dev-only trong runtime plugin.
- Debug dump và log path đều cấu hình được hoặc chọn path tương đối an toàn.

Repo touchpoints:

- `src/sample_company/vms_server_plugins/opencv_object_detection/device_agent.cpp`
- `src/sample_company/vms_server_plugins/opencv_object_detection/object_detector.cpp`
- `src/sample_company/vms_server_plugins/opencv_object_detection/object_detector.h`
- `src/sample_company/vms_server_plugins/opencv_object_detection/logging_utils.h`
- `config/manifest.json`

### Plugin P0.7: Dọn legacy để repo phản ánh đúng kiến trúc Python-service hiện tại

Trạng thái hiện tại: `Partial`

Mục tiêu:

- Khi ai đó đọc repo, họ thấy đúng kiến trúc hiện hành chứ không bị nhiễu bởi di sản cũ.

Hiện trạng repo:

- Không còn thấy dấu vết `yolov5s.onnx` hoặc `MobileNetSSD` qua tìm kiếm text.
- Tuy vậy vẫn còn nhiều dấu hiệu legacy khác:
- `processFrame()` legacy path còn tồn tại.
- `detection.cpp`, manifest và `device_agent.h` vẫn còn `cat` và `dog`.
- `device_agent.cpp` có log "legacy processFrame path".
- `plugin.cpp` và diagnostic strings còn hardcode version/build riêng.

Vì sao task này quan trọng:

- Legacy không chỉ là model name cũ; bất kỳ logic cũ còn sống đều là nguồn bug và làm chậm review.
- Repo lẫn lộn khiến mỗi thay đổi nhỏ trở thành thay đổi rủi ro cao.

Công việc chi tiết:

1. Xác định phần nào là compatibility cần giữ và phần nào là dead code thật.
2. Bỏ hoặc cô lập path cũ không còn phục vụ kiến trúc mới.
3. Dọn object types/comment/cấu hình không còn đúng scope sản phẩm.
4. Chuẩn hóa version/build metadata.

Sau khi hoàn tất sẽ làm được gì:

- Onboard nhanh hơn.
- Review dễ hơn.
- Giảm nguy cơ sửa nhầm vào path không còn dùng.

Cách sử dụng sau khi có task này:

- Repo trở thành tài liệu sống phản ánh pipeline hiện tại.

Cách verify:

- Chạy grep/rg lại để tìm legacy marker.
- Review build artifact và manifest xem còn dấu vết sai kiến trúc hay không.

Done khi:

- Repo không còn kể câu chuyện sai về kiến trúc hiện hành.
- Các path legacy quan trọng hoặc bị xóa, hoặc được đánh dấu compatibility rõ ràng.

Repo touchpoints:

- `src/sample_company/vms_server_plugins/opencv_object_detection/detection.cpp`
- `src/sample_company/vms_server_plugins/opencv_object_detection/device_agent.cpp`
- `src/sample_company/vms_server_plugins/opencv_object_detection/device_agent.h`
- `src/sample_company/vms_server_plugins/opencv_object_detection/plugin.cpp`
- `config/manifest.json`

### Service P0.1: Tách `service.py` thành module config/model/preprocess/tracking/fall/api

Trạng thái hiện tại: `Partial`

Mục tiêu:

- Giảm blast radius khi sửa service.
- Tách rõ concern để test và review dễ hơn.

Hiện trạng repo:

- `python/service.py` hiện đã rất mỏng.
- `python/people_analytics_service/` đã có `config.py`, `model.py`, `preprocess.py`, `tracking.py`, `fall.py`, `api.py`.
- Nghĩa là phần lớn task này đã được làm về mặt structure.
- Việc còn lại là chốt module boundary, bỏ import lắt léo/legacy và cắt tiếp `api.py`.

Vì sao task này quan trọng:

- `api.py` hiện vẫn ôm quá nhiều logic infer, tracking orchestration, debug save và endpoint handling.
- Nếu tiếp tục sửa trực tiếp vào một file lớn, nguy cơ regression sẽ quay lại.

Công việc chi tiết:

1. Chốt `service.py` chỉ còn bootstrapping.
2. Đảm bảo module package không phụ thuộc vào top-level script path hack quá nhiều.
3. Cân nhắc tách thêm `contracts.py`, `readiness.py`, `security.py`, `metrics.py` nếu `api.py` còn quá dày.
4. Dọn import từ `python/fall_detection.py` và các module lịch sử nếu còn vai trò vòng vo.

Sau khi hoàn tất sẽ làm được gì:

- Mỗi PR chỉnh tracking, fall, preprocess, auth hoặc readiness sẽ chạm ít file hơn.
- Viết unit test độc lập theo module dễ hơn.

Cách sử dụng sau khi có task này:

- Dev có thể mở đúng module liên quan thay vì lội cả `api.py`.

Cách verify:

- Chạy service bình thường.
- Chạy smoke test `/health`, `/status`, `/infer`.
- Review import graph để đảm bảo tách lớp rõ ràng.

Done khi:

- `service.py` chỉ là entrypoint.
- Mỗi concern chính của service có module tương ứng rõ ràng.
- `api.py` không còn là "god file" che hết logic.

Repo touchpoints:

- `python/service.py`
- `python/people_analytics_service/api.py`
- `python/people_analytics_service/config.py`
- `python/people_analytics_service/model.py`
- `python/people_analytics_service/preprocess.py`
- `python/people_analytics_service/tracking.py`
- `python/people_analytics_service/fall.py`

### Service P0.2: Đổi `/health` thành readiness thật

Trạng thái hiện tại: `Partial`

Mục tiêu:

- `/health` phải phản ánh khả năng sẵn sàng nhận traffic infer thật, không chỉ nói process đang sống.

Hiện trạng repo:

- `api.py` đã có `HealthResponse` với `model_loaded`, `device`, `warmup_ok`, `auth_enabled`, `ready`.
- Startup đã có warmup model và cập nhật readiness state.
- `/health` đã trả `503` khi chưa ready.
- Đây là nền rất tốt, nhưng vẫn cần chốt semantics và cách plugin/ops dùng kết quả này.

Vì sao task này quan trọng:

- Nếu plugin gọi infer ngay khi service mới bật nhưng model chưa warm, latency đầu tiên sẽ méo nặng.
- Nếu `/health` không đủ giàu thông tin, degraded mode phía plugin sẽ mù.

Công việc chi tiết:

1. Chốt `/health` là readiness endpoint chính thức.
2. Mô tả rõ khi nào `status=healthy`, `degraded`, `not_ready`.
3. Chốt plugin sẽ poll `/health` với hành vi gì cho từng trạng thái.
4. Nếu cần liveness riêng, thêm sau, nhưng không làm mờ vai trò readiness.

Sau khi hoàn tất sẽ làm được gì:

- Service startup an toàn hơn.
- Plugin có thể tránh flood infer trong lúc model chưa sẵn sàng.

Cách sử dụng sau khi có task này:

- Ops dùng `/health` để biết service ready hay chưa.
- Plugin dùng `/health` để vào/ra degraded mode.

Cách verify:

- Startup service chậm và quan sát `/health`.
- Mô phỏng model load fail và warmup fail.
- Xác nhận response fields đúng contract.

Done khi:

- `/health` là readiness thật được tài liệu hóa và dùng đồng nhất.
- Có hành vi chuẩn cho plugin trước `healthy`, `degraded`, `not_ready`.

Repo touchpoints:

- `python/people_analytics_service/api.py`
- `python/people_analytics_service/model.py`
- `python/people_analytics_service/config.py`

### Service P0.3: Bỏ dùng `req_seq` toàn cục cho fall detection, chuyển sang `frame_idx` theo từng camera

Trạng thái hiện tại: `Closure/Hardening`

Mục tiêu:

- Mọi logic TTL/stale/fall phải camera-local, không bị traffic camera khác làm sai nhịp.

Hiện trạng repo:

- `tracking.py` đã có `frame_idx` theo từng camera state.
- `api.py` tăng `state["frame_idx"]` mỗi camera trước khi gọi fall detector.
- `fall_detector.update()` đang nhận `camera_frame_idx`.
- `req_seq` toàn cục vẫn còn, nhưng hiện chủ yếu để log/debug sampling và tên file.

Vì sao task này quan trọng:

- Multi-camera là nơi counter toàn cục gây bug khó nhìn nhất: camera A đông traffic có thể làm TTL ở camera B trôi bất thường.

Công việc chi tiết:

1. Xác nhận không còn nhánh logic nghiệp vụ nào phụ thuộc `req_seq`.
2. Nếu cần sample debug, chuyển sang sample theo camera hoặc theo config rõ ràng.
3. Dọn comment, tên biến và logging để không gây hiểu lầm.

Sau khi hoàn tất sẽ làm được gì:

- Fall/tracking behavior ổn định hơn trong môi trường nhiều camera.

Cách sử dụng sau khi có task này:

- Có thể benchmark 1 camera và 10 camera mà không sợ logic TTL bị méo vì shared counter.

Cách verify:

- Chạy 2 camera với tốc độ request khác nhau.
- Kiểm tra fall tracker và stale cleanup ở từng camera.

Done khi:

- `req_seq` không còn ảnh hưởng đến fall/tracking/business logic.
- Camera-local frame index là nguồn thời gian logic duy nhất cho phần đó.

Repo touchpoints:

- `python/people_analytics_service/api.py`
- `python/people_analytics_service/tracking.py`
- `python/fall_detection.py`

### Service P0.4: Chốt contract tracking

Trạng thái hiện tại: `Partial`

Mục tiêu:

- Định nghĩa rõ `track_id` ổn định nghĩa là gì.
- Định nghĩa rõ vòng đời `tentative`, `confirmed`, `lost`, `deleted`.
- Định nghĩa degraded mode là gì và bị giới hạn ở đâu.

Hiện trạng repo:

- `api.py` đã có model `stable`, `degraded`.
- Tracking logic đã có `tentative`, `confirmed`, `lost`, `deleted`.
- Fallback raw YOLO đã được comment rõ là render-only degraded mode.
- Nhưng tất cả điều này hiện chủ yếu sống trong code và comment, chưa thành contract chung.

Vì sao task này quan trọng:

- Plugin lifecycle event, persistence, alert dedupe, benchmark và E2E test đều phụ thuộc tracking contract.
- Nếu contract không chốt, mỗi nhóm code sẽ suy diễn khác nhau về cùng một `track_id`.

Công việc chi tiết:

1. Ghi thành văn bản rule tạo mới track, confirm track, lost track, delete track, reuse track từ history.
2. Chốt `track_id` có phạm vi camera-local hay global.
3. Chốt degraded mode có được phép sinh event nào hay không.
4. Chốt khi nào `stable=true`.
5. Chốt behavior sau service restart hoặc camera reset.

Sau khi hoàn tất sẽ làm được gì:

- Có thể persist event/person state mà không sợ mỗi nơi hiểu `track_id` khác nhau.
- Plugin có thể quyết định lifecycle event chính xác và test được.

Cách sử dụng sau khi có task này:

- Dùng contract này làm nền cho test recorded frames, P0 gate và persistence schema.

Cách verify:

- Viết integration tests cho các chuỗi: new -> tentative -> confirmed -> lost -> deleted.
- Test degraded fallback và xác nhận plugin không phát lifecycle event từ degraded detections.

Done khi:

- Tracking contract tồn tại dưới dạng tài liệu và test.
- Plugin/service đều bám đúng contract đó.

Repo touchpoints:

- `python/people_analytics_service/api.py`
- `python/people_analytics_service/tracking.py`
- `working_pipeline/` tài liệu contract mới cần tạo

### Service P0.5: Chuẩn hóa log/metrics

Trạng thái hiện tại: `Partial`

Mục tiêu:

- Có thể đo service theo camera và theo thời gian bằng số liệu dùng được, không chỉ log text.

Hiện trạng repo:

- `tracking.py` đã lưu metrics window theo camera.
- `/status` đã trả `request_count`, `error_count`, `p50_inference_ms`, `p95_inference_ms`, `avg_decode_ms`, `avg_preprocess_ms`, `avg_yolo_ms`, `last_track_count`, `avg_track_count`.
- `api.py` đã log summary định kỳ.
- `infra/dev/prometheus/prometheus.yml` vẫn đang comment phần scrape service vì chưa có `/metrics`.

Vì sao task này quan trọng:

- Không đo được thì không thể chốt P0 gate hay SLA P1.
- Log text nhìn được trong tay dev, nhưng không đủ cho monitoring hoặc benchmark lặp lại.

Công việc chi tiết:

1. Giữ metric hiện có nhưng chuẩn hóa tên và semantics.
2. Quyết định expose metrics qua Prometheus format hoặc ít nhất structured endpoint riêng.
3. Thêm breakdown rõ cho infer, decode, preprocess, YOLO, track count, error count theo camera.
4. Nếu còn average-only, cân nhắc histograms/counters để hợp Prometheus hơn.

Sau khi hoàn tất sẽ làm được gì:

- Đo p50/p95 thật sự trong benchmark.
- Phát hiện camera nào đang bất thường thay vì chỉ biết "service chậm".

Cách sử dụng sau khi có task này:

- Dev dùng `/status` để debug nhanh.
- Ops/Prometheus dùng `/metrics` để scrape và cảnh báo.

Cách verify:

- Chạy nhiều camera, xác nhận metrics tách theo camera.
- So sánh số trong log, `/status`, `/metrics`.

Done khi:

- Có metrics endpoint hoặc cơ chế structured metrics dùng được thật.
- Các metric chính có định nghĩa rõ ràng và nhất quán.

Repo touchpoints:

- `python/people_analytics_service/api.py`
- `python/people_analytics_service/tracking.py`
- `infra/dev/prometheus/prometheus.yml`

### Service P0.6: Giữ tương thích transport hiện tại trước, nhưng lên backlog binary/multipart thật

Trạng thái hiện tại: `Partial`

Mục tiêu:

- Không phá plugin hiện tại.
- Đồng thời chuẩn bị đường nâng cấp transport để giảm overhead.

Hiện trạng repo:

- `api.py` đã cô lập decode logic của transport hiện tại.
- `working_pipeline/SERVICE_TRANSPORT_BACKLOG.md` đã mô tả khá rõ vì sao JSON+base64 là technical debt và hướng đi multipart.
- Plugin hiện vẫn POST JSON đến `/infer`.

Vì sao task này quan trọng:

- Nếu nóng vội đổi transport trong P0, pipeline có thể mất ổn định đúng lúc đang cần chốt tracking/health/event contract.
- Nhưng nếu không ghi rõ roadmap tương thích, về sau sẽ lại ngại đụng vì sợ vỡ API.

Công việc chi tiết:

1. Giữ `/infer` hiện tại làm compatibility path chính thức cho P0.
2. Chốt feature flag và rollout shape cho transport mới.
3. Viết rõ transport change là backlog tối ưu hóa, không phải blocker P0.

Sau khi hoàn tất sẽ làm được gì:

- Đội triển khai biết cái gì được phép giữ nguyên để ổn định P0, và cái gì để lại cho wave tối ưu sau.

Cách sử dụng sau khi có task này:

- Dùng tài liệu này như decision log khi ai đó hỏi "tại sao chưa đổi sang multipart ngay".

Cách verify:

- Không cần benchmark transport mới trong P0.
- Chỉ cần đảm bảo backlog và compatibility contract được chốt.

Done khi:

- Có decision record rõ ràng về transport hiện tại và transport tương lai.
- Không còn tranh cãi mở lại cùng một câu hỏi ở nhiều PR.

Repo touchpoints:

- `python/people_analytics_service/api.py`
- `working_pipeline/SERVICE_TRANSPORT_BACKLOG.md`
- `src/sample_company/vms_server_plugins/opencv_object_detection/object_detector.cpp`

### Shared P0.1: Viết một file API contract duy nhất

Trạng thái hiện tại: `Open`

Mục tiêu:

- Có một tài liệu duy nhất mô tả `/infer`, `/health`, `/status`, auth header, error codes, timeout, retry và semantics tracking/degraded.

Vì sao task này quan trọng:

- Hiện thông tin đang rải giữa code, comment, tool test và tài liệu roadmap.
- Không có single source of truth thì plugin và service rất dễ "đều đúng theo cách riêng".

Nội dung file contract nên có:

- Request/response schema của `/infer`.
- Auth header canonical.
- Readiness semantics của `/health`.
- Nội dung `/status` dùng cho dev/ops.
- Error code 401/403/429/5xx và ý nghĩa.
- Timeout/retry expectation phía plugin.
- Định nghĩa `stable`, `degraded`, `track_id`.

Sau khi hoàn tất sẽ làm được gì:

- Review nhanh hơn.
- Test viết đúng hơn.
- Onboarding ít phải hỏi hơn.

Cách sử dụng sau khi có task này:

- Mọi PR liên quan plugin-service phải link tới file contract này.

Done khi:

- Có đúng một file contract được cả plugin, service, tools và tài liệu khác trỏ đến.

Repo touchpoints:

- `working_pipeline/` file contract mới cần tạo
- `config/manifest.json`
- `python/people_analytics_service/api.py`
- `src/sample_company/vms_server_plugins/opencv_object_detection/object_detector.cpp`

### Shared P0.2: Tạo bộ recorded frames để test E2E plugin-service

Trạng thái hiện tại: `Open`

Mục tiêu:

- Có bộ dữ liệu cố định để test pipeline end-to-end mà không phải "nhìn bằng mắt là chính".

Hiện trạng repo:

- Có thư mục `frame_samples/` nhưng chưa phải bộ recorded frames có metadata kỳ vọng.

Nên thiết kế bộ dữ liệu như sau:

- Theo camera hoặc theo scenario.
- Mỗi scenario có frame sequence.
- Có file metadata mô tả expected detections, expected stable tracks, expected fall state, expected events.
- Có scenario tốt và scenario xấu: blur, occlusion, empty frame, service degraded, no-person, fall, re-entry.

Sau khi hoàn tất sẽ làm được gì:

- Regression tracking và fall không còn dựa vào cảm giác.
- Có thể benchmark thay đổi P0/P1 bằng cùng một đầu vào.

Cách sử dụng sau khi có task này:

- Dùng cho test service thuần.
- Dùng cho test plugin-service integration.
- Dùng để so sánh trước/sau thay đổi threshold hoặc tracking contract.

Cách verify:

- Bộ dữ liệu chạy lại nhiều lần phải cho kết quả ổn định trong tolerance đã định.

Done khi:

- Có recorded dataset versioned trong repo hoặc trong artifact test có manifest mô tả rõ.
- Có test harness sử dụng được bộ dữ liệu đó.

Repo touchpoints:

- `frame_samples/` hoặc thư mục test data mới
- `tools/`
- test harness mới cần tạo

### Shared P0.3: Gate P0 chỉ pass khi bbox ổn định, fall/event nhất quán, không còn drift config/manifest/runtime

Trạng thái hiện tại: `Open`

Mục tiêu:

- Định nghĩa "P0 ổn định" thành tiêu chí đo được.

Gate P0 nên bao gồm:

- Bbox không nhảy loạn ở cùng một đối tượng trong scenario chuẩn.
- `track_id` không bị drift vô cớ.
- Fall metadata và fall event nhất quán.
- Detections degraded không sinh person lifecycle event.
- `enabled` hoạt động đúng.
- UI default, runtime default và actual applied config không lệch nhau.
- `/health` phản ánh readiness đúng.

Sau khi hoàn tất sẽ làm được gì:

- Có thể nói "P0 pass" bằng chứng cứ.
- Tránh kéo bug nền từ P0 sang P1.

Cách sử dụng sau khi có task này:

- Dùng như checklist trước khi đóng P0 hoặc cắt release candidate.

Done khi:

- Có file gate hoặc test suite đại diện cho định nghĩa P0 pass.

Repo touchpoints:

- `working_pipeline/` tài liệu gate mới cần tạo
- test scripts/harness mới

## 9. Chi tiết task P1: Production hóa

### Plugin P1.1: Thêm health polling và degraded mode rõ ràng trong NX diagnostic events

Trạng thái hiện tại: `Open`

Mục tiêu:

- Plugin biết service đang `healthy`, `degraded`, `not_ready` và phản ứng khác nhau thay vì chỉ infer until fail.

Tác dụng:

- Hạn chế spam infer khi service chưa ready.
- Cho người vận hành thấy rõ "service sống nhưng degraded" khác với "service down".

Cần làm:

- Poll `/health` định kỳ.
- Cache last known health state.
- Chuyển health state thành NX diagnostic event có reason code.
- Tinh chỉnh infer behavior dựa trên health state.

Sau khi xong sẽ làm được gì:

- Plugin có degraded mode thật chứ không chỉ xử lý exception.

Repo touchpoints:

- `src/sample_company/vms_server_plugins/opencv_object_detection/object_detector.cpp`
- `src/sample_company/vms_server_plugins/opencv_object_detection/device_agent.cpp`

### Plugin P1.2: Chuẩn hóa packaging/build/versioning/release artifact

Trạng thái hiện tại: `Partial`

Mục tiêu:

- Build plugin và đóng gói artifact theo cách lặp lại được, sạch legacy và không phụ thuộc trí nhớ cá nhân.

Hiện trạng repo:

- Có `tools/build_plugin_windows.ps1`.
- Có `config/CMakeLists.txt`.
- Manifest và metadata plugin đang lặp ở nhiều nơi.

Tác dụng:

- Release plugin dễ bàn giao hơn.
- Giảm lỗi "build được trên máy A nhưng không ra artifact giống máy B".

Cần làm:

- Chốt version source duy nhất.
- Chuẩn hóa nội dung release artifact.
- Dọn metadata cũ.
- Ghi quy trình build/release.

Repo touchpoints:

- `tools/build_plugin_windows.ps1`
- `config/CMakeLists.txt`
- `config/manifest.json`
- `src/sample_company/vms_server_plugins/opencv_object_detection/plugin.cpp`

### Plugin P1.3: Thêm integration test cho queue/backpressure/circuit breaker/reconnect/metadata consistency

Trạng thái hiện tại: `Open`

Mục tiêu:

- Chứng minh các cơ chế resilience của plugin bằng test, không chỉ bằng log.

Tác dụng:

- Tự tin hơn khi chỉnh timeout/retry/queue policy.

Cần làm:

- Dựng test harness mô phỏng service chậm, service lỗi, reconnect, response không hợp lệ.
- Kiểm tra metadata/event nhất quán trước và sau lỗi.

### Plugin P1.4: Xem lại queue policy và thêm metric/threshold cảnh báo

Trạng thái hiện tại: `Partial`

Mục tiêu:

- Giữ policy `drop oldest` nếu nó phù hợp realtime, nhưng phải đo và cảnh báo được.

Hiện trạng repo:

- Plugin đã có queue, drop count, warning khi queue full, metrics log định kỳ.

Việc còn thiếu:

- Ngưỡng nào là chấp nhận được.
- Khi nào warning thành error.
- Metric nào cần export hoặc ít nhất đưa vào diagnostic.

## 10. Chi tiết task P1: Service

### Service P1.1: Thêm persistence PostgreSQL cho persons/events/zones/alerts/camera_configs

Trạng thái hiện tại: `Open`

Mục tiêu:

- Đưa metadata nghiệp vụ ra khỏi memory để có lịch sử, audit và API thật.

Tác dụng:

- Không mất state nghiệp vụ sau restart.
- Mở đường cho dashboard, alert, person profile.

Cần làm:

- Chốt schema.
- Chọn ORM/DAL.
- Thêm migrations.
- Giữ `/infer` không phụ thuộc trực tiếp vào DB latency.

### Service P1.2: Thêm API thật cho person/zone/event/reset

Trạng thái hiện tại: `Open`

Mục tiêu:

- Service trở thành backend nghiệp vụ thật, không chỉ là infer daemon.

Tác dụng:

- Có thể cấu hình zone/person/camera bằng API thay vì hardcode/env.
- Có thể tra cứu sự kiện và trạng thái hệ thống ngoài NX.

### Service P1.3: Thêm background worker cho alert/email và event persistence

Trạng thái hiện tại: `Open`

Mục tiêu:

- Tách side effect chậm khỏi luồng infer.

Tác dụng:

- Infer ổn định hơn.
- DB/email/object storage outage không làm nghẽn request infer.

### Service P1.4: Thêm test tự động

Trạng thái hiện tại: `Open`

Mục tiêu:

- Phủ unit test cho tracking/fall/ROI/undistort và integration/load test cho API.

Tác dụng:

- Tốc độ refactor tăng.
- Độ tự tin khi đổi threshold/contract tăng.

### Service P1.5: Thêm observability thực sự

Trạng thái hiện tại: `Partial`

Mục tiêu:

- Chuyển từ "có log và status" sang "monitor được trong production".

Hiện trạng repo:

- Metric nội bộ đã có.
- Prometheus dev stack đã có.
- Chưa có `/metrics` thật.

Tác dụng:

- Cảnh báo, dashboard, SLA mới có cơ sở.

## 11. Chi tiết task P1: Shared

### Shared P1.1: Dựng `.env.example`, deployment config, service supervision, log rotation

Trạng thái hiện tại: `Partial`

Hiện trạng repo:

- `.env.example` đã tồn tại và đã có nhiều biến cho service, PostgreSQL, MinIO, monitoring.
- Tuy nhiên nhiều biến còn là hợp đồng trên giấy hơn là contract đã được code tiêu thụ đầy đủ.

Mục tiêu:

- Biến `.env.example` thành tài liệu vận hành thật.
- Có supervision, log rotation và runtime directories rõ ràng.

### Shared P1.2: Chạy benchmark 1 camera, 5 camera, 10 camera

Trạng thái hiện tại: `Open`

Mục tiêu:

- Biết giới hạn hệ thống ở từng cấp tải bằng cùng một phương pháp đo.

Tác dụng:

- Chốt queue policy, timeout, SLA và sizing.

### Shared P1.3: Chốt SLA

Trạng thái hiện tại: `Open`

Mục tiêu:

- Biến các chỉ số kỹ thuật thành cam kết vận hành nội bộ.

Nên chốt tối thiểu:

- infer p95
- dropped frames
- event latency
- service recovery time

## 12. Chi tiết task P2: Hoàn thiện tính năng

### Plugin P2.1: Enrich metadata gửi NX

Mục tiêu:

- Metadata đi vào NX không chỉ là bbox/fall/count mà còn có person info, zone violation, severity, event details.

Sau khi hoàn tất:

- NX có thể hiển thị và rule hóa ngữ cảnh nghiệp vụ phong phú hơn.

### Plugin P2.2: Hỗ trợ config theo camera từ service

Mục tiêu:

- Plugin nhận config theo camera từ backend thay vì phụ thuộc env/setting cục bộ cố định.

### Plugin P2.3: Hoàn thiện multi-camera behavior phía NX

Mục tiêu:

- Camera identity, reconnect behavior, metadata correlation và track semantics ổn định trong môi trường nhiều camera.

### Service P2.1: Face recognition/ReID thật

Mục tiêu:

- Thay color histogram bằng nhận diện/re-identification thực dụng hơn.

### Service P2.2: Zone engine thật với polygon config/camera calibration/violation rules

Mục tiêu:

- Zone monitoring trở thành một feature sản phẩm đầy đủ, không phải placeholder.

### Service P2.3: Alert pipeline thật

Mục tiêu:

- Email/SMS/push có retry, dedupe, escalation và trạng thái delivery.

### Service P2.4: Person profile + audit log + retention policy

Mục tiêu:

- Dữ liệu người và sự kiện có vòng đời quản trị chuẩn.

### Service P2.5: Dashboard/analytics/reporting

Mục tiêu:

- Có lớp hiển thị/khai thác dữ liệu nếu đây là mục tiêu sản phẩm.

## 13. Phụ thuộc quan trọng cần nhớ

- Không nên làm persistence P1.1 trước khi tracking contract P0.4 được chốt.
- Không nên làm plugin health polling P1.1 trước khi readiness P0.2 và shared API contract P0.1 được chốt.
- Không nên benchmark P1.2 trước khi debug dumping và default drift P0.2/P0.6 đã được dọn.
- Không nên enrich metadata P2.1 trước khi event/fall semantics P0 đã ổn định.

## 14. Những phát hiện thêm ngoài backlog gốc

Đây không hẳn là task mới, nhưng nên được nhớ khi triển khai:

- `logging_utils.h` còn path tuyệt đối dev-only.
- `tools/test_service.py` và `tools/manage_service.ps1` còn hardcode service URL.
- `config/manifest.json` và `detection.cpp` còn `cat`/`dog` dù service đang `classes=[0]` cho person.
- `SAVE_DEBUG_SAMPLES=true` trong service là default không phù hợp production.
- `plugin.cpp` và manifest đang lặp metadata plugin.

## 15. Kết luận thực thi

Nếu phải tóm tắt thứ tự ưu tiên thực dụng nhất cho repo hiện tại, nên làm như sau:

1. Chốt contract plugin-service và role tracking.
2. Dọn drift default, auth, debug path, legacy path.
3. Dựng recorded frames và gate P0.
4. Sau khi P0 qua gate mới đẩy mạnh persistence, worker, observability và release hygiene của P1.
5. Chỉ nên làm P2 khi nền P0/P1 đã đo được, test được và deploy được.

Mục tiêu của toàn bộ tài liệu này là để mỗi thay đổi tiếp theo trong repo đều có ngữ cảnh: sửa cái gì, sửa để làm gì, sửa xong thì dùng thế nào, và kiểm tra bằng cách nào.
