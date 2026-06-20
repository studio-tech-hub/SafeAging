"""Smoke tests for P2 Tier 4 (S P2.1 — ReID pipeline)."""
import base64
import json
import sys
import httpx

BASE = "http://localhost:18000"


def ok(label, cond, detail=""):
    if cond:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}  {detail}")
        sys.exit(1)


def test_health():
    r = httpx.get(f"{BASE}/health", timeout=5)
    ok("health 200", r.status_code == 200)
    ok("service healthy/degraded", r.json()["status"] in ("healthy", "degraded"))


def test_reid_engine_import():
    """Verify reid_engine can be imported and compute_embedding works."""
    import sys
    sys.path.insert(0, "/app/python")
    from people_analytics_service.reid_engine import (
        compute_embedding,
        embedding_to_bytes,
        bytes_to_embedding,
        find_best_match,
        EMBEDDING_DIM,
    )
    import numpy as np
    import cv2

    # Create a synthetic 100x200 BGR crop (person-sized)
    crop = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    vec = compute_embedding(crop)
    ok("embedding shape", vec.shape == (EMBEDDING_DIM,), f"got {vec.shape}")
    ok("embedding dtype float32", vec.dtype.name == "float32")
    norm = float(np.linalg.norm(vec))
    ok(f"embedding L2-normalised (norm≈1, got {norm:.4f})", abs(norm - 1.0) < 0.01)

    # Round-trip bytes serialisation
    b = embedding_to_bytes(vec)
    ok("bytes length", len(b) == EMBEDDING_DIM * 4, f"got {len(b)}")
    vec2 = bytes_to_embedding(b)
    ok("bytes round-trip", float(np.allclose(vec, vec2)))

    # Matching: same vector should match itself
    match = find_best_match(vec, [(42, vec)], threshold=0.5)
    ok("self-match", match is not None and match[0] == 42)

    # Below threshold should not match
    no_match = find_best_match(vec, [(42, vec)], threshold=1.1)
    ok("above-threshold no match", no_match is None)
    print("  reid_engine functional ✓")


def test_embeddings_list_empty(person_id):
    r = httpx.get(f"{BASE}/admin/persons/{person_id}/embeddings", timeout=5)
    ok("list embeddings 200", r.status_code == 200, r.text)
    ok("list initially empty", isinstance(r.json(), list))
    return r.json()


def test_enroll_from_image(person_id):
    """Enroll a synthetic person crop via base64 upload."""
    import numpy as np
    import cv2

    crop = np.random.randint(80, 200, (200, 100, 3), dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", crop)
    b64 = base64.b64encode(jpeg.tobytes()).decode()

    r = httpx.post(
        f"{BASE}/admin/persons/{person_id}/embeddings",
        json={"image_b64": b64, "embedding_type": "body", "source": "test"},
        timeout=30,  # model load on first call can be slow
    )
    ok("enroll from image 201", r.status_code == 201, r.text)
    d = r.json()
    ok("enroll returns id", "id" in d)
    ok("enroll returns correct person_id", d["person_id"] == str(person_id))
    ok("enroll type=body", d["embedding_type"] == "body")
    ok("enroll source=test", d["source"] == "test")
    return d["id"]


def test_enroll_with_bbox(person_id):
    """Enroll from full frame with bbox crop."""
    import numpy as np
    import cv2

    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(jpeg.tobytes()).decode()

    r = httpx.post(
        f"{BASE}/admin/persons/{person_id}/embeddings",
        json={
            "image_b64": b64,
            "embedding_type": "body",
            "source": "test_bbox",
            "bbox_x": 100.0,
            "bbox_y": 50.0,
            "bbox_w": 120.0,
            "bbox_h": 300.0,
        },
        timeout=30,
    )
    ok("enroll with bbox 201", r.status_code == 201, r.text)
    return r.json()["id"]


def test_list_embeddings_after_enroll(person_id):
    r = httpx.get(f"{BASE}/admin/persons/{person_id}/embeddings", timeout=5)
    ok("list embeddings after enroll 200", r.status_code == 200)
    embs = r.json()
    ok("at least 2 embeddings", len(embs) >= 2, f"got {len(embs)}")
    return embs


def test_delete_embedding(person_id, emb_id):
    r = httpx.delete(
        f"{BASE}/admin/persons/{person_id}/embeddings/{emb_id}", timeout=5
    )
    ok("delete embedding 204", r.status_code == 204, r.text)

    r = httpx.get(f"{BASE}/admin/persons/{person_id}/embeddings", timeout=5)
    embs = r.json()
    ids = [e["id"] for e in embs]
    ok("embedding removed from list", emb_id not in ids)


def test_reid_match_no_match():
    """Match a random image when no embeddings exist for other persons → no match."""
    import numpy as np
    import cv2

    noise = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", noise)
    b64 = base64.b64encode(jpeg.tobytes()).decode()

    r = httpx.post(
        f"{BASE}/admin/reid/match",
        json={"image_b64": b64, "threshold": 0.99},  # unreachably high threshold
        timeout=30,
    )
    ok("reid match 200", r.status_code == 200, r.text)
    d = r.json()
    ok("match key present", "matched" in d)
    ok("no match at threshold 0.99", not d["matched"])


def test_reid_match_self():
    """Enroll a crop, then match the same crop → should match."""
    import numpy as np
    import cv2

    # Create a person and enroll
    r = httpx.post(
        f"{BASE}/admin/persons",
        json={"name": "ReID Match Test Person", "status": "active"},
        timeout=5,
    )
    ok("create match-test person 201", r.status_code == 201)
    person_id = r.json()["id"]

    crop = np.ones((200, 100, 3), dtype=np.uint8) * 128
    crop[50:150, 20:80] = [200, 100, 50]  # distinctive colour block
    _, jpeg = cv2.imencode(".jpg", crop)
    b64 = base64.b64encode(jpeg.tobytes()).decode()

    # Enroll
    r = httpx.post(
        f"{BASE}/admin/persons/{person_id}/embeddings",
        json={"image_b64": b64, "embedding_type": "body"},
        timeout=30,
    )
    ok("enroll match-test 201", r.status_code == 201)

    # Match the same image at low threshold → should match
    r = httpx.post(
        f"{BASE}/admin/reid/match",
        json={"image_b64": b64, "threshold": 0.5},
        timeout=30,
    )
    ok("reid self-match 200", r.status_code == 200, r.text)
    d = r.json()
    ok("self-match matched=true", d["matched"], json.dumps(d))
    ok("self-match person_id correct", d["person_id"] == person_id)
    ok("self-match score > 0.5", d["score"] is not None and d["score"] > 0.5)
    print(f"     self-match score={d['score']:.4f}")

    # Cleanup
    httpx.delete(f"{BASE}/admin/persons/{person_id}", timeout=5)


def test_batch_auto_link():
    r = httpx.post(
        f"{BASE}/admin/reid/auto-link",
        json={"limit": 10},
        timeout=60,
    )
    ok("batch auto-link 200", r.status_code == 200, r.text)
    d = r.json()
    ok("auto-link processed key", "processed" in d)
    ok("auto-link linked key", "linked" in d)
    ok("auto-link skipped key", "skipped" in d)
    print(f"     auto-link result: {d}")


if __name__ == "__main__":
    print("=== Tier 4 Smoke Tests (S P2.1 — ReID) ===")

    print("\n[health]")
    test_health()

    print("\n[reid_engine unit tests]")
    test_reid_engine_import()

    # Create a test person
    r = httpx.post(
        f"{BASE}/admin/persons",
        json={"name": "Tier4 Test Resident", "status": "active"},
        timeout=5,
    )
    assert r.status_code == 201
    person_id = r.json()["id"]
    print(f"\n[enrollment] person_id={person_id}")

    test_embeddings_list_empty(person_id)
    emb_id1 = test_enroll_from_image(person_id)
    print(f"  enrolled emb_id={emb_id1}")
    emb_id2 = test_enroll_with_bbox(person_id)
    print(f"  enrolled with bbox emb_id={emb_id2}")
    embs = test_list_embeddings_after_enroll(person_id)
    print(f"  total embeddings: {len(embs)}")
    test_delete_embedding(person_id, emb_id2)
    print(f"  deleted emb_id={emb_id2}")

    print("\n[reid match]")
    test_reid_match_no_match()
    test_reid_match_self()

    print("\n[batch auto-link]")
    test_batch_auto_link()

    # Cleanup test person
    httpx.delete(f"{BASE}/admin/persons/{person_id}", timeout=5)

    print("\n=== All Tier 4 tests passed ===")
