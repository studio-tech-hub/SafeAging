# 📦 Deployment & Operations Guide

Complete guide for deploying, monitoring, and operating the elderly care management system.

---

## Table of Contents

1. [Deployment Strategy](#deployment-strategy)
2. [Docker Containerization](#docker-containerization)
3. [Service Configuration](#service-configuration)
4. [Monitoring & Alerting](#monitoring--alerting)
5. [Database Management](#database-management)
6. [Troubleshooting](#troubleshooting)
7. [Security Best Practices](#security-best-practices)
8. [Disaster Recovery](#disaster-recovery)

---

## 🚀 Deployment Strategy

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│         Production Environment                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  Load Balancer (Nginx/Caddy)                   │
│           ↓                                     │
│  ┌──────────────────────────────────────────┐  │
│  │    Docker Network (Service Stack)        │  │
│  │                                          │  │
│  │  ┌────────────┐  ┌────────────┐        │  │
│  │  │Service-1   │  │Service-2   │        │  │
│  │  │(Port 8001) │  │(Port 8002) │        │  │
│  │  └────────────┘  └────────────┘        │  │
│  │        ↓              ↓                 │  │
│  │  ┌──────────────────────────────┐      │  │
│  │  │  PostgreSQL Database          │      │  │
│  │  │  (Volume: db_data)            │      │  │
│  │  └──────────────────────────────┘      │  │
│  │        ↓                                 │  │
│  │  ┌──────────────────────────────┐      │  │
│  │  │  Redis Cache                  │      │  │
│  │  │  (Volume: cache_data)         │      │  │
│  │  └──────────────────────────────┘      │  │
│  │                                          │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  Monitoring (Prometheus + Grafana)             │
│  Logging (ELK Stack)                           │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Deployment Phases

#### Phase 1: Development Environment
```bash
# Single machine setup for testing
- Service + Database on localhost
- Plugin compiled locally
- Testing with sample videos
- No high availability requirements
```

#### Phase 2: Staging Environment
```bash
# Pre-production testing
- Service in Docker container
- Database on dedicated machine
- Multiple cameras simulated
- Performance testing
- Security scanning
```

#### Phase 3: Production Environment
```bash
# Full production deployment
- Service with 2+ replicas (load balanced)
- High-availability database (replication)
- Monitoring and alerting active
- Automated backups
- 24/7 support procedures
```

---

## 🐳 Docker Containerization

### Dockerfile (Python Service)

```dockerfile
# Dockerfile - Service container
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY service.py .
COPY analytics/ ./analytics/
COPY integrations/ ./integrations/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:18000/health || exit 1

# Expose port
EXPOSE 18000

# Run service
CMD ["python", "service.py"]
```

### docker-compose.yml

```yaml
# docker-compose.yml - Full stack orchestration
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: elderly_care_db
    environment:
      POSTGRES_USER: ${DB_USER:-postgres}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-changeme}
      POSTGRES_DB: ${DB_NAME:-elderly_care}
    ports:
      - "5432:5432"
    volumes:
      - db_data:/var/lib/postgresql/data
      - ./migrations/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - elderly_care_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: elderly_care_cache
    ports:
      - "6379:6379"
    volumes:
      - cache_data:/data
    networks:
      - elderly_care_network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Analytics Service (Instance 1)
  service_1:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: elderly_care_service_1
    environment:
      SERVICE_PORT: 18000
      SERVICE_HOST: 0.0.0.0
      MODEL_PATH: yolov8n.pt
      CONFIDENCE_THRESHOLD: 0.45
      DATABASE_URL: postgresql://${DB_USER:-postgres}:${DB_PASSWORD:-changeme}@postgres:5432/${DB_NAME:-elderly_care}
      REDIS_URL: redis://redis:6379/0
      AZURE_FACE_API_KEY: ${AZURE_FACE_API_KEY}
      AZURE_FACE_ENDPOINT: ${AZURE_FACE_ENDPOINT}
      SENDGRID_API_KEY: ${SENDGRID_API_KEY}
      LOG_LEVEL: INFO
    ports:
      - "18000:18000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./configs:/app/configs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - elderly_care_network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:18000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  # Analytics Service (Instance 2)
  service_2:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: elderly_care_service_2
    environment:
      SERVICE_PORT: 18001
      SERVICE_HOST: 0.0.0.0
      MODEL_PATH: yolov8n.pt
      CONFIDENCE_THRESHOLD: 0.45
      DATABASE_URL: postgresql://${DB_USER:-postgres}:${DB_PASSWORD:-changeme}@postgres:5432/${DB_NAME:-elderly_care}
      REDIS_URL: redis://redis:6379/0
      AZURE_FACE_API_KEY: ${AZURE_FACE_API_KEY}
      AZURE_FACE_ENDPOINT: ${AZURE_FACE_ENDPOINT}
      SENDGRID_API_KEY: ${SENDGRID_API_KEY}
      LOG_LEVEL: INFO
    ports:
      - "18001:18001"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./configs:/app/configs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - elderly_care_network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:18001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    container_name: elderly_care_lb
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - service_1
      - service_2
    networks:
      - elderly_care_network
    restart: unless-stopped

  # Prometheus (Monitoring)
  prometheus:
    image: prom/prometheus:latest
    container_name: elderly_care_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - elderly_care_network
    restart: unless-stopped

  # Grafana (Dashboards)
  grafana:
    image: grafana/grafana:latest
    container_name: elderly_care_grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
      GF_USERS_ALLOW_SIGN_UP: 'false'
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
    depends_on:
      - prometheus
    networks:
      - elderly_care_network
    restart: unless-stopped

  # Elasticsearch (Logging)
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.0.0
    container_name: elderly_care_elasticsearch
    environment:
      discovery.type: single-node
      xpack.security.enabled: 'false'
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - elderly_care_network
    restart: unless-stopped

  # Kibana (Log Analysis)
  kibana:
    image: docker.elastic.co/kibana/kibana:8.0.0
    container_name: elderly_care_kibana
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - elderly_care_network
    restart: unless-stopped

volumes:
  db_data:
    driver: local
  cache_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
  elasticsearch_data:
    driver: local

networks:
  elderly_care_network:
    driver: bridge
```

### Build and Deploy

```bash
# Development
docker-compose -f docker-compose.dev.yml up -d

# Staging
docker-compose -f docker-compose.staging.yml up -d

# Production
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f service_1

# Update service
docker-compose pull service_1
docker-compose up -d service_1

# Cleanup
docker-compose down -v
```

---

## ⚙️ Service Configuration

### Environment Variables (.env)

```env
# Database
DB_USER=postgres
DB_PASSWORD=your_secure_password_here
DB_NAME=elderly_care
DB_HOST=postgres
DB_PORT=5432

# Service
SERVICE_PORT=18000
SERVICE_HOST=0.0.0.0
LOG_LEVEL=INFO
LOG_FILE=/app/logs/service.log

# Model
MODEL_PATH=yolov8n.pt
CONFIDENCE_THRESHOLD=0.45
IOU_THRESHOLD=0.45
MIN_DETECTION_AREA=20

# Feature Flags
ENABLE_FALL_DETECTION=true
ENABLE_ZONE_CHECK=true
ENABLE_FACE_RECOGNITION=true
ENABLE_EMAIL_ALERTS=true

# Azure Face API
AZURE_FACE_API_KEY=your_api_key_here
AZURE_FACE_ENDPOINT=https://your-region.face.cognitive.microsoft.com/

# Email Service (SendGrid)
SENDGRID_API_KEY=your_sendgrid_api_key_here
SENDER_EMAIL=alerts@elderly-care.com

# Redis
REDIS_URL=redis://redis:6379/0

# Twilio (SMS alerts - optional)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890

# Monitoring
PROMETHEUS_PORT=8001
ENABLE_METRICS=true

# Timeouts (milliseconds)
HTTP_TIMEOUT_MS=2500
DB_TIMEOUT_MS=5000
FACE_API_TIMEOUT_MS=10000
```

### Configuration Schema

```python
# config.py
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/elderly_care"
    
    # Service
    service_port: int = 18000
    service_host: str = "0.0.0.0"
    log_level: str = "INFO"
    
    # Model
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    min_detection_area: int = 20
    
    # Features
    enable_fall_detection: bool = True
    enable_zone_check: bool = True
    enable_face_recognition: bool = True
    enable_email_alerts: bool = True
    
    # External APIs
    azure_face_api_key: Optional[str] = None
    azure_face_endpoint: Optional[str] = None
    sendgrid_api_key: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

---

## 📊 Monitoring & Alerting

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'elderly-care-system'

scrape_configs:
  # Service metrics
  - job_name: 'service'
    static_configs:
      - targets: ['service_1:8001', 'service_2:8001']
    scrape_interval: 10s

  # Database metrics
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 30s

  # Redis metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### Key Metrics to Monitor

**Service Metrics:**
```
# Inference Performance
elderly_care_inference_duration_seconds (histogram)
elderly_care_detections_per_frame (gauge)
elderly_care_model_load_time_seconds (gauge)

# Request Performance
elderly_care_http_requests_total (counter)
elderly_care_http_request_duration_seconds (histogram)

# Business Metrics
elderly_care_fall_detections_total (counter)
elderly_care_zone_violations_total (counter)
elderly_care_alerts_sent_total (counter)
elderly_care_active_tracks (gauge)
```

**Database Metrics:**
```
pg_connections (gauge)
pg_database_size_bytes (gauge)
pg_query_duration_seconds (histogram)
pg_write_duration_seconds (histogram)
```

### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
  - name: elderly_care
    rules:
      # Service health
      - alert: ServiceDown
        expr: up{job="service"} == 0
        for: 1m
        annotations:
          summary: "Service is down"

      # High error rate
      - alert: HighErrorRate
        expr: rate(elderly_care_http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "Service error rate > 5%"

      # Database connectivity
      - alert: DatabaseConnectionError
        expr: pg_connections > 90
        for: 1m
        annotations:
          summary: "Database connection pool nearly exhausted"

      # Model inference latency
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, elderly_care_inference_duration_seconds) > 0.1
        for: 5m
        annotations:
          summary: "Model inference p95 latency > 100ms"

      # Alert delivery failure
      - alert: AlertDeliveryFailure
        expr: rate(elderly_care_alerts_failed_total[5m]) > 0.01
        for: 5m
        annotations:
          summary: "Alert delivery failure rate > 1%"
```

### Grafana Dashboards

**Dashboard 1: System Overview**
```
- Service uptime
- Request rate (requests/sec)
- Error rate (%)
- Average inference time
- Active tracks
- Database connections
```

**Dashboard 2: Analytics**
```
- Fall detections (hourly)
- Zone violations (hourly)
- Alerts sent (by type)
- Person recognition accuracy
- Top cameras by activity
- Alert delivery success rate
```

**Dashboard 3: Performance**
```
- Inference time p50, p95, p99
- HTTP request duration
- Database query time
- Cache hit rate
- Model load time
- Memory usage
```

---

## 🗄️ Database Management

### Initial Setup

```bash
# Connect to database
psql -h localhost -U postgres -d elderly_care

# Run migrations
psql -h localhost -U postgres -d elderly_care < migrations/001_init.sql
psql -h localhost -U postgres -d elderly_care < migrations/002_add_zones.sql
```

### Migration Example

```sql
-- migrations/001_init.sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

CREATE TABLE persons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    age INT,
    gender CHAR(1),
    room_number VARCHAR(50),
    emergency_contact VARCHAR(20),
    status VARCHAR(20) DEFAULT 'active',
    face_image BYTEA,
    face_embedding BYTEA,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_persons_name ON persons(name);
CREATE INDEX idx_persons_status ON persons(status);
CREATE INDEX idx_persons_created ON persons(created_at);
```

### Backup Strategy

```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/backups/elderly_care"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.sql.gz"

mkdir -p $BACKUP_DIR

# Full backup
pg_dump -h localhost -U postgres elderly_care | gzip > $BACKUP_FILE

# Keep last 7 days of backups
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE"
```

### Restore from Backup

```bash
# Restore from backup
gunzip -c backup_20260118_120000.sql.gz | psql -h localhost -U postgres elderly_care
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue: Service not connecting to database**
```bash
# Check connectivity
docker-compose exec service_1 \
  python -c "from sqlalchemy import create_engine; \
  engine = create_engine(os.getenv('DATABASE_URL')); \
  print(engine.execute('SELECT 1'))"

# Check logs
docker-compose logs postgres
docker-compose logs service_1
```

**Issue: High inference latency**
```python
# Profile inference
python debug_frame_quality.py

# Check model
python -c "from ultralytics import YOLO; \
  model = YOLO('yolov8n.pt'); \
  print(model.info())"
```

**Issue: Memory leak**
```bash
# Monitor memory
docker stats elderly_care_service_1

# Check for growing processes
ps aux | grep python | sort -k4 -rn
```

**Issue: Database deadlock**
```sql
-- Check active locks
SELECT * FROM pg_stat_activity 
WHERE state = 'active';

-- Kill blocking query
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'active' AND query_start < now() - interval '30 min';
```

---

## 🔐 Security Best Practices

### API Security

```python
# 1. Authentication (JWT)
from fastapi_jwt_auth import AuthJWT
from pydantic import BaseModel

class User(BaseModel):
    username: str
    password: str

@app.post("/login")
async def login(user: User, Authorize: AuthJWT = Depends()):
    # Verify credentials
    if verify_password(user.password):
        access_token = Authorize.create_access_token(
            subject=user.username
        )
        return {"access_token": access_token}

@app.get("/person/{person_id}")
async def get_person(person_id: str, Authorize: AuthJWT = Depends()):
    Authorize.jwt_required()
    # Protected endpoint
```

### Database Security

```yaml
# PostgreSQL hardening
postgresql:
  environment:
    - POSTGRES_INITDB_ARGS=-c max_connections=100 -c shared_buffers=256MB
    - POSTGRES_HOST_AUTH_METHOD=md5
  command: |
    -c ssl=on
    -c ssl_cert_file=/var/lib/postgresql/server.crt
    -c ssl_key_file=/var/lib/postgresql/server.key
```

### Data Encryption

```python
# Encrypt sensitive data at rest
from cryptography.fernet import Fernet

def encrypt_face_embedding(embedding: np.ndarray) -> bytes:
    cipher = Fernet(ENCRYPTION_KEY)
    data = pickle.dumps(embedding)
    return cipher.encrypt(data)

def decrypt_face_embedding(encrypted: bytes) -> np.ndarray:
    cipher = Fernet(ENCRYPTION_KEY)
    data = cipher.decrypt(encrypted)
    return pickle.loads(data)
```

---

## 🆘 Disaster Recovery

### Disaster Recovery Plan

| Scenario | RTO | RPO | Action |
|----------|-----|-----|--------|
| Single service crash | 1 min | 0 sec | Auto-restart via docker |
| Database failure | 5 min | 5 min | Promote replica or restore |
| Data corruption | 30 min | 5 min | Restore from backup |
| Network outage | 15 min | 0 sec | Failover to backup connection |
| Complete site failure | 2 hours | 30 min | Restore to DR site |

### Backup Strategy

```bash
# Hourly incremental backups
0 * * * * /scripts/backup_database.sh

# Daily full backups
0 3 * * * /scripts/backup_full.sh

# Weekly encrypted backups to S3
0 4 * * 0 /scripts/backup_s3.sh

# Monthly retention in cold storage
0 5 1 * * /scripts/backup_archive.sh
```

### Testing Recovery

```bash
# Monthly DR drill
docker-compose -f docker-compose.dr.yml up -d

# Verify data integrity
python verify_recovery.py

# Test failover time
time docker-compose down && time docker-compose up -d
```

---

## 📋 Operational Checklists

### Daily Operations
- [ ] Check service health: `curl http://localhost/health`
- [ ] Review error logs for warnings
- [ ] Monitor database size
- [ ] Check active alert count

### Weekly Operations
- [ ] Review performance metrics
- [ ] Check backup completion
- [ ] Update security patches
- [ ] Verify failover works

### Monthly Operations
- [ ] Full DR drill
- [ ] Database maintenance (VACUUM, ANALYZE)
- [ ] SSL certificate renewal check
- [ ] Security audit

### Quarterly Operations
- [ ] Model performance evaluation
- [ ] Capacity planning
- [ ] Security penetration test
- [ ] Documentation update

---

**Document Version:** 1.0  
**Created:** January 18, 2026
