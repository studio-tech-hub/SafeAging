#!/usr/bin/env bash
# Native PostgreSQL on AI Box (required when Docker postgres cannot use SysV shm).
set -euo pipefail

DB_USER="${POSTGRES_USER:-safeaging}"
DB_PASS="${POSTGRES_PASSWORD:-safeaging_dev_password}"
DB_NAME="${POSTGRES_DB:-safeaging}"

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y postgresql postgresql-contrib

systemctl enable postgresql
systemctl start postgresql

sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1 \
  || sudo -u postgres psql -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';"
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1 \
  || sudo -u postgres psql -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

PG_CONF=$(find /etc/postgresql -name postgresql.conf | head -1)
PG_HBA=$(find /etc/postgresql -name pg_hba.conf | head -1)
sed -i "s/^#*listen_addresses.*/listen_addresses = '*'/" "$PG_CONF"
grep -q '172.17.0.0/16' "$PG_HBA" || echo "host all all 172.17.0.0/16 scram-sha-256" >> "$PG_HBA"
grep -q '127.0.0.1/32' "$PG_HBA" || echo "host all all 127.0.0.1/32 scram-sha-256" >> "$PG_HBA"

systemctl restart postgresql
pg_isready -h 127.0.0.1 -p 5432
echo "Host PostgreSQL ready: ${DB_NAME} @ 127.0.0.1:5432 (Docker uses 172.17.0.1)"
