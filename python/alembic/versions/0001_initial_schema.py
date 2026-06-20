"""Initial schema: persons, person_embeddings, zones, events, alerts, camera_configs

Revision ID: 0001
Revises:
Create Date: 2026-05-26
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── persons ───────────────────────────────────────────────────────────────
    op.create_table(
        "persons",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name",   sa.String(255), nullable=False),
        sa.Column("age",    sa.Integer,     nullable=True),
        sa.Column("gender", sa.String(50),  nullable=True),
        sa.Column("notes",  sa.Text,        nullable=True),
        sa.Column("room",   sa.String(255), nullable=True),
        sa.Column(
            "status",
            sa.String(50),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # ── person_embeddings ─────────────────────────────────────────────────────
    op.create_table(
        "person_embeddings",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "person_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("persons.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("embedding",      sa.LargeBinary, nullable=False),
        sa.Column("embedding_type", sa.String(50),  nullable=False),
        sa.Column("source",         sa.String(255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_person_embeddings_person_id",
        "person_embeddings",
        ["person_id"],
    )

    # ── zones ─────────────────────────────────────────────────────────────────
    op.create_table(
        "zones",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("camera_id",  sa.String(255), nullable=False),
        sa.Column("name",       sa.String(255), nullable=False),
        sa.Column("zone_type",  sa.String(50),  nullable=False),
        sa.Column("geometry",   postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "active",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_zones_camera_id", "zones", ["camera_id"])

    # ── events ────────────────────────────────────────────────────────────────
    op.create_table(
        "events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("event_id",   sa.String(128), nullable=False),
        sa.Column("camera_id",  sa.String(255), nullable=False),
        sa.Column("track_id",   sa.BigInteger,  nullable=True),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("cls",        sa.String(50),  nullable=True),
        sa.Column("confidence", sa.Float,       nullable=True),
        sa.Column("bbox_x",     sa.Float,       nullable=True),
        sa.Column("bbox_y",     sa.Float,       nullable=True),
        sa.Column("bbox_w",     sa.Float,       nullable=True),
        sa.Column("bbox_h",     sa.Float,       nullable=True),
        sa.Column(
            "zone_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("zones.id"),
            nullable=True,
        ),
        sa.Column(
            "person_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("persons.id"),
            nullable=True,
        ),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("event_id", name="uq_events_event_id"),
    )
    op.create_index("ix_events_camera_id",   "events", ["camera_id"])
    op.create_index("ix_events_event_type",  "events", ["event_type"])
    op.create_index("ix_events_occurred_at", "events", ["occurred_at"])

    # ── alerts ────────────────────────────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "event_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("events.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("alert_type", sa.String(50),  nullable=False),
        sa.Column("recipient",  sa.String(500), nullable=True),
        sa.Column(
            "status",
            sa.String(50),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "attempts",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sent_at",         sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_alerts_event_id", "alerts", ["event_id"])
    op.create_index("ix_alerts_status",   "alerts", ["status"])

    # ── camera_configs ────────────────────────────────────────────────────────
    op.create_table(
        "camera_configs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("camera_id",            sa.String(255), nullable=False),
        sa.Column("confidence_threshold", sa.Float,       nullable=True),
        sa.Column("iou_threshold",        sa.Float,       nullable=True),
        sa.Column("frame_period",         sa.Integer,     nullable=True),
        sa.Column("zone_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("extra",    postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("camera_id", name="uq_camera_configs_camera_id"),
    )


def downgrade() -> None:
    op.drop_table("camera_configs")
    op.drop_index("ix_alerts_status",        table_name="alerts")
    op.drop_index("ix_alerts_event_id",      table_name="alerts")
    op.drop_table("alerts")
    op.drop_index("ix_events_occurred_at",   table_name="events")
    op.drop_index("ix_events_event_type",    table_name="events")
    op.drop_index("ix_events_camera_id",     table_name="events")
    op.drop_table("events")
    op.drop_index("ix_zones_camera_id",      table_name="zones")
    op.drop_table("zones")
    op.drop_index("ix_person_embeddings_person_id", table_name="person_embeddings")
    op.drop_table("person_embeddings")
    op.drop_table("persons")
