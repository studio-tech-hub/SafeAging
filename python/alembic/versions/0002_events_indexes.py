"""S P2.4 — Add performance indexes to events table for track/person/time queries.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-31
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_events_camera_track ON events (camera_id, track_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_events_person_id ON events (person_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_events_occurred_at ON events (occurred_at)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_events_occurred_at")
    op.execute("DROP INDEX IF EXISTS ix_events_person_id")
    op.execute("DROP INDEX IF EXISTS ix_events_camera_track")
