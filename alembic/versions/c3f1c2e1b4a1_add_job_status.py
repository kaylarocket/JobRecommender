"""add job status

Revision ID: c3f1c2e1b4a1
Revises: 902138dc949d
Create Date: 2026-01-20 00:00:00.000000

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c3f1c2e1b4a1'
down_revision = '902138dc949d'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'jobs',
        sa.Column('status', sa.String(length=32), server_default='active', nullable=False),
    )
    op.create_index(op.f('ix_jobs_status'), 'jobs', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_jobs_status'), table_name='jobs')
    op.drop_column('jobs', 'status')
