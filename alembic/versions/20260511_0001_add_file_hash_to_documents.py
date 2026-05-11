"""add file_hash to documents

Revision ID: 20260511_0001
Revises:
Create Date: 2026-05-11

Adds a SHA-256 content hash column to the documents table so that
duplicate file uploads can be detected before ingestion.
"""

from alembic import op
import sqlalchemy as sa

revision = "20260511_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("file_hash", sa.String(64), nullable=True),
    )
    op.create_index(
        "idx_documents_user_hash",
        "documents",
        ["user_id", "file_hash"],
        postgresql_where=sa.text("file_hash IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_documents_user_hash", table_name="documents")
    op.drop_column("documents", "file_hash")
