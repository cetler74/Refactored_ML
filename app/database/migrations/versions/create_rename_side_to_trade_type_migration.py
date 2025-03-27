"""Rename side to trade_type

Revision ID: 2f5a1b3c4d5e
Revises: ace6175e1676
Create Date: 2025-03-26 20:15:12.123456

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2f5a1b3c4d5e'
down_revision: Union[str, None] = 'ace6175e1676'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to use trade_type instead of side."""
    # Create a new enum type 'tradetype'
    op.execute("CREATE TYPE tradetype AS ENUM ('buy', 'sell')")
    
    # Add new columns
    op.add_column('trades', sa.Column('trade_type', sa.Enum('BUY', 'SELL', name='tradetype'), nullable=True))
    op.add_column('signals', sa.Column('trade_type', sa.Enum('BUY', 'SELL', name='tradetype'), nullable=True))
    
    # Copy data from old columns to new columns
    op.execute("UPDATE trades SET trade_type = side::text::tradetype")
    op.execute("UPDATE signals SET trade_type = side::text::tradetype")
    
    # Make new columns non-nullable
    op.alter_column('trades', 'trade_type', nullable=False)
    op.alter_column('signals', 'trade_type', nullable=False)
    
    # Drop old columns
    op.drop_column('trades', 'side')
    op.drop_column('signals', 'side')
    
    # Drop old enum type
    op.execute("DROP TYPE orderside")


def downgrade() -> None:
    """Downgrade schema to use side instead of trade_type."""
    # Create the old enum type 'orderside'
    op.execute("CREATE TYPE orderside AS ENUM ('BUY', 'SELL')")
    
    # Add old columns
    op.add_column('trades', sa.Column('side', sa.Enum('BUY', 'SELL', name='orderside'), nullable=True))
    op.add_column('signals', sa.Column('side', sa.Enum('BUY', 'SELL', name='orderside'), nullable=True))
    
    # Copy data from new columns to old columns
    op.execute("UPDATE trades SET side = trade_type::text::orderside")
    op.execute("UPDATE signals SET side = trade_type::text::orderside")
    
    # Make old columns non-nullable
    op.alter_column('trades', 'side', nullable=False)
    op.alter_column('signals', 'side', nullable=False)
    
    # Drop new columns
    op.drop_column('trades', 'trade_type')
    op.drop_column('signals', 'trade_type')
    
    # Drop new enum type
    op.execute("DROP TYPE tradetype") 