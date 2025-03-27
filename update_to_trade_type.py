#!/usr/bin/env python
"""
Script to apply the migration from 'side' to 'trade_type'
"""
import os
import sys
import asyncio
from alembic import command
from alembic.config import Config

def run_migration():
    """Run the database migration to rename side to trade_type"""
    print("Running migration to update 'side' to 'trade_type'")
    try:
        # Get the absolute path to alembic.ini
        base_dir = os.path.dirname(os.path.abspath(__file__))
        alembic_ini = os.path.join(base_dir, 'alembic.ini')
        
        if not os.path.exists(alembic_ini):
            print(f"Error: Could not find alembic.ini at {alembic_ini}")
            sys.exit(1)
        
        # Create the Alembic config
        config = Config(alembic_ini)
        
        # Run the migration
        command.upgrade(config, "head")
        
        print("Migration completed successfully!")
        return True
    except Exception as e:
        print(f"Error running migration: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1) 