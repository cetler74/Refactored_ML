#!/usr/bin/env python3
"""Test script for database connection"""

import asyncio
import os
from dotenv import load_dotenv
from app.database.manager import DatabaseManager

# Load environment variables
load_dotenv()

async def test_db():
    """Test database connection."""
    # Get database URL from environment
    db_url = os.environ.get("DATABASE_URL", "postgresql+asyncpg://carloslarramba@localhost:5432/trading_bot")
    print(f"Testing connection to database: {db_url}")
    
    # Create database manager
    manager = DatabaseManager(db_url, debug=True)
    print("Database manager created")
    
    # Initialize database
    result = await manager.initialize(create_tables=False)
    print(f"Database initialized: {result}")
    
    return result

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_db()) 