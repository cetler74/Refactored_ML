import logging
import redis
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PairCooldownManager:
    """
    Manages trading pair cooldowns using Redis.
    Provides a central location for cooldown functionality used by pair selectors.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, redis_url: Optional[str] = None):
        """
        Initialize the pair cooldown manager.
        
        Args:
            redis_client: Existing Redis client (optional)
            redis_url: Redis connection URL (optional, used if client not provided)
        """
        self.redis_client = redis_client
        
        if self.redis_client is None and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info("Connected to Redis for pair cooldown management")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                self.redis_client = None
    
    def set_cooldown(self, pair: str, hours: int = 24) -> bool:
        """
        Set a cooldown period for a pair.
        
        Args:
            pair: Trading pair symbol
            hours: Number of hours for cooldown
            
        Returns:
            Success status
        """
        if not self.redis_client or not pair:
            return False
            
        try:
            cooldown_key = f"trading_bot:symbol_cooldown:{pair}"
            seconds = hours * 3600
            self.redis_client.setex(cooldown_key, seconds, 1)
            logger.info(f"Set cooldown for {pair} for {hours} hours")
            return True
        except Exception as e:
            logger.error(f"Error setting cooldown for {pair}: {str(e)}")
            return False
    
    def has_cooldown(self, pair: str) -> bool:
        """
        Check if a pair is in cooldown period.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            True if in cooldown, False otherwise
        """
        if not self.redis_client or not pair:
            return False
            
        try:
            cooldown_key = f"trading_bot:symbol_cooldown:{pair}"
            return bool(self.redis_client.exists(cooldown_key))
        except Exception as e:
            logger.error(f"Error checking cooldown for {pair}: {str(e)}")
            return False
    
    def get_cooldown_ttl(self, pair: str) -> int:
        """
        Get the remaining cooldown time in seconds.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            Remaining time in seconds, 0 if not in cooldown or error
        """
        if not self.redis_client or not pair:
            return 0
            
        try:
            cooldown_key = f"trading_bot:symbol_cooldown:{pair}"
            ttl = self.redis_client.ttl(cooldown_key)
            return max(0, ttl)
        except Exception as e:
            logger.error(f"Error getting cooldown TTL for {pair}: {str(e)}")
            return 0
    
    def clear_cooldown(self, pair: str) -> bool:
        """
        Clear the cooldown for a pair.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            Success status
        """
        if not self.redis_client or not pair:
            return False
            
        try:
            cooldown_key = f"trading_bot:symbol_cooldown:{pair}"
            self.redis_client.delete(cooldown_key)
            logger.info(f"Cleared cooldown for {pair}")
            return True
        except Exception as e:
            logger.error(f"Error clearing cooldown for {pair}: {str(e)}")
            return False
    
    def get_all_cooldowns(self) -> Dict[str, int]:
        """
        Get all pairs currently in cooldown with their remaining times.
        
        Returns:
            Dictionary mapping pairs to remaining cooldown time in seconds
        """
        result = {}
        
        if not self.redis_client:
            return result
            
        try:
            # Get all cooldown keys
            pattern = "trading_bot:symbol_cooldown:*"
            keys = self.redis_client.keys(pattern)
            
            for key in keys:
                try:
                    # Extract pair from key
                    pair = key.decode().split(":")[-1]
                    ttl = self.redis_client.ttl(key)
                    
                    if ttl > 0:
                        result[pair] = ttl
                except Exception as e:
                    logger.debug(f"Error processing cooldown key {key}: {str(e)}")
            
            return result
        except Exception as e:
            logger.error(f"Error getting all cooldowns: {str(e)}")
            return {}
    
    def get_formatted_cooldowns(self) -> List[Dict[str, Any]]:
        """
        Get all cooldowns in a formatted list for display.
        
        Returns:
            List of dictionaries with formatted cooldown information
        """
        result = []
        cooldowns = self.get_all_cooldowns()
        
        for pair, ttl in cooldowns.items():
            # Convert seconds to hours and minutes
            hours = ttl // 3600
            minutes = (ttl % 3600) // 60
            
            expires_at = datetime.now() + timedelta(seconds=ttl)
            
            result.append({
                "pair": pair,
                "remaining_seconds": ttl,
                "remaining_time": f"{hours}h {minutes}m",
                "expires_at": expires_at.isoformat()
            })
        
        # Sort by remaining time (ascending)
        result.sort(key=lambda x: x["remaining_seconds"])
        
        return result 