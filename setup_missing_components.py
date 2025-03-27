#!/usr/bin/env python3
"""
Setup script for missing components in the Entry Trade Flow

This script installs the required packages for:
1. TWAP (Time-Weighted Average Price) algorithm
2. SHAP (SHapley Additive exPlanations) for model explainability
3. Notification system
"""

import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages for the missing components."""
    requirements = [
        # SHAP for model explainability
        "shap==0.41.0",
        "matplotlib>=3.5.0",
        
        # Notification system dependencies
        "aiohttp>=3.8.3",
        "redis>=4.3.4",
        
        # Additional ML dependencies
        "joblib>=1.1.0",
        "xgboost>=1.6.2",
        "scikit-learn>=1.0.2",
        
        # Async support
        "asyncio>=3.4.3",
    ]
    
    try:
        logger.info("Installing required packages...")
        
        # Install packages using pip
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + requirements)
        
        logger.info("All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing packages: {str(e)}")
        return False

def verify_installations():
    """Verify that required packages are installed correctly."""
    try:
        logger.info("Verifying installations...")
        
        # Try importing key packages
        import_checks = [
            ("shap", "SHAP for model explainability"),
            ("matplotlib", "Matplotlib for SHAP visualizations"),
            ("aiohttp", "AsyncIO HTTP client for notifications"),
            ("redis", "Redis client for caching and cooldowns"),
            ("joblib", "Joblib for model persistence"),
            ("xgboost", "XGBoost for ML models")
        ]
        
        all_passed = True
        for package, description in import_checks:
            try:
                __import__(package)
                logger.info(f"✓ {description} ({package}) is installed")
            except ImportError:
                logger.error(f"✗ {description} ({package}) is NOT installed")
                all_passed = False
        
        if all_passed:
            logger.info("All required packages are installed correctly.")
        else:
            logger.warning("Some packages are missing. Run this script again or install them manually.")
            
        return all_passed
    except Exception as e:
        logger.error(f"Error verifying installations: {str(e)}")
        return False

def create_config_template():
    """Create a template configuration file for the notification system."""
    config_template = """# Notification System Configuration
# Copy this template to app/config/notification_config.py and modify as needed

NOTIFICATION_CONFIG = {
    # Enable/disable notifications system-wide
    'notifications_enabled': True,
    
    # Channels to use for notifications (console, email, slack, telegram, discord, sms)
    'enabled_channels': ['console'],
    
    # Minimum priority level to trigger notifications (low, medium, high, critical)
    'min_priority': 'low',
    
    # Email configuration
    'email': {
        'sender': 'your-bot@example.com',
        'recipients': ['your-email@example.com'],
        'smtp_server': 'smtp.example.com',
        'smtp_port': 587,
        'username': 'your-smtp-username',
        'password': 'your-smtp-password'
    },
    
    # Slack configuration
    'slack': {
        'webhook_url': 'https://hooks.slack.com/services/your/webhook/url',
        'channel': '#trading-bot-alerts'
    },
    
    # Telegram configuration
    'telegram': {
        'bot_token': 'your-telegram-bot-token',
        'chat_id': 'your-telegram-chat-id'
    },
    
    # Discord configuration
    'discord': {
        'webhook_url': 'https://discord.com/api/webhooks/your/webhook/url'
    },
    
    # SMS configuration (placeholder)
    'sms': {
        'provider': 'twilio',
        'api_key': 'your-api-key',
        'phone_numbers': ['+1234567890']
    },
    
    # Rate limiting settings (per hour)
    'rate_limits': {
        'email': {'count': 10, 'period_minutes': 60},
        'slack': {'count': 30, 'period_minutes': 60},
        'telegram': {'count': 30, 'period_minutes': 60},
        'discord': {'count': 30, 'period_minutes': 60},
        'sms': {'count': 5, 'period_minutes': 60}
    },
    
    # Maximum number of notifications to keep in history
    'max_history_size': 1000
}
"""

    try:
        # Create config directory if it doesn't exist
        os.makedirs('app/config', exist_ok=True)
        
        # Write the template file
        with open('app/config/notification_config_template.py', 'w') as f:
            f.write(config_template)
            
        logger.info("Created notification system configuration template at: app/config/notification_config_template.py")
        logger.info("Copy this file to notification_config.py and modify it with your settings to enable notifications.")
        return True
    except Exception as e:
        logger.error(f"Error creating config template: {str(e)}")
        return False

def main():
    """Run the setup process for missing components."""
    logger.info("Setting up missing components for the Entry Trade Flow...")
    
    # Install required packages
    if not install_requirements():
        logger.error("Failed to install required packages. Exiting.")
        return False
    
    # Verify installations
    if not verify_installations():
        logger.warning("Some packages may not be installed correctly.")
    
    # Create notification config template
    create_config_template()
    
    logger.info("\nSetup complete! The following components are now ready to use:")
    logger.info("1. TWAP algorithm for order execution (app/exchange/manager.py)")
    logger.info("2. SHAP for model explainability (app/ml/model.py)")
    logger.info("3. Notification system (app/utils/notification_system.py)")
    logger.info("\nMake sure to update the notification configuration to enable desired channels.")
    
    return True

if __name__ == "__main__":
    main() 