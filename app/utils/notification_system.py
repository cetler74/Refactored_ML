#!/usr/bin/env python3
"""
Notification System

This module provides a flexible notification system for the trading bot to alert
about important events, trade executions, errors, and performance updates.
"""

import logging
import json
import smtplib
import os
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationCategory(Enum):
    """Notification categories"""
    TRADE_SIGNAL = "trade_signal"
    TRADE_EXECUTION = "trade_execution"
    ERROR = "error"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    ML_UPDATE = "ml_update"
    MARKET_ALERT = "market_alert"

class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SMS = "sms"
    CONSOLE = "console"

class NotificationSystem:
    """
    A flexible notification system that can send alerts and updates
    through multiple channels including email, Slack, Discord, Telegram, and SMS.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the notification system.
        
        Args:
            config: Dictionary with notification configuration settings
        """
        self.config = config
        self.enabled_channels = config.get('enabled_channels', ['console'])
        self.enabled = config.get('notifications_enabled', True)
        self.min_priority = config.get('min_priority', 'medium')
        
        # Email configuration
        self.email_config = config.get('email', {})
        self.email_sender = self.email_config.get('sender')
        self.email_recipients = self.email_config.get('recipients', [])
        self.email_smtp_server = self.email_config.get('smtp_server')
        self.email_smtp_port = self.email_config.get('smtp_port', 587)
        self.email_username = self.email_config.get('username')
        self.email_password = self.email_config.get('password')
        
        # Slack configuration
        self.slack_config = config.get('slack', {})
        self.slack_webhook_url = self.slack_config.get('webhook_url')
        self.slack_channel = self.slack_config.get('channel')
        
        # Telegram configuration
        self.telegram_config = config.get('telegram', {})
        self.telegram_bot_token = self.telegram_config.get('bot_token')
        self.telegram_chat_id = self.telegram_config.get('chat_id')
        
        # Discord configuration
        self.discord_config = config.get('discord', {})
        self.discord_webhook_url = self.discord_config.get('webhook_url')
        
        # SMS configuration
        self.sms_config = config.get('sms', {})
        self.sms_provider = self.sms_config.get('provider')
        self.sms_api_key = self.sms_config.get('api_key')
        self.sms_phone_numbers = self.sms_config.get('phone_numbers', [])
        
        # Initialize notification history
        self.notification_history = []
        self.max_history_size = config.get('max_history_size', 1000)
        
        # Rate limiting
        self.rate_limits = config.get('rate_limits', {
            'email': {'count': 10, 'period_minutes': 60},
            'slack': {'count': 30, 'period_minutes': 60},
            'telegram': {'count': 30, 'period_minutes': 60},
            'discord': {'count': 30, 'period_minutes': 60},
            'sms': {'count': 5, 'period_minutes': 60}
        })
        
        # Track sent notifications for rate limiting
        self.sent_notifications = {
            channel.value: [] for channel in NotificationChannel
        }
        
        # Log initialization
        channels_str = ', '.join(self.enabled_channels)
        logger.info(f"Notification system initialized with channels: {channels_str}")
    
    async def send_notification(self, 
                          message: str, 
                          title: str = None, 
                          category: Union[NotificationCategory, str] = NotificationCategory.SYSTEM,
                          priority: Union[NotificationPriority, str] = NotificationPriority.MEDIUM,
                          channels: List[str] = None,
                          data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send a notification through all configured channels.
        
        Args:
            message: The main notification message
            title: Optional title for the notification
            category: Category of the notification
            priority: Priority level of the notification
            channels: Specific channels to use (if None, uses all enabled channels)
            data: Additional data to include with the notification
        
        Returns:
            Dictionary with results for each channel
        """
        if not self.enabled:
            logger.info(f"Notifications disabled, would have sent: {title}: {message}")
            return {"success": False, "error": "Notifications disabled"}
        
        # Convert enum values if strings are provided
        if isinstance(category, str):
            try:
                category = NotificationCategory(category)
            except ValueError:
                category = NotificationCategory.SYSTEM
                
        if isinstance(priority, str):
            try:
                priority = NotificationPriority(priority)
            except ValueError:
                priority = NotificationPriority.MEDIUM
        
        # Set default title if not provided
        if not title:
            title = f"{category.value.replace('_', ' ').title()} Alert"
            
        # Use specified channels or fall back to all enabled ones
        channels_to_use = channels if channels else self.enabled_channels
        
        # Check if we should send based on priority
        if not self._should_send_by_priority(priority):
            logger.debug(f"Notification skipped due to priority filter: {priority.value}")
            return {"success": False, "error": "Priority below threshold"}
        
        # Prepare the notification
        notification = {
            "title": title,
            "message": message,
            "category": category.value,
            "priority": priority.value,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        
        # Store in history
        self._add_to_history(notification)
        
        # Send through each requested channel
        results = {}
        tasks = []
        
        for channel in channels_to_use:
            # Check rate limiting
            if not self._check_rate_limit(channel):
                results[channel] = {"success": False, "error": "Rate limit exceeded"}
                continue
                
            # Send based on channel type
            if channel == NotificationChannel.EMAIL.value:
                if self._can_send_email():
                    tasks.append(self._send_email(notification))
            elif channel == NotificationChannel.SLACK.value:
                if self.slack_webhook_url:
                    tasks.append(self._send_slack(notification))
            elif channel == NotificationChannel.TELEGRAM.value:
                if self.telegram_bot_token and self.telegram_chat_id:
                    tasks.append(self._send_telegram(notification))
            elif channel == NotificationChannel.DISCORD.value:
                if self.discord_webhook_url:
                    tasks.append(self._send_discord(notification))
            elif channel == NotificationChannel.SMS.value:
                if self._can_send_sms():
                    tasks.append(self._send_sms(notification))
            elif channel == NotificationChannel.CONSOLE.value:
                self._send_console(notification)
                results[channel] = {"success": True}
        
        # Wait for all notification tasks to complete
        if tasks:
            channel_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            channel_index = 0
            for channel in channels_to_use:
                if channel in results:
                    continue  # Already set results for this channel
                    
                if channel_index < len(channel_results):
                    result = channel_results[channel_index]
                    if isinstance(result, Exception):
                        results[channel] = {"success": False, "error": str(result)}
                    else:
                        results[channel] = result
                    channel_index += 1
        
        return {
            "success": any(r.get("success", False) for r in results.values()) if results else False,
            "channels": results,
            "notification": notification
        }
    
    def _add_to_history(self, notification: Dict[str, Any]):
        """Add a notification to the history, respecting the max size."""
        self.notification_history.append(notification)
        if len(self.notification_history) > self.max_history_size:
            self.notification_history = self.notification_history[-self.max_history_size:]
    
    def get_notification_history(self, 
                               limit: int = 100, 
                               category: Optional[str] = None,
                               min_priority: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get notification history, optionally filtered by category and priority.
        
        Args:
            limit: Maximum number of notifications to return
            category: Filter by notification category
            min_priority: Filter by minimum priority level
            
        Returns:
            List of notification dictionaries
        """
        result = self.notification_history
        
        # Apply filters
        if category:
            result = [n for n in result if n.get("category") == category]
            
        if min_priority:
            priorities = [p.value for p in NotificationPriority]
            min_idx = priorities.index(min_priority) if min_priority in priorities else 0
            result = [n for n in result if priorities.index(n.get("priority", "low")) >= min_idx]
            
        # Return the most recent notifications up to the limit
        return result[-limit:]
    
    def _should_send_by_priority(self, priority: NotificationPriority) -> bool:
        """Check if a notification should be sent based on its priority."""
        priorities = [p.value for p in NotificationPriority]
        min_idx = priorities.index(self.min_priority) if self.min_priority in priorities else 0
        current_idx = priorities.index(priority.value)
        
        return current_idx >= min_idx
    
    def _check_rate_limit(self, channel: str) -> bool:
        """
        Check if sending to a channel would exceed the rate limit.
        
        Args:
            channel: The channel to check
            
        Returns:
            True if sending is allowed, False if rate limit would be exceeded
        """
        if channel not in self.rate_limits:
            return True
            
        limit = self.rate_limits[channel]
        count = limit.get('count', 0)
        period_minutes = limit.get('period_minutes', 60)
        
        # Get timestamp threshold
        threshold = datetime.now().timestamp() - (period_minutes * 60)
        
        # Filter out old timestamps
        self.sent_notifications[channel] = [
            ts for ts in self.sent_notifications[channel] 
            if ts > threshold
        ]
        
        # Check if we're under the limit
        if len(self.sent_notifications[channel]) < count:
            # Record this send
            self.sent_notifications[channel].append(datetime.now().timestamp())
            return True
            
        return False
    
    def _can_send_email(self) -> bool:
        """Check if email sending is properly configured."""
        return all([
            self.email_sender,
            self.email_recipients,
            self.email_smtp_server,
            self.email_username,
            self.email_password
        ])
    
    def _can_send_sms(self) -> bool:
        """Check if SMS sending is properly configured."""
        return all([
            self.sms_provider,
            self.sms_api_key,
            self.sms_phone_numbers
        ])
    
    async def _send_email(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send a notification via email."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = ', '.join(self.email_recipients)
            msg['Subject'] = f"[{notification['priority'].upper()}] {notification['title']}"
            
            # Create HTML body
            html = f"""
            <html>
            <body>
                <h2>{notification['title']}</h2>
                <p><strong>Priority:</strong> {notification['priority']}</p>
                <p><strong>Category:</strong> {notification['category']}</p>
                <p><strong>Time:</strong> {notification['timestamp']}</p>
                <p><strong>Message:</strong></p>
                <p>{notification['message']}</p>
            """
            
            # Add data if present
            if notification['data']:
                html += "<h3>Additional Data:</h3><ul>"
                for key, value in notification['data'].items():
                    html += f"<li><strong>{key}:</strong> {value}</li>"
                html += "</ul>"
                
            html += "</body></html>"
            
            msg.attach(MIMEText(html, 'html'))
            
            # Connect to server and send
            server = smtplib.SMTP(self.email_smtp_server, self.email_smtp_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent: {notification['title']}")
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _send_slack(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send a notification to Slack."""
        try:
            # Build attachment fields for additional data
            fields = []
            for key, value in notification.get('data', {}).items():
                fields.append({
                    "title": key,
                    "value": str(value),
                    "short": True
                })
            
            # Set color based on priority
            colors = {
                "low": "#36a64f",  # green
                "medium": "#ecb613",  # yellow
                "high": "#ff9000",  # orange
                "critical": "#dc3545"  # red
            }
            color = colors.get(notification['priority'], "#36a64f")
            
            # Build payload
            payload = {
                "channel": self.slack_channel,
                "text": f"*{notification['title']}*",
                "attachments": [
                    {
                        "color": color,
                        "title": notification['category'].upper(),
                        "text": notification['message'],
                        "fields": fields,
                        "footer": f"Priority: {notification['priority'].upper()} | {notification['timestamp']}"
                    }
                ]
            }
            
            # Send the request
            response = await self._async_post_request(
                self.slack_webhook_url, 
                json.dumps(payload)
            )
            
            if response.status_code == 200:
                logger.info(f"Slack notification sent: {notification['title']}")
                return {"success": True}
            else:
                error = f"Error sending Slack notification: {response.status_code} - {response.text}"
                logger.error(error)
                return {"success": False, "error": error}
                
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _send_telegram(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send a notification to Telegram."""
        try:
            # Format the message
            message = f"*{notification['title']}*\n\n"
            message += f"*Priority:* {notification['priority'].upper()}\n"
            message += f"*Category:* {notification['category'].upper()}\n\n"
            message += f"{notification['message']}\n\n"
            
            # Add data if present
            if notification['data']:
                message += "*Additional Data:*\n"
                for key, value in notification['data'].items():
                    message += f"• {key}: {value}\n"
            
            # Add timestamp
            message += f"\n_Sent at {notification['timestamp']}_"
            
            # Build API URL
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            # Build payload
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            # Send the request
            response = await self._async_post_request(url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Telegram notification sent: {notification['title']}")
                return {"success": True}
            else:
                error = f"Error sending Telegram notification: {response.status_code} - {response.text}"
                logger.error(error)
                return {"success": False, "error": error}
                
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _send_discord(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send a notification to Discord."""
        try:
            # Set color based on priority
            colors = {
                "low": 3066993,  # green
                "medium": 16776960,  # yellow
                "high": 16744192,  # orange
                "critical": 15158332  # red
            }
            color = colors.get(notification['priority'], 3066993)
            
            # Build fields for additional data
            fields = []
            for key, value in notification.get('data', {}).items():
                fields.append({
                    "name": key,
                    "value": str(value),
                    "inline": True
                })
            
            # Build payload
            payload = {
                "embeds": [
                    {
                        "title": notification['title'],
                        "description": notification['message'],
                        "color": color,
                        "fields": [
                            {
                                "name": "Category",
                                "value": notification['category'].upper(),
                                "inline": True
                            },
                            {
                                "name": "Priority",
                                "value": notification['priority'].upper(),
                                "inline": True
                            },
                            *fields
                        ],
                        "footer": {
                            "text": f"Sent at {notification['timestamp']}"
                        }
                    }
                ]
            }
            
            # Send the request
            response = await self._async_post_request(
                self.discord_webhook_url, 
                json.dumps(payload)
            )
            
            if response.status_code == 204:  # Discord returns 204 No Content on success
                logger.info(f"Discord notification sent: {notification['title']}")
                return {"success": True}
            else:
                error = f"Error sending Discord notification: {response.status_code} - {response.text}"
                logger.error(error)
                return {"success": False, "error": error}
                
        except Exception as e:
            logger.error(f"Error sending Discord notification: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _send_sms(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send a notification via SMS."""
        # Simplified implementation for now
        try:
            # Format message (keep it short for SMS)
            message = f"{notification['title']}: {notification['message']}"
            
            # Log instead of actual implementation
            logger.info(f"SMS notification would be sent to {len(self.sms_phone_numbers)} numbers: {message}")
            
            # For real implementation, we would integrate with Twilio, AWS SNS, or other SMS providers
            # This is a placeholder
            
            return {"success": True, "note": "SMS sending is implemented as a placeholder"}
                
        except Exception as e:
            logger.error(f"Error sending SMS notification: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _send_console(self, notification: Dict[str, Any]):
        """Send a notification to the console/logs."""
        priority_prefixes = {
            "low": "[INFO]",
            "medium": "[NOTICE]",
            "high": "[WARNING]",
            "critical": "[ALERT]"
        }
        prefix = priority_prefixes.get(notification['priority'], "[INFO]")
        
        logger.info(f"{prefix} {notification['category'].upper()} - {notification['title']}: {notification['message']}")
        
        # Log additional data if present
        if notification['data']:
            data_str = json.dumps(notification['data'], indent=2)
            logger.info(f"Additional data: {data_str}")
    
    async def _async_post_request(self, url, data=None, json=None):
        """Make an async POST request using aiohttp."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, json=json) as response:
                # Create a response-like object with status_code, text, etc.
                class ResponseLike:
                    def __init__(self, status, text):
                        self.status_code = status
                        self.text = text
                
                return ResponseLike(response.status, await response.text())

    async def notify_trade_signal(self, symbol: str, action: str, price: float, 
                             confidence: float, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send a notification about a new trade signal.
        
        Args:
            symbol: Trading pair symbol
            action: Trade action (buy/sell)
            price: Current price
            confidence: Signal confidence score
            data: Additional data
            
        Returns:
            Notification result
        """
        # Determine priority based on confidence
        priority = NotificationPriority.LOW
        if confidence >= 0.8:
            priority = NotificationPriority.HIGH
        elif confidence >= 0.6:
            priority = NotificationPriority.MEDIUM
            
        title = f"New {action.upper()} Signal for {symbol}"
        message = f"A new {action} signal was generated for {symbol} at price ${price:.2f} with {confidence:.2%} confidence."
        
        # Add data to include with notification
        notification_data = {
            "symbol": symbol,
            "action": action,
            "price": f"${price:.2f}",
            "confidence": f"{confidence:.2%}"
        }
        
        # Add any additional data
        if data:
            notification_data.update(data)
            
        return await self.send_notification(
            message=message,
            title=title,
            category=NotificationCategory.TRADE_SIGNAL,
            priority=priority,
            data=notification_data
        )
    
    async def notify_trade_execution(self, symbol: str, side: str, amount: float, 
                                price: float, order_id: str = None,
                                data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send a notification about a trade execution.
        
        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            amount: Order amount
            price: Execution price
            order_id: Order ID if available
            data: Additional data
            
        Returns:
            Notification result
        """
        title = f"Trade Executed: {side.upper()} {symbol}"
        message = f"Executed {side} order for {amount} {symbol} at ${price:.2f}"
        if order_id:
            message += f" (Order ID: {order_id})"
            
        # Add data to include with notification
        notification_data = {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": f"${price:.2f}",
            "total_value": f"${amount * price:.2f}"
        }
        
        if order_id:
            notification_data["order_id"] = order_id
            
        # Add any additional data
        if data:
            notification_data.update(data)
            
        return await self.send_notification(
            message=message,
            title=title,
            category=NotificationCategory.TRADE_EXECUTION,
            priority=NotificationPriority.MEDIUM,
            data=notification_data
        )
    
    async def notify_error(self, error_message: str, source: str = None, 
                     severity: str = "medium", data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send a notification about an error.
        
        Args:
            error_message: The error message
            source: Source of the error
            severity: Error severity (low/medium/high/critical)
            data: Additional data
            
        Returns:
            Notification result
        """
        # Map severity to priority
        priority_map = {
            "low": NotificationPriority.LOW,
            "medium": NotificationPriority.MEDIUM,
            "high": NotificationPriority.HIGH,
            "critical": NotificationPriority.CRITICAL
        }
        priority = priority_map.get(severity, NotificationPriority.MEDIUM)
        
        title = f"Error: {source or 'System'}"
        message = error_message
        
        # Add data to include with notification
        notification_data = {
            "error": error_message,
            "source": source or "System",
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add any additional data
        if data:
            notification_data.update(data)
            
        return await self.send_notification(
            message=message,
            title=title,
            category=NotificationCategory.ERROR,
            priority=priority,
            data=notification_data
        )
    
    async def notify_performance_update(self, total_trades: int, win_rate: float, 
                                  profit_loss: float, period: str = "daily",
                                  data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send a notification with performance updates.
        
        Args:
            total_trades: Total number of trades in the period
            win_rate: Win rate as a decimal (0.65 = 65%)
            profit_loss: Profit/loss amount
            period: Time period (daily/weekly/monthly)
            data: Additional data
            
        Returns:
            Notification result
        """
        # Determine priority based on performance
        priority = NotificationPriority.MEDIUM
        if profit_loss < 0 and abs(profit_loss) > 1000:
            priority = NotificationPriority.HIGH
        elif profit_loss > 1000:
            priority = NotificationPriority.HIGH
            
        title = f"{period.capitalize()} Performance Update"
        
        # Format the message based on profit/loss
        if profit_loss >= 0:
            message = f"Positive performance for {period} period: ${profit_loss:.2f} profit with {win_rate:.2%} win rate across {total_trades} trades."
        else:
            message = f"Negative performance for {period} period: ${abs(profit_loss):.2f} loss with {win_rate:.2%} win rate across {total_trades} trades."
        
        # Add data to include with notification
        notification_data = {
            "period": period,
            "total_trades": total_trades,
            "win_rate": f"{win_rate:.2%}",
            "profit_loss": f"${profit_loss:.2f}",
            "average_per_trade": f"${profit_loss / total_trades:.2f}" if total_trades > 0 else "$0.00"
        }
        
        # Add any additional data
        if data:
            notification_data.update(data)
            
        return await self.send_notification(
            message=message,
            title=title,
            category=NotificationCategory.PERFORMANCE,
            priority=priority,
            data=notification_data
        ) 