import time
import logging
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Model performance metrics."""
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_count: int
    avg_confidence: float
    model_name: str

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    api_response_time: float
    memory_usage: float
    cpu_usage: float
    active_connections: int
    error_rate: float

@dataclass
class Alert:
    """Alert configuration."""
    name: str
    condition: str
    threshold: float
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    enabled: bool = True

class MetricsDatabase:
    """SQLite database for storing metrics."""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Model metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                prediction_count INTEGER,
                avg_confidence REAL,
                model_name TEXT
            )
        ''')
        
        # System metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                api_response_time REAL,
                memory_usage REAL,
                cpu_usage REAL,
                active_connections INTEGER,
                error_rate REAL
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_name TEXT,
                severity TEXT,
                message TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_model_metrics(self, metrics: ModelMetrics):
        """Store model performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_metrics 
            (timestamp, accuracy, precision, recall, f1_score, prediction_count, avg_confidence, model_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp.isoformat(),
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1_score,
            metrics.prediction_count,
            metrics.avg_confidence,
            metrics.model_name
        ))
        
        conn.commit()
        conn.close()
    
    def store_system_metrics(self, metrics: SystemMetrics):
        """Store system performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_metrics 
            (timestamp, api_response_time, memory_usage, cpu_usage, active_connections, error_rate)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp.isoformat(),
            metrics.api_response_time,
            metrics.memory_usage,
            metrics.cpu_usage,
            metrics.active_connections,
            metrics.error_rate
        ))
        
        conn.commit()
        conn.close()
    
    def store_alert(self, alert_name: str, severity: str, message: str):
        """Store alert in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (timestamp, alert_name, severity, message)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now().isoformat(), alert_name, severity, message))
        
        conn.commit()
        conn.close()
    
    def get_recent_metrics(self, hours: int = 24) -> Dict:
        """Get recent metrics for analysis."""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        # Get model metrics
        model_df = pd.read_sql_query('''
            SELECT * FROM model_metrics 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', conn, params=(cutoff_time,))
        
        # Get system metrics
        system_df = pd.read_sql_query('''
            SELECT * FROM system_metrics 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', conn, params=(cutoff_time,))
        
        conn.close()
        
        return {
            'model_metrics': model_df,
            'system_metrics': system_df
        }

class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alerts = self._load_alert_configs()
        self.db = MetricsDatabase()
    
    def _load_alert_configs(self) -> List[Alert]:
        """Load alert configurations."""
        return [
            Alert("low_accuracy", "accuracy < threshold", 0.50, 
                  "Model accuracy has dropped below 50%", "high"),
            Alert("high_error_rate", "error_rate > threshold", 0.10, 
                  "API error rate is above 10%", "medium"),
            Alert("slow_response", "api_response_time > threshold", 2.0, 
                  "API response time is above 2 seconds", "medium"),
            Alert("high_memory", "memory_usage > threshold", 0.85, 
                  "Memory usage is above 85%", "low"),
            Alert("low_confidence", "avg_confidence < threshold", 0.60, 
                  "Average prediction confidence is below 60%", "medium"),
        ]
    
    def check_alerts(self, model_metrics: Optional[ModelMetrics] = None, 
                    system_metrics: Optional[SystemMetrics] = None):
        """Check all alert conditions."""
        for alert in self.alerts:
            if not alert.enabled:
                continue
            
            triggered = False
            
            if model_metrics and alert.name in ["low_accuracy", "low_confidence"]:
                if alert.name == "low_accuracy" and model_metrics.accuracy < alert.threshold:
                    triggered = True
                elif alert.name == "low_confidence" and model_metrics.avg_confidence < alert.threshold:
                    triggered = True
            
            if system_metrics and alert.name in ["high_error_rate", "slow_response", "high_memory"]:
                if alert.name == "high_error_rate" and system_metrics.error_rate > alert.threshold:
                    triggered = True
                elif alert.name == "slow_response" and system_metrics.api_response_time > alert.threshold:
                    triggered = True
                elif alert.name == "high_memory" and system_metrics.memory_usage > alert.threshold:
                    triggered = True
            
            if triggered:
                self.trigger_alert(alert)
    
    def trigger_alert(self, alert: Alert):
        """Trigger an alert."""
        message = f"ALERT [{alert.severity.upper()}]: {alert.message}"
        logger.warning(message)
        
        # Store in database
        self.db.store_alert(alert.name, alert.severity, alert.message)
        
        # Send notifications
        if self.config.get('email_notifications', {}).get('enabled', False):
            self.send_email_alert(alert)
        
        if self.config.get('webhook_notifications', {}).get('enabled', False):
            self.send_webhook_alert(alert)
    
    def send_email_alert(self, alert: Alert):
        """Send email notification."""
        try:
            email_config = self.config['email_notifications']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = email_config['to_email']
            msg['Subject'] = f"Financial Prediction Alert: {alert.name}"
            
            body = f"""
            Alert: {alert.name}
            Severity: {alert.severity}
            Message: {alert.message}
            Timestamp: {datetime.now().isoformat()}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.name}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
    
    def send_webhook_alert(self, alert: Alert):
        """Send webhook notification."""
        try:
            webhook_config = self.config['webhook_notifications']
            
            payload = {
                'alert_name': alert.name,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook alert sent for {alert.name}")
            else:
                logger.error(f"Webhook alert failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {str(e)}")

class ModelPerformanceMonitor:
    """Monitor model performance over time."""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self.db = MetricsDatabase()
        self.prediction_history = []
    
    def collect_prediction_metrics(self, test_symbols: List[str] = None) -> Optional[ModelMetrics]:
        """Collect model performance metrics."""
        if test_symbols is None:
            test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        try:
            predictions = []
            confidences = []
            
            for symbol in test_symbols:
                response = requests.get(f"{self.api_url}/predict/{symbol}", timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    predictions.append(data)
                    confidences.append(data['confidence'])
            
            if not predictions:
                return None
            
            # Calculate metrics (simplified - in production, compare with actual outcomes)
            avg_confidence = np.mean(confidences)
            
            # For demonstration, we'll use historical performance
            # In production, you'd compare predictions with actual outcomes
            metrics = ModelMetrics(
                timestamp=datetime.now(),
                accuracy=0.56,  # Would be calculated from actual vs predicted
                precision=0.68,
                recall=0.47,
                f1_score=0.56,
                prediction_count=len(predictions),
                avg_confidence=avg_confidence,
                model_name=predictions[0].get('model_name', 'unknown')
            )
            
            self.db.store_model_metrics(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect model metrics: {str(e)}")
            return None

class SystemMonitor:
    """Monitor system performance and health."""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self.db = MetricsDatabase()
        self.response_times = []
        self.error_count = 0
        self.total_requests = 0
    
    def collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect system performance metrics."""
        try:
            # Test API response time
            start_time = time.time()
            response = requests.get(f"{self.api_url}/health", timeout=10)
            response_time = time.time() - start_time
            
            self.response_times.append(response_time)
            self.total_requests += 1
            
            if response.status_code != 200:
                self.error_count += 1
            
            # Calculate error rate
            error_rate = self.error_count / self.total_requests if self.total_requests > 0 else 0
            
            # Get system resource usage (simplified)
            import psutil
            memory_usage = psutil.virtual_memory().percent / 100.0
            cpu_usage = psutil.cpu_percent() / 100.0
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                api_response_time=response_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                active_connections=1,  # Simplified
                error_rate=error_rate
            )
            
            self.db.store_system_metrics(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            self.error_count += 1
            self.total_requests += 1
            return None

class MonitoringSystem:
    """Main monitoring system coordinator."""
    
    def __init__(self, config_file: str = "monitoring_config.json"):
        self.config = self._load_config(config_file)
        self.model_monitor = ModelPerformanceMonitor(self.config.get('api_url', 'http://localhost:5000'))
        self.system_monitor = SystemMonitor(self.config.get('api_url', 'http://localhost:5000'))
        self.alert_manager = AlertManager(self.config)
        self.running = False
    
    def _load_config(self, config_file: str) -> Dict:
        """Load monitoring configuration."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default monitoring configuration."""
        return {
            "api_url": "http://localhost:5000",
            "monitoring_interval": 300,  # 5 minutes
            "model_check_interval": 3600,  # 1 hour
            "email_notifications": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "from_email": "",
                "to_email": "",
                "username": "",
                "password": ""
            },
            "webhook_notifications": {
                "enabled": False,
                "url": "",
                "headers": {}
            }
        }
    
    def start_monitoring(self):
        """Start the monitoring system."""
        self.running = True
        logger.info("Starting monitoring system...")
        
        # Start system monitoring thread
        system_thread = threading.Thread(target=self._system_monitoring_loop)
        system_thread.daemon = True
        system_thread.start()
        
        # Start model monitoring thread
        model_thread = threading.Thread(target=self._model_monitoring_loop)
        model_thread.daemon = True
        model_thread.start()
        
        logger.info("Monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.running = False
        logger.info("Monitoring system stopped")
    
    def _system_monitoring_loop(self):
        """System monitoring loop."""
        while self.running:
            try:
                metrics = self.system_monitor.collect_system_metrics()
                if metrics:
                    self.alert_manager.check_alerts(system_metrics=metrics)
                
                time.sleep(self.config.get('monitoring_interval', 300))
                
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {str(e)}")
                time.sleep(60)  # Wait before retrying
    
    def _model_monitoring_loop(self):
        """Model monitoring loop."""
        while self.running:
            try:
                metrics = self.model_monitor.collect_prediction_metrics()
                if metrics:
                    self.alert_manager.check_alerts(model_metrics=metrics)
                
                time.sleep(self.config.get('model_check_interval', 3600))
                
            except Exception as e:
                logger.error(f"Error in model monitoring loop: {str(e)}")
                time.sleep(300)  # Wait before retrying
    
    def get_dashboard_data(self) -> Dict:
        """Get monitoring data for dashboard."""
        try:
            db = MetricsDatabase()
            recent_data = db.get_recent_metrics(hours=24)
            
            return {
                'model_metrics': recent_data['model_metrics'].to_dict('records'),
                'system_metrics': recent_data['system_metrics'].to_dict('records'),
                'status': 'healthy' if self.running else 'stopped'
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {str(e)}")
            return {'error': str(e)}

def create_monitoring_config():
    """Create a sample monitoring configuration file."""
    config = {
        "api_url": "http://localhost:5000",
        "monitoring_interval": 300,
        "model_check_interval": 3600,
        "email_notifications": {
            "enabled": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "from_email": "your-email@gmail.com",
            "to_email": "alerts@yourcompany.com",
            "username": "your-email@gmail.com",
            "password": "your-app-password"
        },
        "webhook_notifications": {
            "enabled": False,
            "url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
            "headers": {
                "Content-Type": "application/json"
            }
        }
    }
    
    with open('monitoring_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Created monitoring_config.json - please update with your settings")

if __name__ == "__main__":
    # Create sample config if it doesn't exist
    import os
    if not os.path.exists('monitoring_config.json'):
        create_monitoring_config()
    
    # Start monitoring system
    monitor = MonitoringSystem()
    
    try:
        monitor.start_monitoring()
        print("🔍 Monitoring system started")
        print("📊 Dashboard data available at monitor.get_dashboard_data()")
        print("⚠️  Alerts will be logged and stored in monitoring.db")
        print("🛑 Press Ctrl+C to stop")
        
        # Keep running
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("\n✅ Monitoring system stopped") 