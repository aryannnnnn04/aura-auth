"""
MLflow Model Monitoring - MLOps Integration
Tracks model performance, confidence scores, and system health metrics
"""

import mlflow
from mlflow.tracking import MlflowClient
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ModelMonitoring:
    """
    MLflow-based monitoring for face recognition model performance
    Tracks confidence scores, detection rates, and system health
    """
    
    def __init__(self, experiment_name="face_recognition_attendance", tracking_uri="./mlruns"):
        """
        Initialize MLflow monitoring
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: Path to store MLflow data
        """
        try:
            mlflow.set_tracking_uri(tracking_uri)
            self.experiment_name = experiment_name
            
            # Create or get experiment
            try:
                self.experiment = mlflow.get_experiment_by_name(experiment_name)
                if self.experiment is None:
                    exp_id = mlflow.create_experiment(experiment_name)
                    self.experiment = mlflow.get_experiment(exp_id)
            except Exception as e:
                logger.warning(f"Could not get experiment: {e}")
            
            self.client = MlflowClient(tracking_uri)
            
            # Current run tracking
            self.current_run = None
            
            # Metrics accumulator
            self.metrics_buffer = {
                'detection_count': 0,
                'recognition_count': 0,
                'failed_detections': 0,
                'confidence_scores': [],
                'session_start': datetime.now()
            }
            
            logger.info(f"MLflow monitoring initialized for '{experiment_name}'")
            
        except Exception as e:
            logger.error(f"Error initializing MLflow: {e}")
    
    def start_session(self, session_name=None):
        """
        Start a new MLflow run for a recognition session
        
        Args:
            session_name: Optional name for the session
        """
        try:
            mlflow.set_experiment(self.experiment_name)
            
            if session_name is None:
                session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.current_run = mlflow.start_run(run_name=session_name)
            
            # Log initial parameters
            mlflow.log_param("session_start", datetime.now().isoformat())
            mlflow.log_param("model_type", "face_recognition")
            
            logger.info(f"MLflow session started: {session_name}")
            
        except Exception as e:
            logger.error(f"Error starting MLflow session: {e}")
    
    def log_detection(self, detection_result):
        """
        Log a face detection event
        
        Args:
            detection_result: {
                'detected': bool,
                'employee_id': str,
                'confidence': float,
                'timestamp': datetime,
                'frame_id': int
            }
        """
        try:
            self.metrics_buffer['detection_count'] += 1
            
            if detection_result.get('detected'):
                self.metrics_buffer['recognition_count'] += 1
                
                # Log confidence score
                if 'confidence' in detection_result:
                    confidence = detection_result['confidence']
                    self.metrics_buffer['confidence_scores'].append(confidence)
                    
                    # Log to MLflow periodically
                    if self.metrics_buffer['recognition_count'] % 10 == 0:
                        avg_confidence = sum(self.metrics_buffer['confidence_scores']) / len(self.metrics_buffer['confidence_scores'])
                        mlflow.log_metric(
                            "avg_confidence_score",
                            avg_confidence,
                            step=self.metrics_buffer['recognition_count']
                        )
            else:
                self.metrics_buffer['failed_detections'] += 1
            
            # Log detection rate every 50 detections
            if self.metrics_buffer['detection_count'] % 50 == 0:
                detection_rate = (self.metrics_buffer['recognition_count'] / self.metrics_buffer['detection_count']) * 100
                mlflow.log_metric(
                    "detection_rate_percent",
                    detection_rate,
                    step=self.metrics_buffer['detection_count']
                )
            
        except Exception as e:
            logger.error(f"Error logging detection: {e}")
    
    def log_model_health(self, health_metrics):
        """
        Log overall model and system health metrics
        
        Args:
            health_metrics: {
                'total_employees': int,
                'known_encodings': int,
                'avg_frame_processing_time': float (ms),
                'camera_fps': float,
                'system_uptime': int (seconds),
                'errors': int
            }
        """
        try:
            for metric_name, value in health_metrics.items():
                mlflow.log_metric(f"health_{metric_name}", value)
            
            logger.info("Model health metrics logged")
            
        except Exception as e:
            logger.error(f"Error logging health metrics: {e}")
    
    def log_performance_alert(self, alert_type, severity, message):
        """
        Log performance alerts for monitoring
        
        Args:
            alert_type: Type of alert (e.g., 'low_confidence', 'high_failure_rate')
            severity: 'INFO', 'WARNING', 'CRITICAL'
            message: Alert message
        """
        try:
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'type': alert_type,
                'severity': severity,
                'message': message
            }
            
            mlflow.log_dict(alert_data, f"alerts/{alert_type}_{datetime.now().timestamp()}.json")
            
            logger.warning(f"[{severity}] {alert_type}: {message}")
            
        except Exception as e:
            logger.error(f"Error logging alert: {e}")
    
    def check_system_health(self):
        """
        Check if system needs attention based on metrics
        
        Returns:
            dict: Health check results and alerts
        """
        try:
            alerts = []
            
            # Check detection rate
            if self.metrics_buffer['detection_count'] > 0:
                detection_rate = (self.metrics_buffer['recognition_count'] / self.metrics_buffer['detection_count']) * 100
                
                if detection_rate < 50:
                    alerts.append({
                        'type': 'low_detection_rate',
                        'severity': 'WARNING',
                        'message': f"Detection rate is low: {detection_rate:.1f}%. Check camera, lighting, or face database."
                    })
            
            # Check confidence scores
            if self.metrics_buffer['confidence_scores']:
                avg_confidence = sum(self.metrics_buffer['confidence_scores']) / len(self.metrics_buffer['confidence_scores'])
                
                if avg_confidence < 0.6:
                    alerts.append({
                        'type': 'low_confidence_scores',
                        'severity': 'WARNING',
                        'message': f"Average confidence is low: {avg_confidence:.2f}. Consider re-training or adjusting threshold."
                    })
            
            # Check failure rate
            if self.metrics_buffer['detection_count'] > 100:
                failure_rate = (self.metrics_buffer['failed_detections'] / self.metrics_buffer['detection_count']) * 100
                
                if failure_rate > 40:
                    alerts.append({
                        'type': 'high_failure_rate',
                        'severity': 'CRITICAL',
                        'message': f"High failure rate detected: {failure_rate:.1f}%. System may need debugging."
                    })
            
            # Log alerts
            for alert in alerts:
                self.log_performance_alert(alert['type'], alert['severity'], alert['message'])
            
            return {
                'health_status': 'HEALTHY' if not alerts else 'NEEDS_ATTENTION',
                'alerts': alerts,
                'metrics': {
                    'total_detections': self.metrics_buffer['detection_count'],
                    'successful_recognitions': self.metrics_buffer['recognition_count'],
                    'failed_detections': self.metrics_buffer['failed_detections'],
                    'detection_rate': (self.metrics_buffer['recognition_count'] / self.metrics_buffer['detection_count'] * 100) if self.metrics_buffer['detection_count'] > 0 else 0,
                    'avg_confidence': (sum(self.metrics_buffer['confidence_scores']) / len(self.metrics_buffer['confidence_scores'])) if self.metrics_buffer['confidence_scores'] else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {'health_status': 'ERROR', 'error': str(e)}
    
    def end_session(self):
        """End current MLflow run and save summary"""
        try:
            # Log final metrics
            health = self.check_system_health()
            mlflow.log_param("session_end", datetime.now().isoformat())
            mlflow.log_param("health_status", health['health_status'])
            
            # Log session summary
            summary = {
                'total_detections': self.metrics_buffer['detection_count'],
                'successful_recognitions': self.metrics_buffer['recognition_count'],
                'session_duration_seconds': (datetime.now() - self.metrics_buffer['session_start']).total_seconds()
            }
            mlflow.log_dict(summary, "session_summary.json")
            
            mlflow.end_run()
            logger.info("MLflow session ended and logged")
            
        except Exception as e:
            logger.error(f"Error ending MLflow session: {e}")
    
    def get_run_history(self, limit=10):
        """
        Get recent run history
        
        Args:
            limit: Number of recent runs to retrieve
            
        Returns:
            list: Recent run information
        """
        try:
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                max_results=limit,
                order_by=["start_time DESC"]
            )
            
            history = []
            for run in runs:
                history.append({
                    'run_id': run.info.run_id,
                    'run_name': run.info.run_name,
                    'start_time': datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                    'end_time': datetime.fromtimestamp(run.info.end_time / 1000).isoformat() if run.info.end_time else None,
                    'status': run.info.status
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting run history: {e}")
            return []
    
    def get_model_metrics(self):
        """
        Get aggregated model metrics
        
        Returns:
            dict: Summary of model performance
        """
        try:
            return {
                'total_detections': self.metrics_buffer['detection_count'],
                'recognition_count': self.metrics_buffer['recognition_count'],
                'failed_detections': self.metrics_buffer['failed_detections'],
                'detection_rate': (self.metrics_buffer['recognition_count'] / self.metrics_buffer['detection_count'] * 100) if self.metrics_buffer['detection_count'] > 0 else 0,
                'avg_confidence': (sum(self.metrics_buffer['confidence_scores']) / len(self.metrics_buffer['confidence_scores'])) if self.metrics_buffer['confidence_scores'] else 0,
                'min_confidence': min(self.metrics_buffer['confidence_scores']) if self.metrics_buffer['confidence_scores'] else 0,
                'max_confidence': max(self.metrics_buffer['confidence_scores']) if self.metrics_buffer['confidence_scores'] else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {e}")
            return {}
