"""
Advanced Analytics Module - Business Intelligence
Generates insights from attendance data using Pandas and creates interactive visualizations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class AttendanceAnalytics:
    """
    Advanced analytics for attendance data
    Provides heatmaps, trends, and business insights
    """
    
    def __init__(self, db_manager):
        """
        Initialize analytics engine
        
        Args:
            db_manager: Database manager instance for data retrieval
        """
        self.db_manager = db_manager
    
    def get_attendance_heatmap_data(self, days=30):
        """
        Generate attendance heatmap data (which hours are busiest)
        
        Returns:
            dict: Hourly attendance data for visualization
        """
        try:
            records = self.db_manager.get_attendance_records(days=days)
            
            if not records:
                return {}
            
            # Convert to DataFrame
            data = []
            for record in records:
                if record['check_in_time']:
                    try:
                        check_in = record['check_in_time']
                        if isinstance(check_in, str):
                            check_in = datetime.fromisoformat(check_in.replace('Z', '+00:00'))
                        
                        hour = check_in.hour
                        day_of_week = check_in.strftime('%A')
                        
                        data.append({
                            'hour': hour,
                            'day_of_week': day_of_week,
                            'count': 1
                        })
                    except Exception as e:
                        logger.warning(f"Error processing check-in time: {e}")
                        continue
            
            if not data:
                return {}
            
            df = pd.DataFrame(data)
            
            # Create pivot table for heatmap
            heatmap_data = df.pivot_table(
                index='day_of_week',
                columns='hour',
                values='count',
                aggfunc='sum',
                fill_value=0
            )
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex([d for d in day_order if d in heatmap_data.index])
            
            return {
                'heatmap': heatmap_data.to_dict(),
                'peak_hour': int(df['hour'].mode()[0]) if not df['hour'].mode().empty else 9,
                'peak_day': df['day_of_week'].mode()[0] if not df['day_of_week'].mode().empty else 'Monday'
            }
            
        except Exception as e:
            logger.error(f"Error generating heatmap data: {e}")
            return {}
    
    def get_late_arrival_analysis(self, threshold_hour=9, days=30):
        """
        Analyze late arrivals and identify patterns
        
        Args:
            threshold_hour: Consider arrival after this hour as late (default 9 AM)
            days: Number of days to analyze
            
        Returns:
            dict: Late arrival statistics and offenders
        """
        try:
            records = self.db_manager.get_attendance_records(days=days)
            
            late_arrivals = {}
            total_days = {}
            
            for record in records:
                employee_id = record['employee_id']
                
                if employee_id not in late_arrivals:
                    late_arrivals[employee_id] = 0
                    total_days[employee_id] = 0
                
                total_days[employee_id] += 1
                
                if record['check_in_time']:
                    try:
                        check_in = record['check_in_time']
                        if isinstance(check_in, str):
                            check_in = datetime.fromisoformat(check_in.replace('Z', '+00:00'))
                        
                        if check_in.hour >= threshold_hour and check_in.minute > 0:
                            late_arrivals[employee_id] += 1
                    except Exception as e:
                        logger.warning(f"Error processing check-in: {e}")
                        continue
            
            # Calculate late arrival percentage
            late_analysis = []
            for emp_id, late_count in late_arrivals.items():
                total = total_days.get(emp_id, 1)
                percentage = (late_count / total * 100) if total > 0 else 0
                
                # Get employee name
                try:
                    employees = self.db_manager.get_all_employees()
                    emp_name = next((e['name'] for e in employees if e['employee_id'] == emp_id), emp_id)
                except:
                    emp_name = emp_id
                
                if late_count > 0:  # Only include employees with late arrivals
                    late_analysis.append({
                        'employee_id': emp_id,
                        'name': emp_name,
                        'late_days': late_count,
                        'total_days': total,
                        'percentage': round(percentage, 2)
                    })
            
            # Sort by percentage descending
            late_analysis.sort(key=lambda x: x['percentage'], reverse=True)
            
            return {
                'late_arrivals': late_analysis[:10],  # Top 10 chronic late arrivals
                'total_late_incidents': sum(late_arrivals.values()),
                'affected_employees': len(late_arrivals),
                'average_late_percentage': round(np.mean([v/total_days.get(k, 1)*100 for k, v in late_arrivals.items()]) if late_arrivals else 0, 2)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing late arrivals: {e}")
            return {}
    
    def get_total_man_hours(self, days=30):
        """
        Calculate total man-hours worked (crucial for construction/payroll)
        
        Args:
            days: Number of days to analyze
            
        Returns:
            dict: Man-hours statistics by employee and department
        """
        try:
            records = self.db_manager.get_attendance_records(days=days)
            
            man_hours = {}
            department_hours = {}
            
            for record in records:
                employee_id = record['employee_id']
                
                if employee_id not in man_hours:
                    man_hours[employee_id] = 0
                
                # Calculate duration
                if record['check_in_time'] and record['check_out_time']:
                    try:
                        check_in = record['check_in_time']
                        check_out = record['check_out_time']
                        
                        if isinstance(check_in, str):
                            check_in = datetime.fromisoformat(check_in.replace('Z', '+00:00'))
                        if isinstance(check_out, str):
                            check_out = datetime.fromisoformat(check_out.replace('Z', '+00:00'))
                        
                        duration = (check_out - check_in).total_seconds() / 3600
                        man_hours[employee_id] += duration
                        
                    except Exception as e:
                        logger.warning(f"Error calculating duration: {e}")
                        continue
                elif record.get('work_hours'):
                    man_hours[employee_id] += record['work_hours']
            
            # Get employee details and group by department
            employees = self.db_manager.get_all_employees()
            emp_dict = {e['employee_id']: e for e in employees}
            
            for emp_id, hours in man_hours.items():
                emp = emp_dict.get(emp_id, {})
                dept = emp.get('department', 'Unknown')
                
                if dept not in department_hours:
                    department_hours[dept] = 0
                department_hours[dept] += hours
            
            # Create detailed breakdown
            man_hours_list = []
            for emp_id, hours in man_hours.items():
                emp = emp_dict.get(emp_id, {})
                man_hours_list.append({
                    'employee_id': emp_id,
                    'name': emp.get('name', emp_id),
                    'department': emp.get('department', 'Unknown'),
                    'hours_worked': round(hours, 2),
                    'days_worked': len([r for r in records if r['employee_id'] == emp_id and r['check_in_time']])
                })
            
            # Sort by hours worked
            man_hours_list.sort(key=lambda x: x['hours_worked'], reverse=True)
            
            return {
                'total_man_hours': round(sum(man_hours.values()), 2),
                'by_employee': man_hours_list,
                'by_department': {dept: round(hrs, 2) for dept, hrs in department_hours.items()},
                'average_hours_per_employee': round(np.mean(list(man_hours.values())) if man_hours else 0, 2),
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error calculating man-hours: {e}")
            return {}
    
    def get_daily_attendance_trend(self, days=30):
        """
        Get daily attendance trend for visualization
        
        Returns:
            dict: Daily present count and trend
        """
        try:
            analytics_data = self.db_manager.get_analytics_data(
                start_date=datetime.now().date() - timedelta(days=days),
                end_date=datetime.now().date()
            )
            
            daily_data = []
            if 'daily_trend' in analytics_data:
                for trend in analytics_data['daily_trend']:
                    daily_data.append({
                        'date': trend['date'],
                        'present': trend['present_count']
                    })
            
            return {
                'daily_trend': daily_data,
                'average_daily_presence': round(np.mean([d['present'] for d in daily_data]), 2) if daily_data else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting daily trend: {e}")
            return {}
    
    def get_department_analytics(self, days=30):
        """
        Department-wise attendance analysis
        
        Returns:
            dict: Department statistics and metrics
        """
        try:
            analytics_data = self.db_manager.get_analytics_data(
                start_date=datetime.now().date() - timedelta(days=days),
                end_date=datetime.now().date()
            )
            
            dept_stats = []
            if 'department_stats' in analytics_data:
                for stat in analytics_data['department_stats']:
                    dept_stats.append({
                        'department': stat['department'],
                        'present': stat['present_count'],
                        'total': stat['total_employees'],
                        'attendance_rate': round(stat['attendance_rate'], 2)
                    })
            
            return sorted(dept_stats, key=lambda x: x['attendance_rate'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting department analytics: {e}")
            return []
    
    def get_employee_consistency(self, days=30):
        """
        Identify most and least consistent employees
        
        Returns:
            dict: Consistency analysis
        """
        try:
            records = self.db_manager.get_attendance_records(days=days)
            
            attendance_count = Counter()
            for record in records:
                if record['check_in_time']:
                    attendance_count[record['employee_id']] += 1
            
            total_possible_days = (datetime.now().date() - (datetime.now().date() - timedelta(days=days))).days
            
            employees = self.db_manager.get_all_employees()
            emp_dict = {e['employee_id']: e for e in employees}
            
            consistency = []
            for emp_id, attendance_days in attendance_count.items():
                percentage = (attendance_days / total_possible_days) * 100 if total_possible_days > 0 else 0
                emp = emp_dict.get(emp_id, {})
                
                consistency.append({
                    'employee_id': emp_id,
                    'name': emp.get('name', emp_id),
                    'days_present': attendance_days,
                    'consistency_percentage': round(percentage, 2)
                })
            
            # Sort by consistency
            consistency.sort(key=lambda x: x['consistency_percentage'], reverse=True)
            
            return {
                'most_consistent': consistency[:5] if consistency else [],
                'least_consistent': consistency[-5:] if consistency else [],
                'overall_average': round(np.mean([c['consistency_percentage'] for c in consistency]), 2) if consistency else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing consistency: {e}")
            return {}
    
    def generate_executive_summary(self, days=30):
        """
        Generate comprehensive executive summary
        
        Returns:
            dict: Key metrics and insights
        """
        try:
            # Collect all analytics
            man_hours = self.get_total_man_hours(days)
            late_arrivals = self.get_late_arrival_analysis(days=days)
            daily_trend = self.get_daily_attendance_trend(days)
            departments = self.get_department_analytics(days)
            consistency = self.get_employee_consistency(days)
            
            return {
                'period_days': days,
                'generated_at': datetime.now().isoformat(),
                'key_metrics': {
                    'total_man_hours': man_hours.get('total_man_hours', 0),
                    'average_daily_attendance': daily_trend.get('average_daily_presence', 0),
                    'late_incidents': late_arrivals.get('total_late_incidents', 0),
                    'employees_affected': late_arrivals.get('affected_employees', 0)
                },
                'department_performance': departments,
                'employee_insights': {
                    'most_consistent': consistency.get('most_consistent', []),
                    'least_consistent': consistency.get('least_consistent', []),
                    'overall_consistency': consistency.get('overall_average', 0)
                },
                'man_hours_data': man_hours
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {}
