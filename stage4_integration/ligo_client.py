"""Client for accessing LIGO/Virgo gravitational wave data."""

import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import time


class LIGOClient:
    """Client for accessing LIGO/Virgo GraceDB API."""
    
    def __init__(self, base_url: str = "https://gracedb.ligo.org/api/"):
        """
        Initialize LIGO client.
        
        Parameters:
        -----------
        base_url : str
            Base URL for GraceDB API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.last_check_time = None
        self.cached_events = []
    
    def get_recent_events(
        self,
        hours: int = 24,
        min_false_alarm_rate: float = 1e-6,
    ) -> List[Dict]:
        """
        Get recent gravitational wave events from LIGO/Virgo.
        
        Parameters:
        -----------
        hours : int
            Number of hours to look back
        min_false_alarm_rate : float
            Minimum false alarm rate threshold
            
        Returns:
        --------
        List[Dict]
            List of event dictionaries
        """
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Query GraceDB API
            url = f"{self.base_url}/superevents"
            params = {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'orderby': 'gpstime',
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Handle different response formats
            try:
                events = response.json()
                
                # Check if response is a list or dict
                if isinstance(events, dict):
                    # If it's a dict, try to extract events from common keys
                    events = events.get('results', events.get('events', events.get('data', [])))
                
                # Ensure events is a list
                if not isinstance(events, list):
                    events = []
                
                # Filter by false alarm rate
                filtered_events = [
                    event for event in events
                    if isinstance(event, dict) and event.get('far', float('inf')) <= min_false_alarm_rate
                ]
                
                return filtered_events
            except (ValueError, AttributeError, TypeError) as e:
                print(f"Error parsing LIGO API response: {e}")
                return []
            
        except Exception as e:
            print(f"Error fetching LIGO events: {e}")
            return []
    
    def get_event_details(self, event_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific event.
        
        Parameters:
        -----------
        event_id : str
            Event ID (e.g., "S123456")
            
        Returns:
        --------
        Optional[Dict]
            Event details or None if not found
        """
        try:
            url = f"{self.base_url}/superevents/{event_id}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            # Ensure result is a dict
            if isinstance(result, dict):
                return result
            else:
                print(f"Unexpected response format for event {event_id}")
                return None
        except Exception as e:
            print(f"Error fetching event details: {e}")
            return None
    
    def check_new_events(
        self,
        check_interval: int = 60,
        min_false_alarm_rate: float = 1e-6,
    ) -> List[Dict]:
        """
        Check for new events periodically.
        
        Parameters:
        -----------
        check_interval : int
            Interval between checks in seconds
        min_false_alarm_rate : float
            Minimum false alarm rate threshold
            
        Returns:
        --------
        List[Dict]
            List of new events since last check
        """
        current_time = time.time()
        
        # Check if enough time has passed
        if (self.last_check_time is None or
            current_time - self.last_check_time >= check_interval):
            
            # Get recent events
            events = self.get_recent_events(hours=24, min_false_alarm_rate=min_false_alarm_rate)
            
            # Find new events (not in cache)
            cached_ids = {
                e.get('graceid') for e in self.cached_events 
                if isinstance(e, dict) and e.get('graceid')
            }
            new_events = [
                e for e in events 
                if isinstance(e, dict) and e.get('graceid') not in cached_ids
            ]
            
            # Update cache
            self.cached_events = events
            self.last_check_time = current_time
            
            return new_events
        
        return []
    
    def cross_reference(
        self,
        detection_time: datetime,
        time_tolerance: timedelta = timedelta(seconds=10),
        min_false_alarm_rate: float = 1e-6,
    ) -> Optional[Dict]:
        """
        Cross-reference a detection time with LIGO/Virgo events.
        
        Parameters:
        -----------
        detection_time : datetime
            Time of local detection
        time_tolerance : timedelta
            Time window for matching
        min_false_alarm_rate : float
            Minimum false alarm rate threshold
            
        Returns:
        --------
        Optional[Dict]
            Matching LIGO/Virgo event or None
        """
        events = self.get_recent_events(hours=24, min_false_alarm_rate=min_false_alarm_rate)
        
        for event in events:
            # Ensure event is a dict
            if not isinstance(event, dict):
                continue
                
            # Parse event time
            event_time_str = event.get('gpstime') or event.get('gps_time') or event.get('time')
            if event_time_str:
                try:
                    # Convert GPS time to datetime (simplified)
                    if isinstance(event_time_str, (int, float)):
                        # GPS time might be a number (seconds since GPS epoch)
                        # GPS epoch is 1980-01-06 00:00:00 UTC
                        gps_epoch = datetime(1980, 1, 6)
                        event_time = gps_epoch + timedelta(seconds=float(event_time_str))
                    else:
                        # Try parsing as ISO format string
                        event_time_str = str(event_time_str).replace('Z', '+00:00')
                        event_time = datetime.fromisoformat(event_time_str)
                    
                    # Check if within tolerance
                    time_diff = abs((event_time - detection_time).total_seconds())
                    if time_diff <= time_tolerance.total_seconds():
                        return event
                except Exception as e:
                    # Silently skip events with unparseable times
                    continue
        
        return None
