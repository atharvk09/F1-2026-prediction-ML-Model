import pandas as pd
import fastf1
from src.config import fastf1 as f1_config

class F1DataLoader:
    def __init__(self, cache_dir='cache'):
        fastf1.Cache.enable_cache(cache_dir)
        self.cache_dir = cache_dir

    def load_csv_data(self):
        """Load the core project CSVs."""
        return {
            'hist_race': pd.read_csv('data/race_results_2022_2025.csv'),
            'hist_qual': pd.read_csv('data/qualifying_2022_2025.csv'),
            'race_2026': pd.read_csv('data/race_results_2026.csv'),
            'qual_2026': pd.read_csv('data/qualifying_2026.csv')
        }

    def fetch_session_results(self, year, event, session_type='R'):
        """Fetch a specific session from FastF1."""
        try:
            session = fastf1.get_session(year, event, session_type)
            session.load()
            return session.results
        except Exception as e:
            print(f"Error fetching session {year} {event}: {e}")
            return None
