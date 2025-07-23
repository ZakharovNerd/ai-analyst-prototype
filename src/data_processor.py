import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.users_df = None
        self.orders_df = None
        self._load_data()
    
    def _load_data(self):
        try:
            self.users_df = pd.read_csv('data/users.csv')
            self.orders_df = pd.read_csv('data/orders.csv')
            
            self.users_df['registration_date'] = pd.to_datetime(self.users_df['registration_date'])
            self.users_df['last_login_date'] = pd.to_datetime(self.users_df['last_login_date'])
            self.orders_df['order_date'] = pd.to_datetime(self.orders_df['order_date'])
            
        except Exception:
            logger.exception("Failed to load data")
            raise
    
    def get_data_schema(self) -> str:
        return f"""
users_df columns: {list(self.users_df.columns)}
users_df dtypes: {dict(self.users_df.dtypes)}
users_df sample: {self.users_df.head(2).to_dict('records')}

orders_df columns: {list(self.orders_df.columns)}  
orders_df dtypes: {dict(self.orders_df.dtypes)}
orders_df sample: {self.orders_df.head(2).to_dict('records')}
        """
    
    def execute_pandas_query(self, code: str) -> Tuple[Any, str | None]:
        try:
            dangerous_patterns = ['import', '__', 'exec', 'eval', 'open', 'file', 'os', 'sys', 'subprocess']
            code_lower = code.lower()
            for pattern in dangerous_patterns:
                if pattern in code_lower:
                    return None, f"Dangerous operation detected: {pattern}"
            
            local_vars = {
                'users_df': self.users_df.copy(),
                'orders_df': self.orders_df.copy(),
                'pd': pd,
                'datetime': datetime,
                'len': len,
                'sum': sum,
                'min': min,
                'max': max,
                'round': round,
                'set': set
            }
            
            exec(code, {"__builtins__": {}}, local_vars)
            
            if 'result' in local_vars:
                return local_vars['result'], None
            else:
                excluded_vars = {'users_df', 'orders_df', 'pd', 'datetime', 'len', 'sum', 'min', 'max', 'round', 'set'}
                for var_name, var_value in local_vars.items():
                    if var_name not in excluded_vars:
                        if not var_name.startswith('_'):
                            return var_value, None
                
                return "Code executed successfully but no result found", None
                
        except Exception as e:
            return None, str(e)