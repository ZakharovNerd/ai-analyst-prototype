import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)


num_users = 150

user_ids = range(1, num_users + 1)
regions = ['Москва', 'Санкт-Петербург', 'Екатеринбург', 'Новосибирск', 'Казань']
registration_dates = [datetime(2024, 5, 1) + timedelta(days=random.randint(0, 59)) for _ in range(num_users)]
is_active = [random.choice([True, False]) for _ in range(num_users)]
last_login_dates = []
for i in range(num_users):
    if is_active[i]:
        last_login_dates.append(registration_dates[i] + timedelta(days=random.randint(0, 30)))
    else:
        last_login_dates.append(registration_dates[i] + timedelta(days=random.randint(0, 15)))

users_data = pd.DataFrame({
    'user_id': user_ids,
    'region': np.random.choice(regions, num_users),
    'registration_date': [d.strftime('%Y-%m-%d') for d in registration_dates],
    'is_active': is_active,
    'last_login_date': [d.strftime('%Y-%m-%d') for d in last_login_dates]
})

users_data.to_csv('users.csv', index=False)
print("Generated users.csv with", num_users, "rows.")


num_orders = 200

order_ids = range(1001, 1001 + num_orders)

order_user_ids = np.random.choice(user_ids, num_orders)
order_dates = [datetime(2024, 6, 1) + timedelta(days=random.randint(0, 29)) for _ in range(num_orders)]
order_amounts = np.random.randint(500, 15000, num_orders)
statuses = ['completed', 'pending', 'canceled']

orders_data = pd.DataFrame({
    'order_id': order_ids,
    'user_id': order_user_ids,
    'order_date': [d.strftime('%Y-%m-%d') for d in order_dates],
    'order_amount': order_amounts,
    'status': np.random.choice(statuses, num_orders, p=[0.7, 0.15, 0.15])
})

orders_data.to_csv('orders.csv', index=False)
print("Generated orders.csv with", num_orders, "rows.")
