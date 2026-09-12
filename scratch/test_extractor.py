import pandas as pd, re

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=50)

def extract_target_entity_and_filter(q, df, roles):
    q_l = q.lower()
    
    # 1. Target entity from question subject
    target_match = re.search(r'\b(?:which|what|who|name the|tell me the|top|best|worst|lowest)\s+([a-z\s]+?)(?:\s+(?:generated|generates|produced|produces|has|have|had|with|by|in|for|among|from|is|was|leads|ranked|ranks|got)\b|\?|$)', q_l)
    
    target_phrase = target_match.group(1).strip() if target_match else ''
    print(f'Question: {q}')
    print(f'Extracted target phrase: "{target_phrase}"')
    
    def _map_phrase(phrase):
        if re.search(r'\b(shipping\s*mode|ship\s*mode|delivery\s*mode|shipping\s*type)\b', phrase):
            return roles.get('shipping_mode') or 'Shipping Mode'
        if re.search(r'\b(customer\s*segment|client\s*segment|segment)\b', phrase):
            return roles.get('segment') or 'Customer Segment'
        if re.search(r'\b(product\s*name|product|item\s*name|item|sku|goods)\b', phrase):
            return roles.get('product') or 'Product Name'
        if re.search(r'\b(category\s*name|category|product\s*category)\b', phrase):
            return roles.get('category') or 'Category Name'
        if re.search(r'\b(department\s*name|department|dept)\b', phrase):
            return roles.get('department') or 'Department Name'
        if re.search(r'\b(order\s*status)\b', phrase):
            return roles.get('order_status') or 'Order Status'
        if re.search(r'\b(delivery\s*status)\b', phrase):
            return roles.get('delivery_status') or 'Delivery Status'
        if re.search(r'\b(customer|buyer|client|shopper)\b', phrase):
            return roles.get('customer') or 'Customer Fname'
        if re.search(r'\b(region|territory|zone)\b', phrase):
            return roles.get('region') or 'Order Region'
        if re.search(r'\b(country|nation)\b', phrase):
            return roles.get('country') or 'Order Country'
        if re.search(r'\b(state|province)\b', phrase):
            return roles.get('state') or 'Order State'
        if re.search(r'\b(city|town)\b', phrase):
            return roles.get('city') or 'Order City'
        return None

    group_col = _map_phrase(target_phrase)
    if not group_col:
        # Fallback to whole question search
        group_col = _map_phrase(q_l) or roles.get('product') or 'Product Name'
    
    # 2. Extract filter (e.g. 'Consumer segment', 'in Europe', etc.)
    filter_col = None
    filter_val = None
    
    # Check categorical columns for known values appearing in question
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
            if col != group_col:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 100:
                    for v in unique_vals:
                        v_str = str(v).strip().lower()
                        if len(v_str) > 2 and re.search(rf'\b{re.escape(v_str)}\b', q_l):
                            filter_col = col
                            filter_val = str(v).strip()
                            break
                if filter_col:
                    break

    print(f'Result -> group_col: {group_col}, filter: {filter_col} = {filter_val}')
    print()

roles = {
    'shipping_mode': 'Shipping Mode',
    'segment': 'Customer Segment',
    'product': 'Product Name',
    'category': 'Category Name',
    'department': 'Department Name',
    'customer': 'Customer Fname',
    'region': 'Order Region',
}

extract_target_entity_and_filter('Which customer segment generates the most sales?', df, roles)
extract_target_entity_and_filter('Which product generated the highest profit among orders from the Consumer segment?', df, roles)
extract_target_entity_and_filter('Which product generated the highest profit among orders?', df, roles)
extract_target_entity_and_filter('Which product name generated the highest profit among orders?', df, roles)
extract_target_entity_and_filter('Which shipping mode generates the highest average profit per order?', df, roles)
