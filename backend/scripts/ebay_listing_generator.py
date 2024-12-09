import os
import sys
import csv
import json
import statistics
from ebaysdk.finding import Connection as Finding
from ebaysdk.exception import ConnectionError
from dotenv import load_dotenv
from rich.console import Console

# Initialize Rich console
console = Console()

# Load environment variables
load_dotenv()

# eBay API credentials
EBAY_APP_ID = os.getenv('EBAY_APP_ID')
EBAY_DEV_ID = os.getenv('EBAY_DEV_ID')
EBAY_CERT_ID = os.getenv('EBAY_CERT_ID')

def search_ebay(query):
    try:
        api = Finding(appid=EBAY_APP_ID, config_file=None)
        response = api.execute('findCompletedItems', {
            'keywords': query,
            'categoryId': '281',  # Example category ID for Jewelry
            'itemFilter': [
                {'name': 'SoldItemsOnly', 'value': True},
                {'name': 'Condition', 'value': 'New'},
            ],
            'paginationInput': {
                'entriesPerPage': 10,
                'pageNumber': 1
            }
        })
        items = response.dict().get('searchResult', {}).get('item', [])
        prices = [float(item['sellingStatus']['currentPrice']['value']) for item in items if 'sellingStatus' in item and 'currentPrice' in item['sellingStatus']]
        return prices
    except ConnectionError as e:
        console.log(f"[red]eBay API ConnectionError: {e.message}[/red]")
        return []

def generate_listing(product_id, title, description, tags, specifics, prices):
    # Calculate median price
    if prices:
        median_price = statistics.median(prices)
    else:
        median_price = 50.0  # Default price if no data

    listing = {
        'product_id': product_id,
        'title': title,
        'description': description,
        'tags': tags,
        'item_specifics': specifics,
        'price': f"{median_price:.2f}"
    }
    return listing

def generate_csv(augmented_dir, csv_path):
    fieldnames = ['product_id','title','description','tags','item_specifics','price']
    rows = []

    for product_folder in os.listdir(augmented_dir):
        product_path = os.path.join(augmented_dir, product_folder)
        if os.path.isdir(product_path):
            # Example: folder name as product_id
            product_id = product_folder
            title = f"Elegant {product_folder.replace('_', ' ').title()}"
            description = f"This elegant {product_folder.replace('_', ' ')} is perfect for any occasion."
            tags = product_folder.replace('_', ', ')
            specifics = f"Type:{product_folder.replace('_', ' ')},Material:Gold"  # Example specifics

            # Placeholder: Read labels from moondream_classification.py output if available
            # For simplicity, we'll use existing tags
            labels = tags.split(', ')

            # Search eBay for similar items to determine price
            ebay_prices = search_ebay(title)
            listing = generate_listing(product_folder, title, description, ', '.join(labels), specifics, ebay_prices)
            rows.append(listing)
            console.log(f"[green]Generated listing for {product_id}[/green]")

    # Write to CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        console.log("[red]Usage: python ebay_listing_generator.py <augmented_dir> <csv_path>[/red]")
        sys.exit(1)

    augmented_dir = sys.argv[1]
    csv_path = sys.argv[2]

    generate_csv(augmented_dir, csv_path)
    console.log("[green]eBay listing CSV generation complete.[/green]")
