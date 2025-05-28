import boto3
import csv
import json

PROMPT_CHECKS_TABLE = 'PromptChecks'
dynamodb = boto3.resource('dynamodb')

def list_repeats(threshold=0.85):
    table = dynamodb.Table(PROMPT_CHECKS_TABLE)
    response = table.scan()
    repeats = [item for item in response.get('Items', []) if item.get('result') == 0 and float(item.get('similarity_score', 0)) > threshold]
    return repeats

def export_analytics_csv(filename='analytics.csv'):
    from ..src.analytics import get_analytics
    analytics = get_analytics()
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for k, v in analytics.items():
            writer.writerow([k, v])

def export_analytics_json(filename='analytics.json'):
    from ..src.analytics import get_analytics
    analytics = get_analytics()
    with open(filename, 'w') as f:
        json.dump(analytics, f, indent=2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--csv', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    repeats = list_repeats(args.threshold)
    print(f"Repeats above threshold {args.threshold}:")
    for r in repeats:
        print(r)
    if args.csv:
        export_analytics_csv()
    if args.json:
        export_analytics_json() 