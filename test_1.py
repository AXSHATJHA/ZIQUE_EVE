import re

user_prefs = {
    'diet': set(),
    'allergens': set(),
    'cuisine': set(),
    'dislikes': set(),
    'staple': set()
}

# Sample input
content = "palate: Hi, I am Anubhav Mishra. Here is my user palate: diet: Non-Vegetarian allergies: Milk,Eggs cuisine: Italian staple: Chicken,Crab,Salmon dislikes: Mushrooms."

# Define regex patterns for extracting information
patterns = {
    "diet": r"diet:\s*([^\n]+)",
    "allergens": r"allergies:\s*([^\n]+)",
    "cuisine": r"cuisine:\s*([^\n]+)",
    "dislikes": r"dislikes:\s*([^\n]+)",
    "staple": r"staple:\s*([^\n]+)"
}

# Extract data using regex
for key, pattern in patterns.items():
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        values = match.group(1).strip().split(",")  # Split comma-separated values
        user_prefs[key].update(map(str.strip, values))  # Remove extra spaces

# Print extracted preferences
print(f"""
Dietary Identity: {', '.join(user_prefs['diet']) or 'None specified'}
Absolute Exclusions: {', '.join(user_prefs['allergens']) or 'None'}
Preferred Cuisines: {', '.join(user_prefs['cuisine']) or 'Open to all'}
Dislikes: {', '.join(user_prefs['dislikes']) or 'None'}
Staples: {', '.join(user_prefs['staple']) or 'None'}
""")
