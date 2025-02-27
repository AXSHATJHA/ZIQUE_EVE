import os
import pandas as pd
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from typing import Literal, Optional, List, Dict
from typing_extensions import TypedDict
from groq import Groq
from langgraph.graph import END, StateGraph, START # type: ignore
import random
from langgraph.checkpoint.memory import MemorySaver # type: ignore
from google import generativeai as genai
from dotenv import load_dotenv
from openai import OpenAI # type: ignore
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import re

df = pd.read_csv('Eve_Menu.csv')  # Changed filename
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Create the model
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}


class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "general_chatbot"] = Field(
        ...,
        description="Router for food recommendations using Eve's Macros Sheet"
    )


llm = ChatGroq(model="llama3-8b-8192", api_key=os.environ["GROQ_API_KEY"])
structured_llm_router = llm.with_structured_output(RouteQuery)
system = """
      You are an expert dish recommendation router. Analyze user questions and route them to the most appropriate resource:
        1. **Food Menu Database (Vectorstore):**
            - Questions about specific dishes, ingredients, or dietary options
            - Dish Categories:
                * Cold (Salads, Ceviche)
                * Hot (Soups, Grilled Dishes)
                * Desserts (Cakes, Ice Creams)
                * Beverages (Juices, Smoothies)
            - Filters available:
                * Diet Type (Vegetarian, Vegan, Keto, Gluten-Free)
                * Cuisine Type (European, Japanese, Fusion, etc.)
                * Meal Type (Breakfast, Lunch, Dinner)
            - Examples:
                - "Show me vegetarian salads"
                - "Dishes with avocado"
                - "Gluten-free desserts"
                - "Suggest a healthy sushi"
                - "more", "more dishes"
                - "Suggest healthy options"
                - "What dishes do you offer?"
                - "Show me vegetarian dishes under 300 calories"
                - "High-protein keto options"
                - "Gluten-free vegan meals"
        2. **general_chatbot (Direct Response):**
            - Greetings, help with menu navigation, or general inquiries
            - Examples:
                - "Hi"
                - "Hello"
                - "Namaste"
                - "Hi, my name is (name)"
                - "What is my name?"
                - "How do I use the chatbot?"
                - "Which meals are dairy-free?"
                - "What was the previous meal?"
                - "calories in previous meal"
                - "about previous meals"
        Always respond in JSON format:
        {{
          "datasource": "vectorstore|general_chatbot",
          "reasoning": "Brief explanation of routing decision"
        }}
        - Special Considerations:
        - Prioritize vocational dishes (e.g., Cooking techniques, Regional Cuisines)
        - Suggest meal options based on dietary restrictions (e.g., Diabetes-friendly, High-protein)
        - Handle multi-language queries if dish descriptions are available in Hindi/English
        - Consider age restrictions (e.g., Alcoholic beverages for 18+)
        }}
      """

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
question_router = route_prompt | structured_llm_router

api_wrapper = WikipediaAPIWrapper(top_k=3, doc_content_chars_max=1000)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = Field(default_factory=lambda: f"thread_{random.randint(1000, 9999)}")
    chat_history: List[Dict[str, str]] = []


class GraphState(TypedDict):
    question: str
    generation: str
    docs: List[str]
    chat_history: List[Dict[str, str]]  # Add chat history


def route_question(state):
    print("---Route Question----")
    question = state["question"]

    # Check if the question is a general query (e.g., greetings, casual questions)

    # Otherwise, route to database or Wikipedia search
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION To Wiki Search---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION To Vectorstore---")
        return "vectorstore"

    elif source.datasource == "general_chatbot":
        print("---ROUTE QUESTION To General Chatbot---")
        return "general_chatbot"


workflow = StateGraph(GraphState)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "vectorstore": END,
        "general_chatbot": END
    },
)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

api = FastAPI()

# Configure CORS
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])


@api.post("/chat", response_model=Dict)
async def chat_endpoint(request: ChatRequest):

    try:
        question = request.question
        chat_history = request.chat_history
        thread_id = request.thread_id

        # Route the question
        source = question_router.invoke({"question": question})
        print(f"Routing decision: {source.datasource}")

        if source.datasource == "general_chatbot":
            print("---General Chatbot----")

            # Define the enhanced system prompt
            system_prompt = """
            YOUR IDENTITY:
            - Name: Zico (always mention if greeted)
            - Role: Hybrid Assistant (70% dish expert / 30% conversationalist)
            
            RESPONSE PROTOCOL:
            1. PRIORITY QUERIES (Handle these first):
            a) Calorie Inquiries:
            - Pattern: "calories in last dish" / "previous meal calories" / "nutritional info"
            - Action: 
            1. Scan chat history for last recommended dish
            2. If found: "The Dish Name had Calories kcal"
            3. Add context: Compare to average (avg_cal kcal typical)
            4. Not found: "Let me suggest something new!"

            b) Greetings:
            - Patterns: "Hi" / "Hello" / "Hey Zico" / Time-based (Good morning)
            - Response: 
            1. "Hello! [Time-based emoji] Zico here" 
            2. Add food pun: "Ready to dish out recommendations!"

            2. SECONDARY RESPONSES:
            - Food-related questions: "What's tahini?" → Brief answer + recipe connection
            - Personal: "Who made you?" → "I'm Chef Eve's digital sous-chef!"
            
            3. BOUNDARY CASES:
            - Non-food queries: "Python code help" → "While I cook ideas, I only serve dish recommendations!"
            - Multiple intents: "Hi, calories last dish?" → Handle both greeting + calorie check
            """

            # Analyze chat history for previous dishes
            dish_history = [msg for msg in chat_history if "kcal" in msg.get("content", "")]
            last_dish_info = dish_history[-1]["content"] if dish_history else None

            # Prepare conversation context
            messages = [
                {"role": "system", "content": system_prompt},
                *chat_history[-5:],  # Last 5 messages
                {"role": "user", "content": question}
            ]

            # Add nutritional context if available
            if last_dish_info:
                messages.insert(1, {
                    "role": "system", 
                    "content": f"LAST DISH CONTEXT: {last_dish_info}"
                })

            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3,
                max_tokens=100,
                top_p=0.95
            )

            chatbot_response = response.choices[0].message.content

            # Post-processing
            if "kcal" in chatbot_response and last_dish_info:
                # Ensure numerical accuracy from history
                calories_match = re.search(r"(\d+)kcal", last_dish_info)
                if calories_match:
                    calories = calories_match.group(1)
                    chatbot_response = chatbot_response.replace("Calories", calories)

            # Update chat history (keep last 10 exchanges)
            chat_history = chat_history[-8:] + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": chatbot_response}
            ]

            return {"response": chatbot_response, "chat_history": chat_history}


        elif source.datasource == "vectorstore":
            print("---Handling Vectorstore Query---")

            # Load Eve's dataset
            df = pd.read_csv("Eve_Menu.csv")

            user_prefs = {
                'diet': set(),
                'allergens': set(),
                'cuisine': set(),
                'dislikes': set(),
                'staple': set()
            }

            # Extract user palate from chat history
            for msg in chat_history:
                if msg["role"] == "user":
                    content = msg["content"]

                    # Check if the message contains "user palate"
                    if "user palate" in content.lower():
                        try:
                            # Extract the JSON-like palate information
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

                        except Exception as e:
                            print(f"Error parsing user palate: {e}")

            print(f"""
                Dietary Identity: {', '.join(user_prefs['diet']) or 'None specified'}
                Absolute Exclusions: {', '.join(user_prefs['allergens']) or 'None'}
                Preferred Cuisines: {', '.join(user_prefs['cuisine']) or 'Open to all'}
                Dislikes: {', '.join(user_prefs['dislikes']) or 'None'}
                Staples: {', '.join(user_prefs['staple']) or 'None'}
                """
            )

            # Convert dietary columns to flags
            diet_columns = ['Vegetarian', 'Non-vegetarian', 'Jain', 'Vegan', 'Gluten-free', 'Keto']

            # Create dataset string
            dataset_str = "Dish Name | Calories | Protein | Carbs | Fats | Diets | Flavor | Dish Category\n"
            dataset_str += "------------------------------------------------------------------------------------------------\n"
            for _, row in df.iterrows():
                diets = [col for col in diet_columns if row[col] == 'Yes']
                dataset_str += (
                    f"{row['Dish_Name']} | {row['Total_Calories_(kcal)']}kcal | "
                    f"{row['Total_Protein_(g)']}g | {row['Total_Carbs_(g)']}g | "
                    f"{row['Total_Fats_(g)']}g | {', '.join(diets)} | "
                    f"{row['Flavor_Profile']} | {row['Dish_Category']}\n"
                )

            # Extract previously recommended dishes from assistant messages
            assistant_messages = [msg for msg in chat_history if msg["role"] == "assistant"]
            recommended_dishes = set()

            for msg in assistant_messages:
                # Extract dish names from assistant messages
                dish_match = re.search(r"\*\*([^\n]+)\*\*", msg["content"])
                if dish_match:
                    recommended_dishes.add(dish_match.group(1).strip())

            print(f"Previously recommended dishes: {recommended_dishes}")

            # Filter out previously recommended dishes
            filtered_dishes = df[~df['Dish_Name'].isin(recommended_dishes)]

            # Construct prompt with user preferences
            prompt = f"""
                ROLE: You are Zico, Eve's AI culinary assistant. Synthesize menu data, chat history, and user preferences to create perfect dish matches.

                CONTEXTUAL ELEMENTS:
                === DATASET ===
                {dataset_str}

                === CHRONOLOGICAL HISTORY ===
                {chat_history[-5:]}

                === DISHES RECOMMENDED ===
                {assistant_messages}

                === USER PALATE PROFILE ===
                Dietary Identity: {', '.join(user_prefs['diet']) or 'None specified'}
                Absolute Exclusions: {', '.join(user_prefs['allergens']) or 'None'}
                Preferred Cuisines: {', '.join(user_prefs['cuisine']) or 'Open to all'}
                Dislikes: {', '.join(user_prefs['dislikes']) or 'None'}
                Staples: {', '.join(user_prefs['staple']) or 'None'}

                === CURRENT REQUEST ===
                "{question}"

                CONTEXTUAL PRIORITIZATION:

                ALWAYS TRY TO SUGGEST DIFFERENT DISHES, GO TO CORNERS OF THE DATASET TO GET THE DESIRED DISH BUT ALWAYS MATCH THE PALATE AND QUESTION.

                STRICTLY FOLLOW -> IF THE Dietary Identity IS Non-Vegetarian Find the Chicken Dish and Suggest THAT.
                STRICTLY FOLLOW -> IF THE DIETARY IDENTITY IS VEGETARIAN NEVER SUGGEST SOMETHING THAT HAS CHICKEN!!(ALWAYS FOLLOW THIS)
                IF THE CURRENT QUESTION IS YES OR NO LOOK AT THE LAST ANSWER AND SUGGEST DISH.
                FOR EXAMPLE IF THE LAST ANSWER IS : 
                "role": "assistant",
                "content": "Looking for a cheesy delight? Here you go :)\n\n**Burrata Mozzarella with Basil Pesto Pizza**\nCompliant\nNo\nFlavors\nFeatures\n\n600kcal • 20g protein • 40g carbs\nWould you like to pair this pizza with a dessert or another dish for a delightful meal combination?"
                Now the user says 'Yes'. So Suggest a desert from the dataset.
                1. **Current Intent**: Treat the user's latest question as the primary driver. If it contains explicit modifiers (e.g., "vegetarian version of the last dish"), override previous filters.
                2. **Conversational Thread**: Identify patterns in chat history:
                - Repeated flavor mentions → Prioritize those flavors even if not in current query
                - Sequential requests (e.g., "more options" → same category, varied proteins)
                - Implicit preferences (e.g., if 3/5 last dishes were salads → favor light options)
                3. **Temporal Weighting**: Recent messages (last 2-3 exchanges) have 2x impact vs older history.
                -> IF THE USER HAS ASKED FOR A DISH CATEGORY THAT DOES NOT EXIST LIKE A DRINK DO NOT SUGGEST ANY DISH.....
                4. **Pairing Suggestions**: Suggest a dish that pairs well with the current recommendation (e.g., pizza with garlic bread).

                RESPONSE PROTOCOL:
                ->*IF THE USER HAS ASKED FOR MORE OR OTHER SUGGESTIONS, RECOMMEND SOMETHING THAT HAS NOT BEEN RECOMMENDED IN {recommended_dishes} DISHES THAT ARE RECOMMENDED. PLEASE MAKE SURE OF THAT!!
                
                1. **Opening Context**: 
                - Acknowledge previous dish if relevant ("Building on your sushi choice...")
                - Explicitly state *why* the recommendation fits the *current* ask
                2. **Proactive Anticipation**:
                - If history shows repeated "more" requests → Offer 2 alternatives upfront
                - For dietary shifts (e.g., vegan → vegetarian), explain compatibility
                3. **Tone Enforcement**:
                - Use contractions ("you'll love") and food emotiveness ("velvety sauce")
                - Never list numbers → "protein-packed" not "25g protein"
                - For "compare" requests → Use relative terms ("lighter than your last pick") 

                DYNAMIC FILTER ADJUSTMENT:
                - If the current query contradicts previous preferences (e.g., "Ignore my keto rule today"), temporarily disable conflicting filters
                - For vague requests ("Something new"), use history to infer:
                - Last dish category → Suggest adjacent cuisines
                - Average calorie range → Stay within ±15%        

                ANALYTICAL FRAMEWORK:
                1. PALATE-TRIGGERED FILTERS:
                - Hard Exclusions: Automatically reject dishes containing {user_prefs['allergens']}
                - Core Identity: Prioritize {user_prefs['diet']} compliant options
                - Cuisine Matching: Weight dishes from {user_prefs['cuisine']} higher
                - Dislikes: Exclude dishes containing {user_prefs['dislikes']}

                2. NUTRITIONAL INTERPRETATION:
                Language Patterns → Nutritional Logic:
                - "Light"/"Healthy" → <400kcal & <15g fat
                - "Hearty"/"Filling" → >500kcal & >25g protein  
                - "Quick" → Breakfast/Brunch/Lunch categories
                - Comparison Requests: ±20% of last dish's values

                3. TEMPORAL CONTEXT HANDLING:
                - Morning (5-11AM): Boost breakfast items
                - Afternoon (11AM-5PM): Highlight lunch salads/sandwiches
                - Evening (5PM+): Feature dinner entrees
                - "More Like This": Maintain category + nutritional profile

                SELECTION ALGORITHM:
                1. First Pass: Remove allergen-containing dishes
                2. Second Pass: Filter by hard dietary requirements
                3. Third Pass: Score remaining dishes using:
                - 40% match to current request
                - 30% alignment with flavor preferences
                - 20% meal time appropriateness
                - 10% variety from previous suggestions

                FORMATTING TEMPLATE:

                IF DISH NOT FOUND -> NO DISHES MATCHED!

                    **Dish Name**
                    - Culinary Profile: Dish Category | Flavor Profile
                    - Nutrition Spotlight: (Calories)kcal • (Protein)g protein  (Carbs)g carbs (GET THE CARBS AS WELL FROM THE DATASET)
                    - Perfect Match Because: 
                    🎯 Combines your love for (user_flavor) with (diet_type) needs
                    ⏰ Ideal for (meal_type) with (key_characteristic)
                    🌟 Fresh alternative to (last_dish) ((improvement_metric))
                    Suggest a dish from the dataset that goes well with the current dish.

                    EXAMPLE IMPLEMENTATION:
                    **Spicy Tuna Sushi**
                    - Culinary Profile: Japanese Fusion | Spicy Umami
                    - Nutrition Spotlight: 360kcal • 22g protein  12g carbs
                    - Perfect Match Because:
                    🎯 Balances your spice preference with keto compliance
                    ⏰ Light yet satisfying lunch option
                    🌟 20% leaner than your last tempura choice
                    You can pair it with dumplings, or soups.

                    Looking for a vegetarian delight with a spicy kick? Here's a dish that excludes your allergen, mushrooms, and aligns with your love for cottage cheese.\n\n**Mexican Bean and Guacamole Burger**\nA dish that aligns with:\n- Vegetarian compliant\n- Free from mushrooms\n- Spicy flavors\n- Features cottage cheese\n\n**600kcal**•**25g protein**•**80g carbs**\n\nHow about pairing this burger with a fresh salad for a complete meal?

                PROHIBITED MENTIONS:
                - Allergen information
                - Technical diet terms
                - Calorie counting language
                - Previous dish shortcomings
                """

            # OpenAI API Call
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.0
            )

            initial_message = response.choices[0].message.content
            print(initial_message)

            # Refine with GPT-4
            refined_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """
                    Format responses EXACTLY as follows:

                    1. **Dish Name**: Extract the dish name from the initial message. If no dish name is found, respond with: "Kindly try another query, please."

                    2. **Attributes**:
                    - Acknowledge the user's dietary preferences (e.g., Non-Vegetarian, Italian).
                    - Highlight allergens excluded (e.g., Free from Milk, Eggs).
                    - Mention the cuisine and staple ingredients.

                    3. **Nutritional Information**:
                    - Ensure the calories, protein, and carbs are accurate and match the dataset.
                    - Format as: **[NUM]kcal** • **[NUM]g protein** • **[NUM]g carbs**.

                    4. **Follow-up Question**:
                    - Ask a concise, engaging follow-up question (e.g., "Would you like to pair this with a side dish or explore more options?").

                    Example:
                    **Grilled Salmon with Lemon Herb Sauce**
                    A dish that aligns with:
                    - Non-Vegetarian compliant
                    - Free from Milk, Eggs
                    - Italian flavors
                    - Features Salmon

                    **450kcal** • **30g protein** • **20g carbs**

                    Would you like to pair this with a side dish or explore more options?
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Current Question: {question}
                    Raw Suggestion: {initial_message}
                    Previous Prompts: {chat_history[-1]}

                    Analyze the suggestion and refine it according to these principles:
                    1. Extract the dish name from the raw suggestion. If no dish name is found, respond with: "Kindly try another query, please."
                    2. Ensure the nutritional information (calories, protein, carbs) is accurate and matches the dataset.
                    3. Format the response as shown in the example above.
                    4. Conclude with a concise, engaging follow-up question.
                    """
                }
            ],
            temperature=0.2,
            max_tokens=150
        )


            response_text = refined_response.choices[0].message.content

            response_text = response_text.replace(" **", "**") \
                                       .replace("** ", "**") \
                                       .replace(" g ", "g") \
                                       .replace(" kcal", "kcal")

            # Update chat history
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": response_text})
            chat_history = chat_history[-10:]

            return {
                "response": response_text,
                "chat_history": chat_history
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid datasource routing"
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Run the Flask app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)
