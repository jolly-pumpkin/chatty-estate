import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import json
from langchain_openai import OpenAIEmbeddings

load_dotenv()


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_URL = os.getenv('OPENAI_URL')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
MODEL = "gpt-4o-mini",



print("------------Params------------")
print(OPENAI_API_KEY)
print(OPENAI_URL)
print(EMBEDDING_MODEL_NAME)
print("------------Params------------")


openai = OpenAI(
    base_url =OPENAI_URL,
    api_key = OPENAI_API_KEY,
)


def generate_real_estate(listing_count):

    prompt_template = """
        Generate the details for {} properties

        Here is an example of what a listing looks like:
        Neighborhood: Green Oaks
        Price: $800,000
        Bedrooms: 3
        Bathrooms: 2
        House Size: 2,000 sqft

        Description: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 
        2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, 
        highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. 
        Embrace sustainable living without compromising on style in this Green Oaks gem.

        Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. 
        Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.


        response should be in JSON format, all property names should be lower case. should be an array of objects
  
    """

    prompt = prompt = prompt_template.format(listing_count)


    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful real estate assistant. You write realistic real estate listings"},
            {"role": "user", "content": prompt}
        ],
    )   

    response_text = response.choices[0].message.content
    print(response_text)
    dictionary = json.loads(response_text)
    print("dict", dictionary)
    return dictionary


listings = generate_real_estate(2)
print("------------Listings------------")
print(listings)
print("------------Listings------------")




embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model=EMBEDDING_MODEL_NAME,
    openai_api_base=OPENAI_URL
)



def store_listings_in_vectorstore(listings):
    # Prepare texts and metadatas for Chroma
    texts = []
    metadatas = []
    
    for listing in listings:
        # Combine description and neighborhood description for the text to be embedded
        text = f"{listing['description']}\n{listing['neighborhood_description']}"
        texts.append(text)
        
        # Store all other fields as metadata
        metadata = {
            "neighborhood": listing["neighborhood"],
            "price": listing["price"],
            "bedrooms": listing["bedrooms"],
            "bathrooms": listing["bathrooms"],
            "house_size": listing["house_size"],
        }
        metadatas.append(metadata)
    
    # Create and return the vector store
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="./chroma_db"  # This will persist the database to disk
    )
    
    return vectorstore


def search_listings(vectorstore, query: str, k: int = 2):
    # Search for similar documents
    results = vectorstore.similarity_search(
        query,
        k=k,
    )
    return results


# Store listings in vector store
vectorstore = store_listings_in_vectorstore(listings)

# Example search
results = search_listings(vectorstore, "modern waterfront home with luxury features")
print("------------Search Results------------")
print(results)
print("------------End Search Results------------")


questions = [   
    "How big do you want your house to be?" 
    "What are 3 most important things for you in choosing this property?", 
    "How much do you want to spend?", 
    "Which transportation options are important to you?",
    "How urban do you want your neighborhood to be?",   
            ]
answers = [
    "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
    "A quiet neighborhood, good local schools, and convenient shopping options.",
    "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
    "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
    "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
]




