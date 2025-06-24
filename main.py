from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

from vector import retriever  # Import the retriever from vector.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import random
import time

# Init LLM - using the community Ollama integration
model = Ollama(model="llama3.2:1b")

# Memory
memory = ConversationBufferMemory(return_messages=True)

# Prompt with context injection
template="""Tu es un assistant virtuel pour le site TuN. Tu ne réponds qu'aux questions qui concernent ce site, comme la création de compte, les appels d'offres, les signalements, etc. Si une question est hors sujet, dis-le poliment à l'utilisateur sans inventer une réponse.

Ton rôle est d'aider les utilisateurs à comprendre les services, processus, et informations liés à TuN.

Ne réponds jamais si la question ne concerne pas TuN, même si un contexte ou un historique existe. Dis-le poliment à l’utilisateur.

Encourage toujours l'utilisateur à poser d'autres questions en lien avec TuN.
Tu ne dois pas répondre aux questions qui ne concernent pas TuN, et tu dois le signaler à l'utilisateur de manière polie.

Voici le contexte extrait des documents :  
{context}

Voici l'historique de conversation à titre informatif :  
{history}

Question actuelle : {question}

Réponds uniquement à la question actuelle de manière claire et concise.
"""

prompt = PromptTemplate(
    input_variables=["context", "history", "question"],
    template=template,
)

# Détection de ton
sarcasm_keywords = ["super efficace", "génial", "bravo", "n’importe quoi", "top", "incroyable", "une blague", "excellent", "trop bien", "trop fort", "trop cool", "trop marrant", "trop drôle"]
insult_keywords = ["nul", "incompétent", "idiot", "imbécile", "moquerie", "cata", "catastrophe", "stupide", "débile", "connard"]

REPONSES_INSULTES = [
    "Je maintiens un ton respectueux. Comment puis-je vous aider concernant TuN ?",
    "Je suis ici pour aider. Avez-vous une question sur nos services ?",
    "Passons à une discussion constructive. Que puis-je faire pour vous ?"
]

REPONSES_SARCASTIQUES = [
    "Je comprends que vous puissiez être frustré. Comment puis-je mieux vous aider ?",
    "Je note votre feedback. Pouvez-vous reformuler votre demande ?",
    "Mon but est de vous aider efficacement. Quelle est votre question sur TuN ?"
]

fallback_responses = [
    "Je suis désolé, je n'ai pas trouvé d'information claire à ce sujet. Puis-je vous aider sur autre chose ?",
    "Malheureusement, je ne trouve pas de réponse dans les documents. Souhaitez-vous reformuler votre question ?",
    "Je n'ai pas l'information pour le moment, mais je suis à votre disposition si vous avez d'autres questions.",
    "Je ne dispose pas d'information concernant cette question. N'hésitez pas à poser d'autres questions sur TuN !",
]

def detect_sarcasm_or_insult(text):
    text = text.lower()
    for word in sarcasm_keywords:
        if word in text:
            return "sarcasme"
    for word in insult_keywords:
        if word in text:
            return "insulte"
    return "neutre"

def handle_sarcasm_or_insult(user_input):
    sentiment = detect_sarcasm_or_insult(user_input)
    if sentiment == "sarcasme":
        return random.choice(REPONSES_SARCASTIQUES)
    elif sentiment == "insulte":
        return random.choice(REPONSES_INSULTES)
    else:
        return None


def is_off_topic(text):
    # Liste simple de mots-clés hors sujet (tu peux améliorer avec une classification ML plus tard)
    off_topic_keywords = [
        "pizza", "chat", "météo", "sport", "film", "recette", "voiture", "musique", "jeu vidéo",
        "vacances", "voyage", "série", "cinéma"
    ]
    text = text.lower()
    return any(keyword in text for keyword in off_topic_keywords)

# FastAPI setup
app = FastAPI()

origins = ["http://localhost:4200"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuestion(BaseModel):
    question: str



@app.post("/ask")
def ask(user_question: UserQuestion):
    user_input = user_question.question

    try:
        # 1 : Gestion insulte / sarcasme
        custom_response = handle_sarcasm_or_insult(user_input)
        if custom_response:
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(custom_response)
            current_history = memory.load_memory_variables({}).get("history", [])
            return {
                "response": custom_response,
                "history": [
                    {"role": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
                    for msg in current_history
                ],
            }

        # 2 : Recherche documents pertinents
        relevant_docs = retriever.get_relevant_documents(user_input)

        # 3 : Si pas de documents, alors hors sujet => message standard
        if not relevant_docs:
            result = "Désolé, je ne peux répondre qu’aux questions liées à TuN (appels d’offres, signalements, comptes, etc.)."
        else:
            # Construction du contexte pour le prompt
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

            history = memory.load_memory_variables({}).get("history", [])
            history_str = "\n".join([
                f"{'Utilisateur' if isinstance(msg, HumanMessage) else 'Assistant'} : {msg.content}"
                for msg in history[-4:]
            ])

            final_prompt = prompt.format(
                context=context_text,
                history=history_str,
                question=user_input
            )
            start = time.time()
            result = model.invoke(final_prompt)
            print(f"Temps de réponse : {time.time() - start:.2f} secondes")

        # Mémorisation des échanges
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(result)

        current_history = memory.load_memory_variables({}).get("history", [])
        return {
            "response": result,
            "history": [
                {"role": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
                for msg in current_history
            ],
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "response": "Désolé, une erreur technique est survenue. Veuillez réessayer.",
            "history": []
        }
