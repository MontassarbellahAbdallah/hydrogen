import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import json
import re

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_medical_appointment_chain():
    prompt_template = """
Tu es un assistant pour un cabinet médical. Analyse la demande et réponds en format JSON avec :

IMPORTANT: Tu dois d'abord vérifier si la demande concerne le domaine médical (médecin, docteur, consultation, rendez-vous médical, cabinet médical, etc.)

1. "classification" : 
   - "RESERVATION" : si c'est une demande de rendez-vous médical
   - "MESSAGE" : si c'est un message pour un médecin
   - "HORS_SUJET" : si ce n'est pas lié au domaine médical

2. Si RESERVATION, extrais :
   - "heure" : l'heure mentionnée ou "NON_SPECIFIE"
   - "date" : la date mentionnée ou "NON_SPECIFIE" 
   - "nom_docteur" : le nom du médecin ou "NON_SPECIFIE"

3. Si MESSAGE, extrais :
   - "nom_docteur" : le nom du médecin ou "NON_SPECIFIE"

Exemples :
"Je veux un RDV avec Dr Martin lundi 10h" -> {{"classification": "RESERVATION", "heure": "10h", "date": "lundi", "nom_docteur": "Dr Martin"}}
"Message pour Dr Paul" -> {{"classification": "MESSAGE", "heure": "NON_APPLICABLE", "date": "NON_APPLICABLE", "nom_docteur": "Dr Paul"}}
"Je veux réserver un livre à la bibliothèque" -> {{"classification": "HORS_SUJET"}}
"Réserver une table au restaurant" -> {{"classification": "HORS_SUJET"}}
"RDV chez le coiffeur demain" -> {{"classification": "HORS_SUJET"}}

Demande: {user_input}
Réponse JSON:
"""

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["user_input"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def parse_llm_response(response_text):
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            return None
    except json.JSONDecodeError:
        return None

def main():
    st.title("Assistant Réservation Médicale")
    st.subheader("Système de gestion des demandes patients")
    
    st.write("Saisir votre demande")
    user_input = st.text_area("", placeholder="Exemple: Je veux un rendez-vous avec Dr Martin demain à 14h", height=100)
    
    if st.button("Analyser"):
        if user_input.strip():
            try:
                chain = get_medical_appointment_chain()
                response = chain.run(user_input=user_input.strip())
                parsed_result = parse_llm_response(response)
                
                if parsed_result:
                    classification = parsed_result.get('classification', 'INCONNU')
                    
                    if classification == "RESERVATION":
                        st.write("Classification: RESERVATION")
                        heure = parsed_result.get('heure', 'NON_SPECIFIE')
                        date = parsed_result.get('date', 'NON_SPECIFIE')
                        docteur = parsed_result.get('nom_docteur', 'NON_SPECIFIE')
                        st.write(f"Heure: {heure}")
                        st.write(f"Date: {date}")
                        st.write(f"Médecin: {docteur}")
                        
                    elif classification == "MESSAGE":
                        st.write("Classification: MESSAGE")
                        docteur = parsed_result.get('nom_docteur', 'NON_SPECIFIE')
                        st.write(f"Message pour: {docteur}")
                        
                    elif classification == "HORS_SUJET":
                        st.write("Classification: HORS_SUJET")         
                    else:
                        st.write("Classification: Non reconnue")
                else:
                    st.write("Erreur: Impossible d'analyser la demande")
                    
            except Exception as e:
                st.write(f"Erreur: {str(e)}")
        else:
            st.write("Veuillez saisir une demande")

if __name__ == "__main__":
    main()
