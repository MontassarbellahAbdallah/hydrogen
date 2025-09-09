import streamlit as st
import ollama
import json
import re

# Configuration du modèle
MODEL_NAME = "phi3:mini"

def get_few_shot_examples():
    """Exemples few-shot équilibrés pour le prompt"""
    examples = [
        # RÉSERVATIONS
        {
            "input": "Je veux un RDV avec Dr Martin demain à 14h",
            "output": '{"classification": "RESERVATION", "heure": "14h", "date": "demain", "nom_docteur": "Dr Martin"}'
        },
        {
            "input": "Rendez-vous avec le médecin Dupont lundi matin",
            "output": '{"classification": "RESERVATION", "heure": "NON_SPECIFIE", "date": "lundi matin", "nom_docteur": "médecin Dupont"}'
        },
        {
            "input": "Consultation Dr Sophie vendredi à 15h30",
            "output": '{"classification": "RESERVATION", "heure": "15h30", "date": "vendredi", "nom_docteur": "Dr Sophie"}'
        },
        
        # MESSAGES
        {
            "input": "Je veux laisser un message au Dr Martin",
            "output": '{"classification": "MESSAGE", "heure": "NON_SPECIFIE", "date": "NON_SPECIFIE", "nom_docteur": "Dr Martin"}'
        },
        {
            "input": "Message pour le docteur Dupont concernant mes résultats",
            "output": '{"classification": "MESSAGE", "heure": "NON_SPECIFIE", "date": "NON_SPECIFIE", "nom_docteur": "docteur Dupont"}'
        },
        {
            "input": "Transmettez au médecin Sophie que j'ai des effets secondaires",
            "output": '{"classification": "MESSAGE", "heure": "NON_SPECIFIE", "date": "NON_SPECIFIE", "nom_docteur": "médecin Sophie"}'
        },
        
        # HORS_SUJET
        {
            "input": "Je veux réserver une table au restaurant demain soir",
            "output": '{"classification": "HORS_SUJET", "heure": "NON_SPECIFIE", "date": "NON_SPECIFIE", "nom_docteur": "NON_SPECIFIE"}'
        },
        {
            "input": "Réservation d'une chambre d'hôtel pour le weekend",
            "output": '{"classification": "HORS_SUJET", "heure": "NON_SPECIFIE", "date": "NON_SPECIFIE", "nom_docteur": "NON_SPECIFIE"}'
        },
        {
            "input": "RDV chez le coiffeur samedi à 10h",
            "output": '{"classification": "HORS_SUJET", "heure": "NON_SPECIFIE", "date": "NON_SPECIFIE", "nom_docteur": "NON_SPECIFIE"}'
        },
        {
            "input": "Je veux emprunter un livre à la bibliothèque",
            "output": '{"classification": "HORS_SUJET", "heure": "NON_SPECIFIE", "date": "NON_SPECIFIE", "nom_docteur": "NON_SPECIFIE"}'
        }
    ]
    return examples

def create_prompt(user_input):
    """Créer le prompt few-shot pour Ollama"""
    examples = get_few_shot_examples()
    
    prompt = """Tu es un assistant pour un cabinet médical. Analyse les demandes et réponds UNIQUEMENT en format JSON.

RÈGLES :
1. Classification : "RESERVATION" (RDV médical), "MESSAGE" (message pour médecin), "HORS_SUJET" (non médical)
2. Pour RESERVATION/MESSAGE : extrais heure, date, nom_docteur (ou "NON_SPECIFIE" si absent)
3. Pour HORS_SUJET : tout à "NON_SPECIFIE"
4. Réponds SEULEMENT avec le JSON, rien d'autre

EXEMPLES :

"""
    
    # Ajouter les exemples few-shot
    for example in examples:
        prompt += f"Demande: {example['input']}\n"
        prompt += f"Réponse: {example['output']}\n\n"
    
    # Ajouter la vraie demande
    prompt += f"Demande: {user_input}\n"
    prompt += "Réponse: "
    
    return prompt

def query_ollama(prompt):
    """Interroger Ollama avec le prompt"""
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        return response['message']['content']
    except Exception as e:
        return f"Erreur Ollama: {str(e)}"

def parse_response(response_text):
    """Parser la réponse JSON d'Ollama"""
    try:
        # Nettoyer la réponse (enlever texte avant/après JSON)
        response_text = response_text.strip()
        
        # Chercher le JSON dans la réponse
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            return None
    except json.JSONDecodeError:
        return None

def main():
    st.title("Assistant Réservation Médicale - Ollama")
    st.subheader("Système local avec Phi-3")
    
    st.write("Saisir votre demande")
    user_input = st.text_area("", placeholder="Exemple: Je veux un rendez-vous avec Dr Martin demain à 14h", height=100)
    
    if st.button("Analyser"):
        if user_input.strip():
            try:
                with st.spinner("Analyse en cours avec Phi-3..."):
                    # Créer le prompt
                    prompt = create_prompt(user_input.strip())
    
                    # Interroger Ollama
                    response = query_ollama(prompt)
                    
                    # Parser la réponse
                    parsed_result = parse_response(response)
                    
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
                            st.write("Cette demande ne concerne pas le domaine médical")
                            
                        else:
                            st.write(f"Classification: {classification}")
                    else:
                        st.write("Erreur: Impossible d'analyser la réponse")
                        
            except Exception as e:
                st.write(f"Erreur: {str(e)}")
        else:
            st.write("Veuillez saisir une demande")
    
    # Section d'informations
    st.sidebar.write(f"Modèle utilisé: {MODEL_NAME}")
    st.sidebar.write("Type: Few-shot learning local")
    st.sidebar.write("Exemples dans le prompt: 10")

if __name__ == "__main__":
    main()
