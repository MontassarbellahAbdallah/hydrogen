import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import json
import re
import os

class FinetunedModelTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        try:
            print("Chargement du modèle fine-tuné...")
            
            # Charger la configuration LoRA
            config = PeftConfig.from_pretrained(self.model_path)
            
            # Charger le tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Charger le modèle de base
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map=None, 
            )
            
            # Charger le modèle LoRA
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            
            # GPU si disponible
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            print("Modèle chargé avec succès!")
            return True
            
        except Exception as e:
            print(f"Erreur de chargement: {e}")
            return False
    
    def generate_response(self, user_input, max_tokens=120):
        
        prompt = f"""<|user|>
Analyse cette demande médicale et réponds en JSON:
{user_input}
<|end|>
<|assistant|>
"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=400, truncation=True)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            input_ids = inputs["input_ids"]
            generated_text = ""
            json_started = False
            json_ended = False
            
            # Génération token par token avec détection de fin JSON
            with torch.no_grad():
                for i in range(max_tokens):
                    outputs = self.model(input_ids)
                    logits = outputs.logits
                    
                    next_token_id = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
                    
                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Décoder le nouveau token
                    new_token = self.tokenizer.decode(next_token_id.item(), skip_special_tokens=True)
                    generated_text += new_token
                    
                    # Détecter le début du JSON
                    if "{" in new_token:
                        json_started = True
                    
                    # CONDITION D'ARRÊT : Si on trouve } 
                    if json_started and "}" in new_token:
                        json_ended = True
                        # Ajouter ce token et arrêter
                        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                        break
                    
                    input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                    
                    if input_ids.shape[1] > 1000:
                        break
            
            # Décoder la réponse finale
            full_response = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            if "<|assistant|>" in full_response:
                response = full_response.split("<|assistant|>")[-1].strip()
                if "<|end|>" in response:
                    response = response.split("<|end|>")[0].strip()
                return response
            else:
                return full_response.replace(prompt, "").strip()
                
        except Exception as e:
            return f"Erreur de génération: {str(e)}"
    
    def parse_json(self, response):
        """parser le JSON"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                return None
        except:
            return None

# Configuration globale
MODEL_PATH = "./phi3_medical_finetuned"

@st.cache_resource
def load_finetuned_model():
    tester = FinetunedModelTester(MODEL_PATH)
    if tester.load_model():
        return tester
    else:
        return None

def main():
    st.title("Assistant Réservation Médicale")
    st.subheader("Système de gestion des demandes patients")
    
    # Vérifier que le modèle existe
    if not os.path.exists(MODEL_PATH):
        st.error("Modèle fine-tuné non trouvé!")
        st.info("Assurez-vous que le dossier 'phi3_medical_finetuned' est présent")
        return
    
    # Charger le modèle
    with st.spinner("Chargement du modèle fine-tuné..."):
        tester = load_finetuned_model()
    
    if tester is None:
        st.error("Impossible de charger le modèle fine-tuné")
        return
    
    st.write("Saisir votre demande")
    user_input = st.text_area("", placeholder="Exemple: Je veux un rendez-vous avec Dr Martin demain à 14h", height=100)
    
    if st.button("Analyser"):
        if user_input.strip():
            try:
                with st.spinner("Analyse en cours..."):
                    response = tester.generate_response(user_input.strip())
                    
                    # Parser le JSON
                    parsed_result = tester.parse_json(response)
                    
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
                            st.write(f"Classification: {classification}")
                    
                    else:
                        st.error("Le modèle n'a pas produit de JSON valide")
                        st.write("Réponse du modèle:")
                        st.text(response)
                        
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
        else:
            st.write("Veuillez saisir une demande")

if __name__ == "__main__":
    main()
