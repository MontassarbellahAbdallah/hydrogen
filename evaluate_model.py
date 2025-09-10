import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import json
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import pandas as pd
import numpy as np

class ModelEvaluator:
    def __init__(self, model_path, test_data_path):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.model = None
        self.tokenizer = None
        self.test_data = None
        
    def load_model(self):
        """Charger le modèle fine-tuné"""
        try:
            print("Chargement du modèle fine-tuné pour évaluation...")
            
            config = PeftConfig.from_pretrained(self.model_path)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map=None,
            )
            
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            print("Modèle chargé avec succès!")
            return True
            
        except Exception as e:
            print(f"Erreur de chargement: {e}")
            return False
    
    def load_test_data(self):
        """Charger les données de test"""
        try:
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                self.test_data = json.load(f)
            print(f"Données de test chargées: {len(self.test_data)} exemples")
            return True
        except Exception as e:
            print(f"Erreur de chargement des données: {e}")
            return False
    
    def generate_response(self, user_input, max_tokens=120):
        """Générer une réponse avec le modèle fine-tuné"""
        
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
            json_started = False
            
            with torch.no_grad():
                for i in range(max_tokens):
                    outputs = self.model(input_ids)
                    logits = outputs.logits
                    
                    next_token_id = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
                    
                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break
                    
                    new_token = self.tokenizer.decode(next_token_id.item(), skip_special_tokens=True)
                    
                    if "{" in new_token:
                        json_started = True
                    
                    if json_started and "}" in new_token:
                        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                        break
                    
                    input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                    
                    if input_ids.shape[1] > 1000:
                        break
            
            full_response = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            if "<|assistant|>" in full_response:
                response = full_response.split("<|assistant|>")[-1].strip()
                if "<|end|>" in response:
                    response = response.split("<|end|>")[0].strip()
                return response
            else:
                return full_response.replace(prompt, "").strip()
                
        except Exception as e:
            return f"Erreur: {str(e)}"
    
    def parse_json(self, response):
        """Parser le JSON de la réponse"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                return None
        except:
            return None
    
    def evaluate_model(self):
        """Évaluer le modèle sur les données de test"""
        
        if not self.load_model() or not self.load_test_data():
            return None
        
        predictions = []
        ground_truths = []
        detailed_results = []
        
        print(f"Évaluation en cours sur {len(self.test_data)} exemples...")
        
        for i, item in enumerate(self.test_data):
            if i % 50 == 0:
                print(f"Progression: {i}/{len(self.test_data)}")
            
            user_input = item['texte']
            true_classification = item['classification']
            true_entities = item['entites']
            
            # Générer la prédiction
            response = self.generate_response(user_input)
            parsed = self.parse_json(response)
            
            if parsed:
                pred_classification = parsed.get('classification', 'INCONNU')
                pred_entities = {
                    'heure': parsed.get('heure', 'NON_SPECIFIE'),
                    'date': parsed.get('date', 'NON_SPECIFIE'),
                    'nom_docteur': parsed.get('nom_docteur', 'NON_SPECIFIE')
                }
            else:
                pred_classification = 'PARSE_ERROR'
                pred_entities = {
                    'heure': 'PARSE_ERROR',
                    'date': 'PARSE_ERROR',
                    'nom_docteur': 'PARSE_ERROR'
                }
            
            predictions.append(pred_classification)
            ground_truths.append(true_classification)
            
            # Évaluation des entités
            entity_scores = {}
            if pred_classification == true_classification and pred_classification == "RESERVATION":
                for entity in ['heure', 'date', 'nom_docteur']:
                    if true_entities.get(entity, 'NON_SPECIFIE') == pred_entities.get(entity, 'NON_SPECIFIE'):
                        entity_scores[entity] = 1
                    else:
                        entity_scores[entity] = 0
            
            detailed_results.append({
                'texte': user_input,
                'true_class': true_classification,
                'pred_class': pred_classification,
                'true_entities': true_entities,
                'pred_entities': pred_entities,
                'entity_scores': entity_scores,
                'raw_response': response
            })
        
        return self.calculate_metrics(predictions, ground_truths, detailed_results)
    
    def calculate_metrics(self, predictions, ground_truths, detailed_results):
        """Calculer les métriques de performance"""
        
        # Métriques de classification
        accuracy = accuracy_score(ground_truths, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truths, predictions, average='weighted', zero_division=0
        )
        
        # Métriques par classe
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            ground_truths, predictions, average=None, zero_division=0
        )
        
        # Classes uniques
        unique_classes = sorted(list(set(ground_truths + predictions)))
        
        # Matrice de confusion
        cm = confusion_matrix(ground_truths, predictions, labels=unique_classes)
        
        # Métriques d'entités (pour RESERVATION seulement)
        entity_metrics = {}
        reservation_results = [r for r in detailed_results if r['true_class'] == 'RESERVATION' and r['pred_class'] == 'RESERVATION']
        
        if reservation_results:
            for entity in ['heure', 'date', 'nom_docteur']:
                scores = [r['entity_scores'].get(entity, 0) for r in reservation_results]
                entity_metrics[entity] = np.mean(scores) if scores else 0
        
        # Taux de parsing JSON
        parse_success_rate = len([p for p in predictions if p != 'PARSE_ERROR']) / len(predictions)
        
        results = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': dict(zip(unique_classes, precision_per_class)),
            'recall_per_class': dict(zip(unique_classes, recall_per_class)),
            'f1_per_class': dict(zip(unique_classes, f1_per_class)),
            'support_per_class': dict(zip(unique_classes, support_per_class)),
            'confusion_matrix': cm,
            'class_labels': unique_classes,
            'entity_metrics': entity_metrics,
            'parse_success_rate': parse_success_rate,
            'detailed_results': detailed_results
        }
        
        return results
    
    def print_evaluation_report(self, results):
        """Afficher le rapport d'évaluation"""
        
        print("\n" + "="*60)
        print("RAPPORT D'ÉVALUATION DU MODÈLE FINE-TUNÉ")
        print("="*60)
        
        # Métriques globales
        print(f"\nMÉTRIQUES GLOBALES:")
        print(f"Précision (Accuracy): {results['accuracy']:.3f}")
        print(f"Précision pondérée: {results['precision_weighted']:.3f}")
        print(f"Rappel pondéré: {results['recall_weighted']:.3f}")
        print(f"F1-Score pondéré: {results['f1_weighted']:.3f}")
        print(f"Taux de parsing JSON: {results['parse_success_rate']:.3f}")
        
        # Métriques par classe
        print(f"\nMÉTRIQUES PAR CLASSE:")
        for class_name in results['class_labels']:
            print(f"\n{class_name}:")
            print(f"  Précision: {results['precision_per_class'][class_name]:.3f}")
            print(f"  Rappel: {results['recall_per_class'][class_name]:.3f}")
            print(f"  F1-Score: {results['f1_per_class'][class_name]:.3f}")
            print(f"  Support: {results['support_per_class'][class_name]}")
        
        # Métriques d'entités
        if results['entity_metrics']:
            print(f"\nMÉTRIQUES D'EXTRACTION D'ENTITÉS (RESERVATION):")
            for entity, score in results['entity_metrics'].items():
                print(f"  {entity}: {score:.3f}")
        
        # Matrice de confusion
        print(f"\nMATRICE DE CONFUSION:")
        print("Lignes = Vraie classe, Colonnes = Classe prédite")
        print("Classes:", results['class_labels'])
        for i, row in enumerate(results['confusion_matrix']):
            print(f"{results['class_labels'][i]}: {row}")
        
        # Analyse des erreurs
        errors = [r for r in results['detailed_results'] if r['true_class'] != r['pred_class']]
        print(f"\nANALYSE DES ERREURS:")
        print(f"Nombre total d'erreurs: {len(errors)}")
        
        if errors:
            error_types = {}
            for error in errors:
                error_type = f"{error['true_class']} → {error['pred_class']}"
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            print("Types d'erreurs les plus fréquents:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count}")

def create_test_split(data_path, test_ratio=0.2):
    """Créer un split de test à partir des données"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Mélanger les données
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    
    # Split
    test_size = int(len(data) * test_ratio)
    test_indices = indices[:test_size]
    
    test_data = [data[i] for i in test_indices]
    
    # Sauvegarder le split de test
    with open('test_data.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"Split de test créé: {len(test_data)} exemples sauvegardés dans 'test_data.json'")
    return 'test_data.json'

def main():
    """Fonction principale d'évaluation"""
    
    # Paramètres
    model_path = "./phi3_medical_finetuned"  
    data_path = "data.json"  
    
    # Créer un split de test si nécessaire
    print("Création du split de test...")
    test_data_path = create_test_split(data_path, test_ratio=0.2)
    
    # Initialiser l'évaluateur
    evaluator = ModelEvaluator(model_path, test_data_path)
    
    # Lancer l'évaluation
    results = evaluator.evaluate_model()
    
    if results:
        # Afficher le rapport
        evaluator.print_evaluation_report(results)
        
        # Sauvegarder les résultats
        with open('evaluation_results.json', 'w', encoding='utf-8') as f:
            # Convertir les arrays numpy en listes pour la sérialisation JSON
            serializable_results = results.copy()
            serializable_results['confusion_matrix'] = results['confusion_matrix'].tolist()
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"\nRésultats sauvegardés dans 'evaluation_results.json'")
    else:
        print("Échec de l'évaluation")

if __name__ == "__main__":
    main()
