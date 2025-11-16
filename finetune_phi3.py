"""
Script de fine-tuning pour Phi-3 Mini
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

# Configuration
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "./phi3_medical_finetuned"
DATA_FILE = "data.json"

def load_and_prepare_data():
    """Charger et préparer les données"""
    print("Chargement des données...")
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"{len(data)} exemples chargés")
    
    prompts = []
    for item in data:
        user_input = item['texte']
        classification = item['classification']
        entites = item['entites']
        
        response = {
            "classification": classification,
            "heure": entites.get('heure', 'NON_SPECIFIE'),
            "date": entites.get('date', 'NON_SPECIFIE'),
            "nom_docteur": entites.get('nom_docteur', 'NON_SPECIFIE')
        }
        
        prompt = f"""<|user|>
Analyse cette demande médicale et réponds en JSON:
{user_input}
<|end|>
<|assistant|>
{json.dumps(response, ensure_ascii=False)}
<|end|>"""
        
        prompts.append(prompt)
    
    split_idx = int(len(prompts) * 0.8)
    train_prompts = prompts[:split_idx]
    val_prompts = prompts[split_idx:]
    
    print(f"Train: {len(train_prompts)}, Validation: {len(val_prompts)}")
    return train_prompts, val_prompts

def setup_model():
    """Setup modèle avec gradients correctement activés"""
    print("Chargement du modèle Phi-3...")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Modèle SANS optimisations qui causent des problèmes
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Retour au float32 pour éviter les problèmes
        trust_remote_code=True,
    )
    
    # Déplacer sur GPU si disponible
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Modèle sur GPU: {torch.cuda.get_device_name()}")
    
    # ACTIVER les gradients sur le modèle de base
    for param in model.parameters():
        param.requires_grad = False  # Geler d'abord tout
    
    # Configuration LoRA SIMPLE
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Un peu plus élevé pour assurer l'entraînement
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["qkv_proj", "o_proj"]  # Modules simples
    )
    
    # Appliquer LoRA
    model = get_peft_model(model, lora_config)
    
    # FORCER l'activation des gradients sur les paramètres LoRA
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            print(f"Gradient activé pour: {name}")
    
    model.print_trainable_parameters()
    
    return model, tokenizer

def create_datasets(train_prompts, val_prompts, tokenizer):
    """Créer les datasets tokenizés"""
    print("Création des datasets...")
    
    def tokenize_function(examples):
        """Fonction de tokenization"""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors=None,
        )
        
        # Labels pour l'entraînement causal
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    train_dataset = Dataset.from_dict({"text": train_prompts})
    val_dataset = Dataset.from_dict({"text": val_prompts})
    
    print("Tokenization...")
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train"
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing validation"
    )
    
    print("Datasets tokenizés")
    return train_dataset, val_dataset

def train_model(model, tokenizer, train_dataset, val_dataset):
    """Entraîner le modèle"""
    print("Début de l'entraînement...")
    
    # Vérifier que les gradients sont bien activés
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"Paramètres entraînables: {len(trainable_params)}")
    
    if len(trainable_params) == 0:
        raise ValueError("Aucun paramètre entraînable trouvé!")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,  # Learning rate plus élevé pour LoRA
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to=[],
        fp16=False,  # Pas de fp16 pour éviter les problèmes
        optim="adamw_torch",
        label_names=["labels"],
        gradient_checkpointing=False,  # Désactivé car peut causer des problèmes
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Entraînement
    trainer.train()
    
    # Sauvegarde
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    

def main():
    """Pipeline principal"""
    print("FINE-TUNING PHI-3")
    print("=" * 50)
    
    # Nettoyage mémoire
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    try:
        if not os.path.exists(DATA_FILE):
            print(f"Fichier {DATA_FILE} non trouvé!")
            return
        
        print(f"GPU disponible: {torch.cuda.is_available()}")
        
        # Pipeline
        train_prompts, val_prompts = load_and_prepare_data()
        model, tokenizer = setup_model()
        train_dataset, val_dataset = create_datasets(train_prompts, val_prompts, tokenizer)
        train_model(model, tokenizer, train_dataset, val_dataset)
        
        print("FINE-TUNING TERMINÉ!")
        print(f"Modèle sauvegardé: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
