import os
import shutil
import argparse
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

class DocumentClassifier:
    def __init__(self):
        """Inicializa el clasificador de documentos."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Utilizando dispositivo: {self.device}")
        
        if self.device == "cuda":
            # Mostrar información de la GPU
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"GPU detectada: {gpu_name}")
                print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                print(f"Memoria disponible: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
        
    def train_classifier(self, categories, example_texts=None):
        """Configura el clasificador para las categorías dadas.
        
        Args:
            categories: Lista de categorías para clasificar documentos.
            example_texts: Textos de ejemplo para cada categoría (opcional).
        """
        self.categories = categories
        
        # Usamos un modelo más grande y potente para español
        model_name = "PlanTL-GOB-ES/roberta-large-bne"
        print(f"Cargando modelo avanzado para español: {model_name}...")
        
        # Cargar modelo y tokenizador manualmente para mejor control
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "cross-encoder/stsb-roberta-large", 
            num_labels=1
        )
        
        # Mover modelo a GPU si está disponible
        self.model.to(self.device)
        
        # Configurar pipeline de zero-shot
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="Recognai/bert-base-spanish-wwm-cased-xnli",
            device=0 if self.device == "cuda" else -1
        )
        
        print("Modelo cargado correctamente y listo para clasificar.")
        
    def classify_document(self, document_text):
        """Clasifica un documento en una de las categorías configuradas.
        
        Args:
            document_text: Texto del documento a clasificar.
            document_name: Nombre del documento (opcional, para mejorar la clasificación).
            
        Returns:
            Categoría asignada y score de confianza.
        """
        # Combina el nombre del documento y su contenido para mejorar la clasificación
        document_name_weight = 3  # Dar más peso al nombre del documento
        text_to_classify =  document_text
        
        # Truncar el texto si es demasiado largo (límite de tokens del modelo)
        max_length = 2048  # Aumentado ya que tenemos GPU
        text_to_classify = text_to_classify[:max_length]
        
        # Ejecutar clasificación zero-shot
        result = self.classifier(
            text_to_classify, 
            self.categories, 
            multi_label=False,
            hypothesis_template="Este texto es sobre {}."  # Template en español
        )
        
        # Devolver la categoría con mayor puntuación
        top_category = result["labels"][0]
        top_score = result["scores"][0]
        
        return top_category, top_score

def process_files(input_dir, output_dir, categories):
    """Procesa todos los archivos en el directorio de entrada y los clasifica.
    
    Args:
        input_dir: Directorio donde se encuentran los documentos a clasificar.
        output_dir: Directorio donde se organizarán los documentos clasificados.
        categories: Lista de categorías para clasificar documentos.
    """
    # Crear el clasificador
    classifier = DocumentClassifier()
    classifier.train_classifier(categories)
    
    # Crear directorios de salida
    os.makedirs(output_dir, exist_ok=True)
    for category in categories:
        os.makedirs(os.path.join(output_dir, category), exist_ok=True)
    
    # Obtener lista de archivos a procesar
    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.md', '.txt')):
                files_to_process.append(os.path.join(root, file))
    
    print(f"Se encontraron {len(files_to_process)} archivos para clasificar.")
    
    # Mantener una estadística de la clasificación
    category_counts = {category: 0 for category in categories}
    
    # Procesamiento de archivos con barra de progreso
    for file_path in tqdm(files_to_process, desc="Clasificando documentos"):
        try:
            # Leer contenido del archivo
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                content = file.read()
            
            # Obtener nombre del archivo sin extensión
            file_name = os.path.basename(file_path)
            file_name_without_ext = os.path.splitext(file_name)[0]
            
            # Clasificar documento
            category, confidence = classifier.classify_document(content)
            
            # Incrementar contador de categoría
            category_counts[category] += 1
            
            # Copiar archivo a la carpeta correspondiente
            destination = os.path.join(output_dir, category, file_name)
            shutil.copy2(file_path, destination)
            
            print(f"Archivo: {file_name} -> Categoría: {category} (Confianza: {confidence:.2f})")
        
        except Exception as e:
            print(f"Error procesando {file_path}: {str(e)}")
    
    # Mostrar resumen de clasificación
    print("\n=== Resumen de Clasificación ===")
    for category, count in category_counts.items():
        print(f"- {category}: {count} archivos")

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Clasificador de documentos basado en IA con soporte para GPU')
    parser.add_argument('--input', required=True, help='Directorio de entrada con los documentos .md o .txt')
    parser.add_argument('--output', required=True, help='Directorio de salida para los documentos clasificados')
    parser.add_argument('--categories', required=True, nargs='+', help='Categorías para clasificar los documentos')
    
    args = parser.parse_args()
    
    print(f"Directorio de entrada: {args.input}")
    print(f"Directorio de salida: {args.output}")
    print(f"Categorías: {args.categories}")
    
    # Procesar archivos
    process_files(args.input, args.output, args.categories)
    
    print("¡Clasificación completada!")

if __name__ == "__main__":
    main()