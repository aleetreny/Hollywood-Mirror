from huggingface_hub import HfApi
import getpass
import os
import sys

def main():
    print("=== Subidor a Hugging Face Spaces ===")
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        token = getpass.getpass("Pega tu Token de Hugging Face (no se verá al escribir): ").strip()
    
    if not token:
        print("Error: El token no puede estar vacío.")
        sys.exit(1)
        
    api = HfApi(token=token)
    
    print("\nSubiendo archivos... por favor espera (ignorar los archivos pesados temporales).")
    try:
        api.upload_folder(
            folder_path=".",
            repo_id="Imagenbomba/hollywood-mirror",
            repo_type="space",
            ignore_patterns=[
                ".git/*",
                ".venv/*",
                "venv/*",
                "my_env/*",
                "__pycache__/*",
                "**/__pycache__/*",
                "node_modules/*",
                "frontend/*",
                "frontend/dist/*",
                "analysis/*",
                "docs/*",
                ".quarto/*",
                "**/_site/*",
                "**/*_files/*",
                ".vscode/*",
                ".DS_Store",
                "*.csv",
                "*.parquet",
                "data/raw/*",
                "data/metadata/*",
            ]
        )
        print("\n¡SUBIDA COMPLETADA CON ÉXITO! 🎉")
        print("Ve a https://huggingface.co/spaces/Imagenbomba/hollywood-mirror para ver tu API corriendo.")
    except Exception as e:
        print(f"\nError al subir: {e}")

if __name__ == "__main__":
    main()
