import getpass
import os
import sys

from huggingface_hub import HfApi

SPACE_REPO_ID = "Imagenbomba/hollywood-mirror"
MANAGED_PATTERNS = [
    "Dockerfile",
    "requirements.txt",
    "requirements-web.txt",
    "src/*",
    "data/processed/movie_embeddings_*.npy",
    "data/processed/movie_embeddings_*.txt",
]


def resolve_token() -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token
    if sys.stdin.isatty():
        return getpass.getpass(
            "Pega tu Token de Hugging Face (no se verá al escribir): "
        ).strip()

    print("Error: HF_TOKEN es obligatorio en entornos no interactivos.", file=sys.stderr)
    sys.exit(1)


def main():
    print("=== Subidor a Hugging Face Spaces ===")
    token = resolve_token()
    if not token:
        print("Error: El token no puede estar vacío.")
        sys.exit(1)

    repo_id = os.getenv("HF_SPACE_REPO_ID", SPACE_REPO_ID).strip() or SPACE_REPO_ID
    commit_message = os.getenv(
        "HF_COMMIT_MESSAGE",
        "Sync Space deployment from local repository",
    ).strip()
    api = HfApi(token=token)

    print(f"\nSincronizando Space '{repo_id}' con los archivos del backend web...")
    try:
        commit_url = api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            allow_patterns=MANAGED_PATTERNS,
            delete_patterns=MANAGED_PATTERNS,
            commit_message=commit_message,
        )
        print("\n¡SUBIDA COMPLETADA CON ÉXITO! 🎉")
        print(f"Commit creado: {commit_url}")
        print(f"Ve a https://huggingface.co/spaces/{repo_id} para ver tu API corriendo.")
    except Exception as e:
        print(f"\nError al subir: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
