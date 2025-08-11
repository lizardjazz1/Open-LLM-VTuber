import os

# Настройки
ROOT_DIR = "externals/Open-LLM-VTuber-Web"
OUTPUT_FILE = "Open-LLM-VTuber-Web_FULL_SOURCE.txt"

# Что игнорировать (чтобы не включать бинарные/служебные файлы)
IGNORE_DIRS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".vscode",
    ".idea",
    "dist",
    "build",
    ".cache",
    "coverage",
    "tmp",
    "temp",
    "public",
}

IGNORE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",  # изображения
    ".mp3",
    ".wav",
    ".ogg",
    ".webm",  # аудио/видео
    ".zip",
    ".tar",
    ".gz",
    ".7z",  # архивы
    ".log",  # логи
}


def should_ignore(path: str) -> bool:
    name = os.path.basename(path)
    if os.path.isdir(path):
        return name in IGNORE_DIRS
    ext = os.path.splitext(name)[1].lower()
    return ext in IGNORE_EXTENSIONS


def export_all_code():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as output:
        output.write("# FULL SOURCE EXPORT\n")
        output.write("# Project: Open-LLM-VTuber-Web\n")
        output.write(
            f"# Generated on: {__import__('datetime').datetime.now().isoformat()}\n"
        )
        output.write(f"# Path: {os.path.abspath(ROOT_DIR)}\n")
        output.write(f"{'=' * 100}\n\n")

        for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
            dirnames[:] = [
                d for d in dirnames if not should_ignore(os.path.join(dirpath, d))
            ]
            dirnames.sort()
            filenames.sort()

            for file in filenames:
                filepath = os.path.join(dirpath, file)
                if should_ignore(filepath):
                    continue

                relative_path = os.path.relpath(filepath, ROOT_DIR)
                header = f"{'=' * 20} FILE: {relative_path} {'=' * 20}"
                output.write(header + "\n\n")

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        output.write(content)
                except Exception as e:
                    output.write(f"[ERROR: Could not read file - {e}]\n")

                output.write("\n\n")

    print(f"✅ Полный код сохранён в {OUTPUT_FILE}")


if __name__ == "__main__":
    if not os.path.exists(ROOT_DIR):
        print(f"❌ Папка не найдена: {ROOT_DIR}")
        print("Убедитесь, что вы запускаете скрипт из корня Open-LLM-VTuber")
    else:
        export_all_code()
