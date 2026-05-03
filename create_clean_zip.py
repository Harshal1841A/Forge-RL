import zipfile
import os

def create_zip():
    zip_name = 'forge_ma_final.zip'
    exclude_list = ['graphify-out', 'uv.lock', '.env', 'node_modules', '.next', '__pycache__', '.git', 'forge_ma_final.zip']
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if not any(x in d for x in exclude_list)]
            
            for file in files:
                if any(x in file for x in exclude_list):
                    continue
                if file.endswith('.pyc'):
                    continue
                
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, '.')
                zipf.write(full_path, rel_path)

if __name__ == '__main__':
    create_zip()
    print("forge_ma_final.zip created successfully.")
