import os
import sys
import yaml
import zipfile
import subprocess
import json

def run_checks():
    print("=== SUBMISSION READY CHECKLIST ===")
    
    # FORGE-MA: openenv name check
    print("\nChecking openenv.yaml...")
    try:
        c = yaml.safe_load(open('openenv.yaml'))
        assert c['name'] == 'forge-ma', f"name is still {c['name']}"
        assert c['version'] == '2.0.0'
        assert c['reward_range'] == [0.001, 0.999]
        assert c['action_space']['n'] == 13
        assert c['observation_space']['shape'] == [3859]
        print('openenv.yaml: ALL OK')
        print(f'  name={c["name"]} version={c["version"]} reward_range={c["reward_range"]}')
    except Exception as e:
        print(f'FAIL: openenv.yaml check failed: {e}')
        sys.exit(1)

    # FORGE-MA: no junk in zip
    print("\nChecking forge_ma_final.zip...")
    try:
        z = zipfile.ZipFile('forge_ma_final.zip')
        junk = [n for n in z.namelist() if any(x in n for x in ['graphify-out','uv.lock','.env'])]
        if junk:
            print('FAIL: junk files still present:', junk)
            sys.exit(1)
        print('Zip clean: no junk files')
        print(f'Total files: {len(z.namelist())}')
    except Exception as e:
        print(f'FAIL: zip check failed: {e}')
        sys.exit(1)

    # Spatial SaaS: lucide-react version check
    print("\nChecking Spatial SaaS package.json...")
    try:
        pkg_path = os.path.join('spatial-saas', 'package.json')
        with open(pkg_path, 'r', encoding='utf-8') as f:
            pkg = json.load(f)
        v = pkg['dependencies'].get('lucide-react')
        print('lucide-react version:', v)
        if v == '^1.8.0':
            print('FAIL: lucide-react still ^1.8.0 — npm install will fail')
            sys.exit(1)
        print('lucide-react: OK')
    except Exception as e:
        print(f'FAIL: package.json check failed: {e}')
        sys.exit(1)

    # Spatial SaaS: API base URL check
    print("\nChecking Spatial SaaS API base URL...")
    try:
        api_path = os.path.join('spatial-saas', 'src', 'lib', 'api.ts')
        with open(api_path, 'r', encoding='utf-8') as f:
            api = f.read()
        if ': ""' in api:
            print('FAIL: API BASE still defaults to empty string')
            sys.exit(1)
        if 'localhost:7860' not in api:
            print('FAIL: API BASE does not default to port 7860')
            sys.exit(1)
        print('api.ts BASE default: OK (localhost:7860)')
    except Exception as e:
        print(f'FAIL: api.ts check failed: {e}')
        sys.exit(1)

    # Spatial SaaS: name-based action lookup
    print("\nChecking Spatial SaaS action lookup...")
    try:
        dash_path = os.path.join('spatial-saas', 'src', 'components', 'sections', 'DashboardPreviewSection.tsx')
        with open(dash_path, 'r', encoding='utf-8') as f:
            dash = f.read()
        if 'AUTO_SEQUENCE_NAMES' not in dash:
            print('FAIL: AUTO_SEQUENCE_NAMES not found in dashboard')
            sys.exit(1)
        if 'AUTO_SEQUENCE = [0' in dash or 'AUTO_SEQUENCE = [1' in dash:
            print('FAIL: old integer AUTO_SEQUENCE still present')
            sys.exit(1)
        print('DashboardPreviewSection action lookup: OK (name-based)')
    except Exception as e:
        print(f'FAIL: DashboardPreviewSection check failed: {e}')
        sys.exit(1)

    print("\nALL CHECKS PASSED. SYSTEM IS READY FOR SUBMISSION.")

if __name__ == '__main__':
    run_checks()
