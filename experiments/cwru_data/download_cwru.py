import urllib.request, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

files = {
    "normal_0.mat":       "https://engineering.case.edu/sites/default/files/97.mat",
    "inner_race_007.mat": "https://engineering.case.edu/sites/default/files/105.mat",
}

for filename, url in files.items():
    if not os.path.exists(filename):
        print(f"מוריד {filename}...")
        urllib.request.urlretrieve(url, filename)
        size = os.path.getsize(filename)
        print(f"  ✓ {filename} ({size:,} bytes)")
    else:
        size = os.path.getsize(filename)
        print(f"  כבר קיים: {filename} ({size:,} bytes)")
