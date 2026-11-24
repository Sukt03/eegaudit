import sys, json, urllib.request
def ls(url, depth, maxdepth):
    while url:
        d = json.load(urllib.request.urlopen(url, timeout=60))
        for f in d["data"]:
            a = f["attributes"]
            if a["kind"] == "folder":
                print("  "*depth + "[D] " + a["name"])
                if depth < maxdepth:
                    ls(f["relationships"]["files"]["links"]["related"]["href"], depth+1, maxdepth)
            else:
                print("  "*depth + f'{a["name"]}  {a.get("size",0)/1e6:.1f} MB  {f["links"]["download"]}')
        url = d["links"].get("next")
for node in sys.argv[1:]:
    print("==", node)
    ls(f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/?page[size]=100", 0, 2)
