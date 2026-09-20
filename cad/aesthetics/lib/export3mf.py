"""Write a Bambu/Orca project 3MF: one object, N parts, each pre-assigned a filament."""
import sys, zipfile, numpy as np, pickle
from xml.sax.saxutils import escape
sys.path.insert(0,".")
from build123d import import_step
from render3d import tessellate

CT = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>'''
RELS = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''

def write_3mf(parts, out, name="Part", plate=(128.0,128.0)):
    """parts: list of (partname, verts Nx3, tris Mx3, filament_index)"""
    allv = np.vstack([v for _,v,_,_ in parts])
    off = np.array([plate[0]-(allv[:,0].min()+allv[:,0].max())/2,
                    plate[1]-(allv[:,1].min()+allv[:,1].max())/2,
                    -allv[:,2].min()])
    res, asm = [], []
    for i,(pn,V,T,fil) in enumerate(parts, start=1):
        W = V + off
        vx = "".join(f'<vertex x="{a:.5f}" y="{b:.5f}" z="{c:.5f}"/>' for a,b,c in W)
        tx = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a,b,c in T)
        res.append(f'<object id="{i}" type="model"><mesh><vertices>{vx}</vertices>'
                   f'<triangles>{tx}</triangles></mesh></object>')
        asm.append(f'<component objectid="{i}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>')
    aid = len(parts)+1
    res.append(f'<object id="{aid}" type="model"><components>{"".join(asm)}</components></object>')
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">'
        '<metadata name="Application">BuiltByClaude</metadata>'
        f'<resources>{"".join(res)}</resources>'
        f'<build><item objectid="{aid}" transform="1 0 0 0 1 0 0 0 1 0 0 0" printable="1"/></build></model>')
    cfg = ['<?xml version="1.0" encoding="UTF-8"?>', '<config>',
           f'  <object id="{aid}">', f'    <metadata key="name" value="{escape(name)}"/>']
    for i,(pn,_,_,fil) in enumerate(parts, start=1):
        cfg += [f'    <part id="{i}" subtype="normal_part">',
                f'      <metadata key="name" value="{escape(pn)}"/>',
                f'      <metadata key="extruder" value="{fil}"/>',
                f'    </part>']
    cfg += ['  </object>', '</config>']
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CT)
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", model)
        z.writestr("Metadata/model_settings.config", "\n".join(cfg))
    return out

if __name__ == "__main__":
    FIL = {"core":1, "bot":2, "rim":2, "inlay":1, "accent":3}   # 1=graphite 2=white 3=blue
    parts=[]
    for n,f in FIL.items():
        V,T,_ = tessellate(import_step(f"body_{n}.step"), 0.08)
        parts.append((n,V,T,f)); print(f"  {n:7s} filament {f}  {len(V):6d} v {len(T):6d} t")
    p = write_3mf(parts, "Tibia_multicolor.3mf", name="Tibia v4 (multicolour)")
    import os; print(f"wrote {p}  {os.path.getsize(p)/1024:.0f} KB")
    with zipfile.ZipFile(p) as z:
        for i in z.infolist(): print(f"   {i.filename:38s} {i.file_size:>9d} B")
