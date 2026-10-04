"""Step 3 of the SolidWorks automation ladder: read the robot assembly. Changes nothing.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/03_read_assembly.py

Uses ROBOT.SLDASM if it is already open (it is never reopened or saved);
otherwise opens it READ-ONLY.  Prints the configurations, the top-level
components, and every top-level mate with the components it ties together --
which is what step 4 needs to find what drives the hip.
"""
import os
import sys
import math
import win32com.client
from win32com.client import gencache, constants as c

sys.stdout.reconfigure(encoding="utf-8")

sldworks = gencache.EnsureModule("{83A33D31-27C5-11CE-BFD4-00400513BB57}", 0, 31, 0)
gencache.EnsureModule("{4687F359-55D0-4CD3-B6CF-2EB42C11F989}", 0, 31, 0)

HERE = os.path.dirname(os.path.abspath(__file__))
ASM = os.path.normpath(os.path.join(HERE, "..", "v4 Larger Ball bearings", "ROBOT.SLDASM"))


def wrap(obj, cls):
    """Everything SolidWorks returns as a bare IDispatch must be re-typed."""
    return None if obj is None else cls(obj._oleobj_)


sw = sldworks.ISldWorks(win32com.client.GetActiveObject("SldWorks.Application")._oleobj_)
print("connected to SolidWorks", sw.RevisionNumber())

model = None
for d in sw.GetDocuments() or []:
    d = wrap(d, sldworks.IModelDoc2)
    if os.path.normcase(d.GetPathName()) == os.path.normcase(ASM):
        model = d
        print("using the already-open", d.GetTitle(), "(read only, never saved)")
        break
if model is None:
    doc, err, warn = sw.OpenDoc6(ASM, c.swDocASSEMBLY,
                                 c.swOpenDocOptions_ReadOnly | c.swOpenDocOptions_Silent,
                                 "", 0, 0)
    if doc is None:
        raise SystemExit(f"could not open {ASM} (errors={err}, warnings={warn})")
    model = wrap(doc, sldworks.IModelDoc2)
    print("opened READ-ONLY:", model.GetTitle())

# --- configurations ---------------------------------------------------------
active = wrap(model.GetActiveConfiguration(), sldworks.IConfiguration)
print(f"\nconfigurations (active = {active.Name}):")
for n in model.GetConfigurationNames() or []:
    print("  ", n)

# --- top-level components ---------------------------------------------------
asm = wrap(model, sldworks.IAssemblyDoc)
comps = [wrap(x, sldworks.IComponent2) for x in asm.GetComponents(True) or []]
print(f"\n{len(comps)} top-level component(s):")
for cp in sorted(comps, key=lambda k: k.Name2):
    flags = []
    if cp.IsFixed():
        flags.append("FIXED")
    if cp.IsSuppressed():
        flags.append("suppressed")
    if cp.IsHidden(False):
        flags.append("hidden")
    kids = len(cp.GetChildren() or [])
    sub = f"sub-assembly, {kids} children" if kids else "part"
    print(f"  {cp.Name2:34s} {sub:24s} {' '.join(flags)}")

# --- top-level mates --------------------------------------------------------
def mate_detail(feat):
    """Components a mate ties together, plus its value / limits if it has any."""
    m = wrap(feat.GetSpecificFeature2(), sldworks.IMate2)
    names = []
    for i in range(m.GetMateEntityCount()):
        rc = m.MateEntity(i).ReferenceComponent
        names.append(rc.Name2 if rc is not None else "(assembly)")
    extra = ""
    lo, hi = m.MinimumVariation, m.MaximumVariation
    if lo or hi:
        ang = feat.GetTypeName2().find("Angle") >= 0
        f = (lambda v: f"{math.degrees(v):.2f} deg") if ang else (lambda v: f"{v*1e3:.2f} mm")
        extra = f"  LIMITS {f(lo)} .. {f(hi)}"
    dd = m.DisplayDimension
    if dd is not None:
        dim = wrap(wrap(dd, sldworks.IDisplayDimension).GetDimension2(0), sldworks.IDimension)
        extra = f"  value {dim.GetSystemValue3(c.swThisConfiguration, None)[0]:.6g} (SI)" + extra
    return " <-> ".join(names), extra


print("\ntop-level mates:")
n_mates = 0
feat = wrap(model.FirstFeature(), sldworks.IFeature)
while feat is not None:
    if feat.GetTypeName2() == "MateGroup":
        sub = wrap(feat.GetFirstSubFeature(), sldworks.IFeature)
        while sub is not None:
            kind = sub.GetTypeName2()
            sup = "  [SUPPRESSED]" if sub.IsSuppressed() else ""
            if kind.startswith("Mate"):
                n_mates += 1
                try:
                    who, extra = mate_detail(sub)
                except Exception as e:                       # report, don't hide
                    who, extra = f"(could not read: {e})", ""
                print(f"  {sub.Name:30s} {kind:22s} {who}{extra}{sup}")
            else:
                print(f"  {sub.Name:30s} {kind:22s} (not a mate){sup}")
            sub = wrap(sub.GetNextSubFeature(), sldworks.IFeature)
    feat = wrap(feat.GetNextFeature(), sldworks.IFeature)
print(f"\n{n_mates} mate(s)")
