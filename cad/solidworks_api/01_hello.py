"""Step 1 of the SolidWorks automation ladder: connect and read. Changes nothing.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/01_hello.py

Attaches to an already-running SolidWorks (never launches one), prints its
version, and lists the open documents.

Two COM traps, both hit on the first try:
  * Late binding (plain GetActiveObject) runs a zero-argument method the moment
    it is named, so `sw.RevisionNumber()` fails with "'str' is not callable".
  * SolidWorks will not hand out type info from the live object, so
    gencache.EnsureDispatch fails with "Element not found" (the same error that
    stopped the earlier PowerShell attempt).
The fix is to generate wrappers from sldworks.tlb directly (EnsureModule, ~2 s
once, cached after) and wrap each object explicitly in its interface.
Everything SolidWorks returns comes back untyped and must be wrapped again.
"""
import sys
import win32com.client
from win32com.client import gencache

sys.stdout.reconfigure(encoding="utf-8")   # STEP/part names can be non-ASCII

# "SldWorks 2023 Type Library", sldworks.tlb, version 31.0
sldworks = gencache.EnsureModule("{83A33D31-27C5-11CE-BFD4-00400513BB57}", 0, 31, 0)

DOC_TYPES = {1: "part", 2: "assembly", 3: "drawing"}

raw = win32com.client.GetActiveObject("SldWorks.Application")
sw = sldworks.ISldWorks(raw._oleobj_)
print("connected to SolidWorks", sw.RevisionNumber())

active = sw.ActiveDoc
print("active document:",
      sldworks.IModelDoc2(active._oleobj_).GetTitle() if active else "(none)")

docs = sw.GetDocuments() or []
print(f"{len(docs)} open document(s):")
for d in docs:
    d = sldworks.IModelDoc2(d._oleobj_)
    kind = DOC_TYPES.get(d.GetType(), f"type {d.GetType()}")
    print(f"  {kind:9s} {d.GetTitle():40s} {d.GetPathName()}")
