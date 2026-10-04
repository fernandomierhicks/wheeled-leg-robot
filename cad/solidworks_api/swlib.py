"""Shared plumbing for the SolidWorks scripts (steps 4+).

Steps 1-3 inline this on purpose so each reads on its own; from step 4 on the
same few calls repeat in every script.

    sw, sld = connect()
    model = open_v5()          # ROBOT.SLDASM in cad/v5 Ai designed, refs checked
"""
import os
import sys
import math
import numpy as np
import pythoncom
import win32com.client
from win32com.client import gencache, constants as c

sys.stdout.reconfigure(encoding="utf-8")

sld = gencache.EnsureModule("{83A33D31-27C5-11CE-BFD4-00400513BB57}", 0, 31, 0)
gencache.EnsureModule("{4687F359-55D0-4CD3-B6CF-2EB42C11F989}", 0, 31, 0)

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.normpath(os.path.join(HERE, "..", "v5 Ai designed"))
ASM = os.path.join(V5, "ROBOT.SLDASM")
HIP_MATE = "LimitAngle1"
NOT_FOR_COLLISION = ("AK_SIM-1", "WheelHanger-1")     # his call, 2026-10-02


def wrap(obj, cls):
    """Everything SolidWorks returns as a bare IDispatch must be re-typed.

    QueryInterface first: an object's IDispatch only speaks ITS OWN interface.
    A sketch from SketchManager.ActiveSketch is an ISketch; handing that pointer
    to sld.IFeature unconverted sends IFeature's method ids to ISketch, and
    feat.Select2() fails with "Invalid number of parameters".  When the object
    does not support the interface at all, fall back to the raw pointer.
    """
    if obj is None:
        return None
    disp = obj._oleobj_
    try:
        disp = disp.QueryInterface(cls.CLSID, pythoncom.IID_IDispatch)
    except Exception:
        pass
    return cls(disp)


def connect():
    """Attach to the running SolidWorks.

    GetActiveObject only finds a SolidWorks the user started.  One launched
    from code runs with -Embedding and never registers for it ("Operation
    unavailable"); Dispatch reattaches to that same process instead -- and
    starts SolidWorks if none is running at all.
    """
    try:
        raw = win32com.client.GetActiveObject("SldWorks.Application")
    except Exception:
        raw = win32com.client.Dispatch("SldWorks.Application")
    sw = sld.ISldWorks(raw._oleobj_)
    sw.Visible = True
    return sw, sld


def open_v5(sw):
    """Open (or reuse) the v5 ROBOT.SLDASM and PROVE every part came from v5.

    A copied assembly can resolve its references back to the folder it was
    copied from.  Anything edited after that would land in v4, so this refuses
    to hand out the model unless every loaded component's file is under V5.
    """
    model = None
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        if os.path.normcase(d.GetPathName()) == os.path.normcase(ASM):
            model = d
    if model is None:
        doc, err, warn = sw.OpenDoc6(ASM, c.swDocASSEMBLY, c.swOpenDocOptions_Silent,
                                     "", 0, 0)
        if doc is None:
            raise SystemExit(f"could not open {ASM} (errors={err}, warnings={warn})")
        model = wrap(doc, sld.IModelDoc2)
    sw.ActivateDoc3(model.GetTitle(), False, 0, 0)
    bad = []
    for cp in components(model).values():
        p = cp.GetPathName()
        if p and not os.path.normcase(p).startswith(os.path.normcase(V5)):
            bad.append((cp.Name2, p))
    if bad:
        raise SystemExit("REFUSING: these components resolved OUTSIDE v5:\n" +
                         "\n".join(f"  {n}: {p}" for n, p in bad))
    return model


def reload_from_disk(sw, path):
    """Discard a part's unsaved state and re-read it from disk.  Returns the doc.

    ReloadOrReplace does NOTHING to a part that is loaded only as an assembly
    component (no window of its own) -- and returns no error the scripts ever
    looked at.  That is how 10_verify_styled left four links with every GL_
    feature suppressed in memory (its OFF baseline) on 2026-10-03, and a Save
    All in SolidWorks then wrote them to disk unstyled.  So: give the part a
    window first (OpenDoc6 on a loaded doc just opens one), reload, and refuse
    a non-zero result.
    """
    doc, err, warn = sw.OpenDoc6(path, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
    if doc is None:
        raise SystemExit(f"cannot open {path} (err {err})")
    code = wrap(doc, sld.IModelDoc2).ReloadOrReplace(False, path, True)
    if code not in (0, c.swDocumentNotChanged):     # 14: memory already = disk
        raise SystemExit(f"reload of {path} failed, swComponentReloadError_e {code}")
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        if os.path.normcase(d.GetPathName()) == os.path.normcase(path):
            return d
    raise SystemExit(f"{path} vanished after reload")


def components(model):
    """{Name2: IComponent2} for every component at every level."""
    asm = wrap(model, sld.IAssemblyDoc)
    out = {}
    for x in asm.GetComponents(False) or []:
        cp = wrap(x, sld.IComponent2)
        out[cp.Name2] = cp
    return out


def hip_dimension(model):
    # FeatureByName lives on IAssemblyDoc / IPartDoc, NOT on IModelDoc2
    feat = wrap(wrap(model, sld.IAssemblyDoc).FeatureByName(HIP_MATE), sld.IFeature)
    if feat is None:
        raise SystemExit(f"no mate called {HIP_MATE}")
    mate = wrap(feat.GetSpecificFeature2(), sld.IMate2)
    dim = wrap(wrap(mate.DisplayDimension, sld.IDisplayDimension).GetDimension2(0),
               sld.IDimension)
    return mate, dim


def placement(cp):
    """Component -> assembly placement as a 3x4 [R | t] in MILLIMETRES.

    SolidWorks' MathTransform.ArrayData is 9 rotation terms, 3 translation (m),
    then scale.  The rotation is stored for ROW vectors (p' = p R + t), so the
    column-vector matrix the rest of this repo uses is its transpose.  Step 4
    checks that against the STEP exports rather than trusting it.
    """
    a = np.array(cp.Transform2.ArrayData, float)
    R = a[:9].reshape(3, 3).T
    t = a[9:12] * 1000.0
    return np.hstack([R, t[:, None]])


class HipDriver:
    """Drive the hip across the WHOLE stroke.  Use this, not the hip mate's value.

    LimitAngle1 measures the angle between the assembly Top Plane and the femur
    sub-assembly's Top Plane, and its range -28..+57 crosses 0 -- where the two
    solutions of an (unsigned) angle constraint merge, so setting its value
    below 0 lands on the MIRROR solution and then stalls (README gotcha 6).

    The fix is a helper angle mate, AI_HipDrive, between the assembly RIGHT
    Plane and the same femur plane.  It reads 90 + hip, i.e. 62..147 deg over
    the stroke, never near 0 or 180, so the solver cannot jump branches.
    LimitAngle1 stays active and still enforces the hard stops (silently: a
    command past a stop just leaves the femur at the stop -- read it back).

    Measured 2026-10-02: 86 one-degree steps 57 -> -28, femur error 0.0000 deg,
    0.23 s per step.  Femur angle (assembly XY, swlib.placement) = hip - 180.
    """
    NAME = "AI_HipDrive"
    PLANES = ("Right Plane", "Top Plane@Femur-1@ROBOT")
    FEMUR = "Femur-1/Femur-1"

    def __init__(self, model):
        self.model = model
        self.asm = wrap(model, sld.IAssemblyDoc)
        self.femur = components(model)[self.FEMUR]
        mate = self._find()
        if mate is None:
            mate = self._add()
        self.dim = wrap(wrap(mate.DisplayDimension, sld.IDisplayDimension).GetDimension2(0),
                        sld.IDimension)

    def _find(self):
        f = wrap(self.asm.FeatureByName(self.NAME), sld.IFeature)
        return None if f is None else wrap(f.GetSpecificFeature2(), sld.IMate2)

    def _add(self):
        """Add the helper at the angle the leg is already at, and prove it did
        not move the femur (a wrong sign would swing it by 2x the hip angle)."""
        _, ldim = hip_dimension(self.model)
        hip = math.degrees(ldim.GetSystemValue3(c.swThisConfiguration, None)[0])
        before = self.femur_angle()
        if abs(_wrapd(before - (hip - 180.0))) > 0.01:
            raise SystemExit(f"femur at {before:.3f} deg does not match {HIP_MATE} = "
                             f"{hip:.3f}; the femur = hip - 180 relation is broken")
        m = self.model
        m.ClearSelection2(True)
        ok = (m.Extension.SelectByID2(self.PLANES[0], "PLANE", 0, 0, 0, False, 1, None, 0) and
              m.Extension.SelectByID2(self.PLANES[1], "PLANE", 0, 0, 0, True, 1, None, 0))
        if not ok:
            raise SystemExit(f"could not select {self.PLANES}")
        res = self.asm.AddMate5(c.swMateANGLE, c.swMateAlignCLOSEST, False, 0, 0, 0, 0, 0,
                                math.radians(90.0 + hip), 0, 0, False, False, 0)
        mate, err = res if isinstance(res, tuple) else (res, None)
        m.ClearSelection2(True)
        m.EditRebuild3()
        if mate is None or err not in (None, c.swAddMateError_NoError):
            raise SystemExit(f"AddMate5 failed, error {err}")
        if abs(_wrapd(self.femur_angle() - before)) > 0.01:
            raise SystemExit("adding the helper mate MOVED the femur -- sign is wrong")
        self._name_last_mate()
        return wrap(mate, sld.IMate2)

    def _name_last_mate(self):
        last = None
        f = wrap(self.model.FirstFeature(), sld.IFeature)
        while f is not None:
            if f.GetTypeName2() == "MateGroup":
                sub = wrap(f.GetFirstSubFeature(), sld.IFeature)
                while sub is not None:
                    last = sub
                    sub = wrap(sub.GetNextSubFeature(), sld.IFeature)
            f = wrap(f.GetNextFeature(), sld.IFeature)
        last.Name = self.NAME

    def femur_angle(self):
        M = placement(self.femur)
        return math.degrees(math.atan2(M[1, 0], M[0, 0]))

    def hip(self):
        """The hip angle the femur is ACTUALLY at, read from its placement."""
        return _wrapd(self.femur_angle() + 180.0)

    def set(self, hip_deg):
        """Command the hip (deg, SolidWorks sign: -28 retracted .. +57 extended).
        Returns where it actually went -- past a stop it stays at the stop."""
        self.dim.SetSystemValue3(math.radians(90.0 + hip_deg),
                                 c.swSetValue_InThisConfiguration, None)
        self.model.EditRebuild3()
        return self.hip()

    @classmethod
    def remove(cls, model):
        """Delete the helper mate, if present, so ROBOT.SLDASM can be SAVED: saved
        with it, the leg is locked at that angle.  The femur stays where it is."""
        asm = wrap(model, sld.IAssemblyDoc)
        f = wrap(asm.FeatureByName(cls.NAME), sld.IFeature)
        if f is None:
            return False
        model.ClearSelection2(True)
        f.Select2(False, 0)
        model.Extension.DeleteSelection2(0)
        model.ClearSelection2(True)
        model.EditRebuild3()
        if asm.FeatureByName(cls.NAME) is not None:
            raise SystemExit(f"could not delete {cls.NAME}")
        return True


def _wrapd(a):
    return (a + 180.0) % 360.0 - 180.0
