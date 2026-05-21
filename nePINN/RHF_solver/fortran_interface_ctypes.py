"""
Fortran Interface (ctypes) — Bridge between Python PPO and Core-1204 Fortran RHF kernel

Provides:
  FortranRHFCalculator: low-level ctypes wrapper around librhf.so
  FortranStateResult: structured data container for env.py
  FortranOrbitalInfo: orbital metadata compatible with OrbitalInfo
  fortran_state_to_tensor(): convert FortranStateResult → dict for StateTensor
"""
import os, ctypes, numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

# ====== Library & Constants ======
_SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Core-1204', "librhf.so")
_M = 201; _Pfx = "__ddrhf_wrapper_MOD_"
_lib = ctypes.CDLL(_SO)
_Ip = ctypes.POINTER(ctypes.c_int)
_Dp = ctypes.POINTER(ctypes.c_double)
_Fp = np.ctypeslib.ndpointer(dtype=np.float64, shape=(_M,), flags='C_CONTIGUOUS')

def _F(n): return _lib[(_Pfx+n).encode()]

# ====== Data Classes ======

@dataclass
class FortranOrbitalInfo:
    """Orbital info from Fortran kernel (compatible with OrbitalInfo interface)"""
    name: str; n: int; kappa: int; l: int; tau3: int
    energy: float; occ: float; deg: int

@dataclass
class FortranStateResult:
    """
    Complete state from one Fortran init/step call.
    Used by env.py _reset_fortran / _step_fortran.
    """
    E_total: float          # etot (MeV)
    E_per_A: float          # binding per nucleon
    E_kinetic: float        # kinetic energy
    E_direct: float         # Hartree direct
    E_exchange: float       # Fock exchange
    E_rearrange: float      # rearrangement term
    iteration: int          # iteration number
    convergence: float      # si value (max|ΔV|)
    npt: int                # grid points
    dr: float               # grid spacing
    grid: np.ndarray        # radial coordinates (npt,)
    sigma: np.ndarray       # σ field (npt,)
    omega: np.ndarray       # ω field (npt,)
    rho_field: np.ndarray   # ρ field (npt,)
    coulomb: np.ndarray     # Coulomb field (npt,)
    rho_s: np.ndarray       # scalar density (npt,)
    rho_b: np.ndarray       # baryon density (npt,)
    rho_b3: np.ndarray      # isovector density (npt,)
    wavefunctions_n: Dict   # {name: {'G','F'}}
    wavefunctions_p: Dict   # {name: {'G','F'}}
    orbitals_n: List        # [FortranOrbitalInfo]
    orbitals_p: List        # [FortranOrbitalInfo]


# ====== Main Calculator Class ======

class FortranRHFCalculator:
    """ctypes bridge to librhf.so (Core-1204 Fortran RHF engine)"""

    PM = {"PKA1":0,"PKO1":1,"PKO2":2,"PKO3":3,
          "DDME1":4,"DDME2":5,"PKDD":6,"TW99":7,"DDLZ1":8}

    def __init__(self, ps="PKA1", Z=8, N=8, A=16):
        self.ps = ps.upper(); self.Z=Z; self.N=N; self.A=A
        if self.ps not in self.PM: raise ValueError(self.ps)
        self.idx = self.PM[self.ps]
        self._ok = False; self._it = 0
        print(f"[Fortran] {self.ps} Z={Z}N={N}A={A}")

    def initialize(self):
        """Full initialization: Config → PreMedia → PrePotels → DBASE → occup→Densit→Potel→Expect"""
        f=_F("ddrhf_init"); f.argtypes=[_Ip]*4; f.restype=None
        f(ctypes.c_int(self.idx),ctypes.c_int(self.Z),
           ctypes.c_int(self.N),ctypes.c_int(self.A))
        self._ok=True; self._it=0
        return self._build_result()

    def iterate(self):
        """One self-consistent iteration step: occup→Densit→Potel→Expect→Detgff/Dirac"""
        if not self._ok: raise RuntimeError("Call initialize() first")
        f=_F("ddrhf_step"); f.argtypes=[_Dp]; f.restype=None
        si=ctypes.c_double(0); f(ctypes.byref(si)); self._it+=1
        return self._build_result()

    def run_to_convergence(self):
        """Run full convergence loop internally (for validation)"""
        f=_F("ddrhf_run"); f.argtypes=[_Ip]*4; f.restype=None
        f(ctypes.c_int(self.idx),ctypes.c_int(self.Z),
           ctypes.c_int(self.N),ctypes.c_int(self.A))
        self._ok=True; self._it=-1
        return self._build_result()

    # ---- Backward-compatible aliases ----
    init = initialize
    step = iterate
    run = run_to_convergence

    def _build_result(self):
        """Extract all state from Fortran module globals into FortranStateResult"""
        Z=np.zeros(_M,dtype=np.float64);xr=Z.copy();n=ctypes.c_int(0);d=ctypes.c_double(0)
        f=_F("ddrhf_extract_grid");f.argtypes=[_Fp,_Ip,_Dp];f.restype=None;f(xr,n,d);nv=n.value

        s=o=r=c=Z.copy()
        f=_F("ddrhf_extract_fields");f.argtypes=[_Fp]*4;f.restype=None;f(s,o,r,c)

        rs=rb=r3=Z.copy()
        f=_F("ddrhf_extract_densities");f.argtypes=[_Fp]*3;f.restype=None;f(rs,rb,r3)

        wn,on=self._wfs(1);wp,op=self._wfs(2)

        # Each c_double must be separate object (Python chain-alias bug!)
        t=ctypes.c_double(0);p=ctypes.c_double(0);k=ctypes.c_double(0)
        d_=ctypes.c_double(0);e=ctypes.c_double(0);x=ctypes.c_double(0)
        f=_F("ddrhf_get_energy");f.argtypes=[_Dp]*6;f.restype=None;f(t,p,k,d_,e,x)
        re_val = x.value - (t.value + p.value + k.value + d_.value + e.value)

        i=ctypes.c_int(0);cv=ctypes.c_double(0)
        try:
            f=_F("ddrhf_get_iter_info");f.argtypes=[_Ip,_Dp];f.restype=None;f(i,cv)
        except Exception:
            pass

        return FortranStateResult(
            E_total=t.value, E_per_A=p.value,
            E_kinetic=k.value, E_direct=d_.value,
            E_exchange=e.value, E_rearrange=re_val,
            iteration=i.value, convergence=cv.value,
            npt=nv, dr=d.value,
            grid=xr[:nv].copy(),
            sigma=s[:nv].copy(), omega=o[:nv].copy(),
            rho_field=r[:nv].copy(), coulomb=c[:nv].copy(),
            rho_s=rs[:nv].copy(), rho_b=rb[:nv].copy(), rho_b3=r3[:nv].copy(),
            wavefunctions_n=wn, wavefunctions_p=wp,
            orbitals_n=on, orbitals_p=op,
        )

    def _wfs(self, it):
        """Extract all wavefunctions for isospin it (1=n, 2=p)"""
        f=_F("ddrhf_get_norbits");f.argtypes=[_Ip];f.restype=ctypes.c_int
        nt=f(ctypes.c_int(it));wf={};obs=[]
        nb=ctypes.create_string_buffer(32);G=F=np.zeros(_M,dtype=np.float64)
        for i in range(1, nt+1):
            nl=ctypes.c_int(32);ka=l=n=ctypes.c_int(0)
            ee=v=ctypes.c_double(0);mu=ctypes.c_int(0)
            f=_F("ddrhf_extract_wavefunction")
            f.argtypes=[_Ip,_Ip,_Fp,_Fp,ctypes.c_char_p,_Ip,_Ip,_Ip,_Ip,_Dp,_Dp,_Ip]
            f.restype=ctypes.c_int
            f(ctypes.c_int(i),ctypes.c_int(it),G,F,nb,nl,ka,l,n,ee,v,mu)
            nm=nb.value.decode().strip() or f"orb_{i}_{it}"
            wf[nm]={'G':G.copy(),'F':F.copy()}
            obs.append(FortranOrbitalInfo(
                nm, n.value, ka.value, l.value,
                1 if it==1 else -1, ee.value, v.value, mu.value))
        return wf, obs


# ====== Converter for env.py StateTensor ======

def fortran_state_to_tensor(fs: FortranStateResult) -> dict:
    """Convert FortranStateResult → dict compatible with StateTensor constructor.

    Returns dict with keys: wavefunctions_n/p, sigma, omega, rho_field,
                             coulomb, rho_s, rho_b, rho_b3
    """
    return {
        'wavefunctions_n': fs.wavefunctions_n,
        'wavefunctions_p': fs.wavefunctions_p,
        'sigma': fs.sigma, 'omega': fs.omega,
        'rho_field': fs.rho_field, 'coulomb': fs.coulomb,
        'rho_s': fs.rho_s, 'rho_b': fs.rho_b,
        'rho_b3': fs.rho_b3,
    }


# ====== CLI Test ======
if __name__ == "__main__":
    fc = FortranRHFCalculator(ps="PKA1", Z=8, N=8, A=16)
    st = fc.initialize()
    print(f"npt={st.npt} orbits n={len(st.orbitals_n)} p={len(st.orbitals_p)}")
    print(f"E_total={st.E_total:.2f} MeV | B/A={st.E_per_A:+.4f}")
    st2 = fc.iterate()
    print(f"After step: E={st2.E_total:.2f} conv={st2.convergence:.4e}")
    res = fc.run_to_convergence()
    print(f"Final: B/A={res.E_per_A:+.4f} after iter={res.iteration}")
    print("\n=== PASSED ===")
