#===================================================================================================================
# TERMS AND CONDITON: IF YOU ARE USING THIS CODE PLEASE CITE:
#===================================================================================================================

# @article{noii2024efficient,
#  title={An Efficient FEniCS implementation for coupling lithium-ion battery charge/discharge processes with fatigue phase-field fracture},
#  author={Noii, Nima and Milijasevic, Dejan and Khodadadian, Amirreza and Wick, Thomas},
#  journal={Engineering Fracture Mechanics},
#  pages={110251},
#  year={2024},
#  publisher={Elsevier}
#}

#@article{noii2024fatigue,
#  title={Fatigue failure theory for lithium diffusion induced fracture in lithium-ion battery electrode particles},
#  author={Noii, Nima and Milijasevic, Dejan and Waisman, Haim and Khodadadian, Amirreza},
#  journal={Computer Methods in Applied Mechanics and Engineering},
#  volume={428},
#  pages={117068},
#  year={2024},
#  publisher={Elsevier}
#}

#===================================================================================================================
# The code is related to the following paper:
# Implementation of Charge/Discharge Processes for Fracture Mechanism in Lithium-Ion Battery
# submitted to Theoretical and Applied Fracture Mechanics

# The authors are:
# Nima Noii, Dejan Milijasevic, Amirreza Khodadadian and Nikolaos Bouklas

# This script is related to 2D battery fatigue fracture

# @copyright(2024) V1.1-NN

#===============================================================================
# Equation to be solved:
# (E): \[\int_\calB\big[\Bsigma:\Bve(\delta\Bu)
#-\overline{\Bf}\cdot\delta\Bu\big]~\mathrm{d}V
#- \int_{\partial\calB^{\Bu}_N}\overline{\Btau}\cdot\delta\Bu~\mathrm{d}A=0 \]

# (C): \[\displaystyle\int_\calB \Big[\Big(&(c-c_n) 
#-\tau \;\bar{r}_F\Big)\delta c
#+ (\tau \;\BK\MB) \cdot \delta \nabla c \Big] dV \\
#& + \displaystyle\int_{\partial^c_N\calB} \tau \bar{H} \;\delta c\; dA = 0 \]

# (D): \int_\calB \bigg[ \big(\,(1-d)\calH -  {d}+\frac{\eta_d}{\tau} (d-d_n)  \bigg)
#\delta d\,  - l_d^2  \nabla d \cdot \nabla &(\delta d)  \bigg]\,~\mathrm{d}V = 0

#===============================================================================
# Output:

# To approximate the material properties:
# [u(x,t),d(x,t),c(x,t)]

#===============================================================================
# ------------------------------------------------
# Import labraries
# ------------------------------------------------
from __future__ import print_function
from fenics import *
import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import time
import sys
set_log_active(False)

# ------------------------------------------------
# Read mesh file
# ------------------------------------------------

mesh = Mesh()

with XDMFFile("circular_battery_RVEonesize.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 1)

with XDMFFile("circular_battery_RVEonesize_mt.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
# ------------------------------------------------


ds = Measure('ds',subdomain_data=mf)
n = FacetNormal(mesh)


class LogicDomain(SubDomain):
    def __init__(self, condition):
        self.condition = condition
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return self.condition(x)    

def material_function(target_mesh, material_values):
    V = FunctionSpace(target_mesh, 'DG', 0)
    mat_func = Function(V)
    tmp = np.asarray(sub_domains.array(), dtype=np.int32)
    mat_func.vector()[:] = np.choose(tmp, material_values)
    return mat_func

class Problem(NonlinearProblem):
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A) 
            
def project2(v, target_func, bcs=[]):
    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    dX = dx(V.mesh)

    # Define variational problem for projection
    w = TestFunction(V)
    Pv = TrialFunction(V)
    a = form(inner(Pv, w) * dX)
    L = form(inner(v, w) * dX)

    # Assemble linear system
    A = assemble_matrix(a, bcs)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)
            
# ------------------------------------------------
# Define Space
# ------------------------------------------------

# for phase field variable: d(x,t)
V_d = FunctionSpace(mesh,'CG',1)
dd , δd = TrialFunction(V_d), TestFunction(V_d)

# for displacement: u(x,t)
V_u = VectorFunctionSpace(mesh, "CG", 1)
du = TrialFunction(V_u)
δu = TestFunction(V_u)

# for history variable: H_n(x,t)
WW = FunctionSpace(mesh, 'DG', 0)

# for concentration: c(x,t)
V_c = FunctionSpace(mesh, 'CG', 1)
dc, δc  = TrialFunction(V_u), TestFunction(V_c)

# for Stress Tensor field: sig(x,t)
TT = TensorFunctionSpace(mesh, 'CG', 1)


# ----------------------------------------
# Define necessary functions
# ----------------------------------------

#displacement
u, u_n = Function(V_u), Function(V_u)

#damage
d = Function(V_d)

#concentration
c, c_n = Function(V_c), Function(V_c)

#fatigue
alph, alph_n, Psip_n, H_n  = Function(V_d), Function(V_d), Function(V_d), Function(WW)

#stress
sig = Function(TT)

# ------------------------------------------------
# Parameters
# ------------------------------------------------
# phasefield properties
Gc =  1
lc = 2.*mesh.hmax()
kappa_d = 1e-8

# elastic properties
E=9.3e+4
nu=0.3
lmbda = E*nu/(1.0+nu)/(1.0-2.0*nu)
mu = E/2.0/(1.0+nu)
kappa = lmbda + 2./3.*mu

# lithium ion properties
DB = 15e-12
c2 = 1e-23
c0 = 300
ΩB = 3*8e-6

# fatigue properties
epsy  = sqrt(Gc/(3.*lc*E))
alp_T = (0.5*epsy*E*epsy)/5
k_fg  = 0.5

# sigma split
# 1: hybrid method
# 2: Amore's method: vol/dev-split
split = 1

# ------------------------------------------------
# Loading and solving parameters
# ------------------------------------------------

# Battery
delta_c = 10000
c_load = 0
t=0
tau_c_eq= 200


# Cyclic loading
Uincr = delta_c

nq     = 3
nCycle = 400
qper   = nq*Uincr
tper   = 4.*qper
Tmax   = nCycle*tper
nStep  = nCycle*4*nq

usgn   = 1.
Umax   = 37*Uincr
u_max  =  Umax
u_min  =  Umax
u_r    =  0

j = 0
conc = 0
innercounter = 0
outercounter = 0

# ------------------------------------------------
# Constitutive functions
# ------------------------------------------------

def LOG(f_):
    return ln(f_)/2.303

def Pls(u_):
    return (u_+abs(u_))/2

# total strain tensor
def epsilon(u_):
    return sym(grad(u_))

# lithium ion strain tensor
def epsilon_Battery(c_):
    return (ΩB/3)*(c_-c0)*Identity(2)

# elastic strain tensor
def epsilon_elastic(u_, c_):
    return epsilon(u_) - epsilon_Battery(c_)

# deviatoric part of elastic strain tensor
def straindev_conc(u_, c_):
    return epsilon_elastic(u_, c_)-(1.0/3.0)*tr(epsilon_elastic(u_, c_))*Identity(2)

# effective stress tensor
def sigma_eff(u_, c_):
    return  2.0*mu*epsilon_elastic(u_, c_) + lmbda*tr(epsilon_elastic(u_,c_))*Identity(2)

# hydrostatic part of effective stress tensor
def sigma_eff_p(u_, c_):
    return  tr(sigma_eff(u_, c_))/3

# hybdrid approach or Amore's approach
if split == 1:
    def sigma(u_, d_, c_): 
        return g(d_)*sigma_eff(u_, c_)
elif split == 2:
    def sigma(u_, d_, c_): 
        A=2.0*mu_*straindev_conc(u_, c_)
        B=kappa_*tr(epsilon_elastic(u_, c_))*Identity(2)
        cond=tr(epsilon_elastic(u_, c_))
        return conditional(ge(cond, 0.0), g(d_)*(A+B),g(d_)*A+B)

# degradation function
def g(d_):
    return (1-kappa_d)*(1-d_)**2+kappa_d

# fatigue degradation function
def falp(f_):
    tolf = 1E-15
    af = conditional(le(f_,alp_T), 1., tolf) \
        + conditional(gt(f_,alp_T), (1.-k_fg*LOG(f_/alp_T))**2, tolf) \
        + conditional(ge(f_,alp_T*10**(1/k_fg)),-(1-k_fg*LOG(f_/alp_T))**2, tolf)    
    return af

# degraded Griffith's criterion
Gf = Gc*falp(alph)

# effective elastic energy
def psip_eff(u_, c_):
    tre=tr(epsilon_elastic(u_, c_))
    dev_et=dev(epsilon_elastic(u_, c_))
    return (0.5*kappa*(Pls(tre))**2+mu*inner(dev_et,dev_et))

# crack driving force
def psip(u_, c_):   
    return (2*lc/Gf)*psip_eff(u_, c_)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Fatigue

def Psip(u_, d_, c_):
    return g(d_)*psip_eff(u_, c_)

def dalph(u_, d_, c_, Psip_n):
    HSF = conditional(lt(Psip_n,Psip(u_, d_, c_)), 1., 1e-15)
    dalp =  HSF*(Psip(u_, d_, c_) - Psip_n)
    return dalp

# Update fatigue paramater: alpha
def cal_fat_param(u_, d_, c_, Psip_n_,alph_n):
    return alph_n + dalph(u_, d_, c_, Psip_n_)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~History field
def H(u_,c_):
    return conditional(lt(H_n, psip(u_,c_)), psip(u_,c_), H_n)

def dmin(x, y):
    z = as_backend_type(x).vec().duplicate()
    z.pointwiseMin(as_backend_type(x).vec(),
                   as_backend_type(y).vec())
    return PETScVector(z)

def dmax(x, y):
    z = as_backend_type(x).vec().duplicate()
    z.pointwiseMax(as_backend_type(x).vec(),
                   as_backend_type(y).vec())
    return PETScVector(z)

# ----------------------------
# Define Boundary Condition
# ----------------------------

c0_initial = Constant(c0)
c_n = interpolate(c0_initial, V_c)
c_D1 =  Expression("t",t = 0.0, degree = 0)
bc_c = [DirichletBC(V_c, c_D1, mf, 1)]  # label 1 are 2D elements at surface of the circle

bc_u = []

bc_phi = DirichletBC(V_d, Constant(1), mf, 2) # label 2 are 2D elements at initial notches



# --------------------------------------------------------------------------------------
# Define variational form of Displacement + corresponding solver
# --------------------------------------------------------------------------------------

E_u = inner(sigma(du, d, c), epsilon(δu))*dx
problem_disp = LinearVariationalProblem(lhs(E_u), rhs(E_u), u, bc_u)
solver_disp = LinearVariationalSolver(problem_disp)
solver_disp.parameters["linear_solver"] = "mumps"

# --------------------------------------------------------------------------------------
# Define variational form of phi + corresponding solver
# --------------------------------------------------------------------------------------

E_phi = (-lc**2*dot(grad(dd),grad(δd))-dd*δd+(1-kappa_d)*(1-dd)*H(u,c)*δd)*dx
problem_phi = LinearVariationalProblem(lhs(E_phi), rhs(E_phi), d, bc_phi)
solver_phi = LinearVariationalSolver(problem_phi)
solver_phi.parameters["linear_solver"] = "mumps"

# --------------------------------------------------------------------------------------
# Define variational form of Concentration
# --------------------------------------------------------------------------------------

E_conc = (((c - c_n)/tau_c_eq)*δc + (g(d)*DB * dot(grad(c), grad(δc)))-(c*c2)*dot(grad(g(d)*sigma_eff_p(u, c)), grad(δc)))*dx


# ----------------------------------------
# Creating files for saving the results
# ----------------------------------------

file_p = XDMFFile("./results/phi.xdmf")
file_c = XDMFFile("./results/conc.xdmf")
file_d = XDMFFile("./results/disp.xdmf")

File("./results/Facet_tags.pvd").write(mf)


# ----------------------------------------
# Solving
# ----------------------------------------

while (j < nStep):
    print("Time Step " + str(j/(4*nq)))
    
    # ----------------------------------------
    # Update BC/fatigue loading
    # ----------------------------------------
    t += Uincr
    j +=1
    tloc = t
    ncycle = 1
    while(tloc >= tper):
        tloc = tloc - tper
        ncycle+= 1
    if (tloc < qper):
        c_load = usgn*((u_max - u_r)*tloc/qper + u_r)
    elif (tloc >= qper) & (tloc < 2.*qper):
        c_load = usgn*((u_r-u_max)/qper*(tloc-qper) + u_max)
    elif (tloc >= 2.*qper) & (tloc < 3.*qper):
        c_load = usgn*((u_min-u_r)/qper*(tloc-2.*qper) + u_r)
    elif (tloc >= 3.*qper) & (tloc < tper):
        c_load = usgn*((u_r-u_min)/qper*(tloc-3.*qper) + u_min)
    c_D1.t = c0+c_load
    i_ud=0
    
    # ----------------------------------------
    # solving incremental approach for the rate dependent
    # multiphysics chemo-mechanical problem induced 
    # fatigue fracture
    # We aim to solve (u(x,t), d(x,t), c(x,t))
    # ----------------------------------------
    
    while (i_ud < 2):
        i_ud += 1
        i = 0
        while (i < 2):
            i += 1
            
            # ----------------------------------------
            # Solving displacement/concentration problem
            # ----------------------------------------
            solve(E_conc == 0, c, bc_c, solver_parameters={"newton_solver": {'linear_solver' : 'mumps'}})
            
            solver_disp.solve()
            
        # ----------------------------------------
        # Solving phasefield problem
        # ----------------------------------------
        solver_phi.solve()
        
        # ----------------------------------------
        # update fatigue parameter
        # ----------------------------------------
        alph_project = project(cal_fat_param(u, d, c, Psip_n, alph_n), V_d)
        alph.assign(alph_project)
        
    # ----------------------------------------
    # Updating histroy variables
    # ----------------------------------------
    # lithium concentration
    c_n.assign(c)
    
    # history field phase field
    H_project = project(H(u,c), WW)
    H_n.assign(H_project)
    
    # fatigue parameter
    alph_n.assign(alph)
    
    Psip_project = project(Psip(u,d,c), V_d)
    Psip_n.assign(Psip_project)
    
    # ----------------------------------------
    # Saving results
    # ----------------------------------------
    file_p.write(d, t)
    file_c.write(c, t)
    file_d.write(u, t)
        
print ('Simulation completed')
