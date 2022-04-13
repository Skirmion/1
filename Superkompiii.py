import numpy as np

def dump_step(filename, r, i_step):
    f_out = open(filename,'a')
    f_out.write("ITEM: TIMESTEP\n")
    f_out.write("{}\n".format(i_step))
    f_out.write("ITEM: NUMBER OF ATOMS\n")
    f_out.write("{}\n".format(r.shape[0]))
    f_out.write("ITEM: BOX BOUNDS pp pp pp\n")
    f_out.write("{} {}\n".format(0,bounds[0]))
    f_out.write("{} {}\n".format(0,bounds[1]))
    f_out.write("{} {}\n".format(0,bounds[2]))
    f_out.write("ITEM: ATOMS id x y z\n")
    for i in range(r.shape[0]):
        f_out.write("{} {} {} {}\n".format(i,r[i][0],r[i][1],r[i][2]))
    f_out.close()
    return(0)


dt = 0.005
bounds = np.array([40,40,40])
N_atoms = 125
epcilon = 1000
sigma = 1

r = np.array([[i, j, k] for i in np.linspace(10,30,5) for j in np.linspace(10,30,5) for k in np.linspace(10, 30,5)])
v = np.random.normal(loc = 0.0, scale = 0, size = (N_atoms, 3))
r_neig = np.ones((N_atoms, N_atoms, 3))


def neighbor(r_neig):
    for i in range(N_atoms):
        for j in range(N_atoms):
            r_neig[i][j] = np.where(r_neig[i][j] > 20, r_neig[i][j] - 40, r_neig[i][j])
            r_neig[i][j] = np.where(r_neig[i][j] < 20, r_neig[i][j] + 40, r_neig[i][j])
    return(r_neig)


def poten_model(r, r_neig):
    for i in range(N_atoms):
        for j in range(N_atoms):
            if (r_neig[i][j].all() != r[j].all()):
                #r_neig = np.where(r_neig[i] != r, r, r_neig[i])
                r_neig[i][j] = r[j]
    
    r_neig = neighbor(r_neig)
    a = force_calc(r_neig, r)
    return(a)
 
def force_calc(rcut, r):
    a = np.zeros((N_atoms,3))
    for i in range(N_atoms):
        for j in range(i, N_atoms):
            del_a = LJ(r_neig, i, j, r)
            a[i] = a[i] + del_a
            a[j] = a[j] - del_a
    return a

def LJ(r_neig, i, j, r):
    a_12 = np.zeros((1, 3))
    if (i != j):
        rad = len(r[i] - r_neig[i][j])
        a_12 = 4*epcilon*((r[i] - r_neig[i][j])/rad**2)*(6*(sigma/rad)**6 - 12*(sigma/rad)**12)
    return a_12


def leapfrog_step(r, v, r_neig):
    a = poten_model(r, r_neig)
    v += a*dt
    r += v*dt
    for i in range(N_atoms):
        r_neig[i] += v*dt


def wrap_periodic(r):
    r = np.where(r > bounds, r - bounds, r)
    r = np.where(r < 0, r + bounds, r)
    return(r)


def ekin(v):
    v_buf = 0.0
    for i in range(N_atoms):
        v_buf += (len(v[i])**2)/2
    return v_buf


def epot(r, r_neig):
    u_buf = 0.0
    for i in range(N_atoms):
        for j in range(i, N_atoms):
            if (i != j):
                radius = len(r[i] - r_neig[i][j])
                u_buf += 4*epcilon*((sigma/radius)**12 - (sigma/radius)**6)
    return u_buf
    


for i in range(100):
    leapfrog_step(r, v, r_neig)
    r = wrap_periodic(r)
    dump_step("dump.txt", r, 0)
    energy = ekin(v) + epot(r, r_neig)
    #print (energy)
    


def len(r):
    len1 = (r[1]**2 + r[2]**2 + r[3]**2)**0.5
    return(len1)
