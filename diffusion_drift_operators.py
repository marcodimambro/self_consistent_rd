from scipy import sparse as sp
import numpy as np

def build_diff_drift_matrix_sparse(x, D, beta, V):
    """
    builds operator D * (d^2/dx^2 + beta * V'' + beta * V' * grad)
    """
    nx  = len(x)
    dx  = x[1] - x[0]
    dV  = np.gradient(V, x)
    d2V = np.gradient(dV, x)

    #1) Discrete Laplacian
    lap = sp.diags(
        [ np.ones(nx-1), -2*np.ones(nx), np.ones(nx-1) ],
        offsets=[-1,0,1],
        format='csr'
    )
    lap = (1/dx**2) * lap
    lap = lap.tolil()
    lap[0, :] = np.zeros(nx)
    lap[0, 0] = -1/dx**2
    lap[0, 1] = 1/dx**2

    lap[-1, :] = np.zeros(nx)
    lap[-1, -1] = -1/dx**2
    lap[-1, -2] = 1/dx**2
    lap = lap.tocsr()
    # 2) Discrete first-derivative (skew-symmetric central difference)
    grad = sp.diags(
        [-np.ones(nx-1), np.zeros(nx), np.ones(nx-1)],
        offsets=[-1,0,1],
        format='csr'
    )
    grad = (1/(2*dx)) * grad
    grad = grad.tolil()
    grad[0, 0] = -1 / dx
    grad[0, 1] =  1 / dx
    grad[-1, -1] =  1 / dx
    grad[-1, -2] = -1 / dx
    grad = grad.tocsr()

    # dV = grad.dot(V)  # first derivative
    # d2V = grad.dot(dV)  # second derivative
    
    # 3) Derivatives of the potential
    Vprime = sp.diags(dV, 0, format='csr')
    Vsecond = sp.diags(d2V, 0, format='csr')

    # 4) Assemble operator
    #A = D * (lap + beta * Vsecond + beta * (Vprime.dot(grad)))
    A = D * (lap + beta * Vsecond + beta * Vprime * grad)

    #5) Zero flux boundary: grad p + beta * p * grad V = 0
    A = A.tolil()
    
    A[0, :]   = np.zeros(nx)
    A[0, 0]   = -1/dx + beta * dV[0]
    A[0, 1]   =  1/dx
    A[-1, :]  = np.zeros(nx)
    A[-1,-1]  =  1/dx + beta * dV[-1]
    A[-1,-2]  = -1/dx

    A = A.tocsr()
    return A


# Non uniform grid implementation --------------------------------------------------------------------------------------------------------
def laplacian_1d_nonuniform(x):
    n = len(x)
    L = sp.lil_matrix((n, n))
    for i in range(1, n - 1):
        dxm = x[i] - x[i - 1]
        dxp = x[i + 1] - x[i]
        denom = 0.5 * (dxm + dxp) * dxm * dxp
        L[i, i - 1] = 2 / (dxm * (dxm + dxp))
        L[i, i] = -2 / (dxm * dxp)
        L[i, i + 1] = 2 / (dxp * (dxm + dxp))
    return L.tocsr()

def gradient_1d_nonuniform(x):
    n = len(x)
    G = sp.lil_matrix((n, n))

    for i in range(1, n - 1):
        dxm = x[i] - x[i - 1]
        dxp = x[i + 1] - x[i]
        denom = dxm + dxp
        G[i, i - 1] = -dxp / (dxm * denom)
        G[i, i]     = (dxp - dxm) / (dxm * dxp)
        G[i, i + 1] = dxm / (dxp * denom)

    # Forward difference at the left boundary
    G[0, 0] = -1 / (x[1] - x[0])
    G[0, 1] =  1 / (x[1] - x[0])

    # Backward difference at the right boundary
    G[-1, -2] = -1 / (x[-1] - x[-2])
    G[-1, -1] =  1 / (x[-1] - x[-2])

    return G.tocsr()

def build_diff_drift_nonuniform(x, D, beta, V):
    """
    builds operator D * (d^2/dx^2 + beta * V'' + beta * V' * grad)
    for non-uniform grids
    """
    nx  = len(x)
    dx  = np.diff(x)  # prepend to keep the same length

    # Discrete Laplacian
    lap = laplacian_1d_nonuniform(x)

    # Discrete first-derivative (skew-symmetric central difference)
    grad = gradient_1d_nonuniform(x)

    # Derivatives of the potential
    dV = grad @ V  # first derivative
    d2V = lap @ V  # second derivative
    Vprime = sp.diags(dV, 0, format='csr')
    Vsecond = sp.diags(d2V, 0, format='csr')

    # Assemble operator
    A = D * (lap + beta * Vsecond + beta * Vprime * grad)

    # Zero flux boundary: grad p + beta * p * grad V = 0
    A = A.tolil()
    
    A[0, :]   = np.zeros(nx)
    A[0, 0]   = -1/dx[0] + beta * dV[0]
    A[0, 1]   =  1/dx[0]
    A[-1, :]  = np.zeros(nx)
    A[-1,-1]  =  1/dx[-1] + beta * dV[-1]
    A[-1,-2]  = -1/dx[-1]

    A = A.tocsr()
    
    return A

# Gummel method imlementation -------------------------------------------------------------------------------------------------------
def bernoulli(t):
    """Stable Bernoulli function."""
    # return np.where(np.abs(t) < 1e-14, 1 - t / 2 + t**2 /12 ,
    #                  t / (np.exp(t) - 1))
    
    # avoid error: invalid value encountered in true_divide
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            result = np.where(np.abs(t) < 1e-14, 1 - t / 2 + t**2 / 12,
                          t / (np.exp(t) - 1))
        except Exception as e:
            raise ValueError(f"Error in bernoulli function: {e}")
        # handle the case where t is exactly zero
        result[np.abs(t) < 1e-14] = 1.0
  
    return result
    

def build_scharfetter_gummel_operator(x, phi, D, beta):
    """
    Constructs the sparse drift-diffusion operator using the Scharfetter-Gummel method
    with zero-flux (Neumann) boundary conditions.

    Parameters:
        x : ndarray, shape (N,)
            Grid point locations (uniform spacing assumed)
        phi : ndarray, shape (N,)
            Electrostatic potential at each grid point
        D : float
            Diffusion coefficient
        beta : float
            q / (k_B * T), inverse thermal voltage

    Returns:
        A : scipy.sparse.csr_matrix, shape (N, N)
            Sparse matrix representing the discretized drift-diffusion operator
    """
    N = len(phi)
    #dx = np.diff(x)  # Ensure dx is the same length as phi
    dx = x[1] - x[0] 
    dx = np.ones(N - 1) * dx  # Uniform spacing assumed
    t = beta * np.diff(phi)  # Size N-1
    B = bernoulli(t)
    B_neg = bernoulli(-t)

    # Initialize diagonals
    lower = np.zeros(N - 1)
    main = np.zeros(N)
    upper = np.zeros(N - 1)

    # Interior points: i = 1 to N-2
    for i in range(1, N - 1):
        lower[i - 1] = -B[i - 1]
        main[i] = B_neg[i - 1] + B[i]
        upper[i] = -B_neg[i]
    
        #multiply each row by 2/(dx[i-1] + dx[i])
    # This is to ensure the correct scaling of the operator
    for i in range(N - 1):
        factor = 1/dx[0]    #2 / (dx[i-1] + dx[i]) 
        lower[i] *= factor
        main[i] *= factor
        upper[i] *= factor

    # # Left boundary (i = 0): enforce J_{1/2} = 0 ⇒ -B(-t0) C1 + B(t0) C0 = 0
    main[0] = B[0] / dx[0]
    upper[0] = -B_neg[0] / dx[0]
    
    # Right boundary (i = N-1): enforce J_{N-1/2} = 0 ⇒ -B(-t_{N-2}) C_{N-1} + B(t_{N-2}) C_{N-2} = 0
    lower[N - 2] = -B[N - 2] / dx[N - 2]
    main[N - 1] = B_neg[N - 2] / dx[N - 2]
    
    # Assemble sparse matrix
    A = - D  * sp.diags([lower, main, upper], offsets=[-1, 0, 1], format='csr')
    return A



def compute_fluxes_gummel(x, phi, rho, D, beta):
    """
    Compute fluxes on a non-uniform grid using the Scharfetter-Gummel approximation.

    Parameters:
    - x: array of grid points (length N)
    - phi: potential at grid points (length N)
    - rho: concentration at grid points (length N)
    - D: diffusion coefficient (scalar or array of length N−1)
    - beta: mobility factor

    Returns:
    - flux: array of flux values at grid points (length N), 
            interpolated from interface fluxes
    """
    t = beta * np.diff(phi)  # size N-1

    B = bernoulli(t)      # B(t)
    B_neg = bernoulli(-t) # B(-t)
    # Compute fluxes at interfaces (between i and i+1)
    flux_interface = np.zeros(len(x) - 1)
    for i in range(len(x) - 1):
        flux_interface[i] = D * (B[i] * rho[i] - B_neg[i] * rho[i + 1]) #/ (x[i + 1] - x[i])

    return -flux_interface

