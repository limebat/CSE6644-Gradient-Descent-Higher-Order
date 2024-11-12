import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
    
def f(x, y, epsilon, alpha, m, n, eta):
    h = x**(2 * np.max([n, m]) + epsilon) + y**(2 * np.max([n, m]) + epsilon)
    func = y*(x**(2*n) + alpha / (2*m+1) * y**(2*m) + h)
    return func
 
def f_grad(x, y, epsilon, alpha, m, n, eta):
    exp_coeff_max = 2 * np.max([n, m]) + epsilon
    dhdx = exp_coeff_max * x**(exp_coeff_max - 1) 
    dfuncdx = y*(2*n * x**(2*n - 1) + dhdx)
    dhdy = exp_coeff_max * y**(exp_coeff_max - 1) 
    dfuncdy = x**(2*n) + 2*m * alpha / (2*m+1) * y**(2*m) + dhdy
    return dfuncdx, dfuncdy

def case1(x, y, epsilon, alpha, m, n, eta):
    h = x**(2 * m + epsilon) + y**(2 * m + epsilon)
    return h
    
def case2(x, y, epsilon, alpha, m, n, eta):
    dhdx = x**(2 * m + epsilon) 
    return dhdx

def case3(x, y, epsilon, alpha, m, n, eta):
    dhdy = x**(2 * m + epsilon) + y**(2 * m - 1 + epsilon)
    return dhdy

def f_grad_case1(x, y, epsilon, alpha, m, n, eta):
    exp_coeff_max = 2 * m + epsilon
    dhdx = exp_coeff_max * x**(exp_coeff_max - 1) 
    dfuncdx = y*(2*n * x**(2*n - 1) + dhdx)
    dhdy = exp_coeff_max * y**(exp_coeff_max - 1) 
    dfuncdy = x**(2*n) + 2*m * alpha / (2*m+1) * y**(2*m) + dhdy
    return dfuncdx, dfuncdy

def f_grad_case2(x, y, epsilon, alpha, m, n, eta):
    exp_coeff_max = 2 * np.max([n, m]) + epsilon
    dhdx =  case2(x, y, epsilon, alpha, m, n, eta)
    dfuncdx = y*(2*n * x**(2*n - 1) + dhdx)
    dhdy = exp_coeff_max * y**(exp_coeff_max - 1) 
    dfuncdy = x**(2*n) + 2*m * alpha / (2*m+1) * y**(2*m) + dhdy
    return dfuncdx, dfuncdy

def f_grad_case3(x, y, epsilon, alpha, m, n, eta):
    exp_coeff_max = 2 * np.max([n, m]) + epsilon
    dhdx = exp_coeff_max * x**(exp_coeff_max - 1) 
    dfuncdx = y*(2*n * x**(2*n - 1) + dhdx)
    dhdy = case3(x, y, epsilon, alpha, m, n, eta) 
    dfuncdy = x**(2*n) + 2*m * alpha / (2*m+1) * y**(2*m) + dhdy
    return dfuncdx, dfuncdy

def GD(x0, y0, epsilon, alpha, m, n, eta, tol=1e-4, max_iter=1e3):
    err = 1
    k  = 0
    x = [x0]
    y = [y0]
    while err > tol:
        [dfuncdx, dfuncdy] = f_grad(x[k], y[k], epsilon, alpha, m, n, eta)
        
        x.append(x[k] - eta * dfuncdx) 
        y.append(y[k] - eta * dfuncdy)
        
        err = np.sqrt((x[k+1] - x[k])**2 + (y[k+1] - y[k])**2)
        
        k += 1
        
        if k > max_iter:
            print('Did not converge!')
            print('The error is now given as : ', err)
            break
        
    return x, y

m = 2
n = 2
alpha = 2
epsilons = [i * 0.1 for i in range(20)] #Range 0.1, 0.2, ... 1
eta = 0.1
x0 = 1
y0 = 1
dx = dy = 0.01

x = np.arange(0, 1+dx, dx)
y = np.arange(0, 1+dy, dy)
X, Y = np.meshgrid(x, y)

# Now solve for each epsilon -- order magnitude increase each time for each term for further incrases in power

x_val_final = []
y_val_final = []
for epsilon in epsilons:
    x, y = GD(x0, y0, epsilon, alpha, m, n, eta)

    #Printing last 10th position rather than the last position due to the singularity of dhdy preventing an exact value to print
    print('The value for each x and y converged form is: ', x[-10], y[-10], 'For the epsilon value of: ', epsilon)
    x_val_final.append(x[-10])
    y_val_final.append(y[-10])

#Plot the x and y through each iteration

plt.loglog(x, y)
plt.title('Iterations of x and y throughout respective iterations')
plt.xlabel(f'$x_k$')
plt.ylabel(f'$y_k$')
plt.grid(True, which='both', ls='--')

plt.show()

#Now plot its change w.r.t epsilon
fig = plt.figure()
ax =fig.add_subplot(111, projection='3d')

ax.scatter(x_val_final, y_val_final, epsilons, marker='o')
ax.set_xlabel(f'$x_k$')
ax.set_ylabel(f'$y_k$')
ax.set_zlabel(f'$\epsilon$')

plt.show()

#Sketch function: General case

dfuncdx, dfuncdy = f_grad(X, Y, epsilon, alpha, m, n, eta)

plt.quiver(X, Y, -dfuncdx, -dfuncdy)

plt.title('Gradient Field Visualization, General Function', fontsize=16)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.grid(True)

plt.xlim([X.min(), X.max()])
plt.ylim([Y.min(), Y.max()])

plt.show()


#Sketch function: Case 1

dfuncdx, dfuncdy = f_grad_case1(X, Y, epsilon, alpha, m, n, eta)

plt.quiver(X, Y, -dfuncdx, -dfuncdy)

plt.title('Gradient Field Visualization, Case 1', fontsize=16)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.grid(True)

plt.xlim([X.min(), X.max()])
plt.ylim([Y.min(), Y.max()])

plt.show()

#Sketch function: Case 2

dfuncdx, dfuncdy = f_grad_case2(X, Y, epsilon, alpha, m, n, eta)

plt.quiver(X, Y, -dfuncdx, -dfuncdy)

plt.title('Gradient Field Visualization, Case 2', fontsize=16)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.grid(True)

plt.xlim([X.min(), X.max()])
plt.ylim([Y.min(), Y.max()])

plt.show()

dfuncdx, dfuncdy = f_grad_case3(X, Y, epsilon, alpha, m, n, eta)

plt.quiver(X, Y, -dfuncdx, -dfuncdy)

plt.title('Gradient Field Visualization, Case 3', fontsize=16)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.grid(True)

plt.xlim([X.min(), X.max()])
plt.ylim([Y.min(), Y.max()])

plt.show()
