import time
import numpy as np
start_time = time.perf_counter()
import math
#Chapter 1 Matrics
l=[]
print("Modes:\n[1]Row echolen form\n[2]Gauss Seidel\n[3]Eigen Values and Eigen Vectory\n[4]Dominant Eigen Value")
mode=int(input("Enter the mode: "))
m=int(input("Enter the no. of rows : "))
n=int(input("Enter the no.of columns : "))
h=input("Enter the vales of  matrix A: ")

iterations=3
h=h.split()
h=[int(f) for f in h]
for u in range(m):
    l.append(h[u*n:(u+1)*n])#Converting to 2d array format
l=np.array(l)
#Concept -1 : Row Echelon Form
#Note: The vector must be input as the argumented matrix
def row_echelon(matrix):
    row = 0  # current pivot row
    for col in range(n):
        if row >= m:#break the code if the pointer exceeds row value
            break
        pivot = None
        for r in range(row, m):#finding the element with non-zero leading element
            if matrix[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        #Swaping pivote row into position
        matrix[[row,pivot]]=matrix[[pivot,row]]
        if row!=pivot:
            print(f"R{row} <-> R{pivot}")
            print(matrix)
        for r in range(row + 1, m):
            #checking for float as hcf is defind for integer
            if  matrix[row][col]-int(matrix[row][col])==0.0 and matrix[r][col]-int(matrix[r][col])==0.0:
                g=math.gcd(int(matrix[row][col]),int(matrix[r][col]))
                
            else:
                #since it is float we use this method
                pivot_val = matrix[row][col]
                matrix[row] = [x / pivot_val for x in matrix[row]]
                print(f"R{row} -> R{row}/{pivot_val}")
                g=1
            # Getting zero for the row below the non zero leading element
        
            factor1 = (matrix[r][col]/g)
            factor2=(matrix[row][col]/g)
            matrix[r] = [factor2*a - factor1*b for a, b in zip(matrix[r], matrix[row])]
            if factor1>0:
                print(f"R{r+1} -> {factor2}*R{r+1} - {factor1}*R{row+1}")
            elif factor1==0:
                continue
            else:
                print(f"R{r+1}->{factor2}*R{r+1}+ {abs(factor1)}*R{row+1}")
            print(matrix)
        row += 1  # move to next pivot row
    
    return matrix
#Concept 2-Gauss Seidel
#Note: B vector must not be included with A i.e it must not be argumented matrix(which implies the matrix must be a square)

def is_diagonally_dominant(matrix):
    for i in range(n):
        diag = abs(matrix[i][i])
        off_diag = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        if diag < off_diag:
            return False
    return True

def reorder_for_dominance(A, B):
    #ry to reorder rows of A and B to make it diagonally dominant
    for i in range(n):
        for j in range(i, n):
            diag = abs(A[j][i])
            off_diag = sum(abs(A[j][k]) for k in range(n) if k != i)
            if diag >= off_diag:
                # swap rows
                A[[i,j]]=A[[j,i]]
                B[[i,j]]=B[[j,i]]
                
                break
    return A, B
def gauss_seidel(A, B, X):
    for it in range(1, iterations+1):
        print(f"Iteration : {it}")
        for i in range(n):
            sum1 = round(sum(A[i][j] * X[j]for j in range(i)),4)
            sum2 = round(sum(A[i][j] * X[j]for j in range(i+1, n)),4)
            X[i] = (B[i] - sum1 - sum2) / A[i][i]
            X[i]=np.round(X[i],decimals=4)
            if sum1>0:
                c1='-'
            else:
                c1='+'
            if sum2>0:
                c2='-'
            else:
                c2='+'
            print(f"x{i+1}:({B[i]} {c1} {abs(sum1)} {c2} {abs(sum2)})/{A[i][i]}={X[i]}")
        #print(f"Iteration {it}: X = {X.ravel()}")
    return X
def print_equations(A, B):
    print("\nSystem of equations:")
    for i in range(n):
        eq = " + ".join(f"({A[i][j]})*x{j+1}" for j in range(n))
        print(f"{eq} = {B[i]}")
    print()
def bodyofgaussseidel(A,B,X):
    if not is_diagonally_dominant(A):
        print("\nMatrix is not diagonally dominant, trying to reorder...")
        A, B = reorder_for_dominance(A, B)

    if is_diagonally_dominant(A):
        print("\nMatrix is diagonally dominant (or reordered).")
    else:
        print("\nWarning: Matrix is not diagonally dominant, convergence may fail.")
    # Print equations
    print_equations(A, B)

    # Perform Gauss-Seidel
    gauss_seidel(A, B, X)


#Concept 3 - Eigen Value and Eigen Function
#Formula = x^3-tr(A)x^2+tr(M)x-|A|
def get_minor_matrix(matrix,i,j):
    minor_matrix = np.delete(matrix, i, axis=0)#delete row
    minor_matrix = np.delete(minor_matrix, j, axis=1)#delete colum
    return minor_matrix

def eigen_value(matrix):
    trace=0
    sumofm=0
    for tr in range(m):
        trace+=matrix[tr][tr] 
    for row in range(m):
        #Get minor matrix and convert to determinat
        minor=np.linalg.det(get_minor_matrix(matrix, row, row))
        #Making the sum of minors and rounding of to 4 decimal
        sumofm=round(minor,4)+sumofm
    #Determinat of matrix
    det=round(np.linalg.det(matrix),4)
    coeff=[1,-trace,sumofm,-det]
    #Calcuate the roots of the degree len(coegg)
    roots = np.roots(coeff)
    #rounding to 4 decimals
    roots=np.round(roots,decimals=4)
    roots=roots.real
    print("The roots of the equation are : ",roots)
    return roots
def eigen_vector(matrix,roots):
    for roo in roots:
        print(f"For root {roo} The eigne vector is : ")
        #Creating an idenity matrix
        Id = np.identity(matrix.shape[0]) 
        #Multiplying with the root
        Scaler=roo*Id
        #print("Scaler\n",Scaler)
        #print("matrix\n",matrix)
        nmatrix=matrix-Scaler
        if roo>=0:
            print(f"A-{roo}λ=\n",nmatrix)
        else:
            print(f"A+{roo}λ=\n",nmatrix)
        nmatrix=row_echelon(nmatrix)
        print(nmatrix)

#Concept 4:  Dominant Eigne Value
def dominant_eigen_value(A,Xt):
    for rep in range(iterations):
        Xt=A @ Xt
        maxi=np.max(Xt)
        Xt=np.round(Xt/maxi,decimals=4)
        print(f"AX{rep+1}={maxi} * \n{Xt}")
    Xi = Xt.T
    num=A @ Xt
    print(f"AX{iterations+1}=\n{num}")
    num=np.round(Xi @ num,4)
    print(f"X^T{iterations+1}(AX{iterations+1})=\n{num}")
    detn= round(Xi @ Xt,4)
    print(f"X^T{iterations+1}*X{iterations+1}={detn}")
    print("Dominat Eigen Value : ",num/detn)

if mode==1:
    row_echelon(l)
elif mode==2:
    b=input("Enter the values of matix B :")
    x=input("Enter the inital guess :")
    b=b.split()
    x=x.split()
    b=np.array(b,dtype=int)
    x=np.array(x,dtype=float)
    bodyofgaussseidel(l,b,x)
elif mode==3:
    r=eigen_value(l)
    eigen_vector(l,r)
elif mode==4:
    b=input("Enter the values of inital guess :")
    b=np.array(b)
    dominant_eigen_value(l,b) 
else:
    print("Invalid")





