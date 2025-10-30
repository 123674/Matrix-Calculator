from flask import Flask, request
import numpy as np
import math
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

def row_echelon(matrix,prechar):
    matrix = matrix.copy()
    m, n = matrix.shape
    output = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        pivot = None
        for r in range(row, m):
            if matrix[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        matrix[[row, pivot]] = matrix[[pivot, row]]
        if row != pivot:
            output.append(f"R{row+1} ↔ R{pivot+1}")
            output.append(format_matrix(matrix," = "))
        for r in range(row + 1, m):
            if matrix[row][col] - int(matrix[row][col]) == 0.0 and matrix[r][col] - int(matrix[r][col]) == 0.0:
                g = math.gcd(int(matrix[row][col]), int(matrix[r][col]))
            else:
                pivot_val = matrix[row][col]
                matrix[row] = [x / pivot_val for x in matrix[row]]
                output.append(f"R{row+1} → R{row+1}/{pivot_val}")
                output.append(format_matrix(matrix,prechar))
                g = 1
            factor1 = (matrix[r][col] / g)
            factor2 = (matrix[row][col] / g)
            matrix[r] = [factor2 * a - factor1 * b for a, b in zip(matrix[r], matrix[row])]
            if factor1 == 0:
                continue
            else:
                output.append("R" + str(r+1) + " → {:+}".format(factor2) + "*R" + str(r+1) + " {:+}".format(-factor1)+ "*R" + str(row+1))
                output.append(format_matrix(matrix,prechar))
                
        row += 1
    return "\n".join(output)

def is_diagonally_dominant(matrix):
    n = matrix.shape[1]
    for i in range(n):
        diag = abs(matrix[i][i])
        off_diag = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        if diag < off_diag:
            return False
    return True

def reorder_for_dominance(A, B):
    n = A.shape[1]
    for i in range(n):
        for j in range(i, n):
            diag = abs(A[j][i])
            off_diag = sum(abs(A[j][k]) for k in range(n) if k != i)
            if diag >= off_diag:
                A[[i, j]] = A[[j, i]]
                B[[i, j]] = B[[j, i]]
                break
    return A, B

def gauss_seidel(A, B, X, iterations):
    output = []
    n = A.shape[1]
    for it in range(1, iterations + 1):
        output.append(f"Iteration : {it}")
        for i in range(n):
            sum1 = sum(A[i][j] * X[j][0] for j in range(i))
            sum2 = sum(A[i][j] * X[j][0] for j in range(i + 1, n))
            sum1=round(sum1,4)
            sum2=round(sum2,4)
            X[i][0] = (B[i][0] - sum1 - sum2) / A[i][i]
            X[i]=np.round(X[i],decimals=4)
            output.append("x" + str(i+1) + " : ( " + str(B[i][0]) + " {:+}".format(-sum1)+" {:+}".format(-sum2)+" ) / " + str(A[i][i]) + " = " + str(X[i][0]))
        #output.append(f"Iteration {it}: X = {X.ravel()}")
        output.append("\n")
    return "\n".join(output)

def eigen_value(matrix):
    output=[]
    m = matrix.shape[0]
    trace = sum(matrix[i][i] for i in range(m))
    sumofm = 0
    for row in range(m):
        minor = np.linalg.det(np.delete(np.delete(matrix, row, axis=0), row, axis=1))
        sumofm = round(minor, 4) + sumofm
    det = round(np.linalg.det(matrix), 4)
    output.append(f"For the given matrix : \ntr(A) = {trace}\nΣ(M) = {sumofm}\n|A| = {det}")
    output.append(f"Using the formula to calculate the eigen value of the matrix :\n =  λ³ - tr(A)λ² + Σ(M)λ - |A| = 0")
    output.append("We get,\n =  λ³ {:+}".format(-trace)+"λ² {:+}".format(sumofm)+"λ  {:+}".format(-det))
    coeff = [1, -trace, sumofm, -det]
    roots = np.roots(coeff)
    roots = np.round(roots, decimals=4)
    roots = roots.real
    output.append(f"On solving the equation,\nλ = {roots}")
    for roo in roots:
        output.append(f"\nFor root {roo} The eigen vector is:")
        Id = np.identity(matrix.shape[0])
        Scaler = roo * Id
        nmatrix = matrix - Scaler
        output.append("A {:+}".format(-roo) + "λ =")
        output.append(format_matrix(nmatrix,"foreigen"))
        output.append(row_echelon(nmatrix,"foreigen"))
    return "\n".join(output)

def dominant_eigen_value(A, Xt, iterations):
    output = []
    for rep in range(iterations):
        Xt = A @ Xt
        maxi = np.max(Xt)
        Xt = np.round(Xt / maxi, decimals=4)
        #output.append(f"AX{rep+1} = {maxi} x ")
        output.append(format_matrix(Xt,f"AX{rep+1} = {maxi} x "))
    #print(Xt,type(Xt),flush=True)
    Xi = Xt.T
    num = A @ Xt
    #print(num,type(num),flush=True)
    #output.append(f"AX{iterations+1} =")
    output.append(format_matrix(num,f"AX{iterations+1} = "))
    num = np.round(Xi @ num, decimals=4)
    #print("Num2",num,type(num),flush=True)
    output.append(f"Xᵀ{iterations+1}(*A*X{iterations+1}) = {num}")
    detn = np.round(Xi @ Xt, decimals=4)
    #print("dent : ",detn,type(detn),flush=True)
    output.append(f"Xᵀ{iterations+1} * X{iterations+1} = {detn}")
    t=num / detn
    #print(t,type(t),flush=True)
    output.append(f"Dominant Eigen Value: {np.round(t,4)}")
    return "\n".join(output)

def format_matrix(matrix,prechar):
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    rows_str = []
    for row in matrix:
        row_values = ' '.join([str(round(val, 4)) for val in row])
        if prechar == "foreigen":
            row_values=row_values+' 0'
        rows_str.append(row_values)
        #rows_str.append("0")
    #print("Hey",flush=True)
    prechar="="
    return prechar + " [[" + "|".join(rows_str) + "]]"

@app.route("/api/matrix-calc", methods=["POST"])
def matrix_calc_api():
    data = request.get_json(force=True)
    matrix = np.array(data["matrix"], dtype=float)
    mode = int(data["mode"])
    iterations = int(data.get("iterations", 3))
    try:
        if mode == 1:
            result = row_echelon(matrix," = ")
        elif mode == 2:
            B = np.array(data["vectorB"], dtype=float).reshape(-1, 1)
            X = np.array(data["vectorX"], dtype=float).reshape(-1, 1)
            if not is_diagonally_dominant(matrix):
                matrix, B = reorder_for_dominance(matrix, B)
            result = gauss_seidel(matrix, B, X, iterations)
        elif mode == 3:
            result = eigen_value(matrix)
        elif mode == 4:
            
            B = np.array(data["vectorB"], dtype=float).reshape(-1, 1)
            result = dominant_eigen_value(matrix, B, iterations)
        else:
            result = "Invalid mode selected."
        return result, 200, {"Content-Type": "text/plain; charset=utf-8"}
    except Exception as e:
        Flask.logger.info(e)
        return f"Error Here: {e}", 400
if __name__ == '__main__':
    app.run(debug=True, port=7432)
#https://student-projects-pn.conferit.com/api/matrix-calc