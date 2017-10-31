% LU Decomposition with complete pivoting
%
% Mauricio de Oliveira
% September 2013

disp('> LU Decomposition with Complete Pivoting ')

disp('> Problem data')
A = [2 1 3; 2 4 8; 4 -7 4]
b = [6 -1 1]'

n = size(A,2)

LU = A;

disp('> Pivot 1')
k = 1;
P1 = [0 1 0; 1 0 0; 0 0 1]
Q1 = P1
LU = P1 * LU * Q1

disp('> Step 1')
rows = (k+1) : n;
M1 = eye(n);
M1(rows,k) = -LU(rows,k) / LU(k,k)
LU(rows,k) = LU(rows,k) / LU(k,k);
LU(rows,rows) = LU(rows,rows) - LU(rows,k)*LU(k,rows)

disp('> Pivot 2')
k = k + 1;
P2 = [1 0 0; 0 1 0; 0 0 1]
Q2 = [1 0 0; 0 0 1; 0 1 0]
LU = P2 * LU * Q2

disp('> Step 2')
rows = (k+1) : n;
M2 = eye(n);
M2(rows,k) = -LU(rows,k) / LU(k,k)
LU(rows,k) = LU(rows,k) / LU(k,k);
LU(rows,rows) = LU(rows,rows) - LU(rows,k)*LU(k,rows)

disp('> LU')
LU

disp('> Extract L and U')
U = triu(LU)
L = eye(n) + tril(LU,-1)

disp('> Compute P = P2 * P1')
P = P2 * P1

disp('> Compute Q = Q1 * Q2')
Q = Q1 * Q2

disp('> L U - P A Q')
L * U - P * A * Q