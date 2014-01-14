#function [f,s,m,mv,mvd,unA] = 

function expmv(t,
               A,
               b;
               M = [], 
               prec = "double", 
               shift = true, 
               # bal = false, 
               full_term = false, 
               prnt = false)

#EXPMV   Matrix exponential times vector or matrix.
#   [F,S,M,MV,MVD] = EXPMV(t,A,B,[],PREC) computes EXPM(t*A)*B without
#   explicitly forming EXPM(t*A). PREC is the required accuracy, 'double',
#   'single' or 'half', and defaults to CLASS(A).
#
#   A total of MV products with A or A^* are used, of which MVD are
#   for norm estimation.
#
#   The full syntax is
#
#     [f,s,m,mv,mvd,unA] = expmv(t,A,b,M,prec,shift,bal,full_term,prnt).
#
#   unA = 1 if the alpha_p were used instead of norm(A).
#
#   If repeated invocation of EXPMV is required for several values of t
#   or B, it is recommended to provide M as an external parameter as
#   M = SELECT_TAYLOR_DEGREE(A,m_max,p_max,prec,shift,bal,true).
#   This also allows choosing different m_max and p_max.

#   Reference: A. H. Al-Mohy and N. J. Higham, Computing the action of
#   the matrix exponential, with an application to exponential
#   integrators. MIMS EPrint 2010.30, The University of Manchester, 2010.

#   Awad H. Al-Mohy and Nicholas J. Higham, October 26, 2010.

# if bal
#     [D,B] = balance(A)
#     if norm(B,1) < norm(A,1) 
#         A = B
#         b = D\b
#     else 
#         bal = false
#     end
# end

n = size(A,1)

if shift
    mu = trace(A)/n 
    mu = full(mu) # Much slower without the full!
    A = A-mu*speye(n)
end

if nargin < 5 || isempty(prec), prec = class(A); end

if isempty(M)
    tt = 1
    (M,mvd,alpha,unA) = select_taylor_degree(t*A,b,[],[],prec,false,false)
    mv = mvd
else
    tt = t
    mv = 0
    mvd = 0
end

tol =   
    if prec == "double"
        2^(-53)
    elseif prec == "single"
        2^(-24)
    elseif prec == "half"   
        2^(-10)
    end

s = 1;

if t == 0
    m = 0;
else
    (m_max,p) = size(M);
    U = diagm(1:m_max);
    C = ( (ceil(abs(tt)*M))'*U );
    zero_els = find(x->x==0, C)
    for el in zero_els
        C[el] = Inf
    end
    if p > 1 
        cost,m = findmin(minimum(C,1)); # cost is the overall cost.
    else
        cost,m = findmin(C);  # when C is one column. Happens if p_max = 2.
    end
    if cost == Inf
        cost = 0 
    end
    s = max(cost/m,1);
end

eta = 1;

if shift 
    eta = exp(t*mu/s)
end

f = b;

if prnt 
    fprintf('m = %2.0f, s = %g, m_actual = ', m, s) 
end

for i = 1:s
    c1 = norm(b,Inf);
    for k = 1:m
        b = (t/(s*k))*(A*b);
        mv = mv + 1;
        f =  f + b;
        c2 = norm(b,Inf);
        if !full_term
            if c1 + c2 <= tol*norm(f,Inf)
                if prnt
                    fprintf(' %2.0f, ', k)
                end
                break
            end
            c1 = c2;
        end

    end
    f = eta*f
    b = f
end

if prnt
    fprintf('\n')
end

#if bal 
#    f = D*f
#end
