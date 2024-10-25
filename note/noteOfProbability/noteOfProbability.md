# Approximate inference

## Variational Inference
**functional**, a mapping that takes function as the input and returns the value of the functional.
$H[p]=\int p(x)[lnp(x)]dx$

marginal probability:
$$ln[p(x)]=L(q)+KL(q||p)$$
where we have defined
$$
L(q) = \int q(Z)ln{\frac{p(X,Z)}{q(Z)}}dZ
$$
$$
KL(q||p) = -\int q(Z)ln{\frac{p(Z|X)}{q(Z)}}dZ
$$
The maximum of lower bound occurs when KL divergence vanish, which occurs when $q(Z)$ equals the posterior distribution $p(Z|X)$

Use a parametric distribution $q(Z|w)$ governed by a set of parameters $w$. The lower bound $L(q)$ becomes a function of $w$.

### Factorized distributions
restrict the family of distribution $q(z)$. 
$$
q(Z) = \prod_{i=1}^M q_i(Z_i)
$$
do variational inference using ***mean field theory***

Denoting $q_j(Z_j)$ by simply $q_j$
$$
L(q) = \int q(Z) ln\frac{P(X,Z)}{q(Z)}dZ
\\= \int \prod_i q_i \{lnp(X, Z)-\sum_i lnq_i \}dZ
\\ = \int q_j \{\int lnp(X, Z)\prod_{(i \neq j)}q_i dZ_i \}dZ_j - \int \prod_i q_i (\sum_i lnq_i) dZ
\\ =\int q_j \{\int lnp(X, Z)\prod_{(i \neq j)}q_i dZ_i \}dZ_j - \int q_j ln q_j dZ_j + const
$$
define a new distribution 
$$
ln\widetilde{p}(X,Z_j) = E_{(i, j)}[lnp(X,Z)] + const
$$
So the $L(q)$ becomes
$$
L(q) = \int q_j ln\tilde{p}(X, Z) dZ_j - \int q_j lnq_j dZ_j + const
\\ = \int q_j ln\frac{\tilde{p}(X,Z)}{q_j}dZ_j + const
$$
This equation shows that maxmizing $L(q)$ is equal to minimizing the KL divergence, and the minimum occures when $q_j(Z_j) = \tilde{p}(X, Z_j)$

So the optimal solution $q^*_j(Z_j)$ is given by
$$
lnq^*_j(Z_j) = E_{(i\neq j)}[lnp(X,Z)] + const
$$
Initialize the $q_i(Z_i)$ and replace all the factors $q_i(Z_i)$ each in turn with a revised estimate given by the right-hand side of this equation.
Convergence is guaranteed(Boyd al. 2004)

### Properties of factorized approximations
Approximates a Gaussian distribution
$$
\mu = \bigl(\begin{matrix}\mu_1 \\\mu_2\end{matrix}\bigr)
\\ \Lambda=\bigl(\begin{matrix}\Lambda_{11} & \Lambda_{12}\\
\Lambda_{21} & \Lambda_{22} \end{matrix})
$$ 
approximate this distribution using $q(z) = q_1(z_1)q_2(z_2)$
$$
lnq^*_1(z_1) = E_{z2}[lnp(z)] + const
\\ =E_{z2}[-\frac{1}{2}(z_1-\mu_1)^2 \Lambda_{11}-(z_1-\mu_q)\Lambda_{12}(z_2-\mu_2)] + const
\\=-\frac{1}{2}z_1^2+z_1\mu_1\Lambda_{11}-z_1\Lambda_{12}(E[z_2] - \mu_2) + const
$$
So the $q^*(z_1)$
$$
q^*(z_1) = N(z_1|m_1, \Lambda_{11}^{-1})
$$
where,
$$
m_1 = \mu_1 - \Lambda_{11}^{-1} \Lambda_{12}(E[z_2]-\mu_2)
$$
symmetrily,
$$
q^*(z_2) = N(z_2|m_2, \Lambda_{22}^{-1})
$$
where,
$$
m_2 = \mu_2 - \Lambda_{22}^{-1} \Lambda_{21}(E[z_1]-\mu_1)
$$
The solution is
$$
E[z_1] = \mu_1,\ E[z_2] = \mu_2
$$
