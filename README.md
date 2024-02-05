We propose an infeasible interior-point algorithm for constrained linear least-squares problems based on the primal-dual regularization of convex programs of Friedlander and Orban. Regularization allows us to dispense with the assumption that the active gradients are linearly independent. At each iteration, a linear system with a symmetric quasi-definite (SQD) matrix is solved. This matrix is shown to be uniformly bounded and nonsingular. While the linear system may be solved using sparse LDLT
 factorization, we observe that other approaches may be used. In particular, we build on the connection between SQD linear systems and unconstrained linear least-squares problems to solve the linear system with sparse QR factorization. We establish conditions under which a solution of the original, constrained least-squares problem is recovered. We report computational experience with the sparse QR factorization and illustrate the potential advantages of our approach.
