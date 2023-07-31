# Application of PINN to a Linear Elasticity problem in Pytorch

This project utilizes the boundary conditions, body forces, and overall problem set-up and solution shown by:
Haghighat, Ehsan, et al. "A deep learning framework for solution and discovery in solid mechanics." [arXiv preprint
arXiv:2003.02751(2020).](https://arxiv.org/abs/2003.02751)
Rather than using the SciANN deep learning library, the problem is solved using a Pytorch implementation.

The governing equations:
![gov eq.](https://github.com/annapachecohartman/PINN/assets/126839762/0f7e02f7-9c44-47be-aa18-7b0ecade8b29)

The body forces:
![body forces](https://github.com/annapachecohartman/PINN/assets/126839762/4a2f20a7-4672-4b0a-9a2d-aa1362a35516)

The boundary conditions:

![boundary cond.](https://github.com/annapachecohartman/PINN/assets/126839762/527006e9-2eec-4316-902c-c7aa1158fcc7)

