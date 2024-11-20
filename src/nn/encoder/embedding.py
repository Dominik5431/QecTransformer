from math import pi

import torch

""" Comments not correct! """

def cm(x, y):
    """ Complex multiplication between two torch tensors

        Args:
            x (torch.tensor): A (n+1)-dimensional torch float tensor with the
                real and imaginary part stored in the last dimension of the
                tensor; i.e. with shape (d_0, d_1, ..., d_{n-1}, 2)
            y (torch.tensor): A (n+1)-dimensional torch float tensor with the
                real and imaginary part stored in the last dimension of the
                tensor; i.e. with shape (d_0, d_1, ..., d_{n-1}, 2)

        Returns:
            torch.tensor: A (n+1)-dimensional torch float tensor with the
                real and imaginary part stored in the last dimension of the
                tensor; i.e. with shape (d_0, d_1, ..., d_{n-1}, 2)
    """
    result = torch.stack(
        [
            x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1],
            x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0],
        ],
        -1,
    )
    return result


class ModReLU(torch.nn.Module):
    """ A modular ReLU activation function for complex-valued tensors """

    def __init__(self, size):
        """ ModReLU

        Args:
            size (torch.tensor): the number of features of the expected input tensor
        """
        super(ModReLU, self).__init__()
        self.bias = torch.nn.Parameter(torch.rand(1, size))
        self.relu = torch.nn.ReLU()

    def forward(self, x, eps=1e-8):
        """ ModReLU forward

        Args:
            x (torch.tensor): A 3-dimensional torch float tensor with the
                real and imaginary part stored in the last dimension of the
                tensor; i.e. with shape (batch_size, features, 2)
            eps (optional, float): A small number added to the norm of the
                complex tensor for numerical stability.

        Returns:
            torch.tensor: A 3-dimensional torch float tensor with the real and
                imaginary part stored in the last dimension of the tensor; i.e.
                with shape (batch_size, features, 2)
        """

        norm = torch.norm(x) + eps
        sgn = x / norm
        activated_norm = self.relu(norm + self.bias)
        modrelu = torch.stack(
            [activated_norm * sgn], -1
        )
        return modrelu




class EUNN(torch.nn.Module):
    """ Efficient Unitary Neural Network layer

    This layer works similarly as a torch.nn.Linear layer. The difference in this case
    is however that the action of this layer can be represented by a unitary matrix.

    This EUNN is loosely based on the tunable version of the EUNN proposed in
    https://arxiv.org/abs/1612.05231. However, the last diagonal matrix of phases
    was removed in favor of periodic boundary conditions. This makes the algorithm
    considerably faster and more stable.

    """

    def __init__(self, hidden_size, capacity=None):
        """ Efficient Unitary Neural Network layer

        Args:
            hidden_size (int): the size of the unitary matrix this cell represents.
            capacity (int): 0 < capacity <= hidden_size. This number represents the
                number of layers containing unitary rotations. The higher the capacity,
                the more of the unitary matrix space can be filled. This obviously
                introduces a speed penalty. In recurrent neural networks, a small
                capacity is usually preferred.
        """
        # validate parameters
        if hidden_size % 2 != 0:
            raise ValueError("EUNN `hidden_size` should be even")
        if capacity is None:
            capacity = hidden_size
        elif capacity % 2 != 0:
            raise ValueError("EUNN `capacity` should be even")

        self.hidden_size = int(round(hidden_size))
        self.capacity = int(round(capacity))

        # initialize
        super(EUNN, self).__init__()

        # monolithic block of angles:
        self.angles = torch.nn.Parameter(
            2 * pi * torch.randn(self.hidden_size, self.capacity)
        )

    def forward(self, x):
        """ forward pass through the layer

        Args:
            x (torch.tensor): A 3-dimensional torch float tensor with the
                real and imaginary part stored in the last dimension of the
                tensor; i.e. with shape (batch_size, features, 2)

        Returns:
            torch.tensor: A 3-dimensional torch float tensor with the real and
                imaginary part stored in the last dimension of the tensor; i.e.
                with shape (batch_size, features, 2)
        """

        # get and validate shape of input tensor:
        z = torch.zeros(x.shape, 2)
        z[..., 0] = x
        x = z
        batch_size, hidden_size, ri = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Input tensor for EUNN layer has size %i, "
                "but the EUNN layer expects a size of %i"
                % (hidden_size, self.hidden_size)
            )
        elif ri != 2:
            raise ValueError(
                "Input tensor for EUNN layer should be complex, "
                "with the complex components stored in the last dimension (x.shape[2]==2)"
            )

        # references to parts in the monolithic block of angles:

        # phi and theta for the even layers
        phi0 = self.angles[::2, ::2]
        theta0 = self.angles[1::2, ::2]

        # phi and theta for the odd layers
        phi1 = self.angles[::2, 1::2]
        theta1 = self.angles[1::2, 1::2]

        # calculate the sin and cos of rotation angles
        cos_phi0 = torch.cos(phi0)
        sin_phi0 = torch.sin(phi0)
        cos_theta0 = torch.cos(theta0)
        sin_theta0 = torch.sin(theta0)
        cos_phi1 = torch.cos(phi1)
        sin_phi1 = torch.sin(phi1)
        cos_theta1 = torch.cos(theta1)
        sin_theta1 = torch.sin(theta1)

        # calculate the rotation vectors
        # shape = (capacity//2, 1, hidden_size, 2=(real|imag))
        zeros = torch.zeros_like(cos_theta0)
        diag0 = (
            torch.stack(
                [
                    torch.stack([cos_phi0 * cos_theta0, cos_theta0], 1).view(
                        -1, self.capacity // 2
                    ),
                    torch.stack([sin_phi0 * cos_theta0, zeros], 1).view(
                        -1, self.capacity // 2
                    ),
                ],
                -1,
            )
            .unsqueeze(0)
            .permute(2, 0, 1, 3)
        )
        offdiag0 = (
            torch.stack(
                [
                    torch.stack([-cos_phi0 * sin_theta0, sin_theta0], 1).view(
                        -1, self.capacity // 2
                    ),
                    torch.stack([-sin_phi0 * sin_theta0, zeros], 1).view(
                        -1, self.capacity // 2
                    ),
                ],
                -1,
            )
            .unsqueeze(0)
            .permute(2, 0, 1, 3)
        )
        diag1 = (
            torch.stack(
                [
                    torch.stack([cos_phi1 * cos_theta1, cos_theta1], 1).view(
                        -1, self.capacity // 2
                    ),
                    torch.stack([sin_phi1 * cos_theta1, zeros], 1).view(
                        -1, self.capacity // 2
                    ),
                ],
                -1,
            )
            .unsqueeze(0)
            .permute(2, 0, 1, 3)
        )
        offdiag1 = (
            torch.stack(
                [
                    torch.stack([-cos_phi1 * sin_theta1, sin_theta1], 1).view(
                        -1, self.capacity // 2
                    ),
                    torch.stack([-sin_phi1 * sin_theta1, zeros], 1).view(
                        -1, self.capacity // 2
                    ),
                ],
                -1,
            )
            .unsqueeze(0)
            .permute(2, 0, 1, 3)
        )

        # loop over the capacity
        for d0, d1, o0, o1 in zip(diag0, diag1, offdiag0, offdiag1):
            # first layer
            x_perm = torch.stack([x[:, 1::2], x[:, ::2]], 2).view(
                batch_size, self.hidden_size, 2
            )
            x = cm(x, d0) + cm(x_perm, o0)

            # periodic boundary conditions
            x = torch.cat([x[:, 1:], x[:, :1]], 1)

            # second layer
            x_perm = torch.stack([x[:, 1::2], x[:, ::2]], 2).view(
                batch_size, self.hidden_size, 2
            )
            x = cm(x, d1) + cm(x_perm, o1)

            # periodic boundary conditions
            x = torch.cat([x[:, -1:], x[:, :-1]], 1)

        return x

