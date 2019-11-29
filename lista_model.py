import torch
import torch.nn as nn
# from torch.nn.parameter import Parameter
# import tensorflow as tf
# import utils.train
#
# from utils.tf import shrink
# from models.LISTA_base import LISTA_base

############################################################
####################   Shrinkage   #########################
############################################################
def shrink(input_, theta_):
    """
    Soft thresholding function with input input_ and threshold theta_.
    """
    theta_ = torch.max(theta_, torch.tensor([0.0]))
    return torch.sign(input_) * torch.max(torch.abs(input_) - theta_, torch.tensor([0.0]))


class LISTA(nn.Module):

    """
    Implementation of LISTA model proposed by LeCun in 2010.
    """

    def __init__(self, A, T, lam, untied=False):
        """
        :A      : [M, N]    Numpy ndarray. Dictionary/Sensing matrix.
        :T      : Integer. Number of layers (depth) of this LISTA model.
        :lam    : Float. The initial weight of l1 loss term in LASSO.
        :untied : Boolean. Flag of whether weights are shared within layers.
        """
        super(LISTA, self).__init__()
        self._A = torch.tensor(A, dtype=torch.float32) # A.astype(torch.float32)
        self._T = T
        self._lam = lam
        self._M = self._A.shape[0]
        self._N = self._A.shape[1]

        self._scale = 1.001 * torch.norm(self._A, p=2)**2
        self._theta = torch.tensor(self._lam / self._scale.numpy(), dtype=torch.float32)

        self._untied = untied

        self.vars_in_layer = None

        """ Set up layers."""
        self.setup_layers()


    def setup_layers(self):
        """
        Implementation of LISTA model proposed by LeCun in 2010.
        :prob: Problem setting.
        :T: Number of layers in LISTA.
        :returns:
            :layers: List of tuples ( name, xh_, var_list )
                :name: description of layers.
                :xh: estimation of sparse code at current layer.
                :var_list: list of variables to be trained seperately.
        """
        Bs_     = []
        Ws_     = []
        thetas_ = []

        B = torch.transpose(self._A, 0, 1) / self._scale
        W = torch.eye(self._N, dtype=torch.float32) - torch.matmul(B, self._A)

        # with tf.variable_scope (self._scope, reuse=False) as vs:
            # constant

        self._kA_ = nn.Parameter(self._A, requires_grad=False)

        B = B.transpose(0, 1)
        Bs_.append(nn.Parameter(B.clone(), requires_grad=True))
        Bs_ = Bs_ * self._T

        W = W.transpose(0, 1)
        if not self._untied: # tied model
            Ws_.append(nn.Parameter(W.clone(), requires_grad=True))
            Ws_ = Ws_ * self._T

        for t in range(self._T):
            thetas_.append(nn.Parameter(self._theta, requires_grad=True))
            if self._untied:  # untied model
                Ws_.append(nn.Parameter(W.clone(), requires_grad=True))

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]


        self.vars_in_layer = list(zip(Bs_, Ws_, thetas_))

    # def setup_parameters(self):


    def forward(self, y_, x0_=None):
        # '''
        # batch_size last
        # :param y_:  [M, B]
        # :param x0_: [N, B]
        # :return:
        # '''
        #
        # xhs_ = [] # collection of the regressed sparse codes

        '''
        pytorch version, batch_size first
        :param y_:  [B, M]
        :param x0_: [B, N]
        :return:
        '''

        xhs_ = []  # collection of the regressed sparse codes

        if x0_ is None:
            batch_size = y_.shape[0]
            xh_ = torch.zeros(batch_size, self._N, dtype=torch.float32)
        else:
            xh_ = x0_
        xhs_.append(xh_)

        for t in range(self._T):
            # B_: [N, M]    W_: [N, N]   theta_:
            B_, W_, theta_ = self.vars_in_layer[t]

            By_ = torch.matmul(y_, B_)  # By_: [B, N]
            xh_ = shrink(torch.matmul(xh_, W_) + By_, theta_)  # xh_: [B, N]
            xhs_.append(xh_)

        return xhs_