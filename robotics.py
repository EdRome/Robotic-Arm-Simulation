import numpy as np
import functools
import math

class bodies_2d:

    def __init__(self, x, y, tetha):
        self.x = x
        self.y = y
        self.tetha = tetha

    def translate(self, trans_matrix, x, y):
        r_axis = np.matrix([[x],[y],[1]])
        return np.matmul(trans_matrix, r_axis)
        

class bodies_3d:

    def __init__(self, alpha, gamma, beta):
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta


class rotation_3d(bodies_3d):

    def __init__(self, alpha, gamma, beta):
        super(rotation_3d, self).__init__(alpha, gamma, beta)
        
    def _R_alpha(self):
        pos_alpha_cos = np.cos(self.alpha)
        pos_alpha_sin = np.sin(self.alpha)
        R_z = np.matrix([[pos_alpha_cos, -1*pos_alpha_sin, 0],
                         [pos_alpha_sin, pos_alpha_cos, 0],
                         [0, 0, 1]])
        return R_z

    def _R_beta(self):
        pos_beta_cos = np.cos(self.beta)
        pos_beta_sin = np.sin(self.beta)
        R_y = np.matrix([[pos_beta_cos, 0, pos_beta_sin],
                         [0, 1, 0],
                         [-1*pos_beta_sin, 0, pos_beta_cos]])
        return R_y

    def _R_gamma(self):
        pos_gamma_cos = np.cos(self.gamma)
        pos_gamma_sin = np.sin(self.gamma)
        R_x = np.matrix([[1, 0, 0],
                         [0, pos_gamma_cos, -1*pos_gamma_sin],
                         [0, pos_gamma_sin, pos_gamma_cos]])
        return R_x

    def transformation(self):
        r_gamma = self._R_gamma()
        r_alpha = self._R_alpha()
        r_beta = self._R_beta()
        return np.matmul(np.matmul(r_alpha, r_beta), r_gamma)

    def translate(self, trans_matrix):
        r_axis = np.matrix([[1], [1], [1]])
        return np.matmul(trans_matrix, r_axis)
        
class rotation_2d(bodies_2d):

    def __init__(self, x_t, y_t, tetha):
        super(rotation_2d, self).__init__(x_t, y_t, tetha)

    def rot_matrix(self):
        pos_tetha_cos = np.cos(self.tetha[0])
        pos_tetha_sin = np.sin(self.tetha[0])
        return np.matrix(data=[[pos_tetha_cos.item(0), -1*pos_tetha_sin.item(0), 0],
                               [pos_tetha_sin.item(0), pos_tetha_cos.item(0), 0],
                               [0, 0, 1]])
    
class trans_rot_2d(bodies_2d):

    def __init__(self, x_t, y_t, tetha):
        super(trans_rot_2d, self).__init__(x_t, y_t, tetha)

    def r_t_matrix(self):
        pos_tetha_cos = np.cos(self.tetha[0])
        pos_tetha_sin = np.sin(self.tetha[0])
        return (np.matrix(data=[[pos_tetha_cos, -1*pos_tetha_sin, self.x],
                                [pos_tetha_sin, pos_tetha_cos, self.y],
                                [0, 0, 1]]))

class kinematics_2d():

    def __init__(self, angle, x, y):
        self.angle = angle
        self.x = x
        self.y = y

    def homogeneous_matrix(self):
        #pos_tetha_cos = np.cos(self.angle)
        #pos_tetha_sin = np.sin(self.angle)
        pos_tetha_cos = self.angle.item(0)
        pos_tetha_sin = self.angle.item(1)
        hom_matrices = np.matrix([
            [pos_tetha_cos,-1*pos_tetha_sin, self.x],
            [pos_tetha_sin,   pos_tetha_cos, self.y],
            [0, 0, 1]])
        return hom_matrices

    def translate(self, mat2trans, x0, y0):
        axis = np.matrix([[x0],[y0],[1]])
        return np.matmul(mat2trans, axis)
        

"""
alpha = 15
gamma = 20
beta = 5
t = rotation_3d(alpha, gamma, beta)
r_trans = t.transformation()
print(r_trans)
print(t.translate(r_trans))
"""

"""
x_t = 10
y_t = 20
tetha = 30
b2d = trans_rot_2d(x_t, y_t, tetha)
r_trans = b2d.r_t_matrix()
print(r_trans)
print(b2d.translate(r_trans))
"""
"""
x_t = 221
y_t = 324
tetha = np.matrix([[0]])
kin_2d = rotation_2d(x_t, y_t, tetha)
rot_m = kin_2d.rot_matrix()
trans_m = kin_2d.translate(rot_m, 150, 150)
x_w = np.cos(tetha[0]).item(0)*(trans_m[0]/trans_m[2])-np.sin(tetha[0]).item(0)*(trans_m[1]/trans_m[2])
y_w = np.sin(tetha[0]).item(0)+(trans_m[0]/trans_m[2])+np.cos(tetha[0]).item(0)*(trans_m[1]/trans_m[2])
print(x_w)
print(y_w)
"""
"""
x_t = 593
y_t = 668
a1_x = 150
a1_y = 150
a2_x = 250
a2_y = 250
a3_x = 350
a3_y = 350
##a4_x = 450
##a4_y = 450
tetha = np.matrix([[0], [0], [0]])
length = np.matrix([[100],[100]])
kin_2d = kinematics_2d(x_t, y_t, tetha, length)
trans_mat = kin_2d.homogenous_matrix()

a1_trans = kin_2d.translate(kin_2d.hom_matrices[0], a1_x, a1_y)
print(a1_trans)
a2_trans = kin_2d.translate(kin_2d.hom_matrices[0].dot(kin_2d.hom_matrices[1]), a2_x, a2_y)
print(a2_trans)
a3_trans = kin_2d.translate(kin_2d.hom_matrices[1].dot(kin_2d.hom_matrices[2]), a3_x, a3_y)
print(a3_trans)
##a4_trans = kin_2d.translate(kin_2d.hom_matrices[2].dot(kin_2d.hom_matrices[3]), a4_x, a4_y)
##print(a4_trans)
p_a1 = np.matrix([a1_trans.item(0)-a1_x, a1_trans.item(1)-a1_y])
p_a2 = np.matrix([a2_trans.item(0)-a2_x, a2_trans.item(1)-a2_y])
p_a3 = np.matrix([a3_trans.item(0)-a3_x, a3_trans.item(1)-a3_y])
##p_a4 = np.matrix([a4_trans.item(0)-a4_x, a4_trans.item(1)-a4_y])
print(math.degrees(math.atan2(p_a1.item(0), p_a1.item(1))))
print(math.degrees(math.atan2(p_a2.item(0), p_a2.item(1))))
print(math.degrees(math.atan2(p_a3.item(0), p_a3.item(1))))
##print(math.degrees(math.atan2(p_a4.item(0), p_a4.item(1))))
"""
