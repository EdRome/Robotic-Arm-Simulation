import numpy as np
import math
import robotics as rob

class IK_2D_Jacobian:
	
	def  __init__(self, p_xy, t_xy, tetha_0):
		#Matrix of joints including end effector
		#Considering 2D kinetatics, then a vector with z component = 1 is created
		self._p_xy = p_xy
		self._t_xy = t_xy
		self._a = np.matrix([[0],[0],[1]]).T
		self._alpha = 1
		self._tetha_0 = tetha_0
		
	def _create_jacobian(self):
		"""
		p_xy
		e |x y 0|
		p1|x y 0|
		p2|x y 0|
		"""
		#size = len(self._p_xy)-1
		#self.new_jacobian = np.zeros((3,size))
		#for i in range(size,-1,-1):
		#	self.new_jacobian[:,i] = np.cross(self._a, self._p_xy[size] - self._p_xy[size-i]).flatten() if i == 0 else np.cross(self._a, self._p_xy[i]).flatten()
			
		#print(self._p_xy)
		diff_e    = np.cross(self._a, self._p_xy[0])
		
		diff_e_p1 = np.cross(self._a, (self._p_xy[0] - self._p_xy[1]))
		
		diff_e_p2 = np.cross(self._a, (self._p_xy[0] - self._p_xy[2]))
		
		diff_e_p3 = np.cross(self._a, (self._p_xy[0] - self._p_xy[3]))
		
		jacobian_1 = [diff_e.item(0), diff_e_p1.item(0), diff_e_p2.item(0), diff_e_p3.item(0)]
		jacobian_2 = [diff_e.item(1), diff_e_p1.item(1), diff_e_p2.item(1), diff_e_p3.item(1)]
		jacobian_3 = [diff_e.item(2), diff_e_p1.item(2), diff_e_p2.item(2), diff_e_p3.item(2)]
		
		self.jacobian = np.matrix([
			jacobian_1,
			jacobian_2,
			jacobian_3
		])
		#print("jacobian \n{}\nnew jacobian \n{}".format(self.jacobian, self.new_jacobian))
		
		
	def _transpose_jacobian(self):
		self.jacobian_t = self.jacobian.T
		#print("jacobian transpose \n{}".format(self.jacobian_t))
		
	def _inverse_jacobian(self, val):
		self.jacobian_i = np.linalg.pinv(val)
		#print("jacobian inverse \n{}".format(self.jacobian_i))
		
	def _pseudo_inverse(self):
		self._inverse_jacobian(self.jacobian*self.jacobian_t)
		self.jacobian_pI = self.jacobian_t*self.jacobian_i
		#print("jacobian pseudoinverse \n{}".format(self.jacobian_pI))
		
	def _calc_diff_x(self):
		self.diff_x = self._t_xy - self._p_xy[0]
		#print(self.diff_x)
		
	def _calc_df_tetha(self):
		self.diff_t =  self._tetha_0[0] - self._tetha_0
		"""
		diff_e    = np.cross(self._p_xy[0], self._a)
		diff_e_p1 = np.cross((self._p_xy[0] - self._p_xy[1]), self._a)
		diff_e_p2 = np.cross((self._p_xy[0] - self._p_xy[2]), self._a)
		diff_e_p3 = np.cross((self._p_xy[0] - self._p_xy[3]), self._a)
		diff_t_1 = [diff_e.item(0), diff_e_p1.item(0), diff_e_p2.item(0), diff_e_p3.item(0)]
		diff_t_2 = [diff_e.item(1), diff_e_p1.item(1), diff_e_p2.item(1), diff_e_p3.item(1)]
		diff_t_3 = [diff_e.item(2), diff_e_p1.item(2), diff_e_p2.item(2), diff_e_p3.item(2)]
		
		self.diff_t = np.matrix([
			diff_t_1,
			diff_t_2,
			diff_t_3
		])
		"""
		
	def calc_diff_tetha(self):
		self._create_jacobian()
		self._transpose_jacobian()
		self._pseudo_inverse()
		self._calc_diff_x()
		self._calc_df_tetha()
		x_dot = self.jacobian_pI*self.diff_x.T
		op = self.jacobian_pI*self.jacobian
		ident = np.identity(op.shape[0])
		self.diff_tetha = self._alpha*x_dot#+(ident-op)*self.diff_t
		
	def degrees_2_radians(self, val):
		return (val*math.pi)/180

	def derivative_method(self):
		self._create_jacobian()
		self._transpose_jacobian()
		self._pseudo_inverse()
		self._calc_diff_x()
		self._calc_df_tetha()
		op = self.jacobian_pI*self.diff_x.T
		ident = np.identity(op.shape[0])
		x_dot = self.jacobian_pI*self.diff_x.T
		self._inverse_jacobian(ident-op)
		self.diff_tetha = x_dot.T+self.jacobian_i*np.power(self.diff_t/2,2)
		#print(self.diff_t)
	