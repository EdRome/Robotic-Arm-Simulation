import sys

import pygame
from pygame.locals import *
from pygame.color import  *
from pygame.gfxdraw import *

import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util

import robotics as rob
import jacobian as jac
import numpy as np
import math
#import msvcrt

class TowArtAnalytic:

    def __init__(self, x, y, art_m, angle_1 = 0, angle_2 = 0,
                 length_1 = 0, length_2 = 0):
        self._angle_1 = angle_1
        self._angle_2 = angle_2
        self._solve_pos_ang_2 = False
        self._length_1 = length_1
        self._length_2 = length_2
        self._target_x = x
        self._target_y = y
        self._art_m = art_m
        self._epsilon = 0.0001 #Avoid division by small numbers

    @property
    def angle_1(self):
        return self._angle_1

    @angle_1.setter
    def angle_1(self, val):
        self._angle_1 = val

    @property
    def angle_2(self):
        return self._angle_2

    @angle_2.setter
    def angle_2(self, val):
        self._angle_2 = val

    @property
    def solve_pos_ang_2(self):
        return self._solve_pos_ang_2

    @solve_pos_ang_2.setter
    def solve_pos_ang_2(self, val):
        self._solve_pos_ang_2 = val

    @property
    def length_1(self):
        return self._length_1

    @length_1.setter
    def length_1(self, val):
        if val < 0:
            raise ValueError("Length 1 must be greater than 0")
        self._lenght_1 = val

    @property
    def length_2(self):
        return self._length_2

    @length_2.setter
    def lenght_2(self, val):
        if val < 0:
            raise ValueError("Length 2 must ve greater than 0")
        self._length_2 = val

    @property
    def target_x(self):
        return self._target_x

    @target_x.setter
    def target_x(self, val):
        self._target_x = val

    @property
    def target_y(self):
        return self._target_y

    @target_y.setter
    def target_y(self, val):
        self._target_y = val

    @property
    def art_m(self):
        return self._art_m

    @art_m.setter
    def art_m(self, val):
        self._art_m = val

    def calc_analytic(self):
        valid_solution = True
        target_dist_sqrt = np.power(self._target_x,2)+np.power(self._target_y,2)

        #Compute new value for angle 2
        cos_angle2_denom = 2*self._length_1*self._length_2
        if cos_angle2_denom > self._epsilon:
            cos_angle2 = (target_dist_sqrt - np.power(self._length_1,2) - np.power(self._length_2,2))/cos_angle2_denom

            #If solution is out of range, then it's not legal solution
            if ((cos_angle2 < -1.0) or (cos_angle2 > 1.0) ):
                valid_solution = False

            cos_angle2 = max(-1., min(1., cos_angle2))
            self._angle_2 = math.acos(cos_angle2)

            if not self._solve_pos_ang_2:
                self._angle_2 = -1*self._angle_2

            sin_angle2 = np.sin(self._angle_2)
        else:
            total_len_sqr = np.power((self._length_1 + self._length_2),2)
            if ( total_dist_sqrt < (total_len_sqr-self._epsilon) or
                 target_dis_sqrt > (total_len_sqr+self._epsilon) ):
                valid_solution = False

            self._angle_2 = 0.0
            cos_angle2 = 1.0
            sin_angle2 = 0.0

        tri_adjacent = self._length_1 + self._length_2*cos_angle2
        tri_opposite = self._length_2*sin_angle2

        tan_y = self._target_y*tri_adjacent - self._target_x*tri_opposite
        tan_x = self._target_x*tri_adjacent + self._target_y*tri_opposite

        self._angle_1 = math.atan2(tan_y, tan_x)

        return valid_solution

    def solving_sine(self, j_xy, e_xy):
        sin_num = (e_xy.item(0)-j_xy.item(0))*(self._target_y-j_xy.item(1))-(e_xy.item(1)-j_xy.item(1))*(self._target_x-j_xy.item(0))
        sin_den = magnitude(e_xy-j_xy)*magnitude(t_xy-j_xy)
        return sin_num/sin_den

    def solving_cosine(self, j_xy, e_xy):
        t_xy = np.matri([[self._target_x],[self._target_y]])
        cos_num = (e_xy-j_xy).dot(t_xy-j_xy)
        cos_den = magnitude(e_xy-j_xy)*magnitude(t_xy-j_xy)
        return cos_num/cos_den

    def simplify_angle(self, angle):
        angle = angle % (1.5*math.pi)
        if angle < -math.pi:
            angle += (1.5*math.pi)
        elif angle > math.pi:
            angle -= (1.5*math.pi)
        return angle
            

    def magnitude(self, vector):
        return math.sqrt(math.pow(vector.item(0),2)+math.pow(vector.item(1),2))

    def mod_space(self, arrival_dist):
        arrival_dist_sqrt = arrival_dist
        trivial_arc_length = 0.00001
        length = len(self._art_m)-2
        end_x = self._art_m[length][0].position.int_tuple[0]
        end_y = self._art_m[length][0].position.int_tuple[1]
        modified_bones = False
        for idx in range(length-1, -1, -1):
            cur_x = self._art_m[idx][0].position.int_tuple[0]
            cur_y = self._art_m[idx][0].position.int_tuple[1]
            
            cur_to_end_x = end_x - cur_x
            cur_to_end_y = end_y - cur_y
            cur_to_end = np.matrix([[cur_to_end_x],[cur_to_end_y]])
            cur_to_end_mag = self.magnitude(cur_to_end)

            cur_to_tar_x = self._target_x - cur_x
            cur_to_tar_y = self._target_y - cur_y
            cur_to_tar = np.matrix([[cur_to_tar_x],[cur_to_tar_y]])
            cur_to_tar_mag = self.magnitude(cur_to_tar)

            cos_rot_ang = 0.
            sin_rot_ang = 0.
            end_tar_mag = cur_to_end_mag*cur_to_tar_mag
            if end_tar_mag <= self._epsilon:
                cos_rot_ang = 1
                sin_rot_ang = 0
            else:
                cos_rot_ang = (cur_to_end_x*cur_to_tar_x + cur_to_end_y*cur_to_tar_y)/end_tar_mag
                sin_rot_ang = (cur_to_end_x*cur_to_tar_y - cur_to_end_y*cur_to_tar_x)/end_tar_mag
                
            
            rot_ang = math.acos(max(-1, min(1,cos_rot_ang)))
            if sin_rot_ang < 0.0:
                rot_ang = -rot_ang

            end_x = cur_x + cos_rot_ang*cur_to_end_x - sin_rot_ang*cur_to_end_y
            end_y = cur_y + sin_rot_ang*cur_to_end_x + cos_rot_ang*cur_to_end_y
            self._art_m[idx][0].angle = self.simplify_angle(self._art_m[idx][0].angle + rot_ang)
            
            end_to_tar_x = (self._target_x-end_x)
            end_to_tar_y = (self._target_y-end_y)
            if (end_to_tar_x**2 + end_to_tar_y**2) <= arrival_dist_sqrt and (end_to_tar_x**2 + end_to_tar_y**2) <= arrival_dist_sqrt**2:
                print("Return: "+str((end_to_tar_x**2 + end_to_tar_y**2))+" <= " + str(arrival_dist_sqrt))
                print("Success")
                return 0

            if (not modified_bones) and ((np.absolute(rot_ang)*cur_to_end_mag) > trivial_arc_length):
                modified_bones = True

        if modified_bones:
            print("Processing")
            return 1
        else:
            print("Fail")
            return 2
                

#def wait_key():
#    msvcrt.getch()

def calc_pos(xy_t, xy_0, idx):
    dx = xy_t.item(0)-xy_0[idx].item(0)
    dy = xy_t.item(1)-xy_0[idx].item(1)
    ang = math.atan2(dy, dx)
    kin_2d = rob.kinematics_2d(ang, dx, dy)
    hom_mat = kin_2d.homogenous_matrix()
    #print(hom_mat)
    A_m = kin_2d.translate(hom_mat, xy_0[idx].item(0), xy_0[idx].item(1))
    return A_m, ang

def create_arm():
    art_m = [[]]
    poi_m = [[]]
    joi_m = [[]]

    ## Origin Fixed Point
    base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    base_shape = pymunk.Circle(base_body, 10)
    base_shape.color = (255,50,50)
    base_body.position = 150,150

    poi_m.insert(0, [base_body, base_shape])

    art01_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    art01_shape = pymunk.Segment(art01_body, (150,150), (250,150), 1)
    art01_shape.color = (50,255,50)

    art_m.insert(0, [art01_body, art01_shape])

    p1_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    p1_shape = pymunk.Circle(p1_body, 10)
    p1_shape.color = (255,50,50)
    p1_body.position = 250,150

    poi_m.insert(1, [p1_body, p1_shape])

    art12_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    art12_shape = pymunk.Segment(art12_body, (250,150), (350,150), 1)
    art12_shape.color = (50,255,50)

    art_m.insert(1, [art12_body, art12_shape])

    p2_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    p2_shape = pymunk.Circle(p2_body, 10)
    p2_shape.color = (255,50,50)
    p2_body.position = 350,150

    poi_m.insert(2, [p2_body, p2_shape])

    art23_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    art23_shape = pymunk.Segment(art23_body, (350,150), (450, 150), 1)
    art23_shape.color = (50,255,50)

    art_m.insert(2, [art23_body, art23_shape])

    p3_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    p3_shape = pymunk.Circle(p3_body, 10)
    p3_shape.color = (255,50,50)
    p3_body.position = 450,150

    poi_m.insert(3, [p3_body, p3_shape])

    art34_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    art34_shape = pymunk.Segment(art23_body, (450,150), (550, 150), 1)
    art34_shape.color = (50,255,50)

    art_m.insert(3, [art34_body, art34_shape])    

    joi_m.insert(0, pymunk.PinJoint(base_body, p1_body, (0,0), (0,0)))
    joi_m.insert(1,pymunk.PinJoint(p1_body, p2_body, (0,0), (0,0)))
    joi_m.insert(2,pymunk.PinJoint(p2_body, p3_body, (0,0), (0,0)))

    
    return poi_m, art_m, joi_m

width, height = 650,600

def simplify_angle(angle):
    angle = angle % (1.5*math.pi)
    if angle < -math.pi:
        angle += (1.5*math.pi)
    elif angle > math.pi:
        angle -= (1.5*math.pi)
    return angle

def main():
    pygame.init()
    running = True
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    screen = pygame.display.set_mode((width,height))
    space = pymunk.Space()
    space.gravity = 0,-1000
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    cannon_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    cannon_shape = pymunk.Circle(cannon_body, 10)
    cannon_shape.color = (50,50,255)
    cannon_body.position = 300,50
    space.add(cannon_shape)

    poi_m, art_m, joi_m = create_arm()
    size = len(poi_m)-1
    xy_0 = np.zeros((size,2))
    for i in range(size):
        space.add(poi_m[i][1])
        space.add(art_m[i][1])
        for  j in range(2):
            xy_0[i][j] = art_m[i][0].position.int_tuple[j]

    space.add(joi_m[0], joi_m[1], joi_m[2])

    art_analytic = TowArtAnalytic(x=cannon_body.position.int_tuple[0], 
                                  y=cannon_body.position.int_tuple[1],
                                  length_1=100, length_2=100,
                                  art_m=poi_m)
    running = True

    screen.fill(pygame.color.THECOLORS["black"])
    space.debug_draw(draw_options)
    pygame.display.flip()
    count = 1
    
    t_xy = np.matrix([cannon_body.position.int_tuple[0],cannon_body.position.int_tuple[1],1])
    e = math.sqrt(np.power(cannon_body.position.int_tuple[0] - poi_m[3][0].position.int_tuple[0],2) 
                  + np.power(cannon_body.position.int_tuple[1] - poi_m[3][0].position.int_tuple[1],2))
    while e > 0.001:        
        p_xy = np.matrix([
            [poi_m[3][0].position.int_tuple[0],poi_m[3][0].position.int_tuple[1],1],
	        [poi_m[0][0].position.int_tuple[0],poi_m[0][0].position.int_tuple[1],1],
	        [poi_m[1][0].position.int_tuple[0],poi_m[1][0].position.int_tuple[1],1],
            [poi_m[2][0].position.int_tuple[0],poi_m[2][0].position.int_tuple[1],1]])
        tetha_0 = np.matrix([[poi_m[3][0].angle], 
                             [poi_m[0][0].angle], 
                             [poi_m[1][0].angle], 
                             [poi_m[2][0].angle]])
        j = jac.IK_2D_Jacobian(p_xy, t_xy, tetha_0)
        #j.calc_diff_tetha()
        j.derivative_method()
        ang_1 = j.degrees_2_radians(j.diff_tetha.item(0))+poi_m[3][0].angle
        ang_1 = simplify_angle(ang_1)
        ang_2 = j.degrees_2_radians(j.diff_tetha.item(1))+poi_m[0][0].angle
        ang_2 = simplify_angle(ang_2)
        ang_3 = j.degrees_2_radians(j.diff_tetha.item(2))+poi_m[1][0].angle
        ang_3 = simplify_angle(ang_3)
        ang_4 = j.degrees_2_radians(j.diff_tetha.item(3))+poi_m[2][0].angle
        ang_4 = simplify_angle(ang_4)
        
        poi_m[0][0].position = Vec2d(150,150)
        poi_m[0][0].angle = ang_2
        poi_m[0][0].angular_velocity = 1
           
        poi_m[1][0].angle = ang_3
        poi_m[1][0].angular_velocity = 1
        poi_m[1][0].position = poi_m[0][0].position + Vec2d(100,0).rotated(poi_m[0][0].angle)
            
        poi_m[2][0].angle = ang_1
        poi_m[2][0].angular_velocity = 1
        poi_m[2][0].position = poi_m[1][0].position + Vec2d(100,0).rotated(poi_m[1][0].angle)

        poi_m[3][0].angle = ang_4
        poi_m[3][0].angular_velocity = 1
        poi_m[3][0].position = poi_m[2][0].position + Vec2d(100,0).rotated(poi_m[2][0].angle)       
        
        e = math.sqrt(np.power(poi_m[3][0].position.int_tuple[0]-cannon_body.position.int_tuple[0],2) 
                      + np.power(poi_m[3][0].position.int_tuple[1]-cannon_body.position.int_tuple[1],2))        
        
    print("ang_2 {}, ang_3 {}, ang_1 {}, ang_4 {}".format(ang_2, ang_3, ang_1, ang_4))
    ang_2_aux = 0
    ang_3_aux = 0
    ang_1_aux = 0
    ang_4_aux = 0
    #paso = 0.0000000000000001
    paso = 0.01
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    sys.exit()

        #if art_analytic.art_m != poi_m:
        #    art_analytic.art_m = poi_m
        #arrival_dist_x = cannon_body.position.int_tuple[0]-poi_m[2][0].position.int_tuple[0]
        #arrival_dist_y = cannon_body.position.int_tuple[1]-poi_m[2][0].position.int_tuple[1]
        #arrival_dist = math.sqrt(arrival_dist_x**2+arrival_dist_y**2)
        
        #result = art_analytic.mod_space(arrival_dist)
        
        #poi_m[0][0].position = art_analytic.art_m[0][0].position
        #poi_m[0][0].angle = art_analytic.art_m[0][0].angle
        #poi_m[0][0].angular_velocity = 1
            
        #poi_m[1][0].angle = art_analytic.art_m[1][0].angle
        #poi_m[1][0].angular_velocity = 1
        #poi_m[1][0].position = poi_m[0][0].position + Vec2d(100,0).rotated(poi_m[0][0].angle)
            
        #poi_m[2][0].angle = art_analytic.art_m[2][0].angle
        #poi_m[2][0].angular_velocity = 1
        #poi_m[2][0].position = poi_m[1][0].position + Vec2d(100,0).rotated(poi_m[1][0].angle)

        #poi_m[3][0].angle = art_analytic.art_m[3][0].angle
        #poi_m[3][0].angular_velocity = 1
        #poi_m[3][0].position = poi_m[2][0].position + Vec2d(100,0).rotated(poi_m[2][0].angle)
        """
        p_xy = np.matrix([
            [poi_m[3][0].position.int_tuple[0],poi_m[3][0].position.int_tuple[1],1],
	        [poi_m[0][0].position.int_tuple[0],poi_m[0][0].position.int_tuple[1],1],
	        [poi_m[1][0].position.int_tuple[0],poi_m[1][0].position.int_tuple[1],1],
            [poi_m[2][0].position.int_tuple[0],poi_m[2][0].position.int_tuple[1],1]])
        t_xy = np.matrix([cannon_body.position.int_tuple[0],cannon_body.position.int_tuple[1],1])
        tetha_0 = np.matrix([[poi_m[3][0].angle], 
                             [poi_m[0][0].angle], 
                             [poi_m[1][0].angle], 
                             [poi_m[2][0].angle]])
        j = jac.IK_2D_Jacobian(p_xy, t_xy, tetha_0)
        j.calc_diff_tetha()
        #j.derivative_method()
        ang_1 = j.degrees_2_radians(j.diff_tetha.item(0))+poi_m[3][0].angle
        ang_1 = simplify_angle(ang_1)
        ang_2 = j.degrees_2_radians(j.diff_tetha.item(1))+poi_m[0][0].angle
        ang_2 = simplify_angle(ang_2)
        ang_3 = j.degrees_2_radians(j.diff_tetha.item(2))+poi_m[1][0].angle
        ang_3 = simplify_angle(ang_3)
        ang_4 = j.degrees_2_radians(j.diff_tetha.item(3))+poi_m[2][0].angle
        ang_4 = simplify_angle(ang_4)
        """
        #kin_2d = rob.kinematics_2d(np.matrix([ang_2, ang_3]))
        #kin_2d
        
        poi_m[0][0].position = Vec2d(150,150)
        poi_m[0][0].angle = ang_2_aux
        poi_m[0][0].angular_velocity = 1
           
        poi_m[1][0].angle = ang_3_aux
        poi_m[1][0].angular_velocity = 1
        poi_m[1][0].position = poi_m[0][0].position + Vec2d(100,0).rotated(poi_m[0][0].angle)
            
        poi_m[2][0].angle = ang_1_aux
        poi_m[2][0].angular_velocity = 1
        poi_m[2][0].position = poi_m[1][0].position + Vec2d(100,0).rotated(poi_m[1][0].angle)

        poi_m[3][0].angle = ang_4_aux
        poi_m[3][0].angular_velocity = 1
        poi_m[3][0].position = poi_m[2][0].position + Vec2d(100,0).rotated(poi_m[2][0].angle)

        
        if ang_2_aux != ang_2:
            ang_2_aux = ang_2_aux+(paso*-1) if ang_2 < 0 else ang_2_aux+paso
            print(ang_2_aux)
        if ang_3_aux != ang_3:
            ang_3_aux = ang_3_aux+(paso*-1) if ang_3 < 0 else ang_3_aux+paso
            print(ang_3_aux)
        if ang_4_aux != ang_4:
            ang_4_aux = ang_4_aux+(paso*-1) if ang_4 < 0 else ang_4_aux+paso
            print(ang_4_aux)
        if ang_1_aux != ang_1:
            ang_1_aux = ang_1_aux+(paso*-1) if ang_1 < 0 else ang_1_aux+paso
            print(ang_1_aux)
    
        screen.fill(pygame.color.THECOLORS["black"])
        space.debug_draw(draw_options)
        pygame.display.flip()
        
        fps = 60
        dt = 1./fps
        space.step(dt)
        
        clock.tick(fps)

if __name__ == '__main__':
    sys.exit(main())    
