import os
import select
import sys

#librerie importate da me 
import math
import struct
import numpy as np

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import PointCloud2
import rclpy
from rclpy.clock import Clock
from rclpy.qos import QoSProfile

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

BURGER_MAX_LIN_VEL = 2.22
BURGER_MAX_ANG_VEL = 2.84

WAFFLE_MAX_LIN_VEL = 2.26
WAFFLE_MAX_ANG_VEL = 1.82

LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.01

TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'waffle')

msg = """
Control Your TurtleBot3!
---------------------------
Moving around:
        w
   a    s    d
        x

w/x : increase/decrease linear velocity (Burger : ~ 0.22, Waffle and Waffle Pi : ~ 0.26)
a/d : increase/decrease angular velocity (Burger : ~ 2.84, Waffle and Waffle Pi : ~ 1.82)

space key, s : force stop

CTRL-C to quit
"""

e = """
Communications Failed
"""

# vettore dei guadagni 
APF_CONFIG = {
    'repulsive_gain': 2.0,      # Guadagno forza repulsiva
    'attractive_gain': 1.0,     # Guadagno forza attrattiva  
    'LIN_VELOCITY_GAIN': 3.0,
    'ANG_VELOCITY_GAIN': 3.0,
}

# get_key serve a prendere il singolo tasto premuto.
def get_key(settings):
    if os.name == 'nt': #definisce quale sistema operativo sta usando  
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def print_vels(target_linear_velocity, target_angular_velocity):
    print('currently:\tlinear velocity {0}\t angular velocity {1} '.format(
        target_linear_velocity,
        target_angular_velocity))

#introduce una rampa per aumentare la vel in maniera incrementale
def make_simple_profile(output_vel, input_vel, slop):
    if input_vel > output_vel:
        output_vel = min(input_vel, output_vel + slop)
    elif input_vel < output_vel:
        output_vel = max(input_vel, output_vel - slop)
    else:
        output_vel = input_vel

    return output_vel
    

# make_simple_profile serve a saturare l'uscita nel caso in cui chiediamo un input troppo alto 
def constrain(input_vel, low_bound, high_bound):
    if input_vel < low_bound:
        input_vel = low_bound
    elif input_vel > high_bound:
        input_vel = high_bound
    else:
        input_vel = input_vel

    return input_vel


def check_linear_limit_velocity(velocity):
    if TURTLEBOT3_MODEL == 'burger':
        return constrain(velocity, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
    else:
        return constrain(velocity, -WAFFLE_MAX_LIN_VEL, WAFFLE_MAX_LIN_VEL)


def check_angular_limit_velocity(velocity):
    if TURTLEBOT3_MODEL == 'burger':
        return constrain(velocity, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)
    else:
        return constrain(velocity, -WAFFLE_MAX_ANG_VEL, WAFFLE_MAX_ANG_VEL)
    
def update_velocity(pub,ROS_DISTRO,lin_velocity, ang_velocity):
    if ROS_DISTRO == 'humble':
        twist = Twist()
        twist.linear.x = lin_velocity
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = ang_velocity
        pub.publish(twist)

    else:
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = Clock().now().to_msg()
        twist_stamped.header.frame_id = ''
        twist_stamped.twist.linear.x = lin_velocity
        twist_stamped.twist.linear.y = 0.0
        twist_stamped.twist.linear.z = 0.0

        twist_stamped.twist.angular.x = 0.0
        twist_stamped.twist.angular.y = 0.0
        twist_stamped.twist.angular.z = ang_velocity
        pub.publish(twist)


class ArtificialPotentialField:
    #definisco le varie funzioni della classe 
    # nella funzione init definisco la configurazione e il vettore degli ostacoli 
    def __init__(self, config):
        self.config = config
        self.obstacles = []  
    
    def compute_attractive_forces(self, v_lin, x_dist, y_dist):
        
        #attractive_force = self.config['attractive_gain'] * (v_lin * np.sin (np.arctan2(y_dist, x_dist)))
        attractive_force_x = self.config['attractive_gain'] * v_lin
        attractive_force_y = attractive_force_x * (y_dist / x_dist) # angolo verde = atan di (y/x) e si semplifica con tan
        attractive_force = [attractive_force_x , attractive_force_y]
        return attractive_force
    
    def compute_repulsive_forces(self, x_dist, y_dist):

        gamma = 3
        eta_0 = 1
        eta_i = x_dist * np.sin (np.arctan2(y_dist, x_dist))
        eta_i = max(abs(eta_i), 0.1)
        APF_computation = (self.config['repulsive_gain']/eta_i**2) * ((1/eta_i - 1/eta_0)**(gamma - 1))
        repulsive_force_y = APF_computation / np.cos (np.atan2 (y_dist, x_dist))
        repulsive_force_x = repulsive_force_y * np.sin (np.atan2 (y_dist, x_dist))
        repulsive_force = [repulsive_force_x , repulsive_force_y]
        return repulsive_force

    def compute_proportional_velocities(self, control_linear_velocity, distance):
        
        attractive_force = self.compute_attractive_forces (control_linear_velocity,distance[0], distance[1])
        repulsive_force = self.compute_repulsive_forces(distance[0], distance[1])
        total_forces = [attractive_force[0] - repulsive_force[0] , attractive_force[1] - repulsive_force[1]]
        
        if attractive_force[0] != 0:
            prop_lin_vel = total_forces[0] / attractive_force[0]
        else:
            prop_lin_vel = 0
        
        if attractive_force[1] !=0:
            prop_ang_vel = total_forces[1] / attractive_force[1]
        else:
            prop_ang_vel = 0
            
        prop_velocities = [prop_lin_vel , prop_ang_vel]
         
        return prop_velocities


class APFController:

    def __init__(self, node, ros_distro, control_linear_velocity, control_angular_velocity):
        self.node = node
        self.ros_distro = ros_distro
        self.control_linear_velocity = control_linear_velocity
        self.control_angular_velocity = control_angular_velocity
        self.apf = ArtificialPotentialField(APF_CONFIG)

    def point_cloud_callback(self, msg):
        try:
            obstacles = self.get_obstacle_positions(msg)
            self.apf.obstacles = obstacles

        except Exception as e:
            self.node.get_logger().error(f"Errore pointcloud: {e}")

    def get_obstacle_positions(self, msg):
        
        obstacles = []
        
        # 1. PRENDI I PRIMI 3 CAMPI (di solito sono x,y,z)
        x_offset = 0    # Il primo campo è quasi sempre X
        y_offset = 4    # Il secondo è Y (4 bytes dopo)  
        z_offset = 8    # Il terzo è Z (8 bytes dopo) - QUESTA È LA PROFONDITÀ!
        
         # 2. SCORRI TUTTI I PUNTI
        for i in range(0, len(msg.data), 30 * msg.point_step):
            point_data = msg.data[i:i + msg.point_step]
            #print('point_data:', point_data)
            
            # 3. ESTRAI COORDINATE 3D
            x = struct.unpack('f', point_data[x_offset:x_offset+4])[0] 
            y = struct.unpack('f', point_data[y_offset:y_offset+4])[0]  
            z = struct.unpack('f', point_data[z_offset:z_offset+4])[0] 
            
            #if not np.isnan(x) and not np.isinf(x) and \
               #not np.isnan(y) and not np.isinf(y) and \
               #not np.isnan(z) and not np.isinf(z):  # Solo primi 5 punti per non intasare
            #if not np.isinf(x):
            #    print(f"Punto {i//msg.point_step}: x={x:.3f}, y={y:.3f}, z={z:.3f}")
            
            if 0.3 < math.sqrt(x**2 + y**2) <= 1.0 and z >= 0.4:   
                obstacles.append([x, y, z])
       
        
        return obstacles


def main():
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    rclpy.init()
    ROS_DISTRO = os.environ.get('ROS_DISTRO')
    qos = QoSProfile(depth=10)
    node = rclpy.create_node('apt_keyboard')

    status = 0
    target_linear_velocity = 0.0
    target_angular_velocity = 0.0
    control_linear_velocity = 0.0
    control_angular_velocity = 0.0

    # Crea il controller APF
   #apf_controller = APFController(node, ROS_DISTRO, control_linear_velocity, control_angular_velocity,pub)

   
    if ROS_DISTRO == 'humble':
        pub = node.create_publisher(Twist, 'cmd_vel', qos)
            # Crea il controller APF
        apf_controller = APFController(node, ROS_DISTRO, control_linear_velocity, control_angular_velocity)
        sub_pointcloud = node.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            apf_controller.point_cloud_callback,
            qos)
    else:
        pub = node.create_publisher(TwistStamped, 'cmd_vel', qos)
            # Crea il controller APF
        apf_controller = APFController(node, ROS_DISTRO, control_linear_velocity, control_angular_velocity)
        sub_pointcloud = node.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            apf_controller.point_cloud_callback,
            qos)

    try:
        print(msg)

        while True:
            rclpy.spin_once(node, timeout_sec=0.01)
           
            if len(apf_controller.apf.obstacles) > 0:
                print("APF attivo, numero ostacoli:", len(apf_controller.apf.obstacles))

                # Modalità APF
                for obstacle in apf_controller.apf.obstacles:
                    x, y, z = obstacle
                    distance = [x, y, z]

                    velocities = apf_controller.apf.compute_proportional_velocities(
                        control_linear_velocity, 
                        distance
                    )
                    
                    final_lin_vel = control_linear_velocity - (np.sign (control_linear_velocity) * velocities[0] * APF_CONFIG['LIN_VELOCITY_GAIN']) 
                    final_ang_vel = control_angular_velocity - (np.sign (control_linear_velocity) * velocities[1] * APF_CONFIG['ANG_VELOCITY_GAIN'])

                update_velocity(pub,ROS_DISTRO, final_lin_vel, final_ang_vel)

            else:

                key = get_key(settings)
                if key == 'w':
                    target_linear_velocity =\
                        check_linear_limit_velocity(target_linear_velocity + LIN_VEL_STEP_SIZE)
                    status = status + 1
                    print_vels(target_linear_velocity, target_angular_velocity)
                elif key == 'x':
                    target_linear_velocity =\
                        check_linear_limit_velocity(target_linear_velocity - LIN_VEL_STEP_SIZE)
                    status = status + 1
                    print_vels(target_linear_velocity, target_angular_velocity)
                elif key == 'a':
                    target_angular_velocity =\
                        check_angular_limit_velocity(target_angular_velocity + ANG_VEL_STEP_SIZE)
                    status = status + 1
                    print_vels(target_linear_velocity, target_angular_velocity)
                elif key == 'd':
                    target_angular_velocity =\
                        check_angular_limit_velocity(target_angular_velocity - ANG_VEL_STEP_SIZE)
                    status = status + 1
                    print_vels(target_linear_velocity, target_angular_velocity)
                elif key == ' ' or key == 's':
                    target_linear_velocity = 0.0
                    control_linear_velocity = 0.0
                    target_angular_velocity = 0.0
                    control_angular_velocity = 0.0
                    print_vels(target_linear_velocity, target_angular_velocity)
                else:
                    if (key == '\x03'):
                        break

                if status == 20:
                    print(msg)
                    status = 0

                control_linear_velocity = make_simple_profile(
                    control_linear_velocity,
                    target_linear_velocity,
                    (LIN_VEL_STEP_SIZE / 2.0))
  
                control_angular_velocity = make_simple_profile(
                    control_angular_velocity,
                    target_angular_velocity,
                    (ANG_VEL_STEP_SIZE / 2.0))
                
                update_velocity(pub, ROS_DISTRO, control_linear_velocity, control_angular_velocity)
            
            


    except Exception as e:
        print(e)

    finally:
        if ROS_DISTRO == 'humble':
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            pub.publish(twist)
        else:
            twist_stamped = TwistStamped()
            twist_stamped.header.stamp = Clock().now().to_msg()
            twist_stamped.header.frame_id = ''
            twist_stamped.twist.linear.x = control_linear_velocity
            twist_stamped.twist.linear.y = 0.0
            twist_stamped.twist.linear.z = 0.0
            twist_stamped.twist.angular.x = 0.0
            twist_stamped.twist.angular.y = 0.0
            twist_stamped.twist.angular.z = control_angular_velocity
            pub.publish(twist_stamped)

        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == '__main__': 
    main()