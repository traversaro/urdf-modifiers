
from urdfpy import xyz_rpy_to_matrix, matrix_to_xyz_rpy
from urdfModifiers.core import modifier
from urdfModifiers.geometry import * 
from dataclasses import dataclass

import numpy as np


from enum import Enum

class LinkAndJointModifierV1Type(Enum):
    PURE_SCALING = 1
    SCALING_WITH_INITIAL_OFFSET_MANTAINED = 2
    SCALING_WITH_BOTH_INITIAL_AND_FINAL_OFFSET_MANTAINED = 3

## This class will be eventually ported in urdf-modifiers, it is just reported here to have a self-contained example 
@dataclass
class LinkAndJointModifierV1():
    """Class to modify link and coherently its parent joint in a URDF"""
    def __init__(self, link_element, joint_element):
        self.link_element = link_element
        self.joint_element = joint_element
        self.dimension = geometry.Side.DEPTH
        # Specify the case to use
        # self.scaling_type = LinkAndJointModifierV1Type.PURE_SCALING
        # self.scaling_type = LinkAndJointModifierV1Type.SCALING_WITH_INITIAL_OFFSET_MANTAINED
        self.scaling_type = LinkAndJointModifierV1Type.SCALING_WITH_BOTH_INITIAL_AND_FINAL_OFFSET_MANTAINED

        # The fact that we hardcode that self.dimension is geometry.Side.DEPTH means a few assumptions, let's enforce them
        
        # We assume that rpy of joint transform is 0.0 0.0 0.0, i.e. identity 
        if (not np.all(np.isclose(self.joint_element.origin[0:3:1,0:3:1], np.eye(3,3)))):
           raise RuntimeError("Link " + link_element.name + " does not respect assumption as it has non-zero rpy values in joint origin transform.")

        # We assume that the joint biggest offset is in the z direction
        # TODO: this creates problem on L_SH_Y_link, double check
        # joint_el_xyz_rpy = matrix_to_xyz_rpy(self.joint_element.origin)
        # print(joint_el_xyz_rpy)
        # index_to_change = self.get_index_to_change_in_link_frame()
        # if (not (max(joint_el_xyz_rpy[0:3]) == joint_el_xyz_rpy[index_to_change])):
        #    raise RuntimeError("Link " + link_element.name + " does not respect assumption that the direction of the joint offet is z.")

        # Compute index_to_change_in_visual_frame, or raise an expection if it is not possible
        deformation_direction_in_visual_frame = self.get_deformation_direction_in_visual_frame()

        if (np.all(np.isclose(deformation_direction_in_visual_frame,np.array([1.0,0.0,0.0])))):
            self.index_to_change_in_visual_frame = 0
        elif (np.all(np.isclose(deformation_direction_in_visual_frame,np.array([0.0,1.0,0.0])))):
            self.index_to_change_in_visual_frame = 1
        elif (np.all(np.isclose(deformation_direction_in_visual_frame,np.array([0.0,0.0,1.0])))):
            self.index_to_change_in_visual_frame = 2
        else:
            raise RuntimeError(f"Cannod add LinkAndJointModifier for {link_element.name} as the rpy of the visual origin does not permit to modify it.")

        geometry_type, visual_data = self.get_geometry(self.get_visual())
        if (self.index_to_change_in_visual_frame != 2 and geometry_type == geometry.Geometry.CYLINDER):
            raise RuntimeError(f"Cannod add LinkAndJointModifier for {link_element.name} as deformation direction is not parallel to the cylinder direction.")



    @classmethod
    def from_link_name(cls, link_name, robot):
        """Creates an instance of LinkAndJointModifier by passing the robot object and link name"""
        return cls(LinkAndJointModifierV1.get_link_element_by_name(link_name, robot), LinkAndJointModifierV1.get_joint_element_by_parent_link_name(link_name, robot) )

    @staticmethod
    def get_link_element_by_name(link_name, robot):
        """Explores the robot looking for the link whose name matches the first argument"""
        link_list = [corresponding_link for corresponding_link in robot.links if corresponding_link.name == link_name]
        if len(link_list) != 0:
            return link_list[0]
        else:
            return None

    @staticmethod
    def get_joint_element_by_parent_link_name(link_name, robot):
        """Explores the robot looking for the link whose name matches the first argument, and then return the joint connecting it to its child"""
        for jnt in robot.joints:
            if jnt.parent == link_name: 
                return jnt
        return None

    def modify(self, modifications):
        """Performs the dimension and density modifications to the current link"""
        # print(f"Before modifying {self.link_element.name}: visual_shape_length: {self.get_visual_shape_length()} joint_offset: {self.get_joint_offset()} visual_offset: {self.get_visual_offset()}  ")
        # print(f"Before modifying {self.link_element.name}: starting offset: {self.get_visual_offset()-self.get_visual_shape_length()/2} end_offset: {self.get_visual_offset()+self.get_visual_shape_length()/2-self.get_joint_offset()}")

        original_density = self.calculate_density()
        original_visual_shape_length = self.get_visual_shape_length()
        original_joint_offset = self.get_joint_offset()
        original_visual_offset = self.get_visual_offset()
        original_mass = self.get_mass()
        if "radius" in modifications:
            raise Exception('radius modification not supported by LinkAndJointModifier')
        if "dimension" in modifications:
            if modifications["dimension"][1]:
                raise Exception('Absolute dimention modification not supported by LinkAndJointModifier')
            else:
                self.modify_visual_shape_length(modifications, original_visual_shape_length, original_joint_offset, original_visual_offset)
                self.modify_visual_link_origin(modifications, original_visual_shape_length, original_joint_offset, original_visual_offset)
                self.modify_joint_origin(modifications, original_visual_shape_length, original_joint_offset, original_visual_offset)
        if "density" in modifications:
            raise Exception('density modification not supported by LinkAndJointModifier')
        if "mass" in modifications:
            raise Exception('mass modification not supported by LinkAndJointModifier')
        self.update_inertia()

        # print(f"After modifying {self.link_element.name}: visual_shape_length: {self.get_visual_shape_length()} joint_offset: {self.get_joint_offset()} visual_offset: {self.get_visual_offset()}  ")
        # print(f"After modifying {self.link_element.name}: starting offset: {self.get_visual_offset()-self.get_visual_shape_length()/2} end_offset: {self.get_visual_offset()+self.get_visual_shape_length()/2-self.get_joint_offset()}")


    def get_visual(self):
        """Returns the visual object of a link"""
        return self.link_element.visuals[0]

    def get_visual_shape_length(self):
        """Gets the significant length for a cylinder or box geometry"""
        geometry_type, visual_data = self.get_geometry(self.get_visual())
        if (geometry_type == geometry.Geometry.BOX):
            return visual_data.size[self.get_index_to_change_in_visual_frame()]
        elif (geometry_type == geometry.Geometry.CYLINDER):
            return visual_data.length
        else:
            return None

    def get_deformation_direction_in_link_frame(self):
        """Get the direction of the deformation in link frame"""
        direction = np.zeros(3)
        direction[self.get_index_to_change_in_link_frame()] = 1.0
        return direction

    def get_deformation_direction_in_visual_frame(self):
        """Get the direction of the deformation in visual frame"""
        link_R_visual = self.get_visual().origin[0:3:1, 0:3:1]
        visual_R_link = np.transpose(link_R_visual)
        return visual_R_link @ self.get_deformation_direction_in_link_frame()

    def get_index_to_change_in_link_frame(self):
        if (self.dimension == geometry.Side.WIDTH):
            index_to_change = 0
        if (self.dimension == geometry.Side.HEIGHT):
            index_to_change = 1
        if (self.dimension == geometry.Side.DEPTH):
            index_to_change = 2

        return index_to_change

    def get_index_to_change_in_visual_frame(self):
        # This value is cached in the construsctor
        return self.index_to_change_in_visual_frame

    def get_joint_offset(self):
        index_to_change_in_link_frame = self.get_index_to_change_in_link_frame()
        joint_el_xyz_rpy = matrix_to_xyz_rpy(self.joint_element.origin)
        return joint_el_xyz_rpy[index_to_change_in_link_frame]

    def get_visual_offset(self):
        visual_obj = self.get_visual()
        geometry_type, visual_data = self.get_geometry(visual_obj)
        index_to_change_in_link_frame = self.get_index_to_change_in_link_frame()
        visual_offset_xyz_rpy = matrix_to_xyz_rpy(visual_obj.origin)
        return visual_offset_xyz_rpy[index_to_change_in_link_frame]

    def modify_visual_shape_length(self, modifications, original_visual_shape_length, original_joint_offset, original_visual_offset):
        """Modifies a link's length, in a manner that is logical with its geometry"""
        geometry_type, visual_data = self.get_geometry(self.get_visual())
        scale = modifications["dimension"][0]
        if (self.scaling_type == LinkAndJointModifierV1Type.PURE_SCALING):
            new_length = scale*original_visual_shape_length
        elif (self.scaling_type == LinkAndJointModifierV1Type.SCALING_WITH_INITIAL_OFFSET_MANTAINED):
            new_length = scale*original_visual_shape_length
        elif (self.scaling_type == LinkAndJointModifierV1Type.SCALING_WITH_BOTH_INITIAL_AND_FINAL_OFFSET_MANTAINED):
            new_length = original_visual_shape_length - (1 - scale)*abs(original_joint_offset)

        if (geometry_type == geometry.Geometry.BOX):
            index_to_change_in_visual_frame = self.get_index_to_change_in_visual_frame()
            visual_data.size[index_to_change_in_visual_frame] = new_length
        elif (geometry_type == geometry.Geometry.CYLINDER):
            visual_data.length = new_length

    @staticmethod
    def get_visual_static(link):
        """Static method that returns the visual of a link"""
        return link.visuals[0]

    @staticmethod
    def get_geometry(visual_obj):
        """Returns the geometry type and the corresponding geometry object for a given visual"""
        if (visual_obj.geometry.box is not None):
            return [geometry.Geometry.BOX, visual_obj.geometry.box]
        if (visual_obj.geometry.cylinder is not None):
            return [geometry.Geometry.CYLINDER, visual_obj.geometry.cylinder]
        if (visual_obj.geometry.sphere is not None):
            return [geometry.Geometry.SPHERE, visual_obj.geometry.sphere]

    def calculate_volume(self, geometry_type, visual_data):
        """Calculates volume with the formula that corresponds to the geometry"""
        if (geometry_type == geometry.Geometry.BOX):
            return visual_data.size[0] * visual_data.size[1] * visual_data.size[2]
        elif (geometry_type == geometry.Geometry.CYLINDER):
            return math.pi * visual_data.radius ** 2 * visual_data.length
        elif (geometry_type == geometry.Geometry.SPHERE):
            return 4 * math.pi * visual_data.radius ** 3 / 3

    def get_mass(self):
        """Returns the link's mass"""
        return self.link_element.inertial.mass

    def set_mass(self, new_mass):
        """Sets the mass value to a new value"""
        self.link_element.inertial.mass = new_mass

    def calculate_density(self):
        """Calculates density from mass and volume"""
        geometry_type, visual_data = self.get_geometry(self.get_visual())
        return self.get_mass() / self.calculate_volume(geometry_type, visual_data)

    def modify_visual_link_origin(self, modifications, original_visual_shape_length, original_joint_offset, original_visual_offset):
        """Modifies the position of the origin by a given amount"""
        visual_obj = self.get_visual()
        geometry_type, visual_data = self.get_geometry(visual_obj)
        xyz_rpy = matrix_to_xyz_rpy(visual_obj.origin)
        if (geometry_type == geometry.Geometry.BOX):
            if (self.dimension is not None):
                if (self.dimension == geometry.Side.WIDTH):
                    index_to_change = 0
                if (self.dimension == geometry.Side.HEIGHT):
                    index_to_change = 1
                if (self.dimension == geometry.Side.DEPTH):
                    index_to_change = 2
                scale = modifications["dimension"][0]
                joint_el_xyz_rpy = matrix_to_xyz_rpy(self.joint_element.origin)
                # Case 0: 
                if (self.scaling_type == LinkAndJointModifierV1Type.PURE_SCALING):
                    xyz_rpy[index_to_change] = scale*original_visual_offset
                # Case 1:
                elif (self.scaling_type == LinkAndJointModifierV1Type.SCALING_WITH_INITIAL_OFFSET_MANTAINED):
                    xyz_rpy[index_to_change] = original_visual_offset + np.sign(original_joint_offset)*(scale - 1)*original_visual_shape_length/2
                # Case 2:
                elif (self.scaling_type == LinkAndJointModifierV1Type.SCALING_WITH_BOTH_INITIAL_AND_FINAL_OFFSET_MANTAINED):
                    xyz_rpy[index_to_change] = original_visual_offset + (scale - 1)*original_joint_offset/2
                visual_obj.origin = xyz_rpy_to_matrix(xyz_rpy) 
            else:
                print(f"Error modifying link {self.link_element.name}'s origin: Box geometry with no dimension")
        elif (geometry_type == geometry.Geometry.CYLINDER):
            # The cylinder is always aligned with the z direction
            index_to_change = 2
            scale = modifications["dimension"][0]
            joint_el_xyz_rpy = matrix_to_xyz_rpy(self.joint_element.origin)
            # xyz_rpy[index_to_change] = xyz_rpy[index_to_change] + (1 - scale)*joint_el_xyz_rpy[index_to_change] - np.sign(joint_el_xyz_rpy[index_to_change])*(1 - scale)*original_length/2
            # Case 0: 
            if (self.scaling_type == LinkAndJointModifierType.PURE_SCALING):
                xyz_rpy[index_to_change] = scale*original_visual_offset
            # Case 1:
            elif (self.scaling_type == LinkAndJointModifierType.SCALING_WITH_INITIAL_OFFSET_MANTAINED):
                xyz_rpy[index_to_change] = original_visual_offset + np.sign(original_joint_offset)*(scale - 1)*original_visual_shape_length/2
            # Case 2:
            elif (self.scaling_type == LinkAndJointModifierType.SCALING_WITH_BOTH_INITIAL_AND_FINAL_OFFSET_MANTAINED):
                xyz_rpy[index_to_change] = original_visual_offset + (scale - 1)*original_joint_offset/2
            visual_obj.origin = xyz_rpy_to_matrix(xyz_rpy) 
        elif (geometry_type == geometry.Geometry.SPHERE):
            return

    def modify_joint_origin(self, modifications, original_visual_shape_length, original_joint_offset, original_visual_offset):
        """Modifies the position of the origin by a given amount"""
        xyz_rpy = matrix_to_xyz_rpy(self.joint_element.origin)
        index_to_change_in_link_frame = self.get_index_to_change_in_link_frame()
        scale = modifications["dimension"][0]
        # For all cases, the joint offset is always just scale
        xyz_rpy[index_to_change_in_link_frame] =  scale * original_joint_offset
        self.joint_element.origin = xyz_rpy_to_matrix(xyz_rpy)

    def set_density(self, density):
        """Changes the mass of a link by preserving a given density."""
        geometry_type, visual_data = self.get_geometry(self.get_visual())
        volume = self.calculate_volume(geometry_type, visual_data)
        self.link_element.inertial.mass = volume * density

    def calculate_inertia(self):
        """Calculates inertia (ixx, iyy and izz) with the formula that corresponds to the geometry
        Formulas retrieved from https://en.wikipedia.org/wiki/List_of_moments_of_inertia"""
        geometry_type, visual_data = self.get_geometry(self.get_visual())
        mass = self.get_mass()
        if (geometry_type == geometry.Geometry.BOX):
            return mass / 12 * np.array([visual_data.size[1] ** 2 + visual_data.size[2] ** 2, 
                                visual_data.size[0] ** 2 + visual_data.size[2] ** 2,
                                visual_data.size[0] ** 2 + visual_data.size[1] ** 2])
        elif (geometry_type == geometry.Geometry.CYLINDER):
            i_xy_incomplete = (3 * visual_data.radius ** 2 + visual_data.length ** 2) / 12
            return mass * np.array([i_xy_incomplete, i_xy_incomplete, visual_data.radius ** 2 / 2])
        elif (geometry_type == geometry.Geometry.SPHERE):
            inertia = 2 * mass * visual_data.radius ** 2 / 5
            return np.array([inertia, inertia, inertia])

    def update_inertia(self):
        """Updates the inertia of a link to match its volume and mass."""
        if (self.link_element.inertial is not None):
            inertia = self.link_element.inertial.inertia
            new_inertia = self.calculate_inertia()
            new_inertia[new_inertia < 0.01] = 0.01
            for i in range(3):
                for j in range(3):
                    if (i == j):
                        inertia[i,j] = new_inertia[i]
                    else:
                        inertia[i,j] = 0

    def __str__(self):
        return f"LinkAndJointModifierV1 with link {self.link_element.name}, joint {self.joint_element.name} and dimension {self.dimension}"
