import time 
import numpy as np

import urdfModifiers
import idyntree.bindings as iDynTree

from dataclasses import dataclass
from urdfpy import xyz_rpy_to_matrix, matrix_to_xyz_rpy
from urdfModifiers.core import modifier
import math
import numpy as np
from urdfModifiers.geometry import * 

from enum import Enum

class LinkAndJointModifierType(Enum):
    PURE_SCALING = 1
    SCALING_WITH_INITIAL_OFFSET_MANTAINED = 2
    SCALING_WITH_BOTH_INITIAL_AND_FINAL_OFFSET_MANTAINED = 3

## This class will be eventually ported in urdf-modifiers, it is just reported here to have a self-contained example 
@dataclass
class LinkAndJointModifier():
    """Class to modify link and coherently its parent joint in a URDF"""
    def __init__(self, link_element, joint_element):
        self.link_element = link_element
        self.joint_element = joint_element
        # TODO: this means that we have this assumptions:
        # * that all the link have the same orientation (i.e. rpy element of all joints is 0.0 0.0 0.0)
        # * that the dimension along which the modification is done is always along the z
        self.dimension = geometry.Side.DEPTH
        # Specify the case to use
        # self.scaling_type = LinkAndJointModifierType.PURE_SCALING
        # self.scaling_type = LinkAndJointModifierType.SCALING_WITH_INITIAL_OFFSET_MANTAINED
        self.scaling_type = LinkAndJointModifierType.SCALING_WITH_BOTH_INITIAL_AND_FINAL_OFFSET_MANTAINED

    @classmethod
    def from_link_name(cls, link_name, robot):
        """Creates an instance of LinkAndJointModifier by passing the robot object and link name"""
        return cls(LinkAndJointModifier.get_link_element_by_name(link_name, robot), LinkAndJointModifier.get_joint_element_by_parent_link_name(link_name, robot) )

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
        if modifications.radius:
            raise Exception('radius modification not supported by LinkAndJointModifier')
        if modifications.dimension:
            if modifications.dimension.absolute:
                raise Exception('Absolute dimention modification not supported by LinkAndJointModifier')
            else:
                self.modify_visual_shape_length(modifications, original_visual_shape_length, original_joint_offset, original_visual_offset)
                self.modify_visual_link_origin(modifications, original_visual_shape_length, original_joint_offset, original_visual_offset)
                self.modify_joint_origin(modifications, original_visual_shape_length, original_joint_offset, original_visual_offset)
        if modifications.density:
            raise Exception('density modification not supported by LinkAndJointModifier')
        if modifications.mass:
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
            if (self.dimension is not None):
                if (self.dimension == geometry.Side.WIDTH):
                    return visual_data.size[0]
                elif (self.dimension == geometry.Side.HEIGHT):
                    return visual_data.size[1]
                elif (self.dimension == geometry.Side.DEPTH):
                    return visual_data.size[2]
            else:
                print(f"Error getting length for link {self.link_element.name}'s volume: Box geometry with no dimension")
        elif (geometry_type == geometry.Geometry.CYLINDER):
            return visual_data.length
        else:
            return None

    def get_index_to_change(self):
        visual_obj = self.get_visual()
        geometry_type, visual_data = self.get_geometry(visual_obj)
        if (geometry_type == geometry.Geometry.CYLINDER):
            index_to_change = 2
        else:
            if (self.dimension == geometry.Side.WIDTH):
                index_to_change = 0
            if (self.dimension == geometry.Side.HEIGHT):
                index_to_change = 1
            if (self.dimension == geometry.Side.DEPTH):
                index_to_change = 2

        return index_to_change

    def get_joint_offset(self):
        index_to_change = self.get_index_to_change()
        joint_el_xyz_rpy = matrix_to_xyz_rpy(self.joint_element.origin)
        return joint_el_xyz_rpy[index_to_change]

    def get_visual_offset(self):
        visual_obj = self.get_visual()
        geometry_type, visual_data = self.get_geometry(visual_obj)
        index_to_change = self.get_index_to_change()
        visual_offset_xyz_rpy = matrix_to_xyz_rpy(visual_obj.origin)
        return visual_offset_xyz_rpy[index_to_change]

    def modify_visual_shape_length(self, modifications, original_visual_shape_length, original_joint_offset, original_visual_offset):
        """Modifies a link's length, in a manner that is logical with its geometry"""
        geometry_type, visual_data = self.get_geometry(self.get_visual())
        scale = modifications.dimension.value
        if (self.scaling_type == LinkAndJointModifierType.PURE_SCALING):
            new_length = scale*original_visual_shape_length
        elif (self.scaling_type == LinkAndJointModifierType.SCALING_WITH_INITIAL_OFFSET_MANTAINED):
            new_length = scale*original_visual_shape_length
        elif (self.scaling_type == LinkAndJointModifierType.SCALING_WITH_BOTH_INITIAL_AND_FINAL_OFFSET_MANTAINED):
            new_length = original_visual_shape_length - (1 - scale)*abs(original_joint_offset)

        if (geometry_type == geometry.Geometry.BOX):
            index_to_change = self.get_index_to_change()
            visual_data.size[index_to_change] = new_length
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
                scale = modifications.dimension.value
                joint_el_xyz_rpy = matrix_to_xyz_rpy(self.joint_element.origin)
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
            else:
                print(f"Error modifying link {self.link_element.name}'s origin: Box geometry with no dimension")
        elif (geometry_type == geometry.Geometry.CYLINDER):
            # The cylinder is always aligned with the z direction
            index_to_change = 2
            scale = modifications.dimension.value
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
        index_to_change = self.get_index_to_change()
        scale = modifications.dimension.value
        # For all cases, the joint offset is always just scale
        xyz_rpy[index_to_change] =  scale * original_joint_offset
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
        return f"Link modifier with name {self.link_element.name}, dimension {self.dimension}"

def GetIDynTreeTransfMatrix(s,H): 
    
    N_DoF = len(s)
    s_idyntree =iDynTree.VectorDynSize(N_DoF)
    pos_iDynTree = iDynTree.Position()
    R_iDynTree = iDynTree.Rotation()
        
    for i in range(N_DoF):
        s_idyntree.setVal(i,s[i])
    
    R_iDynTree.FromPython(H[:3,:3])
    for i in range(3):
        pos_iDynTree.setVal(i,float(H[i,3]))
        for j in range(3):
            R_iDynTree.setVal(j,i,float(H[j,i]))
    
    return s_idyntree, pos_iDynTree, R_iDynTree

## Visualization helper
class OutputVisualization(): 
    def __init__(self) -> None:
        pass

    def prepare_visualization(self):
        self.viz = iDynTree.Visualizer()  
        vizOpt = iDynTree.VisualizerOptions()
        vizOpt.winWidth = 1500
        vizOpt.winHeight = 1500 
        self.viz.init(vizOpt)

        self.env = self.viz.enviroment()
        self.env.setElementVisibility('floor_grid',True)
        self.env.setElementVisibility('world_frame',False)
        self.viz.setColorPalette("meshcat")
        self.env.setElementVisibility('world_frame',False)
        self.frames = self.viz.frames()  
        cam = self.viz.camera()
        cam.setPosition(iDynTree.Position(10.0,0.0,0.5))
        self.viz.camera().animator().enableMouseControl(True)
    
    def add_and_update_model(self, model_name,urdf_path, joints_list, root_link, joint_position, lenght_vector,  delta = 0, ChangeColor = False): 
        mdlLoader = iDynTree.ModelLoader()
        mdlLoader.loadReducedModelFromFile(urdf_path,joints_list, root_link)
        vizModel = mdlLoader.model()
        vizModel.setDefaultBaseLink(vizModel.getLinkIndex(root_link))
        self.viz.addModel(vizModel,model_name)
        
        H_b_vis = np.eye(4)
        H_b_vis[1,3] =H_b_vis[1,3]- delta
        T_b = iDynTree.Transform()
        [s_idyntree,pos,R_iDynTree] = GetIDynTreeTransfMatrix(joint_position, H_b_vis)
        T_b.setPosition(pos)
        T_b.setRotation(R_iDynTree)
        T_b.setPosition(pos)

        if(ChangeColor):
            self.viz.modelViz(model_name).setModelColor(iDynTree.ColorViz(iDynTree.Vector4_FromPython([1,0.2,0.2,0.2])))
        
        self.viz.modelViz(model_name).setPositions(T_b,s_idyntree)
        return

    # As iDynTree visualizer do not have a way to remove a model, we just make it transparent 
    def make_model_transparent(self, model_name):
        # This in theory should work, but in practice it will not work due to https://github.com/robotology/idyntree/issues/966
        # self.viz.modelViz(model_name).setModelColor(iDynTree.ColorViz(iDynTree.Vector4_FromPython([0.0,0.0,0.0,0.0])))
        self.viz.modelViz(model_name).setModelVisibility(False)




def get_joint_name_from_child_link(urdfpy_robot, child_link_name):
    for jnt in urdfpy_robot.joints:
        if jnt.child == child_link_name:
            return jnt.name
    return None

def get_joint_name_from_parent_link(urdfpy_robot, parent_link_name):
    for jnt in urdfpy_robot.joints:
        if jnt.parent == parent_link_name:
            return jnt.name
    return None

def modify_urdf(urdf_path_original, urdf_path_modified, link_groups_list, lenghts_vector): 
    dummy_file =  "./dummy_file.txt"
    robot, gazebo_plugin_text = urdfModifiers.utils.utils.load_robot_and_gazebo_plugins(urdf_path_original, dummy_file)

    # Add link modifiers
    use_new_modifiers = True
    if use_new_modifiers:
        for i in range(len(link_groups_list)):
            for j in range(len(link_groups_list[i])):
                link_name = link_groups_list[i][j]
                # Once the class is available in urf-modifiers, this line will be:
                # link_modifier = urdfModifiers.core.linkAndJointModifier.LinkAndJointModifier.from_link_name(link_name, robot)
                link_modifier = LinkAndJointModifier.from_link_name(link_name, robot)
                modifications = urdfModifiers.core.modification.Modification()
                modifications.add_dimension(lenghts_vector[i], absolute=False)
                link_modifier.modify(modifications)
    else:
        # Copied (with modifications) from https://github.com/ami-iit/element_hardware-intelligence/blob/b8d0f41cf8571ab03b3df74418b9387b6efb5292/Software/NonLinearOptimization/Optimizer.py#L402
        # The only modification is that the joint list is automatically computed as we can compute it from the specified links
        # Furthermore, we remove the density modifications as they are not relevant here
        for i in range(len(link_groups_list)):
            link_name = link_groups_list[i][0]
            link_modifier = urdfModifiers.core.linkModifier.LinkModifier.from_name(link_name, robot, 0.0, dimension=geometry.Side.DEPTH)
            modifications = urdfModifiers.core.modification.modification.Modification()
            modifications.add_dimension(lenghts_vector[i], absolute=False)
            link_modifier.modify(modifications)

        for i in range(len(link_groups_list)):
            joint_name = get_joint_name_from_parent_link(robot, link_groups_list[i][0])
            joint_modifier = urdfModifiers.core.jointModifier.JointModifier.from_name(joint_name, robot, 0.0)
            urdfModifiers.add_dimension(lenghts_vector[i], absolute=False)
            joint_modifier.modify(modifications)     

    urdfModifiers.utils.utils.write_urdf_to_file(robot, urdf_path_modified, gazebo_plugin_text)


originalModel = "original"

urdf_path = "./models/twoLinks/model.urdf"
urdf_path_modified ="./modified_cache.urdf"

## Joint control list (joints of which we control the position)
joints_ctrl_name_list = [
    'joint1', 'joint2'
]

# Parametrized links (group of links of which we are changing the length)
link_groups_list = [['link1'], ['link2']]

# This is just used for visualization, i.e. to make sure that even with the modification, 
# the models have the foot on the ground
root_link = 'root_link'

# Number of random models that we want to visualize
nrOfRandomModels = 1

# Visualizing the output 
VisualizationHelper = OutputVisualization()
VisualizationHelper.prepare_visualization()
joint_positions = np.zeros(len(joints_ctrl_name_list))
joint_offsets = np.zeros(len(joints_ctrl_name_list))
# Add original model to visualziation
VisualizationHelper.add_and_update_model(originalModel, urdf_path, joints_ctrl_name_list, root_link, joint_positions, joint_offsets, 0.25, True)

# Set seed to obtain a deterministic sequence
np.random.seed(0)

# This is to visualize a single model
for i in range(nrOfRandomModels):
    time_now = time.time()
    # Generate random model
    modifiedModelName = "modified" + str(i)
    # Generate random multiplies in the range [0.5,2.0]
    lengths_multiplier = 1.5*np.random.rand(len(link_groups_list)) + 0.5
    # For debug, one may want to set all length multiplier to 1 or constant number
    lengths_multiplier = 3*np.ones(len(link_groups_list))
    modify_urdf(urdf_path, urdf_path_modified, link_groups_list, lengths_multiplier)
    VisualizationHelper.add_and_update_model(modifiedModelName, urdf_path_modified, joints_ctrl_name_list, root_link, joint_positions, joint_offsets, -0.25)

    # Visualize the modified model for some time
    while((time.time()-time_now)<100.5 and VisualizationHelper.viz.run()):
        VisualizationHelper.viz.draw()
        # This should save the image to file, for some reason is just saving 
        # an empty black image, see https://github.com/robotology/idyntree/issues/965

    VisualizationHelper.viz.drawToFile("ProvaView.png")

    # Make the modified model transparent (as the visualizer API does not permit to modify it)
    VisualizationHelper.make_model_transparent(modifiedModelName)

