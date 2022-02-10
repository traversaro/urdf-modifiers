import time 
import numpy as np

import urdfModifiers
import idyntree.bindings as iDynTree

from dataclasses import dataclass
import math
import numpy as np

from LinkAndJointModifierV1 import LinkAndJointModifierV1

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
                link_modifier = LinkAndJointModifierV1.from_link_name(link_name, robot)
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

