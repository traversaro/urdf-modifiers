# urdf-modifiers

This library allows to modify a urdf kinematic chain links and joint and creating a new  urdf model out of it. 
The possible modifications are related to : 
- **Mass** (relative and absolute);
- **Density** (relative and absolute);
- **Dimension** (relative and absolute);
- **Radius** (relative and absolute). 

An interface for reading the modifications from a configuration file is provided together with the library (TODO) 

## :hammer: Dependencies

- [`python3`](https://wiki.python.org/moin/BeginnersGuide)

Other requisites are:

- [`urdfpy`](https://github.com/mmatl/urdfpy)
- [`dataclasses`](https://pypi.org/project/dataclasses/)

They will be installed in the installation step!

## :floppy_disk: Installation

Install `python3`, if not installed (in **Ubuntu 20.04**):

```bash
sudo apt install python3.8
```

Clone the repo and install the library:

```bash

pip install "urdf-modifiers @ git+https://github.com/icub-tech-iit/urdf-modifiers"

```

preferably in a [virtual environment](https://docs.python.org/3/library/venv.html#venv-def). For example:

```bash
pip install virtualenv
python3 -m venv your_virtual_env
source your_virtual_env/bin/activate
```

## :rocket: Usage

```python
from urdfModifiers.core.linkModifier import LinkModifier
from urdfModifiers.core.jointModifier import JointModifier
from urdfModifiers import utils

urdf_path ="./models/stickBot/model.urdf"
output_file = "./models/stickBotModified.urdf"

# TODO make a unique utils function out of it 
dummy_file = 'no_gazebo_plugins.urdf'
main_urdf, gazebo_plugin_text = utils.separate_gazebo_plugins(urdf_path)
utils.create_dummy_file(dummy_file, main_urdf)
robot = URDF.load(dummy_file)
utils.erase_dummy_file(dummy_file)

modifications= {}
modifications["dimension"] = [2.0, False] # Relative modification
modifications["density"] = [2.0, False] # Relative modification

modifiers = [LinkModifier.from_name('r_upper_arm',robot, 0.022),
                JointModifier.from_name('r_elbow',robot, 0.0344)]
                
for item in modifiers:
    item.modify(modificationsArms)
utils.write_urdf_to_file(robot, output_file, gazebo_plugin_text)    

```
