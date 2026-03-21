import mujoco
import mujoco.viewer
import os

XML_PATH = os.path.join(os.path.dirname(__file__), "fourbar.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
