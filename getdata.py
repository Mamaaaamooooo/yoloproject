from roboflow import Roboflow
rf = Roboflow(api_key="Q43msPBBKqxsOGrelQpS")
project = rf.workspace("upul-jayawardana").project("aqua_proj")
dataset = project.version(2).download("yolov8")