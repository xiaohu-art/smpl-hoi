from active_adaptation.assets.asset_cfg import (
    AssetCfg,
    InitialStateCfg,
    ActuatorCfg,
    ContactSensorCfg,
    RigidObjectCfg
)
from active_adaptation.registry import Registry
from pathlib import Path

registry = Registry.instance()

SMPL = AssetCfg(
    usd_path=Path(__file__).parent / "assets" / "smplh" / "sub12.usda",
    mjcf_path=Path(__file__).parent / "assets" / "smplh" / "sub12.xml",
    self_collisions=False,
    init_state=InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ActuatorCfg(
            joint_names_expr=["L_Hip_.", "R_Hip_.", "L_Knee_.", "R_Knee_.", "L_Ankle_.", "R_Ankle_.", "L_Toe_.", "R_Toe_."],
            effort_limit=500,
            velocity_limit=32.0,
            stiffness={
                "L_Hip_.": 800,
                "R_Hip_.": 800,
                "L_Knee_.": 800,
                "R_Knee_.": 800,
                "L_Ankle_.": 800,
                "R_Ankle_.": 800,
                "L_Toe_.": 500,
                "R_Toe_.": 500,
            },
            damping={
                "L_Hip_.": 80,
                "R_Hip_.": 80,
                "L_Knee_.": 80,
                "R_Knee_.": 80,
                "L_Ankle_.": 80,
                "R_Ankle_.": 80,
                "L_Toe_.": 50,
                "R_Toe_.": 50,
                
            },
            friction=0.01,
            armature=0.01,
        ),
        "torso": ActuatorCfg(
            joint_names_expr=["Torso_.", "Spine_.", "Chest_.", "Neck_.", "Head_.", "L_Thorax_.", "R_Thorax_."],
            effort_limit=500,
            velocity_limit=32.0,
            stiffness={
                "Torso_.": 1000,
                "Spine_.": 1000,
                "Chest_.": 1000,
                "Neck_.": 500,
                "Head_.": 500,
                "L_Thorax_.": 500,
                "R_Thorax_.": 500,
            },
            damping={
                "Torso_.": 100,
                "Spine_.": 100,
                "Chest_.": 100,
                "Neck_.": 50,
                "Head_.": 50,
                "L_Thorax_.": 50,
                "R_Thorax_.": 50,
            },
            friction=0.01,
            armature=0.01,
        ),
        "arms": ActuatorCfg(
            joint_names_expr=["L_Shoulder_.", "R_Shoulder_.", "L_Elbow_.", "R_Elbow_.", "L_Wrist_.", "R_Wrist_."],
            effort_limit=300,
            velocity_limit=32.0,
            stiffness={
                "L_Shoulder_.": 500,
                "R_Shoulder_.": 500,
                "L_Elbow_.": 300,
                "R_Elbow_.": 300,
                "L_Wrist_.": 300,
                "R_Wrist_.": 300,
            },
            damping={
                "L_Shoulder_.": 50,
                "R_Shoulder_.": 50,
                "L_Elbow_.": 30,
                "R_Elbow_.": 30,
                "L_Wrist_.": 30,
                "R_Wrist_.": 30,
            },
            friction=0.01,
            armature=0.01,
        ),
        
        "hands": ActuatorCfg(
            joint_names_expr=[
                "L_Index1_.", "L_Index2_.", "L_Index3_.",
                "L_Middle1_.", "L_Middle2_.", "L_Middle3_.",
                "L_Pinky1_.", "L_Pinky2_.", "L_Pinky3_.",
                "L_Ring1_.", "L_Ring2_.", "L_Ring3_.",
                "L_Thumb1_.", "L_Thumb2_.", "L_Thumb3_.",
                "R_Index1_.", "R_Index2_.", "R_Index3_.",
                "R_Middle1_.", "R_Middle2_.", "R_Middle3_.",
                "R_Pinky1_.", "R_Pinky2_.", "R_Pinky3_.",
                "R_Ring1_.", "R_Ring2_.", "R_Ring3_.",
                "R_Thumb1_.", "R_Thumb2_.", "R_Thumb3_."
            ],
            effort_limit=100,
            velocity_limit=10.0,
            stiffness=100.0,
            damping=10.0,
            friction=0.01,
            armature=0.01,
        ),
    },
    sensors_isaaclab=[
        ContactSensorCfg(
            name="contact_forces",
            primary=".*",
            secondary=[],
            track_air_time=True,
            history_length=3
        )
    ]
)

LARGEBOX = RigidObjectCfg(
    usd_path=Path(__file__).parent / "assets" / "objects" / "largebox" / "largebox.usd",
    activate_contact_sensors=True,
    disable_gravity=False,
)

MONITOR = RigidObjectCfg(
    usd_path=Path(__file__).parent / "assets" / "objects" / "monitor" / "monitor.usd",
    activate_contact_sensors=True,
    disable_gravity=False,
)

PLASTICBOX = RigidObjectCfg(
    usd_path=Path(__file__).parent / "assets" / "objects" / "plasticbox" / "plasticbox.usd",
    activate_contact_sensors=True,
    disable_gravity=False,
)

SMALLTABLE = RigidObjectCfg(
    usd_path=Path(__file__).parent / "assets" / "objects" / "smalltable" / "smalltable.usd",
    activate_contact_sensors=True,
    disable_gravity=False,
)

TRASHCAN = RigidObjectCfg(
    usd_path=Path(__file__).parent / "assets" / "objects" / "trashcan" / "trashcan.usd",
    activate_contact_sensors=True,
    disable_gravity=False,
)

registry.register("asset", "smpl", SMPL)
registry.register("asset", "largebox", LARGEBOX)
registry.register("asset", "monitor", MONITOR)
registry.register("asset", "plasticbox", PLASTICBOX)
registry.register("asset", "smalltable", SMALLTABLE)
registry.register("asset", "trashcan", TRASHCAN)