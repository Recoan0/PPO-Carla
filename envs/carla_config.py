renderers = [
    dict(type='Road', rgb_color=(128, 64, 128), class_label=7),
    dict(type='Route', rgb_color=(255, 128, 255), class_label=43),
    dict(
        type='Lane',
        rgb_color=(157, 234, 50),
        class_label=6,
        render_lanes_on_junctions=False),
    dict(type='Vehicle', rgb_color=(123, 101, 222), class_label=46),
    dict(type='Agent', rgb_color=(66, 217, 237), class_label=45),
    dict(
        type='GreenTrafficLight',
        rgb_color=(0, 255, 0),
        class_label=41,
        size=2.0),
    dict(
        type='YellowTrafficLight',
        rgb_color=(255, 255, 0),
        class_label=40,
        size=2.0),
    dict(
        type='RedTrafficLight',
        rgb_color=(255, 0, 0),
        class_label=39,
        size=2.0),
    dict(
        type='Pedestrian', rgb_color=(123, 101, 222), class_label=4, size=2.0),
    dict(type='StopSign', rgb_color=(147, 207, 35), class_label=35, size=2.0),
    dict(type='YieldSign', rgb_color=(219, 106, 35), class_label=36, size=2.0),
    dict(
        type='AnimalCrossingSign',
        rgb_color=(108, 173, 23),
        class_label=23,
        size=2.0),
    dict(
        type='LaneReductSign',
        rgb_color=(168, 133, 25),
        class_label=24,
        size=2.0),
    dict(type='NoTurnSign', rgb_color=(168, 66, 25), class_label=25, size=2.0),
    dict(type='OneWaySign', rgb_color=(201, 16, 62), class_label=26, size=2.0),
    dict(
        type='SpeedLimit100Sign',
        rgb_color=(76, 184, 22),
        class_label=34,
        size=2.0),
    dict(
        type='SpeedLimit90Sign',
        rgb_color=(38, 209, 163),
        class_label=33,
        size=2.0),
    dict(
        type='SpeedLimit80Sign',
        rgb_color=(38, 209, 203),
        class_label=32,
        size=2.0),
    dict(
        type='SpeedLimit70Sign',
        rgb_color=(101, 173, 240),
        class_label=31,
        size=2.0),
    dict(
        type='SpeedLimit60Sign',
        rgb_color=(64, 102, 255),
        class_label=30,
        size=2.0),
    dict(
        type='SpeedLimit50Sign',
        rgb_color=(97, 66, 189),
        class_label=29,
        size=2.0),
    dict(
        type='SpeedLimit40Sign',
        rgb_color=(133, 63, 166),
        class_label=28,
        size=2.0),
    dict(
        type='SpeedLimit30Sign',
        rgb_color=(201, 16, 170),
        class_label=27,
        size=2.0),
    dict(type='Obstacle', rgb_color=(66, 135, 245), class_label=38)
]

sensors = [
    dict(
        type='MultiAgentBEV',
        # ppm=12.8,
        ppm=6.,
        renderers=renderers,
        representation='RGB',
        bev_range=dict(front=40, right=20, rear=20, left=20)),
]

randomizer_cfg = dict(
    type='BasicRandomizer',
    obstacles=dict(
        frequency=0.0, mean=0, std=0, xrange=20, yrange=20, min_dist=3.0))

env = dict(
    randomizer=randomizer_cfg,
    sensors=sensors,
    controller=dict(type='Discrete9'))
