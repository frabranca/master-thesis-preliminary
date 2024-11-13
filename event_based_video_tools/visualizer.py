from multiprocessing import Process

import samna
import samnagui


def open_speck2e():
    return samna.device.open_device("Speck2eDevKit:0")


def build_samna_event_route(dk, dvs_graph, streamer_endpoint):
    # build a graph in samna to show dvs
    _, _, streamer = dvs_graph.sequential(
        [dk.get_model_source_node(), "Speck2eDvsToVizConverter", "VizEventStreamer"]
    )

    config_source, _ = dvs_graph.sequential(
        [samna.BasicSourceNode_ui_event(), streamer]
    )

    streamer.set_streamer_endpoint(streamer_endpoint)
    if streamer.wait_for_receiver_count() == 0:
        raise Exception(f"connecting to visualizer on {streamer_endpoint} fails")

    return config_source


def open_visualizer(window_width, window_height, receiver_endpoint):
    # start visualizer in a isolated process which is required on mac, intead of a sub process.
    gui_process = Process(
        target=samnagui.run_visualizer,
        args=(receiver_endpoint, window_width, window_height),
    )
    gui_process.start()

    return gui_process


streamer_endpoint = "tcp://0.0.0.0:40000"

gui_process = open_visualizer(0.75, 0.75, streamer_endpoint)

dk = open_speck2e()

stopWatch = dk.get_stop_watch()
stopWatch.set_enable_value(True)  # open timestamp

# route events
dvs_graph = samna.graph.EventFilterGraph()
config_source = build_samna_event_route(dk, dvs_graph, streamer_endpoint)

dvs_graph.start()

config_source.write(
    [
        samna.ui.VisualizerConfiguration(
            plots=[samna.ui.ActivityPlotConfiguration(128, 128, "DVS Layer")]
        )
    ]
)

# modify configuration
config = samna.speck2e.configuration.SpeckConfiguration()
config.dvs_layer.monitor_enable = True
dk.get_model().apply_configuration(config)

# wait until visualizer window destroys.
gui_process.join()

dvs_graph.stop()