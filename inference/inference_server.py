import requests
import os
import logging

from flask import Flask, jsonify
import hydra
from omegaconf import DictConfig

import observation_message_pb2
import control_message_pb2
from logging.config import dictConfig
from flask.logging import default_handler


app = Flask(__name__)

config = None

@app.route('/init', methods=['POST'])
def init():
    return jsonify({"message": "Initialization successful"}), 200

@app.route('/control', methods=['GET'])
def control():
    try:
        if config.invoke_observation_server_url:
            # Replace this URL with the URL of the second system
            second_system_url = config.observation_server_url

            # Make the GET request to the second system
            response = requests.get(second_system_url)
            response.raise_for_status()  # Raise an HTTPError if the response was unsuccessful

            # Deserialize the response to an ObservationMessage
            observation_message = observation_message_pb2.ObservationMessage()
            observation_message.ParseFromString(response.content)
            # TODO: Use checkpoint to establish a control

        control_message = control_message_pb2.ControlMessage()
        control_message.roll = 0.0
        control_message.pitch = 1.0
        control_message.throttle = 60.0

        # TODO: Serialize and return
        response = control_message.SerializeToString()
        # Return a response to the client
        return response, 200, {'Content-Type': 'application/octet-stream'}

    except requests.RequestException as e:
        return jsonify({"status": "error", "error": str(e)}), 500
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@hydra.main(version_base="1.1", config_path=".", config_name="inference_config")
def main(cfg: DictConfig):
    global config
    config = cfg
    log_dir = os.path.dirname(cfg.log_file_path)
    os.makedirs(log_dir, exist_ok=True)
    app.logger.removeHandler(default_handler)

    # Clear all handlers for the root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    logging.basicConfig(
        filename=cfg.log_file_path,
        level=logging.DEBUG,  
    )


    flask_logger = logging.getLogger("flask.app")
    flask_logger.propagate = False
    flask_logger.setLevel(logging.DEBUG)
    flask_logger.handlers.clear()
    flask_logger.addHandler(logging.FileHandler(cfg.log_file_path))

    # Start the Flask server
    app.run(debug=True, host=cfg.inference_host, port=cfg.inference_port)
    

if __name__ == '__main__':
    main()
    
