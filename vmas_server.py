# vmas_server.py
import numpy as np, vmas, zmq, pickle
env = vmas.make_env(scenario="balance", device="cpu")

ctx = zmq.Context()
sock = ctx.socket(zmq.REP)
sock.bind("tcp://127.0.0.1:5557")

while True:
    msg = pickle.loads(sock.recv())
    if msg["cmd"] == "reset":
        obs = env.reset()
        sock.send(pickle.dumps(obs))
    elif msg["cmd"] == "step":
        obs, rew, done, trunc, info = env.step(msg["actions"])
        sock.send(pickle.dumps((obs, rew, done, info)))
