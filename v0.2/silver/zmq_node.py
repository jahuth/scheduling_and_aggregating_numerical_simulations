import zmq
import json
import time
import sys, os
from  multiprocessing import Process
import silver
from silver.nodes import ZMQObjectRelay, ZMQFrontendRelay, CoreNode
import uuid


def server(port="5556"):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)
    print "Running server on port: ", port
    # serves only 5 request and dies
    while True:
        # Wait for next request from client
        message = socket.recv()
        socket.send("Not know: {0} [{1}]".format(message,port))
         
def client(ports=["5556"],stdin=None):
    context = zmq.Context()
    print "Connecting to server with ports %s" % ports
    socket = context.socket(zmq.REQ)
    for port in ports:
        socket.connect ("tcp://localhost:%s" % port)
    while True:
        if stdin is not None:
            status = stdin.readline()
        try:
            s = json.loads(status)
            socket.send(status)
            message = socket.recv()
            print "[Out] ", message, ""
        except:
            print "[Err] Not valid json!"
        time.sleep(1) 

         
def zclient(ports=["5556"],stdin=None):
    rel = ZMQFrontendRelay(ports)
    rel.Experiment("Experiments/tuning_curve.py", "some_experiment")
    while True:
        if stdin is not None:
            status = stdin.readline()
        try:
            s = json.loads(status)
            socket.send(status)
            message = socket.recv()
        except:
            print "Not valid json!"
        time.sleep(1) 

if __name__ == "__main__":
    # Now we can run a few servers 
    #server_ports = range(5550,5558,2)
    #for server_port in server_ports:
    #    Process(target=server, args=(server_port,)).start()
    newstdin = os.fdopen(os.dup(sys.stdin.fileno()))
    #z = silver.nodes.SessionNode('A Whole New Test.json')
    node_name = "core"
    argv = {}
    for k,v in enumerate(sys.argv):
        argv[k] = v
    node_name = argv.get(1,"core")
    if node_name =="core":
        z = silver.nodes.CoreNode()
        new_port = argv.get(2,None)
    elif node_name =="session":
        z = silver.nodes.SessionNode(argv.get(2,None))
        new_port = argv.get(3,None)
    elif node_name =="runner":
        z = silver.nodes.RunnerNode(argv.get(2,None),argv.get(3,None))
        new_port = argv.get(4,None)
    else:
        print "unrecognized node type:",node_name
        #z = silver.nodes.CoreNode()
        #new_port = argv.get(2,None)
    port = 5556
    try:
        assert(new_port is not None)
        port = int(new_port)
        assert(port > 1000)
    except:
        port = 5556
    z.createProcesses(str(port))
    while z.port is None:
        time.sleep(0.01)
    print "Server open.\n\nPort:"+str(z.port)+"\n"
    # Now we can connect a client to all these servers
    if not 'nolisten' in sys.argv:
        Process(target=client, args=([str(z.port)],newstdin)).start()


