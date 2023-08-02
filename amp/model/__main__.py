"""Directly calling this module; apparently from another node.
Calls should come as

python -m amp.model id hostname:port

This session will then start a zmq session with that socket, labeling
itself with id. Instructions on what to do will come from the socket.
"""
import os
import sys
import tempfile
import zmq

from ..utilities import MessageDictionary, string2dict, Logger
from .. import importhelper


hostsocket = sys.argv[-1]
proc_id = sys.argv[-2]
msg = MessageDictionary(proc_id)

# Send standard lines to stdout signaling process started and where
# error is directed.
print('<amp-connect>')  # Signal that program started.
if not os.path.exists('tempfiles'):
    os.mkdir('tempfiles')
sys.stderr = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.stderr',
                                         dir="%s/tempfiles" % os.getcwd())
print('Log and stderr written to %s<stderr>' % sys.stderr.name)

# Also send logger output to stderr to aid in debugging.
log = Logger(file=sys.stderr)

# Establish client session via zmq; find purpose.
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://%s' % hostsocket)
socket.send_pyobj(msg('purpose'))
purpose = socket.recv_string()

if purpose == 'calculate_loss_function':
    # Parameters will be sent via a publisher socket; get address.
    socket.send_pyobj(msg('request', 'publisher'))
    publisher_address = socket.recv_pyobj()
    # Request variables.
    socket.send_pyobj(msg('request', 'fortran'))
    fortran = socket.recv_pyobj()
    socket.send_pyobj(msg('request', 'modelstring'))
    modelstring = socket.recv_pyobj()
    dictionary = string2dict(modelstring)
    log(str(dictionary))
    Model = importhelper(dictionary.pop('importname'))
    log('Model received:')
    model = Model(fortran=fortran, **dictionary)
    model.log = log
    log('Model set up.')

    socket.send_pyobj(msg('request', 'args'))
    args = socket.recv_pyobj()
    d = args['d']
    socket.send_pyobj(msg('request', 'lossfunctionstring'))
    lossfunctionstring = socket.recv_pyobj()
    dictionary = string2dict(lossfunctionstring)
    log(str(dictionary))
    LossFunction = importhelper(dictionary.pop('importname'))
    lossfunction = LossFunction(parallel={'cores': 1},
                                raise_ConvergenceOccurred=False,
                                d=d, **dictionary)
    log('Loss function set up.')

    socket.send_pyobj(msg('request', 'images'))
    images = socket.recv_pyobj()
    log('Images received.')

    fingerprints = None
    socket.send_pyobj(msg('request', 'fingerprints'))
    fingerprints = socket.recv_pyobj()
    log('Fingerprints received.')

    fingerprintprimes = None
    socket.send_pyobj(msg('request', 'fingerprintprimes'))
    fingerprintprimes = socket.recv_pyobj()
    log('Fingerprintprimes received.')

    charge_fp_append = None
    socket.send_pyobj(msg('request', 'charge_fp_append'))
    charge_fp_append = socket.recv_pyobj()
    log('Charge fingerprints received.')

    charge_fpprime_append = None
    socket.send_pyobj(msg('request', 'charge_fpprime_append'))
    charge_fpprime_append = socket.recv_pyobj()
    log('Charge fingerprintprimes received.')

    # Set up local loss function.
    model.lossfunction = lossfunction
    lossfunction.attach_model(model,
                              fingerprints=fingerprints,
                              fingerprintprimes=fingerprintprimes,
                              charge_fp_appends=charge_fp_append,
                              charge_fpprime_appends=charge_fpprime_append,
                              images=images)
    log('charge_fp_append is %s' % str(charge_fp_append is not None))
    log('Images, fingerprints, and fingerprintprimes '
        'attached to the loss function.')
    socket.send_pyobj(msg('request', 'args'))
    args = socket.recv_pyobj()

    if model.fortran:
        log('Fortran will be used to evaluate loss function.')
    else:
        log('Fortran will not be used to evaluate loss function.')
    # Now wait for parameters, and send the component of the loss function.
    socket.send_pyobj(msg('setup complete'))
    socket.recv_pyobj()

    log('Establishing subscriber at {}.'.format(publisher_address))
    subscriber = context.socket(zmq.SUB)
    subscriber.connect('tcp://%s' % publisher_address)
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')
    log('Subscriber established.')

    test_message = subscriber.recv_pyobj()
    log('Received a test message. Checking.')
    if test_message == 'test message':
        log('Correct; sending response.')
        socket.send_pyobj(msg('subscriber working'))
        socket.recv_pyobj()
    else:
        raise RuntimeError()

    while True:
        # Drain any test messages from the broadcast.
        test_message = subscriber.recv_pyobj()
        if test_message == 'done':
            break

    while True:
        parameters = subscriber.recv_pyobj()
        if parameters == '<stop>':
            # FIXME/ap: I removed an fmodules.deallocate_variables() call
            # here. Do we need to add this to LossFunction?
            break
        output = lossfunction.get_loss(parameters,
                                       lossprime=args['lossprime'])

        socket.send_pyobj(msg('result', output))
        socket.recv_pyobj()

    socket.close()  # May be needed in python3 / ZMQ.
    subscriber.close()

else:
    raise NotImplementedError('Purpose "%s" unknown.' % purpose)
