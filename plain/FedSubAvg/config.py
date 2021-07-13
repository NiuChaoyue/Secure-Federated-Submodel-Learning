import hashlib

SEND_RECEIVE_CONF = lambda x: x
SEND_RECEIVE_CONF.key = b'4C5jwen4wpNEjBeq1YmdBayIQ1oD'
SEND_RECEIVE_CONF.hashfunction = hashlib.sha1
SEND_RECEIVE_CONF.hashsize = int(160 / 8)
SEND_RECEIVE_CONF.error = b'ERROR'
SEND_RECEIVE_CONF.recv = b'RECEIVED'
SEND_RECEIVE_CONF.signal = b'go!go!go!'
SEND_RECEIVE_CONF.purpose = b'PURPOSE'
SEND_RECEIVE_CONF.init = b'INIT'
SEND_RECEIVE_CONF.update = b'UPDATE'
SEND_RECEIVE_CONF.please_send_update = b'PSU'
SEND_RECEIVE_CONF.please_send_batches_info = b'PSBI'
SEND_RECEIVE_CONF.please_send_model = b'PSM'
SEND_RECEIVE_CONF.heartbeat= b'HEARTBEAT'
SEND_RECEIVE_CONF.buffer = 8192*2

SSL_CONF = lambda x: x
SSL_CONF.key_path = 'server.key'
SSL_CONF.cert_path = 'server.pem'
