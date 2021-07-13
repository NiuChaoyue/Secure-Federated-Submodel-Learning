# -*- coding: UTF-8 -*-

import sys
import socket
import time
import ssl
import hmac
import tensorflow as tf
import numpy as np
import random
from config import SSL_CONF as SC
from config import SEND_RECEIVE_CONF as SRC

try:
    import cPickle as pickle
except ImportError:
    import pickle


class Communication(object):

    def __init__(self, is_ps, private_ip, public_ip):
        """Construcs a FederatedHook object
            Args:
              private_ip (str): complete local ip in which the chief is going to
                    serve its socket. Example: 172.134.65.123:7777
              public_ip (str): ip to which the workers are going to connect.
         """
        self._is_ps = is_ps
        self._private_ip = private_ip.split(':')[0]
        self._private_port = int(private_ip.split(':')[1])
        self._public_ip = public_ip.split(':')[0]
        self._public_port = int(public_ip.split(':')[1])
        if self._is_ps:
            self.ps_socket = self.start_socket_ps()

    def start_socket_ps(self):
        """Creates a socket with ssl protection that will act as server.
            Returns:
              sever_socket (socket): ssl secured socket that will act as server.
         """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 地址复用
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1  # optional
        context.set_ciphers('EECDH+AESGCM:EDH+AESGCM:AES256+EECDH:AES256+EDH')
        server_socket.bind((self._private_ip, self._private_port))
        server_socket.listen(1000)
        return server_socket

    def start_socket_client(self):
        """Creates a socket with ssl protection that will act as client.
           Returns:
              sever_socket (socket): ssl secured socket that will work as client.
         """
        to_wrap_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1  # optional

        client_socket = ssl.wrap_socket(to_wrap_socket)
        client_socket.connect((self._public_ip, self._public_port))
        return client_socket

    def receiving_subroutine(self, connection_socket):
        """Subroutine inside _get_np_array to recieve a list of numpy arrays.
        If the sending was not correctly recieved it sends back an error message
        to the sender in order to try it again.
        Args:
          connection_socket (socket): a socket with a connection already
              established.
         """
        timeout = 1.0
        while True:
            ultimate_buffer = b''
            connection_socket.settimeout(1240)
            first_round = True
            while True:
                try:
                    receiving_buffer = connection_socket.recv(SRC.buffer)
                except socket.timeout:
                    print("client time out")
                    break
                except Exception as e:
                    if e.message != 'The read operation timed out':
                        print(e.message)
                    break
                if first_round:
                    connection_socket.settimeout(timeout)
                    first_round = False
                if not receiving_buffer:
                    break
                ultimate_buffer += receiving_buffer

            pos_signature = SRC.hashsize
            signature = ultimate_buffer[:pos_signature]
            message = ultimate_buffer[pos_signature:]
            good_signature = hmac.new(SRC.key, message, SRC.hashfunction).digest()

            if signature != good_signature:
                connection_socket.send(SRC.error)
                timeout += 0.5
                continue
            else:
                connection_socket.send(SRC.recv)
                return message

    def get_np_array(self, connection_socket):
        """Routine to recieve a list of numpy arrays.
            Args:
              connection_socket (socket): a socket with a connection already
                  established.
         """

        message = self.receiving_subroutine(connection_socket)
        final_message = pickle.loads(message)
        return final_message

    def send_np_array(self, arrays_to_send, connection_socket):
        """Routine to send a list of numpy arrays. It sends it as many time as necessary
            Args:
              connection_socket (socket): a socket with a connection already
                  established.
         """
        serialized = pickle.dumps(arrays_to_send)
        signature = hmac.new(SRC.key, serialized, SRC.hashfunction).digest()
        assert len(signature) == SRC.hashsize
        message = signature + serialized
        connection_socket.settimeout(1240)#原来是240
        connection_socket.sendall(message)
        while True:
            check = connection_socket.recv(10)
            if check == SRC.error:
                connection_socket.sendall(message)
            elif check == SRC.recv:
                break
