import json
import socket

from threading import Lock
from threading import Thread

from src.utils.logger import logger
from src.service.synchronized import safe_exec

class SocketServer():
    # Add the socket timeout
    _SOCKET_TIMEOUT = 10
    
    def __init__(self, server_name, server_port, classifier):
        # Store the parameters
        self.server_name = server_name
        self.server_port = server_port
        self.classifier = classifier
        
        # Set the socker to None
        self.sock = None

        # Define the lock instances for multithreading safery
        self.__ss_lock = Lock()
        self.__is_accept_connections = False
        self.__server_thread = None
    
    def __handle_request(self, connection, client_address):
        logger.info(f'Start handling a new request from client: {client_address}')
        try:
            # Receive the data in small chunks and retransmit it
            req_data = b''
            while self.__is_accept_connections:
                req_chunk_data = connection.recv(1024)
                if req_chunk_data:
                    req_data += req_chunk_data
                else:
                    break
            logger.info(f'Got client: "{client_address}" request data: {req_data}')
                    
            # TODO: Compute the response data
            resp_data = ""
            logger.info(f'Got client: "{client_address}" response data: {resp_data}')
            
            # Send the response data
            connection.sendall(resp_data.encode())
        finally:
            # Clean up the connection
            connection.close()
        
    def __bind_socket_to_port(self):
        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the port
        server_address = (self.server_name, self.server_port)
        logger.info(f'Binding TCP socket to: {server_address}')
        self.sock.bind(server_address)
        
        # Stop listening every N seconds to allow for termination
        self.sock.settimeout(self._SOCKET_TIMEOUT)
        
        # Listen for incoming connections
        self.sock.listen(1)
    
    def __handle_connections(self):
        # First bind socker to port
        self.__bind_socket_to_port()
        
        while self.__is_accept_connections:
            try:
                # Wait for a connection
                logger.info(f'Socket is waiting for a new connection...')
                connection, client_address = self.sock.accept()
                
                # Handle the new connection in a separate daemon thread
                connection = Thread(target = self.__handle_request, args =(connection, client_address), daemon = True)
                connection.start()
            except:
                pass
        
        # Close the socket
        self.sock.close()
        self.sock = None
            
        logger.warning(f'The SockerServer is not accepting new connections, terminating!')
        
    def __start(self):
        if self.__server_thread is None:
            logger.info(f'Starting the ServerSocker thread...')
            self.__server_thread = Thread(target = self.__handle_connections, args =())
            self.__is_accept_connections = True
            self.__server_thread.start()
            logger.info(f'The ServerSocker thread, is started')
        else:
            logger.warning(f'Trying to start the SockerServer but it is already running!')

    def __stop(self):
        if self.__server_thread is not None:
            logger.info(f'Stopping the ServerSocker thread, waiting it to finish (up to {self._SOCKET_TIMEOUT} sec)...')
            self.__is_accept_connections = False
            self.__server_thread.join()
            self.__server_thread = None
            logger.info(f'The ServerSocker thread, has finished, terminating')
        else:
            logger.warning(f'Trying to stop the SockerServer but it is already stopped!')
    
    def start(self):
        safe_exec(self.__ss_lock).execute(self.__start)

    def stop(self):
        safe_exec(self.__ss_lock).execute(self.__stop)
        