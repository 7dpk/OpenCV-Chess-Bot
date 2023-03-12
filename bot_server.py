import socket, threading, os, chess, chess.engine, ssl, multiprocessing, json, requests

SERVER = socket.gethostbyname(socket.gethostname())
# SERVER = requests.get('https://api.ipify.org').text
print(f"Server on internal address: {SERVER}")
PORT = 6751
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

engine = chess.engine.SimpleEngine.popen_uci('./stockfish')
engine.configure({"Threads": multiprocessing.cpu_count() - 1})
engine.configure({"Hash": 8192})
url = "http://tablebase.lichess.ovh/standard"

def no_of_pieces(fen):
	n = 0
	for i in fen:
		if ord(i) in range(97,115) or ord(i) in range(65, 83):
			n += 1
	return n

def handle_client(conn, addr):
	connected = True
	while connected:
		try:
			msg = conn.recv(128).decode(FORMAT)
			try:
				fen, tc, limit = [i.strip() for i in msg.split(',')]
				if no_of_pieces(fen) < 8:
					data = requests.get(url, params={'fen': fen.replace(" ", "_")})
					if data.status_code == 200:
						data = json.loads(data.text)
						move = data['moves'][0]['uci']
						conn.sendall(move.encode(FORMAT))
						print(f"Sent move from cloud API {move} :)")
						continue
			except ValueError:
				print("Sorry Bot misbehaved closing it's connection...")
				break
			if tc == 'depth':
				move = str(engine.play(chess.Board(fen), chess.engine.Limit(depth=limit), ponder=True).move)
			elif tc == 'time':
				move = str(engine.play(chess.Board(fen), chess.engine.Limit(time=float(limit)), ponder=True).move)
			else:
				move = str(engine.play(chess.Board(fen), chess.engine.Limit(time=0.1), ponder=True).move)
			conn.sendall(move.encode(FORMAT))
			print(f"Sent move: {move}")
		except ConnectionResetError:
			print(f"Client: {addr} unexpectedly closed the connection")
			connected = False
	conn.close()
	print(f"Total connections: {threading.activeCount() - 2}")

if __name__ == '__main__':
	server.listen()
	print(f"Server listening on port {PORT}")
	while True:
		conn, addr = server.accept()
		thread = threading.Thread(target = handle_client, args = (conn, addr))
		thread.start()
		print(f"Connected to {addr}")
		print(f"Total connections: {threading.activeCount() - 1}")